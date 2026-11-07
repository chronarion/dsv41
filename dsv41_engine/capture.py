"""Capturing vLLM 0.32's own decode step as one CUDA graph (C3; engine/harness/whole_step.py is the test harness).

StepPatches makes the V2 runner's execute_model + sample_tokens capturable for a fixed decode batch:
  - target and drafter take their eager (NONE) cudagraph path, so both forwards are captured inline (a FULL-graph
    replay cannot be captured inside another capture);
  - the per-step host->device copies of prepare_inputs (idx_mapping, cu_num_logits, query_start_loc) come from
    persistent device tensors created by an eager "record" step and refreshed by the owner when the batch changes;
  - the host's sequence-length upper bound is set to max_model_len (metadata sized for the worst case, as vLLM's own
    FULL capture does);
  - in "replay" (capture) mode the async D2H output copy is replaced by a stub that only keeps the output tensors.
The eager record step and the capture must run on the same stream inside one vllm graph_capture() context: DeepGEMM
keeps a sparse-MQA workspace per stream and refuses to create one under capture."""
import contextlib


class NoCopyOutput:
    """AsyncOutput stand-in during capture: keeps the device tensors the graph writes, copies nothing"""

    def __init__(self, model_runner_output, sampler_output, num_sampled_tokens, *a, **k):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens


class StepPatches:
    def __init__(self, runner):
        import vllm.v1.worker.gpu.model_runner as mr
        import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dfs
        self.runner, self.mr, self.dfs = runner, mr, dfs
        self.h2d = []            # persistent device tensors, in call order
        self.h2d_np = []         # their host values at record time (None for the out= buffer)
        self.mode = "record"
        self.i = 0

    def _h2d(self, orig):
        def f(x, device=None, dtype=None, out=None, **kw):
            if out is not None:
                # query_start_loc: a shared input buffer that only the host's eager prepare_inputs writes. A graph must
                # not trust what the last eager step left there (a stock 32-row suffix step leaves [0, 32, ...] under
                # the 1x5 graph, whose logits gather then reads rows 26..31 of a 6-row tensor), so the record step
                # keeps a graph-owned copy of this key's values (fixed by the key: n requests x K+1 rows) and the
                # captured graph restores the buffer from it before anything reads it.
                if self.mode == "record":
                    r = orig(x, device=device, dtype=dtype, out=out, **kw)
                    self.qsl = orig(x.copy(), device=out.device, dtype=out.dtype, **kw)
                    return r
                out.copy_(self.qsl)
                return out
            if self.mode == "record":
                t = orig(x, device=device, dtype=dtype, **kw)
                self.h2d.append(t)
                self.h2d_np.append(x)
                return t
            t = self.h2d[self.i]
            self.i += 1
            return t
        return f

    @contextlib.contextmanager
    def active(self, mode, stub_output=None):
        """mode: record (a real eager step that allocates the persistent tensors) | replay (under capture)"""
        mr, dfs, runner = self.mr, self.dfs, self.runner
        self.mode, self.i = mode, 0
        if stub_output is None:
            stub_output = mode == "replay"
        saved = dict(h2d=mr.async_tensor_h2d, disp=mr.dispatch_cg_and_sync_dp, ddisp=dfs.dispatch_cg_and_sync_dp,
                     ao=mr.AsyncOutput, prep=runner.prepare_inputs)
        max_len = runner.max_model_len

        def eager(orig):
            def f(*a, **k):
                k["need_eager"] = True
                return orig(*a, **k)
            return f

        def prep(*a, **k):
            ib = saved["prep"](*a, **k)
            ib.seq_lens_cpu_upper_bound[: ib.num_reqs] = max_len
            return ib

        mr.async_tensor_h2d = self._h2d(saved["h2d"])
        mr.dispatch_cg_and_sync_dp = eager(saved["disp"])
        dfs.dispatch_cg_and_sync_dp = eager(saved["ddisp"])
        if stub_output:
            mr.AsyncOutput = NoCopyOutput
        runner.prepare_inputs = prep
        try:
            yield
        finally:
            mr.async_tensor_h2d, mr.dispatch_cg_and_sync_dp = saved["h2d"], saved["disp"]
            dfs.dispatch_cg_and_sync_dp, mr.AsyncOutput = saved["ddisp"], saved["ao"]
            runner.prepare_inputs = saved["prep"]
            if mode == "replay":
                assert self.i == len(self.h2d), (self.i, len(self.h2d))


def steady_scheduler_output(so, runner):
    """a decode SchedulerOutput with the same requests and token counts but no state changes (no new / finished /
    preempted requests, no new blocks, nothing to zero or copy): what the capture pass replays every step"""
    import copy
    s = copy.deepcopy(so)
    s.scheduled_new_reqs = []
    c = s.scheduled_cached_reqs
    n = len(c.req_ids)
    c.resumed_req_ids = set()
    c.new_token_ids = [] if not c.new_token_ids else c.new_token_ids
    c.all_token_ids = {}
    c.new_block_ids = [None] * n
    s.finished_req_ids = set()
    s.preempted_req_ids = None
    s.new_block_ids_to_zero = None
    s.kv_cache_block_copies = None
    s.free_encoder_mm_hashes = []
    return s
