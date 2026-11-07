"""C3 v0 harness: one whole decode step of vLLM 0.32's V2 runner (input prep, attention metadata, target forward, head,
rejection, post-update, DSpark drafter) captured as ONE CUDA graph and replayed back to back for one live request (c=1,
K=5, R=6). Every input of the graph is device state written by the previous replay (num_computed_tokens,
last_sampled_tokens, draft_tokens, all_token_ids), so per step the host only launches the graph.

How: vLLM's own execute_model + sample_tokens run under torch.cuda.graph with
  - the target and the drafter forced onto their eager (NONE) path, so both forwards are captured inline (a FULL-graph
    replay cannot be captured inside another capture);
  - the per-step host->device copies (idx_mapping, cu_num_logits, query_start_loc) served from persistent device
    tensors filled by an eager step before the capture (constant for a fixed batch);
  - the host's sequence-length upper bound set to max_model_len, so metadata is sized for the worst case as vLLM's own
    FULL capture does;
  - the async D2H output copy stubbed out (the harness reads tokens from the device);
  - blocks pre-allocated for the whole run, so block tables are constant.
Test: greedy tokens of the replay loop == vLLM's normal decode of the same prompt; then per-step time, synced and
launch-ahead.
    python3 whole_step.py OUT [--tokens N] [--steps S]"""
import argparse, contextlib, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session  # noqa: E402

PROMPT = ("Write a Python module that parses a CSV file of sales records (date, region, product, units, price), "
          "computes monthly revenue per region, and prints a formatted report. Include type hints and docstrings.")


class StepPatches:
    """Patches that make execute_model + sample_tokens capturable for a fixed decode batch (see the module doc)."""

    def __init__(self, runner):
        import vllm.v1.worker.gpu.model_runner as mr
        import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dfs
        self.runner, self.mr, self.dfs = runner, mr, dfs
        self.h2d = []            # persistent device tensors, in call order
        self.mode = "record"     # record (eager step, allocate) | replay (capture, reuse)
        self.i = 0

    def _h2d(self, orig):
        import torch

        def f(x, device=None, dtype=None, out=None, **kw):
            if out is not None:          # query_start_loc: a persistent buffer already
                if self.mode == "record":
                    return orig(x, device=device, dtype=dtype, out=out, **kw)
                return out
            if self.mode == "record":
                t = orig(x, device=device, dtype=dtype, **kw)
                self.h2d.append(t)
                return t
            t = self.h2d[self.i]
            self.i += 1
            return t
        return f

    @contextlib.contextmanager
    def active(self, mode):
        import torch
        mr, dfs, runner = self.mr, self.dfs, self.runner
        self.mode, self.i = mode, 0
        saved = dict(h2d=mr.async_tensor_h2d, disp=mr.dispatch_cg_and_sync_dp, ddisp=dfs.dispatch_cg_and_sync_dp,
                     ao=mr.AsyncOutput, prep=runner.prepare_inputs)
        max_len = runner.max_model_len

        def eager(orig):
            def f(*a, **k):
                k["need_eager"] = True
                return orig(*a, **k)
            return f

        class NoCopyOutput:
            def __init__(self, model_runner_output, sampler_output, num_sampled_tokens, *a, **k):
                self.model_runner_output, self.sampler_output = model_runner_output, sampler_output
                self.num_sampled_tokens = num_sampled_tokens

        def prep(*a, **k):
            ib = saved["prep"](*a, **k)
            ib.seq_lens_cpu_upper_bound[: ib.num_reqs] = max_len    # worst-case metadata sizes, as a FULL capture
            return ib

        mr.async_tensor_h2d = self._h2d(saved["h2d"])
        mr.dispatch_cg_and_sync_dp = eager(saved["disp"])
        dfs.dispatch_cg_and_sync_dp = eager(saved["ddisp"])
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


@contextlib.contextmanager
def force_eager():
    """vLLM's normal step, target and drafter on their eager (NONE) cudagraph route (no FULL replays)"""
    import vllm.v1.worker.gpu.model_runner as mr
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dfs
    saved = (mr.dispatch_cg_and_sync_dp, dfs.dispatch_cg_and_sync_dp)

    def eager(orig):
        def f(*a, **k):
            k["need_eager"] = True
            return orig(*a, **k)
        return f
    mr.dispatch_cg_and_sync_dp, dfs.dispatch_cg_and_sync_dp = eager(saved[0]), eager(saved[1])
    try:
        yield
    finally:
        mr.dispatch_cg_and_sync_dp, dfs.dispatch_cg_and_sync_dp = saved


def engine_parts(llm):
    core = llm.llm_engine.engine_core.engine_core
    return core, core.scheduler


class LogitTap:
    """Records each verify step's logits rows with their positions and the accepted count. On the normal path the
    rows are recomputed eagerly after vLLM's sample(); under capture the same calls land in the graph, so after each
    replay the graph-pool tensors hold that replay's rows. collect() returns {position: fp32 row} for the accepted
    rows, whose inputs are the true sequence (row i of a step sees the bonus token and drafts 1..i)."""

    def __init__(self, runner):
        self.runner, self.orig, self.cur, self.on = runner, runner.sample, None, True
        runner.sample = self._sample

    def _sample(self, hidden_states, input_batch, grammar_output):
        so, ns, nr = self.orig(hidden_states, input_batch, grammar_output)
        if self.on and input_batch.num_draft_tokens > 0:
            rows = input_batch.logits_indices
            self.cur = (self.runner.model.compute_logits(hidden_states[rows]), input_batch.positions[rows], ns)
        return so, ns, nr

    def collect(self):
        if self.cur is None:
            return {}
        lg, pos, ns = self.cur
        n = int(ns[0].item())
        rows, ps = lg[:n].float().cpu(), pos[:n].cpu().tolist()
        return {int(p): rows[i] for i, p in enumerate(ps)}


def compare_rows(a_rows, b_rows, upto):
    """KLD(A||B), max |dlogit|, argmaxes and A's top-2 gap at the positions both runs computed on the same prefix"""
    import torch
    out = []
    for p in sorted(set(a_rows) & set(b_rows)):
        if p > upto:
            break
        a, b = a_rows[p].double(), b_rows[p].double()
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        top2 = torch.topk(a, 2).values
        out.append(dict(pos=p, kld=float((la.exp() * (la - lb)).sum()), max_dlogit=float((a - b).abs().max()),
                        argmax_a=int(a.argmax()), argmax_b=int(b.argmax()), gap_a=float(top2[0] - top2[1]),
                        bit_identical=bool(torch.equal(a_rows[p], b_rows[p]))))
    return out


def run_to_end(llm, rid, prompt, n, tap=None):
    """vLLM's normal path: greedy, ignore_eos, n tokens (tap: a LogitTap collecting every step's rows)"""
    from vllm import SamplingParams
    eng = llm.llm_engine
    eng.add_request(rid, prompt, SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    out = None
    t0 = time.time()
    steps = []      # (seconds, tokens so far) per engine step: the normal path's per-step time
    n_now = 0
    rows = {}
    while eng.has_unfinished_requests():
        ts = time.perf_counter()
        for o in eng.step():
            if o.request_id == rid:
                n_now = len(o.outputs[0].token_ids)
                if o.finished:
                    out = list(o.outputs[0].token_ids)
        steps.append((time.perf_counter() - ts, n_now))
        if tap is not None:
            rows.update(tap.collect())
            tap.cur = None
    run_to_end.steps = steps
    run_to_end.rows = rows
    return out, time.time() - t0


def start_and_stop(llm, runner, rid, prompt, n, decode_steps=3):
    """vLLM's normal path until the request has run `decode_steps` decode steps; returns the tokens so far and a
    deep copy of the last decode SchedulerOutput"""
    from vllm import SamplingParams
    eng = llm.llm_engine
    stash = {}
    orig = runner.execute_model

    def em(scheduler_output, *a, **k):
        if scheduler_output.scheduled_spec_decode_tokens and iid[0] in scheduler_output.num_scheduled_tokens:
            stash["so"] = copy.deepcopy(scheduler_output)
            stash["n"] = stash.get("n", 0) + 1
        return orig(scheduler_output, *a, **k)
    runner.execute_model = em
    iid = [None]
    iid[0] = eng.add_request(rid, prompt, SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    toks = []
    try:
        while stash.get("n", 0) < decode_steps:
            for o in eng.step():
                if o.request_id == rid:
                    toks = list(o.outputs[0].token_ids)
    finally:
        runner.execute_model = orig
    return toks, stash["so"], iid[0]   # iid: the engine-internal request id (runner and scheduler key)


def preallocate(llm, runner, rid, n_ahead):
    """allocate KV blocks for n_ahead more positions through the scheduler's manager and hand them to the runner"""
    core, sched = engine_parts(llm)
    req = sched.requests[rid]
    mgr = sched.kv_cache_manager
    before = [list(g) for g in mgr.get_block_ids(rid)]
    blocks = mgr.allocate_slots(req, n_ahead)
    assert blocks is not None, "allocate_slots failed (KV cache full?)"
    after = mgr.get_block_ids(rid)
    new = tuple(list(a[len(b):]) for a, b in zip(after, before))
    idx = runner.req_states.req_id_to_index[rid]
    runner.block_tables.append_block_ids(idx, new, overwrite=False)
    zero = mgr.take_new_block_ids() if hasattr(mgr, "take_new_block_ids") else [b for g in new for b in g]
    if zero and runner.kv_block_zeroer is not None:
        runner.kv_block_zeroer.zero_block_ids(zero)
    runner.block_tables.apply_staged_writes()
    return [len(g) for g in new]


def decode_sched_output(so, rid, runner):
    """the stashed decode SchedulerOutput turned into a steady one: no new blocks, no finished/new requests"""
    import numpy as np
    s = copy.deepcopy(so)
    s.scheduled_new_reqs = []
    c = s.scheduled_cached_reqs
    keep = [i for i, r in enumerate(c.req_ids) if r == rid]
    assert keep, "request not in the stashed output"
    i = keep[0]
    c.req_ids = [rid]
    c.resumed_req_ids = set()
    c.new_token_ids = [] if not c.new_token_ids else [c.new_token_ids[i]]
    c.all_token_ids = {}
    c.new_block_ids = [None]
    nc = int(runner.req_states.num_computed_tokens.gpu[runner.req_states.req_id_to_index[rid]].item())
    c.num_computed_tokens = [nc]
    c.num_output_tokens = [c.num_output_tokens[i]] if c.num_output_tokens else []
    s.num_scheduled_tokens = {rid: so.num_scheduled_tokens[rid]}
    s.total_num_scheduled_tokens = so.num_scheduled_tokens[rid]
    s.scheduled_spec_decode_tokens = {rid: [-1] * len(so.scheduled_spec_decode_tokens[rid])}
    s.finished_req_ids = set()
    s.preempted_req_ids = None
    s.new_block_ids_to_zero = None
    s.kv_cache_block_copies = None
    s.free_encoder_mm_hashes = []
    return s


class WholeStep:
    def __init__(self, llm, runner, rid, so):
        import torch
        self.llm, self.runner, self.rid, self.so = llm, runner, rid, so
        self.p = StepPatches(runner)
        self.idx = runner.req_states.req_id_to_index[rid]
        self.graph = None
        self.out = None

    def _step(self):
        r = self.runner
        r.execute_model(self.so)
        return r.sample_tokens(None)

    def eager_then_capture(self, on_eager=None):
        """As vLLM's own FULL capture: inside one graph_capture context (which makes vLLM's current_stream() the
        capture stream), first one real eager step on that stream (it allocates the persistent H2D tensors and warms
        the per-stream state: DeepGEMM keeps a workspace per stream and refuses to create one under capture), then the
        capture on the same stream. on_eager(self) runs after the eager step (e.g. to read its tokens)."""
        import torch
        from vllm.distributed.parallel_state import graph_capture
        from vllm.utils.torch_utils import current_stream
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with graph_capture(device=self.runner.device):
            s = current_stream()
            with self.p.active("record"):
                self.out = self._step()
            torch.cuda.synchronize()
            if on_eager is not None:
                on_eager(self)
            t0 = time.time()
            with self.p.active("replay"), torch.cuda.graph(g, stream=s):
                ao = self._step()
            self.capture_s = time.time() - t0
        torch.cuda.synchronize()
        self.graph, self.out = g, ao

    def tokens(self):
        """this step's accepted tokens (host sync)"""
        so = self.out.sampler_output
        n = int(self.out.num_sampled_tokens[0].item())
        return so.sampled_token_ids[0, :n].tolist()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--timing-steps", type=int, default=200)
    ap.add_argument("--profile", type=int, default=20, help="replays under the torch profiler for a per-kernel table (0: off)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = {}

    def save():
        json.dump(rep, open(os.path.join(a.out, "whole_step.json"), "w"), indent=1)

    llm, runner = session.load(async_scheduling=False)
    tap = LogitTap(runner)
    # A: vLLM's normal path, the whole generation (verify rows recorded)
    ref, dt = run_to_end(llm, "A", PROMPT, a.tokens, tap=tap)
    a_rows = run_to_end.rows
    st = run_to_end.steps[2:]                    # past the prefill and the first decode step
    st_ms = sorted(s * 1e3 for s, _ in st)
    tps = (st[-1][1] - st[0][1]) / max(len(st) - 1, 1) if len(st) > 1 else 0
    rep["A"] = dict(tokens=len(ref), seconds=round(dt, 2), tok_s=round(len(ref) / dt, 1), steps=len(run_to_end.steps),
                    step_ms_median=round(st_ms[len(st_ms) // 2], 3) if st_ms else None,
                    step_ms_mean=round(sum(st_ms) / len(st_ms), 3) if st_ms else None, tokens_per_step=round(tps, 2))
    print(f"A (vLLM normal path): {len(ref)} tokens in {dt:.2f} s = {len(ref) / dt:.1f} tok/s; "
          f"{len(run_to_end.steps)} engine steps, median {rep['A']['step_ms_median']} ms, {tps:.2f} tokens/step",
          flush=True)
    save()
    # C: vLLM's normal path forced onto the eager cudagraph route (what the captured step records), rows recorded
    with force_eager():
        ref_c, dt_c = run_to_end(llm, "C", PROMPT, a.tokens, tap=tap)
    c_rows = run_to_end.rows
    nc_ = min(len(ref), len(ref_c))
    fd_c = next((i for i in range(nc_) if ref[i] != ref_c[i]), None)
    rep["C"] = dict(tokens=len(ref_c), first_diff_vs_A=fd_c, tok_s=round(len(ref_c) / dt_c, 1))
    print(f"C (vLLM normal path, eager route): first diff vs A at {fd_c}", flush=True)
    save()
    # B: same prompt, vLLM for 3 decode steps, then the captured whole step
    toks, so, iid = start_and_stop(llm, runner, "B", PROMPT, a.tokens + 64)
    print(f"B: {len(toks)} tokens from vLLM before the takeover; matches A: {toks == ref[:len(toks)]}", flush=True)
    rep["B_prefix"] = dict(n=len(toks), match=toks == ref[:len(toks)])
    newb = preallocate(llm, runner, iid, a.tokens + 6 * (a.timing_steps + 8) + 64)
    rep["preallocated_blocks"] = newb
    sso = decode_sched_output(so, iid, runner)
    ws = WholeStep(llm, runner, iid, sso)
    # one eager step through the patched path, then the capture (same stream)

    b_rows = {}

    def on_eager(w):
        t1 = w.tokens()
        toks.extend(t1)
        b_rows.update(tap.collect())
        print(f"eager patched step: +{len(t1)} tokens, match so far: {toks == ref[:len(toks)]}", flush=True)
    ws.eager_then_capture(on_eager)
    print(f"captured the whole step in {ws.capture_s:.1f} s", flush=True)
    rep["capture_s"] = round(ws.capture_s, 1)
    # correctness: replay with a sync per step until the reference length
    steps = 0
    while len(toks) < len(ref):
        ws.graph.replay()
        torch.cuda.synchronize()
        toks += ws.tokens()
        b_rows.update(tap.collect())
        steps += 1
    n = min(len(toks), len(ref))
    first_diff = next((i for i in range(n) if toks[i] != ref[i]), None)
    P = engine_parts(llm)[1].requests[iid].num_prompt_tokens
    upto = P + (first_diff if first_diff is not None else n) - 1     # the row that predicts the first differing token
    for name, x, y, u in (("A-B", a_rows, b_rows, upto),
                          ("A-C", a_rows, c_rows, P + (fd_c if fd_c is not None else nc_) - 1),
                          ("C-B", c_rows, b_rows, P + next((i for i in range(min(len(toks), len(ref_c)))
                                                             if toks[i] != ref_c[i]), min(len(toks), len(ref_c))) - 1)):
        cmp = compare_rows(x, y, u)
        rep["rows_" + name] = cmp
        print(f"{name}: {len(cmp)} common rows", flush=True)
        for c in cmp[:24]:
            print(f"  {name} pos {c['pos']}: KLD {c['kld']:.3e}, max|dlogit| {c['max_dlogit']:.4f}, argmax "
                  f"{c['argmax_a']}/{c['argmax_b']}, top-2 gap {c['gap_a']:.4f}, bit-identical {c['bit_identical']}",
                  flush=True)
    tk = llm.get_tokenizer()
    rep["text_A"], rep["text_B"] = tk.decode(ref[:80]), tk.decode(toks[:80])
    print("A text:", repr(rep["text_A"][:300]), "\nB text:", repr(rep["text_B"][:300]), flush=True)
    rep["B"] = dict(tokens=n, steps=steps, tokens_per_step=round((n - rep["B_prefix"]["n"]) / max(steps, 1), 2),
                    identical=first_diff is None, first_diff=first_diff)
    print(f"B (captured whole step): {n} tokens, {steps} replays, identical to A: {first_diff is None}"
          f" (first diff {first_diff})", flush=True)
    save()
    # timing on a second capture without the logit tap (the tap adds an LM head per step)
    tap.on = False
    ws = WholeStep(llm, runner, iid, sso)
    ws.eager_then_capture()
    # timing: S replays back to back (launch-ahead: no sync between steps), tokens read at the end
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    e0.record()
    for _ in range(a.timing_steps):
        ws.graph.replay()
    e1.record()
    t_launch = time.perf_counter() - t0
    torch.cuda.synchronize()
    gpu_ms = e0.elapsed_time(e1) / a.timing_steps
    rep["timing"] = dict(steps=a.timing_steps, ms_per_step=round(gpu_ms, 3),
                         host_launch_ms_per_step=round(t_launch * 1e3 / a.timing_steps, 3))
    print(f"launch-ahead: {gpu_ms:.3f} ms per step on the GPU, {t_launch * 1e3 / a.timing_steps:.3f} ms host launch "
          f"per step", flush=True)
    save()
    if os.environ.get("DSV41_NSYS") == "1":
        # nsys --capture-range=cudaProfilerApi: 20 back-to-back replays, nothing else, inside the capture range
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(20):
            ws.graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print("NSYS RANGE DONE: 20 replays", flush=True)
    if a.profile:
        # per-kernel GPU time of the captured step (CUPTI through the torch profiler), summed over the replays
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.profile):
                ws.graph.replay()
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            if t:
                rows.append((t / a.profile, ev.count / a.profile, ev.key))
        rows.sort(reverse=True)
        tot = sum(r[0] for r in rows)
        rep["profile"] = dict(replays=a.profile, kernel_us_per_step=round(tot, 1),
                              top=[dict(us=round(u, 1), calls=round(c, 1), name=n[:160]) for u, c, n in rows[:60]])
        print(f"profile: {tot / 1e3:.3f} ms of kernel time per step (sum over streams); top kernels:", flush=True)
        for u, c, n in rows[:25]:
            print(f"  {u:9.1f} us  {c:6.1f} calls  {n[:120]}", flush=True)
        save()


if __name__ == "__main__":
    main()
