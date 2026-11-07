"""C4: vLLM 0.32's V2 GPU model runner with decode-only steps served from a captured whole-step CUDA graph.

A decode-only batch (every request verifies K drafts, no prefill, no logprobs, no logits processors or penalties, no
structured output) is keyed by (requests, K, the sampler's host-side branches). The first step of a key runs for real
through the capturable path on a fresh capture stream (dsv41_engine.capture), then the same step is captured with a
steady scheduler output. Later steps do the stock host bookkeeping (finished / added requests, block appends and
zeroing), refresh the graph's persistent batch constants and sampling parameters when they change, and replay the
graph: target forward, LM head, rejection, post-update and the DSpark drafter in one launch. Everything else runs the
stock path. DSV41_WHOLE_STEP=0 disables it."""
import os
import sys
import time

import numpy as np
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids

from .capture import StepPatches, steady_scheduler_output

_LOG = lambda m: sys.stderr.write(f"DSV41_WHOLE_STEP {m}\n")
_PARAMS = ("temperature", "top_k", "top_p", "min_p", "seeds")
# vLLM's WorkspaceManager frees its buffer and allocates a larger one whenever a step needs more (the V2 runner never
# locks it), so a graph captured before a growth keeps pointers into freed memory: the first wide (32-row) suffix
# step grew it and the 1x5 graph's next replay read garbage (an out-of-range gather). Each growth bumps this
# generation; a graph captured under an older one is retired and captured again on its next use.
_WS_GEN = [0]


def _install_workspace_hook():
    from vllm.v1.worker import workspace as wsm
    cls = wsm.WorkspaceManager
    if getattr(cls, "_dsv41_gen_hook", False):
        return
    orig = cls._ensure_workspace_size

    def _ensure_workspace_size(self, required_bytes):
        cur = self._current_workspaces[self._get_workspace_id()]
        before = self._workspace_size_bytes(cur)
        out = orig(self, required_bytes)
        if before < required_bytes:
            _WS_GEN[0] += 1
            _LOG(f"workspace grew {before / 2**20:.1f} -> {required_bytes / 2**20:.1f} MB (generation {_WS_GEN[0]}): "
                 "graphs captured before it are retired")
        return out
    cls._ensure_workspace_size = _ensure_workspace_size
    cls._dsv41_gen_hook = True


def _so_width(so, w):
    """K3b: the scheduler output of a batch-1 step with its drafts cut to the first w (a w-draft graph is captured
    from it; the scheduler itself scheduled K, and sees the drafts beyond w as rejected)"""
    import copy
    s = copy.copy(so)
    s.scheduled_spec_decode_tokens = {r: list(t)[:w] for r, t in so.scheduled_spec_decode_tokens.items()}
    s.num_scheduled_tokens = {r: min(n, w + 1) for r, n in so.num_scheduled_tokens.items()}
    s.total_num_scheduled_tokens = sum(s.num_scheduled_tokens.values())
    return s


class _Graph:
    def __init__(self, key):
        self.key = key
        self.graph = None
        self.patches = None
        self.sampler_output = None
        self.num_sampled = None
        self.consts_np = None    # host values of the persistent H2D tensors
        self.replays = 0
        self.gen = None          # workspace generation the graph was captured under


class WholeStepRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self._ws_on = os.environ.get("DSV41_WHOLE_STEP", "1") == "1"
        self._ws_maxk = int(os.environ.get("DSV41_WS_MAXK", "1000000"))
        self._ws_dbg = os.environ.get("DSV41_WS_DEBUG") == "1"
        # K3 measurement: DSV41_CONF_LOG=path logs, per batch-1 replay, the verified draft width, the
        # tokens the step emitted and the max draft probability of each drafted position (the drafts the next step
        # verifies); a pinned ring, written out at each 30-s log. Off by default (it adds a softmax per step).
        self._conf_path = os.environ.get("DSV41_CONF_LOG")
        self._conf = None
        # K3b (dsv41_engine.conf_width, DSV41_CONF_WIDTH_MODE=graphs): a batch-1 step scheduled with K drafts replays
        # the graph captured for the width W the previous step's drafter confidences chose (W = 1..K): the host
        # bookkeeping runs first (it does not depend on W), then the runner waits for the previous step and reads W
        self._k3b = (os.environ.get("DSV41_CONF_WIDTH_MODE") == "graphs"
                     and float(os.environ.get("DSV41_CONF_WIDTH", "0") or 0) > 0)
        self._k3b_k = int(os.environ.get("DSV41_DSPARK_K", "7"))
        self._k3b_hist = [0] * (self._k3b_k + 1)
        self._k3b_wait = 0.0
        self._k3b_ahead = os.environ.get("DSV41_K3B_AHEAD", "1") == "1"
        self._k3b_run = 0         # consecutive full widths read from the word
        self._k3b_seen = 0        # the sequence last read from the width word
        self._k3b_pending = 0     # batch-1 replays launched since (each one's drafter advances the sequence by one)
        self._k3b_fast = False    # every step since the last read was such a replay: spin on the word alone
        self._ws_graphs: dict = {}
        self._ws_dev: dict = {}      # name -> [device tensor every graph reads, its last written host values]
        self._ws_streak = 0          # consecutive replays
        self._ws_retired: list = []  # graphs retired by a workspace growth, kept alive until no replay can be in flight
        _install_workspace_hook()
        self._ws_pending = None      # (graph, req_ids) for the following sample_tokens
        self._ws_eager_out = None    # the first step's real output, returned by sample_tokens
        self._ws_failed: set = set()
        self._ws_t = time.time()
        self._ws_counts = {"stock": 0, "replay": 0, "capture": 0}
        self._ws_why: dict = {}            # why steps went to the stock path
        self._ws_tl = self._ws_tl_new()
        self._ws_pin = {}                  # pinned staging buffers for the refresh copies
        self._ws_ev = {}                   # the event after each staging buffer's last copy
        # per request slot: does a logits processor (penalties, bad words, logit bias, plugins) or a thinking budget
        # touch its logits? The sampler's own temperature / top-k / top-p / min-p handling is captured in the graph
        # (its host-side branches are part of the graph key), so only these exclude a request.
        self._ws_proc = np.zeros(self.max_num_reqs, dtype=bool)
        smp = self.sampler
        if smp is not None:
            for proc in smp.logits_processors:
                def proc_add(req_idx, params, _o=proc.add_request):
                    r = _o(req_idx, params)
                    if r:
                        self._ws_proc[req_idx] = True
                    return r
                proc.add_request = proc_add

            def smp_add(req_idx, params, _o=smp.add_request):
                self._ws_proc[req_idx] = False
                return _o(req_idx, params)
            smp.add_request = smp_add
        if os.environ.get("DSV41_SUFFIX") == "1":
            from .suffix import register_runner
            register_runner(self)    # the drafter's matcher and the scheduler read the request states
        if self._ws_on:
            _LOG("enabled: decode-only steps replay a captured whole-step graph")

    def load_model(self, *args, **kwargs):
        if os.environ.get("DSV41_LL_GEMM") == "1":
            from .ll_gemm import install   # G3: low-latency MXFP8 GEMM tactics, before warm-up and capture
            install()
        r = super().load_model(*args, **kwargs)
        if os.environ.get("DSV41_HEAD_FP8") in ("1", "dg"):
            self._ws_install_fp8_head()
        return r

    def _ws_install_fp8_head(self):
        """lossy option F9/S10: the verify head as a weight-only FP8 GEMV for <= 32 rows (dsv41_engine.fp8_head);
        DSV41_HEAD_FP8=dg: the drafter's MXFP8 DeepGEMM copy of the shared lm_head instead"""
        from .fp8_head import FP8Head, drafter_dg_head
        lm = getattr(self.model, "language_model", self.model)
        if os.environ.get("DSV41_HEAD_FP8") == "dg":
            head = drafter_dg_head(self)
            if head is None:
                raise RuntimeError("DSV41_HEAD_FP8=dg needs the drafter's DGHead (fastpath tail_drafthead_mxfp8)")
            head.V, head.H = head.v, head.k
        else:
            head = FP8Head(lm.lm_head.weight)
        lp = lm.logits_processor
        vocab = getattr(lp, "org_vocab_size", None) or getattr(lp, "vocab_size", head.V)
        orig = lm.compute_logits

        def compute_logits(hidden_states):
            if hidden_states.shape[0] <= 32:
                return head(hidden_states)[:, :vocab]
            return orig(hidden_states)
        lm.compute_logits = compute_logits
        _LOG(f"FP8 verify head on: [{head.V}, {head.H}] "
             f"{'MXFP8 (the drafter DGHead copy)' if os.environ.get('DSV41_HEAD_FP8') == 'dg' else 'e4m3 per-row scales'}, "
             f"vocab {vocab}")

    # ---------------------------------------------------------------------------------------------- eligibility
    def _no(self, why):
        self._ws_why[why] = self._ws_why.get(why, 0) + 1
        return None

    def _ws_key(self, so):
        if so.total_num_scheduled_tokens == 0:
            return self._no("empty")
        if so.scheduled_new_reqs:
            return self._no("new requests")
        if so.has_structured_output_requests:
            return self._no("structured output")
        if so.kv_connector_metadata is not None or so.preempted_req_ids or so.kv_cache_block_copies:
            return self._no("kv connector / preempted / block copies")
        spec = so.scheduled_spec_decode_tokens
        reqs = list(so.num_scheduled_tokens)
        if not spec or any(r not in spec for r in reqs):
            return self._no("no drafts")
        K = len(spec[reqs[0]])
        if K == 0 or any(len(spec[r]) != K or so.num_scheduled_tokens[r] != K + 1 for r in reqs):
            return self._no("non-uniform drafts")
        if K > self._ws_maxk:   # DSV41_WS_MAXK: wider steps (suffix's 32-row verify) take the stock path
            return self._no(f"K={K} above DSV41_WS_MAXK")
        rs = self.req_states
        try:
            slots = np.fromiter((rs.req_id_to_index[r] for r in reqs), dtype=np.intp, count=len(reqs))
        except KeyError:
            return self._no("unknown request")
        # also true on the first decode after a prefill (this counter trails by a step): that stock step zeroes the
        # SWA replay starts graphs do not refresh
        if np.any(rs.num_computed_prefill_tokens[slots] < rs.prefill_len.np[slots]):
            return self._no("prefilling")
        smp = self.sampler
        if smp is None or np.any(self._ws_proc[slots]):
            return self._no("logits processor")
        tb = getattr(smp, "thinking_budget_state", None)
        if tb is not None and getattr(tb, "enabled", False) and np.any(np.asarray(tb.use_thinking_budget)[slots]):
            return self._no("thinking budget")
        st = smp.sampling_states
        if np.any(st.num_logprobs[slots] != -1):
            return self._no("logprobs")
        temp = st.temperature.np[slots]
        top_k = st.top_k.np[slots]
        do_top_k = bool(np.any(top_k != st.vocab_size))
        sig = (len(reqs), K, bool(np.all((temp == 0.0) | (temp == 1.0))), bool(np.any(temp == 0.0)),
               bool(np.all(temp == 0.0)),
               bool(np.any(st.seeds_set[slots])), do_top_k, int(np.max(top_k)) if do_top_k else 0,
               bool(np.any(st.top_p.np[slots] != 1.0)), bool(np.all(st.min_p.np[slots] == 0.0)),
               bool(np.any(smp.needs_logits_processing[slots])))   # the sampler's branch is per request, not these aggregates
        return None if sig in self._ws_failed else sig

    # ------------------------------------------------------------------------------------------------ the step
    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, *args, **kwargs):
        key = None
        if self._ws_on and not dummy_run and not args and not kwargs and intermediate_tensors is None:
            key = self._ws_key(scheduler_output)
        if key is None or key[0] != 1:
            self._k3b_fast = False
        if key is None:
            self._ws_streak = 0
            self._ws_counts["stock"] += 1
            return super().execute_model(scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs)
        if self._k3b and key[0] == 1 and key[1] == self._k3b_k:
            return self._k3b_step(key, scheduler_output)
        if self._ws_usable(key) is None:
            return self._ws_capture(key, scheduler_output)
        return self._ws_replay(scheduler_output, key)

    def _ws_usable(self, key):
        """the key's graph, or None when there is none (a graph captured before a workspace growth is retired here)"""
        g = self._ws_graphs.get(key)
        if g is not None and g.gen != _WS_GEN[0]:
            self._ws_retired.append(self._ws_graphs.pop(key))
            self._ws_counts["retired"] = self._ws_counts.get("retired", 0) + 1
            _LOG(f"retired {key} (captured under workspace generation {g.gen}, now {_WS_GEN[0]})")
            g = None
        return g

    # ------------------------------------------------------------------------------------------------ K3b
    def _k3b_width(self):
        """the width of the drafts the previous step proposed. Its graph writes the word (sequence << 8 | width) right
        after the drafter, so the steady path spins on pinned memory, no driver call, and launches while the step's
        last kernels finish; after any other kind of step, or if the word does not advance, wait for the stream. Any
        width is correct (a prefix of the drafts verifies exactly): a late word costs speed, never output."""
        cw = sys.modules.get("dsv41_engine.conf_width")
        h = None if cw is None else cw.STATE.get("width_np")
        t0 = time.perf_counter()
        if h is None:
            return self._k3b_k
        v, target, spins = int(h[0]), self._k3b_seen + self._k3b_pending, 0
        if self._k3b_fast:
            while (v >> 8) < target and spins < 20000000:
                v = int(h[0])
                spins += 1
        if not self._k3b_fast or (v >> 8) < target:
            ev = torch.cuda.Event()
            ev.record()
            while not ev.query():
                pass
            v = int(h[0])
        self._k3b_wait += time.perf_counter() - t0
        self._k3b_seen, self._k3b_pending, self._k3b_fast = v >> 8, 0, True
        w = min(max(v & 255, 1), self._k3b_k)
        self._k3b_hist[w] += 1
        return w

    def _k3b_step(self, key, so):
        K = self._k3b_k
        keys = {w: (key[0], w) + key[2:] for w in range(1, K + 1)}
        ok = {w: self._ws_usable(k) is not None for w, k in keys.items()}
        if ok[K] and all(ok[w] or keys[w] in self._ws_failed for w in keys):
            # steady state: the host bookkeeping overlaps the previous step, then the width picks the graph
            def pick():
                # launch ahead on K: after two full widths the next step is most likely one too, and verifying all K drafts
                # is exact whatever the width would have been, so don't wait for the word (the step's sync is the
                # price of K3b on high-acceptance text). A published width below K goes back to waiting
                if self._k3b_run >= 2 and self._k3b_ahead:
                    if int(sys.modules["dsv41_engine.conf_width"].STATE["width_np"][0]) & 255 == K:
                        self._k3b_hist[K] += 1
                        return keys[K]
                    self._k3b_run = 0
                w = self._k3b_width()
                self._k3b_run = self._k3b_run + 1 if w == K else 0
                return keys[w] if ok[w] else keys[K]
            return self._ws_replay(so, None, sched_k=K, pick=pick, refresh=[keys[w] for w in keys if ok[w]])
        # warm-up: a missing width is captured on this step, and the capture runs its own bookkeeping: wait first
        w = self._k3b_width()
        if keys[w] in self._ws_failed:
            w = K
        if ok[w]:
            return self._ws_replay(so, keys[w], sched_k=K)
        if keys[w] in self._ws_failed:
            self._ws_streak = 0
            self._ws_counts["stock"] += 1
            return super().execute_model(so)
        return self._ws_capture(keys[w], _so_width(so, w) if w < K else so)

    def _ws_replay(self, so, key, sched_k=None, pick=None, refresh=None):
        """the stock host bookkeeping, then one replay of the key's graph (pick, when given, chooses the key after the
        bookkeeping, and every graph in `refresh` is refreshed before it, so that only the launch follows the wait);
        sched_k: the drafts the scheduler gets back as this step's draft width (K3b: always K)"""
        t_enter = time.perf_counter()
        # the stock host bookkeeping (execute_model's prologue)
        self.update_pp_decode_requests()
        self.finish_requests(so)
        self.free_states(so)
        self.add_requests(so)
        self.update_requests(so)
        self.block_tables.apply_staged_writes()
        self._ws_sync()
        self._ws_streak += 1
        req_ids = sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, self.decode_query_len)
        slots = np.fromiter((self.req_states.req_id_to_index[r] for r in req_ids), dtype=np.intp, count=len(req_ids))
        if pick is not None:
            for k in refresh:
                self._ws_refresh(self._ws_graphs[k], slots, k)
            key = pick()
        g = self._ws_graphs[key]
        if pick is None:
            self._ws_refresh(g, slots, key)
        tl = self._ws_tl
        # the GPU period: an event at each replay's start on the launch stream; start-to-start of two consecutive
        # replays of one key (no stock step between) is that replay's GPU time plus any idle gap before the next
        ev = torch.cuda.Event(enable_timing=True)
        if pick is None:
            ev.record()
        if self._ws_dbg and g.replays < 3:   # DSV41_WS_DEBUG: name each graph's first replays, fault on the spot
            _LOG(f"debug: replay {g.replays} of {key}, slots {slots.tolist()}, stock steps so far "
                 f"{self._ws_counts['stock']}")
            sys.stderr.flush()
            torch.cuda.synchronize()
        g.graph.replay()
        if key[0] == 1:
            self._k3b_pending += 1
        if pick is not None:
            ev.record()    # K3b: after the launch, off the critical path (end-to-end periods instead of start-to-start)
        if self._ws_dbg and g.replays < 3:
            torch.cuda.synchronize()
            _LOG(f"debug: replay {g.replays} of {key} ok")
            sys.stderr.flush()
        if self._conf_path and key[0] == 1:
            self._ws_conf_record(g, slots, key)
        g.replays += 1
        self._ws_counts["replay"] += 1
        self._ws_pending = (g, req_ids, key[1] if sched_k is None else sched_k)
        now = time.perf_counter()
        tl["n"] += 1
        tl["host"] += now - t_enter
        if tl["last"] is not None and t_enter - tl["last"] < 1.0:
            tl["period"] += t_enter - tl["last"]
            tl["periods"].append(t_enter - tl["last"])
        tl["last"] = t_enter
        prev = tl["ev_last"]
        if prev is not None and prev[1] == key and prev[2] == self._ws_counts["stock"]:
            tl["ev_pairs"].append((prev[0], ev, key[:2]))  # (batch, draft width): per-key medians in the log
        tl["ev_last"] = (ev, key, self._ws_counts["stock"])
        if time.time() - tl["t_log"] > 30:
            n = max(tl["n"], 1)
            ev.synchronize()
            gpu = sorted(a.elapsed_time(b) for a, b, _ in tl["ev_pairs"])
            per_key = {}
            for a, b, k in tl["ev_pairs"]:
                per_key.setdefault(k, []).append(a.elapsed_time(b))
            keys_s = ", ".join(f"{k[0]}x{k[1]}: {sorted(v)[len(v) // 2]:.3f} ms ({len(v)})"
                               for k, v in sorted(per_key.items()) if len(v) >= 20)
            host_p = sorted(tl["periods"])
            med = lambda v: v[len(v) // 2] if v else float("nan")
            _LOG(f"steps {self._ws_counts}; last {tl['n']} replays: host in execute_model {tl['host'] / n * 1e3:.3f} ms, "
                 f"engine period {tl['period'] / n * 1e3:.3f} ms (median {med(host_p) * 1e3:.3f}), GPU period "
                 f"median {med(gpu):.3f} ms / mean {sum(gpu) / max(len(gpu), 1):.3f} over {len(gpu)} replay pairs, "
                 f"refresh copies {tl['copies']}; stock reasons {self._ws_why}; graphs {sorted(self._ws_graphs)}")
            if keys_s:
                _LOG(f"GPU period median per (batch x draft width): {keys_s}")
            if self._k3b:
                nk = max(sum(self._k3b_hist), 1)
                _LOG(f"K3b widths 1..{self._k3b_k}: {self._k3b_hist[1:]} (mean "
                     f"{sum(w * c for w, c in enumerate(self._k3b_hist)) / nk:.2f}), mean wait for the previous step "
                     f"{self._k3b_wait / nk * 1e3:.3f} ms")
            if self._conf is not None:
                self._ws_conf_dump()
            self._ws_tl = self._ws_tl_new(tl["last"])
        return None

    def _ws_conf_record(self, g, slots, key):
        spec = getattr(self, "speculator", None)
        dl = getattr(spec, "draft_logits", None)
        if dl is None or g.num_sampled is None:
            return
        c = self._conf
        if c is None:
            kd = int(os.environ.get("DSV41_DSPARK_K", "5"))
            c = self._conf = {"kd": kd, "n": 65536, "i": 0, "done": 0, "ts": [],
                              "ring": torch.zeros(65536, 2 + 2 * kd, dtype=torch.float32, pin_memory=True),
                              "row": torch.zeros(2 + 2 * kd, dtype=torch.float32, device=dl.device)}
            _LOG(f"K3 confidence log -> {self._conf_path} (draft positions {kd})")
        kd, row = c["kd"], c["row"]
        row[0] = float(key[1])
        row[1:2].copy_(g.num_sampled[:1].float())
        torch.amax(torch.softmax(dl[int(slots[0]), :kd].float(), dim=-1), dim=-1, out=row[2:2 + kd])
        cw = sys.modules.get("dsv41_engine.conf_width")
        if cw is not None and cw.STATE["head"] is not None:   # the checkpoint's confidence head, same positions
            row[2 + kd:2 + 2 * kd].copy_(cw.STATE["head"][int(slots[0]), :kd])
        c["ring"][c["i"] % c["n"]].copy_(row, non_blocking=True)
        c["ts"].append(time.time())
        c["i"] += 1

    def _ws_conf_dump(self):
        c = self._conf
        torch.cuda.current_stream().synchronize()
        lo, hi = max(c["done"], c["i"] - c["n"]), c["i"]
        if hi <= lo:
            return
        rows = c["ring"].numpy()
        with open(self._conf_path, "a") as f:
            for j in range(lo, hi):
                r = rows[j % c["n"]]
                f.write(f"{c['ts'][j]:.3f} {int(r[0])} {int(r[1])} " + " ".join(f"{v:.4f}" for v in r[2:]) + "\n")
        c["done"] = hi

    @staticmethod
    def _ws_tl_new(last=None):
        return {"n": 0, "host": 0.0, "period": 0.0, "periods": [], "last": last, "copies": 0, "t_log": time.time(),
                "ev_pairs": [], "ev_last": None}

    def sample_tokens(self, grammar_output):
        if self._ws_eager_out is not None:
            out, self._ws_eager_out = self._ws_eager_out, None
            return out
        if self._ws_pending is None:
            return super().sample_tokens(grammar_output)
        g, req_ids, K = self._ws_pending
        self._ws_pending = None
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.worker.gpu.model_runner import AsyncOutput
        mro = ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                sampled_token_ids=None, prompt_logprobs_dict={}, prompt_token_id_logprobs_dict={})
        h = self.draft_tokens_handler   # the scheduler gets placeholder drafts (no structured output here)
        h.req_ids, h.num_draft_tokens, h.draft_tokens_np = req_ids, K, None
        return AsyncOutput(model_runner_output=mro, sampler_output=g.sampler_output, num_sampled_tokens=g.num_sampled,
                           main_stream=self.main_stream, copy_stream=self.output_copy_stream,
                           check_ep_fault=self.check_ep_fault, pending_aux_output=None)

    # ----------------------------------------------------------------------------------- capture and refresh
    def _ws_consts(self, slots, K):
        """the host values of the persistent H2D tensors for this batch, in prepare_inputs' call order:
        idx_mapping (int64), cu_num_logits (int32)"""
        n = len(slots)
        cu = np.arange(0, (K + 1) * (n + 1), K + 1, dtype=np.int32)
        return [np.asarray(slots, dtype=np.int64), cu]

    def _ws_live(self):
        """the UVA-backed tensors a step reads, by name"""
        st = self.sampler.sampling_states
        live = {name: getattr(st, name) for name in _PARAMS}
        live.update(prefill_len=self.req_states.prefill_len, prompt_len=self.req_states.prompt_len,
                    num_blocks=self.block_tables.num_blocks)
        return live

    def _ws_sync(self):
        """every graph reads these as device tensors, one per name, written here in stream order. vLLM's UVA pools would
        leave a graph on the buffer current at capture: prefill_len / prompt_len rotate only on a step that adds a request, so
        that buffer can stay a request behind, and a slot's longer previous prompt makes the combine, gather and sampled-count
        kernels treat a new request's rows as prefill (rows without its tokens, nothing sampled, until it passes that length);
        the others rotate every step, but graphs share the buffers, so no per-graph record can say what a buffer holds"""
        for name, t in self._ws_live().items():
            ent = self._ws_dev.get(name)
            if ent is None:
                ent = self._ws_dev[name] = [torch.zeros(t.np.shape, dtype=t.dtype, device=self.device), None]
            if ent[1] is None or not np.array_equal(ent[1], t.np):
                self._ws_copy(("dev", name), ent[0], t.np)
                ent[1] = t.np.copy()

    def _ws_hold(self, dev):
        """a capture's stand-in for UvaBackedTensor.copy_to_uva: `gpu` stays on the device tensor; the record step's own
        bookkeeping still reaches it, the captured pass adds nothing to the graph"""
        def copy_to_uva(n=None):
            if not torch.cuda.is_current_stream_capturing():
                self._ws_sync()
            return dev
        return copy_to_uva

    def _ws_refresh(self, g, slots, key):
        consts = self._ws_consts(slots, key[1])
        for i, (t, v) in enumerate(zip(g.patches.h2d, consts)):
            if g.consts_np is None or not np.array_equal(g.consts_np[i], v):
                self._ws_copy(("c", id(t)), t, v)
        g.consts_np = consts

    def _ws_copy(self, key, dst, src_np):
        """stream-ordered, non-blocking host->device copy through a pinned staging buffer (one per destination; a
        new copy into the same destination is ordered after the previous one, so reusing its staging buffer would
        race only if the host rewrote it before the earlier copy ran: each destination gets a pair, alternated)"""
        pair = self._ws_pin.get(key)
        if pair is None:
            pair = [torch.empty(dst.shape, dtype=dst.dtype, pin_memory=True) for _ in range(2)] + [0]
            self._ws_pin[key] = pair
        i = pair[2]
        pair[2] ^= 1
        buf = pair[i]
        if self._ws_ev.get(key + (i,)) is not None:
            self._ws_ev[key + (i,)].synchronize()   # the copy that last used this staging buffer has run
        buf.copy_(torch.from_numpy(np.ascontiguousarray(src_np)).view(buf.shape).to(buf.dtype))
        dst.copy_(buf, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._ws_ev[key + (i,)] = ev
        self._ws_tl["copies"] += 1

    def _ws_warm_mega_gate(self, mega_gate):
        """call DeepGEMM Mega Gate once, eagerly, on the current (capture) stream with a 128-expert drafter gate and
        a 384-expert target gate, so its per-stream score barriers exist before the capture records it"""
        seen = set()
        for root in (getattr(self, "model", None), getattr(getattr(self, "speculator", None), "model", None)):
            if root is None:
                continue
            for m in root.modules():
                gate = getattr(m, "gate", None)
                if not getattr(m, "use_native_mega_moe", False) or gate is None or not hasattr(gate, "weight"):
                    continue
                E = gate.weight.shape[0]
                if E in seen:
                    continue
                seen.add(E)
                x = torch.zeros(8, gate.weight.shape[1], dtype=torch.bfloat16, device=gate.weight.device)
                bias = getattr(gate, "e_score_correction_bias", None)
                try:
                    mega_gate(x, gate.weight, m.n_activated_experts, scoring_func=m.scoring_func,
                              routed_scaling_factor=m.routed_scaling_factor, ep_rank=getattr(m, "ep_rank", 0),
                              bias=None if bias is None else bias.data)
                except Exception as e:
                    _LOG(f"Mega Gate warm-up for E={E} failed: {e!r}")

    def _ws_capture(self, key, so):
        self._k3b_fast = False   # the record step's drafter advances the width word outside the replay count
        self._ws_streak = 0
        # dsv41 C19: the boundary's lossy routing options (DSV41_DROP_DECODE_ONLY) apply only while a decode graph is
        # recorded and captured; stock steps (prompt prefill, mixed steps) route exactly
        flag = getattr(sys.modules.get("fastpath_boundary"), "WS_DECODE", None)
        if flag is not None:
            flag[0] = True
        self._ws_sync()
        live = self._ws_live()
        pool = {name: t.gpu for name, t in live.items()}
        for name, t in live.items():
            t.gpu = self._ws_dev[name][0]
            t.copy_to_uva = self._ws_hold(t.gpu)
        # dsv41 D7: graphs of batches that sample with top-p draft from the request's top-p (draft_topp.py)
        dt = sys.modules.get("dsv41_engine.draft_topp")
        scope = dt.capture_scope(self, key) if dt is not None and dt._installed else None
        if scope is not None:
            scope[0]()
        try:
            return self._ws_capture_inner(key, so)
        finally:
            for name, t in live.items():
                del t.copy_to_uva
                t.gpu = pool[name]
            if flag is not None:
                flag[0] = False
            if scope is not None:
                scope[1]()

    def _ws_capture_inner(self, key, so):
        from vllm.distributed.parallel_state import graph_capture
        from vllm.utils.torch_utils import current_stream
        g = _Graph(key)
        g.patches = StepPatches(self)
        steady = steady_scheduler_output(so, self)
        t0 = time.time()
        torch.cuda.synchronize()
        self._ws_retired.clear()     # the GPU is idle here: no retired graph can still be running
        with graph_capture(device=self.device):
            s = current_stream()
            # this step for real, through the capturable path, on the capture stream (real async output)
            import vllm.utils.deep_gemm as vdg
            mg_orig, mg_calls = vdg.bf16_mega_gate, []

            def mg_logged(x, *a, **k):       # which Mega Gate calls happen on which stream (record vs capture)
                mg_calls.append((tuple(x.shape), torch.cuda.current_stream().cuda_stream,
                                 torch.cuda.is_current_stream_capturing()))
                return mg_orig(x, *a, **k)
            vdg.bf16_mega_gate = mg_logged
            try:
                # the real step, all of it on the capture stream: the output copy is stubbed here too (a real
                # AsyncOutput's stream context restores the runner's main stream, which would move the drafter off
                # the capture stream and leave its per-stream DeepGEMM state unwarmed); the real async output is
                # built from the recorded tensors below
                with g.patches.active("record", stub_output=True):
                    super().execute_model(so)
                    rec = super().sample_tokens(None)
                torch.cuda.synchronize()
                self._ws_warm_mega_gate(mg_orig)   # DeepGEMM Mega Gate keeps score barriers per stream
                torch.cuda.synchronize()
            finally:
                vdg.bf16_mega_gate = mg_orig
            if not getattr(self, "_ws_logged_mg", False):
                self._ws_logged_mg = True
                _LOG(f"record step Mega Gate calls (shape, stream, capturing): {mg_calls[:8]} (n={len(mg_calls)}); "
                     f"capture stream {s.cuda_stream}")

            graph = torch.cuda.CUDAGraph()
            failed = False
            try:
                with g.patches.active("replay"), torch.cuda.graph(graph, stream=s):
                    super().execute_model(steady)
                    ao = super().sample_tokens(None)
            except Exception as e:   # the step above already ran for real: serve its output, keep this key stock
                import traceback
                self._ws_failed.add(key)
                _LOG(f"capture FAILED for {key}: {e!r}; this key stays on the stock path\n{traceback.format_exc()}")
                failed = True
        if failed:
            from vllm.v1.worker.gpu.model_runner import AsyncOutput
            torch.cuda.synchronize()
            self._ws_eager_out = AsyncOutput(model_runner_output=rec.model_runner_output,
                                             sampler_output=rec.sampler_output,
                                             num_sampled_tokens=rec.num_sampled_tokens, main_stream=self.main_stream,
                                             copy_stream=self.output_copy_stream,
                                             check_ep_fault=self.check_ep_fault, pending_aux_output=None)
            return None
        torch.cuda.synchronize()
        from vllm.v1.worker.gpu.model_runner import AsyncOutput
        out = AsyncOutput(model_runner_output=rec.model_runner_output, sampler_output=rec.sampler_output,
                          num_sampled_tokens=rec.num_sampled_tokens, main_stream=self.main_stream,
                          copy_stream=self.output_copy_stream, check_ep_fault=self.check_ep_fault,
                          pending_aux_output=None)
        g.graph, g.sampler_output, g.num_sampled = graph, ao.sampler_output, ao.num_sampled_tokens
        assert len(g.patches.h2d) == 2, f"expected idx_mapping + cu_num_logits H2D tensors, got {len(g.patches.h2d)}"
        req_ids = sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, self.decode_query_len)
        g.consts_np = self._ws_consts(
            np.fromiter((self.req_states.req_id_to_index[r] for r in req_ids), dtype=np.intp, count=len(req_ids)),
            key[1])
        g.gen = _WS_GEN[0]
        self._ws_graphs[key] = g
        self._ws_counts["capture"] += 1
        self._ws_eager_out = out
        _LOG(f"captured {key} in {time.time() - t0:.2f} s")
        return None
