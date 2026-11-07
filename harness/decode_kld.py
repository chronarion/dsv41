"""C1 forced-acceptance hook: decode-path logits of the served kernels on the KLD corpus, teacher-forced.

For each corpus window: prefill ids[:P] through vLLM's normal path (the prefill's sample is forced to ids[P]), then
every verify step's drafts are overwritten with the corpus tokens and its acceptance is forced (sampled = the next 6
corpus tokens), so each step yields K+1 = 6 decode-path logit rows (positions p..p+5, the verify graph exactly as
served) and the KV cache holds the corpus. The rejection sampler and the drafter still run (their outputs are
overwritten), so the step is the served step. Output per window: OUT/{id}.safetensors (logits [R, V] fp32,
target_ids, positions), the ref_logits.py format, compared by position with tools/kld/kld_rows.py.
Run in the server's container setup:
    python3 decode_kld.py OUT [--prefill 1024] [--rows 510] [--windows a,b,...]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session  # noqa: E402

CORPUS = os.environ.get("DSV41_KLD_CORPUS", "corpus.json")


class Forcer:
    """runner hooks: overwrite the drafts before each forward, record the verify logits and force the sampled tokens"""

    def __init__(self, runner):
        self.runner = runner
        self.active = {}   # internal request id -> state dict
        self._orig_em, self._orig_sample = runner.execute_model, runner.sample
        runner.execute_model, runner.sample = self._execute_model, self._sample

    def _execute_model(self, scheduler_output, *a, **k):
        import torch
        rs = self.runner.req_states
        for rid, st in self.active.items():
            if rid in scheduler_output.scheduled_spec_decode_tokens and st["phase"] == "decode":
                if st.get("stash"):
                    import copy
                    st["so"] = copy.deepcopy(scheduler_output)
                s = rs.req_id_to_index[rid]
                p, ids, K = st["p"], st["ids"], st["K"]
                d = torch.tensor(ids[p + 1:p + 1 + K], dtype=rs.draft_tokens.dtype, device=rs.draft_tokens.device)
                rs.draft_tokens[s, :d.numel()] = d
                rs.last_sampled_tokens[s] = ids[p]
        return self._orig_em(scheduler_output, *a, **k)

    def _sample(self, hidden_states, input_batch, grammar_output):
        import torch
        so, ns, nr = self._orig_sample(hidden_states, input_batch, grammar_output)
        for b, rid in enumerate(input_batch.req_ids):
            st = self.active.get(rid)
            if st is None:
                continue
            ids = st["ids"]
            if st["phase"] == "prefill":
                # the single prefill chunk (P <= max_num_batched_tokens) samples the first token: force it to ids[P]
                so.sampled_token_ids[b, 0] = ids[st["P"]]
                ns[b] = 1
                nr[b] = 0
                st["phase"], st["p"] = "decode", st["P"]
                continue
            p, K = st["p"], st["K"]
            c0 = int(input_batch.cu_num_logits_np[b])
            c1 = int(input_batch.cu_num_logits_np[b + 1])
            assert c1 - c0 == K + 1, (c0, c1, K)
            rows = input_batch.logits_indices[c0:c1]
            lg = self.runner.model.compute_logits(hidden_states[rows]).float()
            st["logits"].append(lg.cpu())
            st["pos"].extend(range(p, p + K + 1))
            st["tgt"].extend(ids[p + 1:p + K + 2])
            so.sampled_token_ids[b, :K + 1] = torch.tensor(ids[p + 1:p + K + 2], dtype=so.sampled_token_ids.dtype,
                                                           device=so.sampled_token_ids.device)
            ns[b] = K + 1
            nr[b] = 0
            st["p"] = p + K + 1
        return so, ns, nr


def captured_rows(llm, runner, forcer, iid, st, n_rows):
    """the rest of the window through the captured whole step (C3): the step is captured with a sampler that runs the
    stock rejection, then overwrites the sampled tokens from a persistent device buffer (K+1 corpus tokens) and
    records the verify logits; before each replay the host writes that buffer and the drafts (stream-ordered)"""
    import torch
    from whole_step import WholeStep, preallocate, decode_sched_output
    K, ids = st["K"], st["ids"]
    rs = runner.req_states
    slot = rs.req_id_to_index[iid]
    forcer.active.pop(iid)                       # the forced sampler below owns this request now
    preallocate(llm, runner, iid, n_rows + 8 * (K + 1))
    sso = decode_sched_output(st["so"], iid, runner)
    orig_sample = forcer._orig_sample
    buf = {}

    def forced_sample(hidden_states, input_batch, grammar_output):
        so, ns, nr = orig_sample(hidden_states, input_batch, grammar_output)
        if "forced" not in buf:
            buf["forced"] = torch.zeros(K + 1, dtype=so.sampled_token_ids.dtype, device=so.sampled_token_ids.device)
        rows = input_batch.logits_indices
        buf["logits"] = runner.model.compute_logits(hidden_states[rows])
        so.sampled_token_ids[0, :K + 1].copy_(buf["forced"])
        ns.fill_(K + 1)
        nr.fill_(0)
        return so, ns, nr

    def prime(p):
        f = buf.get("forced")
        if f is not None:
            f.copy_(torch.tensor(ids[p + 1:p + K + 2], dtype=f.dtype))
        rs.draft_tokens[slot, :K].copy_(torch.tensor(ids[p + 1:p + 1 + K], dtype=rs.draft_tokens.dtype))
        rs.last_sampled_tokens[slot] = ids[p]

    def take(p):
        torch.cuda.synchronize()
        st["logits"].append(buf["logits"][:K + 1].float().cpu())
        st["pos"].extend(range(p, p + K + 1))
        st["tgt"].extend(ids[p + 1:p + K + 2])
        return p + K + 1

    saved = runner.sample
    runner.sample = forced_sample
    try:
        ws = WholeStep(llm, runner, iid, sso)
        p = st["p"]
        # the forced buffer exists only after the first sample(): create it before the eager step reads it
        buf["forced"] = torch.zeros(K + 1, dtype=torch.int64, device=rs.draft_tokens.device)
        prime(p)

        def on_eager(w):
            nonlocal p
            p = take(p)
            prime(p)
        ws.eager_then_capture(on_eager)
        while len(st["pos"]) < n_rows:
            prime(p)
            ws.graph.replay()
            p = take(p)
        st["p"] = p
    finally:
        runner.sample = saved


def main():
    import torch
    from safetensors.torch import save_file
    from vllm import SamplingParams
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prefill", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=510, help="decode rows per window (a multiple of K+1 = 6)")
    ap.add_argument("--windows", default="")
    ap.add_argument("--batch", type=int, default=1, help="windows decoded together on the full / eager routes, so the "
                    "verify steps run at batch N (N x (K+1) rows; the captured route stays at batch 1)")
    ap.add_argument("--routes", default="full,eager", help="full: FULL-graph replays (served default); eager: the NONE "
                    "cudagraph route, which the captured whole step records; captured: the captured whole step itself "
                    "(engine/harness/whole_step.py) with a forced sampler reading the corpus from a device buffer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from whole_step import force_eager
    import contextlib
    W = [w for w in json.load(open(CORPUS)) if len(w["ids"]) == 2048]
    if a.windows:
        W = [w for w in W if w["id"] in a.windows.split(",")]
    llm, runner = session.load(async_scheduling=False)
    K = runner.num_speculative_steps
    f = Forcer(runner)
    eng = llm.llm_engine
    meta = dict(prefill=a.prefill, rows=a.rows, K=K, windows={})
    for route in a.routes.split(","):
      od = os.path.join(a.out, route)
      os.makedirs(od, exist_ok=True)
      groups = [W[i:i + a.batch] for i in range(0, len(W), a.batch)] if route != "captured" else [[w] for w in W]
      for grp in groups:
          t0 = time.time()
          sts = []
          for w in grp:
              ids, P = w["ids"], a.prefill
              assert P + a.rows + K + 1 < len(ids)
              iid = eng.add_request(f"{route}-{w['id']}", {"prompt_token_ids": ids[:P], "cache_salt": f"dkld-{route}-{w['id']}-{t0}"},
                                    SamplingParams(temperature=0, max_tokens=a.rows + 4 * (K + 1), ignore_eos=True))
              st = dict(phase="prefill", P=P, p=None, K=K, ids=ids, logits=[], pos=[], tgt=[])
              f.active[iid] = st
              sts.append((w, iid, st))
          if route == "captured":
              w, iid, st = sts[0]
              st["stash"] = True
              while not (st["phase"] == "decode" and st.get("so") is not None and len(st["pos"]) >= K + 1):
                  eng.step()
              captured_rows(llm, runner, f, iid, st, a.rows)
          else:
              with (force_eager() if route == "eager" else contextlib.nullcontext()):
                  while min(len(st["pos"]) for _, _, st in sts) < a.rows:
                      eng.step()
          eng.abort_request([f"{route}-{w['id']}" for w, _, _ in sts])
          while eng.has_unfinished_requests():
              eng.step()
          for w, iid, st in sts:
              f.active.pop(iid, None)
              n = a.rows
              lg = torch.cat(st["logits"])[:n].contiguous()
              save_file({"logits": lg, "target_ids": torch.tensor(st["tgt"][:n]), "positions": torch.tensor(st["pos"][:n])},
                        os.path.join(od, f"{w['id']}.safetensors"))
              meta["windows"][f"{route}/{w['id']}"] = dict(rows=n, first=st["pos"][0], seconds=round(time.time() - t0, 1),
                                                         batch=len(sts))
              print(f"{route} {w['id']}: {n} decode rows at positions {st['pos'][0]}..{st['pos'][n - 1]} "
                    f"(batch {len(sts)}, {time.time() - t0:.1f} s)", flush=True)
          json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("DECODE KLD DONE", flush=True)


if __name__ == "__main__":
    main()
