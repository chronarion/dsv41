"""M5 (Fable round 10, 5.2): shadow acceptance log for draft-side shaping. For every verified draft position of a
sampled (T > 0) request, the exact token-level acceptance probability sum_x min(p(x), q(x)) of the processed target
row p (after temperature and top-p) against the draft distribution q as drafted today, and against shaped drafts:
q truncated to the request's top-p nucleus (D7), q at another draft temperature, and both. No sampling noise, so a
few thousand steps rank the options before any kernel work. Expected tokens per step follow from the per-position
means (chain product). Measurement only: Python per step, so run it with DSV41_WHOLE_STEP=0.
   DSV41_ACC_LOG=/path/to/file.json (rewritten every 30 s)"""
import importlib
import json
import os
import sys
import time

PATH = os.environ.get("DSV41_ACC_LOG")
TAUS = (0.8, 0.9, 1.1)
STATE = {"t": time.time(), "steps": 0, "acc": {}, "cnt": {}, "chain": {}}
_LOG = lambda m: sys.stderr.write(f"DSV41_ACC_LOG {m}\n")


def _nucleus(probs, top_p):
    import torch
    sp, idx = probs.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p[:, None]
    out = torch.zeros_like(probs).scatter_(-1, idx, sp * keep)
    return out / out.sum(-1, keepdim=True)


def _record(self, processed, draft_logits, expanded_idx_mapping, expanded_local_pos, cu_num_logits):
    import torch
    if draft_logits is None:
        return
    K = draft_logits.shape[1]
    lp = expanded_local_pos.long()
    st = self.sampler.sampling_states
    variants = {}
    sel = ((lp >= 1) & (lp <= K)).nonzero().squeeze(1)
    if sel.numel() == 0:
        return
    req = expanded_idx_mapping[sel].long()
    T = st.temperature.gpu[req].float()
    keep = T > 0
    sel, req, T = sel[keep], req[keep], T[keep]
    if sel.numel() == 0:
        return
    top_p = st.top_p.gpu[req].float()
    p = torch.softmax(processed[sel - 1].float(), -1)
    ql = draft_logits[req, lp[sel] - 1].float() / T[:, None]
    q = torch.softmax(ql, -1)
    variants["today"] = q
    variants["nucleus"] = _nucleus(q, top_p)
    for tau in TAUS:
        qt = torch.softmax(ql / tau, -1)
        variants[f"tau{tau}"] = qt
        variants[f"nucleus+tau{tau}"] = _nucleus(qt, top_p)
    pos = lp[sel].cpu().tolist()
    # per request: its rows in order, for the chain product (expected accepted drafts this step)
    reqs = req.cpu().tolist()
    for name, qv in variants.items():
        a = torch.minimum(p, qv).sum(-1).cpu().tolist()
        acc, cnt = STATE["acc"].setdefault(name, [0.0] * K), STATE["cnt"].setdefault(name, [0] * K)
        chain = {}
        for r, j, v in zip(reqs, pos, a):
            acc[j - 1] += v
            cnt[j - 1] += 1
            prev = chain.get(r, (0, 1.0, 0.0))
            if prev[0] == j - 1:
                run = prev[1] * v
                chain[r] = (j, run, prev[2] + run)
        c = STATE["chain"].setdefault(name, [0.0, 0])
        for _, (_, _, e) in chain.items():
            c[0] += e
            c[1] += 1
    STATE["steps"] += 1
    if time.time() - STATE["t"] > 30:
        STATE["t"] = time.time()
        out = {"steps": STATE["steps"]}
        for name in STATE["acc"]:
            out[name] = {"acc_by_pos": [round(s / n, 4) if n else None for s, n in zip(STATE["acc"][name], STATE["cnt"][name])],
                         "rows_by_pos": STATE["cnt"][name],
                         "expected_accepted_per_request_step": round(STATE["chain"][name][0] / max(1, STATE["chain"][name][1]), 4)}
        with open(PATH + ".tmp", "w") as f:
            json.dump(out, f, indent=1)
        os.replace(PATH + ".tmp", PATH)


def install():
    mod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.rejection_sampler")
    cls = mod.RejectionSampler
    orig = cls._verify

    def _verify(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                expanded_idx_mapping, expanded_local_pos, *args, **kwargs):
        out = orig(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                   expanded_idx_mapping, expanded_local_pos, *args, **kwargs)
        try:
            _record(self, out[0], draft_logits, expanded_idx_mapping, expanded_local_pos, cu_num_logits)
        except Exception as e:  # noqa: BLE001  (measurement only)
            _LOG(f"record failed: {e!r}")
        return out

    cls._verify = _verify
    _LOG(f"installed -> {PATH}")
