"""D7 / K4 / S69: drafts from the request's top-p at T > 0 (distribution-preserving).

vLLM 0.32's DSpark drafter samples at the request's temperature but ignores its top-p ("for simplicity and
performance"): draft tokens in the target's excluded tail are always rejected, and on reasoning text most positions
have a one-token nucleus at top_p 0.95, where the acceptance of an untruncated draft is only q(top-1). Here, for the
whole-step graphs of batches that sample with top-p, the drafter takes vLLM's top-k candidate path
(`_sample_sequential_topk`: the base logits' top-k candidates, the Markov bias on those only) and cuts each position's
candidates to the request's top-p of q at the request's temperature before the draw. The cut logits (-inf outside) are
what the draw uses and what the q cache holds, so the rejection sampler's test stays exact for any q.

Greedy graphs, graphs without top-p and stock steps keep ds41's fused full-vocabulary Markov loop. The V4.1 drafter
shares DSparkMarkovHead (`apply_bias_gathered`) with Qwen3 DSpark but has no `apply_markov_bias_gathered` wrapper; the
loop below calls the head directly.

Enable with DSV41_DRAFT_TOPP=1 (DSV41_DRAFT_TOPK = candidates per position, default 128); install() before the model
runner is built. The runner sets STATE["use"] / STATE["top_p"] while it records and captures a decode graph."""
import importlib
import os
import sys

import torch

TOPK = int(os.environ.get("DSV41_DRAFT_TOPK", "128"))
STATE = {"use": False, "top_p": None, "graphs": 0}
_LOG = lambda m: sys.stderr.write(f"DSV41_DRAFT_TOPP {m}\n")
_installed = False


def apply_top_p(logits, cand, idx_map, temperature, top_p):
    """In place: candidates outside the request's top-p nucleus of softmax(logits / T) become -inf (T > 0 rows only;
    the most likely candidate always stays)."""
    rows = idx_map.clamp_min(0)
    t = temperature[rows].float().unsqueeze(1)
    p = top_p[rows].float().unsqueeze(1)
    vals = logits.gather(1, cand).float()
    sampled = t > 0
    probs = torch.softmax(vals / torch.where(sampled, t, torch.ones_like(t)), dim=1)
    sp, order = probs.sort(dim=1, descending=True)
    drop_sorted = ((sp.cumsum(1) - sp) >= p) & sampled
    drop = torch.zeros_like(drop_sorted).scatter_(1, order, drop_sorted)
    logits.scatter_(1, cand, torch.where(drop, torch.full_like(vals, float("-inf")), vals).to(logits.dtype))


def _sample_sequential_topk(self, num_reqs, head_hidden):
    """vLLM 0.32's DSparkSpeculator._sample_sequential_topk (no confidence head) through the drafter's shared Markov
    head, with apply_top_p on each position's candidates before the draw."""
    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
    base_logits = self.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, -1)
    base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
    base_logits.fill_(float("-inf"))
    idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
    head = self.model.model.markov_head
    scale = self.model.logits_processor.scale
    top_p = STATE["top_p"]
    for i in range(n_spec):
        markov_embed = self.model.markov_embed(prev)
        logits_i = head.apply_bias_gathered(markov_embed, base_logits[:, i], base_values[:, i], draft_indices[:, i], scale)
        if top_p is not None:
            apply_top_p(logits_i, draft_indices[:, i], idx_map[:, i], self.temperature, top_p)
        draft_sampled_i = self._sample_logits(logits_i, idx_map[:, i], sample_pos[:, i], i)
        self.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i


def install():
    global _installed
    if _installed:
        return
    mod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    cls = mod.DSparkSpeculator
    orig = cls._sample_sequential

    def _sample_sequential(self, num_reqs, head_hidden):
        use = STATE["use"] and not getattr(self, "use_confidence_head", False)
        prev = getattr(self, "_draft_topk", None)
        self._draft_topk = TOPK if use else prev
        try:
            if use:
                return _sample_sequential_topk(self, num_reqs, head_hidden)
            return orig(self, num_reqs, head_hidden)
        finally:
            self._draft_topk = prev

    cls._sample_sequential = _sample_sequential
    _installed = True
    _LOG(f"installed: batches that sample with top-p draft from the request's top-p of the top-{TOPK} candidates")


def capture_scope(runner, key):
    """(enter, exit) callables for the runner's capture of graph `key`: on for keys that sample (not all greedy) with
    a top-p below 1 (the key's flags 4 = all greedy, 8 = any top_p != 1)."""
    use = bool(key is not None and len(key) > 8 and not key[4] and key[8])

    def enter():
        STATE["use"] = use
        STATE["top_p"] = runner.sampler.sampling_states.top_p.gpu if use else None
        if use:
            STATE["graphs"] += 1
            _LOG(f"graph {key}: drafts from the request's top-p (graphs so far {STATE['graphs']})")

    def leave():
        STATE["use"] = False
        STATE["top_p"] = None

    return enter, leave
