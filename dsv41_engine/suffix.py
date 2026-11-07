"""G6 / S27 / R9: suffix (copy) drafting on top of DSpark, with a wide batch-1 verify, in the C4 whole-step graph.

A port of ds41's `suffix` component (fastpath_w9_suffix.py with its device kernels w9_suffix_kernels.py, MIT,
vendored under kernels/vendor/ds41/fastpath) to vLLM 0.32 and to graph capture:

  every batch-1 step, after DSpark's 5 drafts: one launch matches the request's history (longest common suffix of
  the history, and of the history extended by the 5 drafts, against every earlier position) and writes a 31-token
  chain into the draft row: a copy of what followed the best match when it is >= GPU_TH tokens long, else DSpark's
  5 drafts followed by the copy after the extended match (or a filler token). Copied positions get q = one-hot in the
  rejection sampler's draft-logits cache, so the output distribution is the target's at any temperature (a
  deterministic draft verified with q(x) = 1 accepts with min(1, p(x)), and the residual is p without x).
  The scheduler reads the match length (host-mapped, one step old) and verifies 32 rows when it is >= TRIGGER,
  else 6. Batches of 2+ requests are untouched (DSpark only, 5 drafts).

Differences from ds41 (0.31 + its hook): the launch's speculative config already carries K = 31 and the batch-1
schedule entry (`speculative_config()` below builds it), so no EngineArgs patch; the matcher scans the runner's
UVA token history directly (no HBM mirror kept from the host: its window copies are sized on the host and cannot
live in a captured graph; the scan costs ~110 us at 512K tokens, ~2 us at 8K, so a device-side mirror is the
follow-up for very long contexts); DSpark's block is DSV41_DSPARK_K (5, or 7 with dynk.py choosing 5 or 7 per
step); the scheduler side lives in dynk.py, which widens a step only when the previous step ran the request alone
(so the chain past DSpark's block is this request's); the drafter's confidence buffer keeps DSpark's columns (0.32
writes it as one [n, K] block). Under the whole-step capture the
matcher's grid comes from the patched sequence-length upper bound (max_model_len), so one graph serves any length.
Enable with DSV41_SUFFIX=1 (install() before the model runner is built; the launch must use speculative_config())."""
import copy
import importlib
import importlib.util
import os
import sys

K_WIDE = int(os.environ.get("DSV41_SUFFIX_KWIDE", "31"))  # verify width K_WIDE + 1 at batch 1 when the history repeats
# (I15: 23 = 24-row copy chains, below the 32-row width where the RTX band is late)
NORMAL_K = int(os.environ.get("DSV41_DSPARK_K", "5"))  # DSpark's block (the checkpoint's 5, or 7 with dynamic K)
TRIGGER = int(os.environ.get("DSV41_SUFFIX_TRIGGER", "24"))  # the scheduler goes wide when the last GPU match length is >= this
GPU_TH = 16        # the chain is a pure copy when the history match is >= this, else DSpark + continuation
VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels/vendor/ds41/fastpath")
STATE = {"runner": None, "drafter": None, "wide": 0, "normal": 0, "logged": -1}
_LOG = lambda m: sys.stderr.write(f"DSV41_SUFFIX {m}\n")
_installed = False


def speculative_config(base: dict) -> dict:
    """The launch's speculative config: K = 31 with a batch-1 schedule entry, other batch sizes as in `base`."""
    sc = dict(base)
    out = []
    for lo, hi, k in sorted(([int(a), int(b), int(c)] for a, b, c in
                             (sc.get("num_speculative_tokens_per_batch_size") or [[1, 1, NORMAL_K]])),
                            key=lambda e: e[0]):
        if lo == 1:
            out.append([1, 1, K_WIDE])
            if hi > 1:
                out.append([2, hi, k])
        else:
            out.append([lo, hi, k])
    if not out or out[0][0] != 1:
        out.insert(0, [1, 1, K_WIDE])
    sc["num_speculative_tokens_per_batch_size"] = out
    sc["num_speculative_tokens"] = K_WIDE
    return sc


def _kernels_module():
    spec = importlib.util.spec_from_file_location("w9_suffix_kernels", os.path.join(VENDOR, "w9_suffix_kernels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dspark_view(vllm_config):
    """the same config with num_speculative_tokens = DSpark's block, for the drafter's own buffers and graphs"""
    cfg = copy.copy(vllm_config)
    spec = copy.copy(vllm_config.speculative_config)
    object.__setattr__(spec, "num_speculative_tokens", NORMAL_K)
    object.__setattr__(cfg, "speculative_config", spec)
    return cfg


def _patch_speculator(module):
    import torch

    cls = module.DSparkSpeculator
    if getattr(cls, "_dsv41_suffix", False):
        return
    orig_init, orig_propose = cls.__init__, cls.propose
    kmod = _kernels_module()

    def __init__(self, vllm_config, device):
        k_total = vllm_config.speculative_config.num_speculative_tokens
        self._sfx_k = k_total
        if k_total <= NORMAL_K:
            orig_init(self, vllm_config, device)
            self._sfx = None
            return
        orig_init(self, _dspark_view(vllm_config), device)
        # the rows the runner and the rejection sampler index: [max_num_reqs, K]; DSpark writes columns 0..4
        self.draft_tokens = torch.zeros(self.max_num_reqs, k_total, dtype=torch.int64, device=device)
        if self.draft_logits is not None:
            old = self.draft_logits
            new = torch.full((old.shape[0], k_total, old.shape[2]), float("-inf"), dtype=old.dtype, device=device)
            new[:, : old.shape[1]].copy_(old)
            self.draft_logits = new
            del old
        filler = int(getattr(self.draft_model_config.hf_config, "dspark_noise_token_id", 0))
        self._sfx = kmod.SuffixDrafter(self.max_num_reqs, k_total, NORMAL_K, filler, 2, device, trigger=TRIGGER)
        STATE["drafter"] = self._sfx
        nbytes = 0 if self.draft_logits is None else self.draft_logits.numel() * self.draft_logits.element_size()
        _LOG(f"DSpark block {NORMAL_K}, draft rows {k_total}, draft-logits cache {nbytes:,} B; trigger {TRIGGER}, "
             f"copy chain at {GPU_TH}")

    def propose(self, input_batch, *args, **kwargs):
        sfx = getattr(self, "_sfx", None)
        dummy = kwargs.get("dummy_run", False) or kwargs.get("is_profile", False)
        if sfx is not None and self.draft_logits is not None and not dummy:
            sfx.reset_onehot(self.draft_logits)
        out = orig_propose(self, input_batch, *args, **kwargs)
        runner = STATE["runner"]
        if sfx is None or dummy or runner is None or input_batch.num_reqs != 1:
            return out
        rs = runner.req_states
        # the matcher's grid covers the sequence-length upper bound (max_model_len under the whole-step capture)
        ub = int(input_batch.seq_lens_cpu_upper_bound[0])
        sfx.draft(rs.all_token_ids.gpu, input_batch.idx_mapping, rs.total_len.gpu, input_batch.idx_mapping,
                  self.draft_tokens[0], self.draft_logits, ub + self._sfx_k, GPU_TH)
        if not torch.cuda.is_current_stream_capturing():
            n = int(sfx.stats_host[0])
            if n // 20000 != STATE["logged"]:
                STATE["logged"] = n // 20000
                s = sfx.stats_host.tolist()
                _LOG(f"batch-1 proposals {s[0]}, copy chains {s[1]}, extensions {s[2]}, mean match "
                     f"{s[3] / max(1, s[0]):.2f}; scheduled wide {STATE['wide']}, normal {STATE['normal']}")
        return out

    cls.__init__ = __init__
    cls.propose = propose
    cls._dsv41_suffix = True


def _wide(req):
    """the request's next step verifies the K_WIDE copy chain: its last GPU match length is >= TRIGGER"""
    drafter, runner = STATE["drafter"], STATE["runner"]
    if drafter is None or runner is None:
        return False
    slot = runner.req_states.req_id_to_index.get(req.request_id)
    # the one-hot q rows cover a flag one step old; after a stock step (its output precedes its drafter) the host can be two behind
    wide = slot is not None and getattr(runner, "_ws_streak", 2) >= 2 and int(drafter.lflag_host[slot]) >= TRIGGER
    STATE["wide" if wide else "normal"] += 1
    return wide


def _patch_states(module):
    cls = module.RequestState
    if getattr(cls, "_dsv41_suffix", False):
        return
    orig = cls.add_request

    def add_request(self, req_id, *args, **kwargs):
        orig(self, req_id, *args, **kwargs)
        drafter = STATE["drafter"]
        if drafter is not None:  # a new request in a slot starts in normal mode
            drafter.lflag_host[self.req_id_to_index[req_id]] = 0

    cls.add_request = add_request
    cls._dsv41_suffix = True


def install():
    """patch the DSpark speculator, the scheduler and the request states (before the model runner is built)"""
    global _installed
    if _installed:
        return
    from . import dynk

    _patch_speculator(importlib.import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator"))
    dynk.register_wide(_wide, K_WIDE, NORMAL_K)
    dynk.install()
    _patch_states(importlib.import_module("vllm.v1.worker.gpu.states"))
    _installed = True
    _LOG(f"installed: batch-1 verify {1 + NORMAL_K} rows, or {1 + K_WIDE} when the history match is >= {TRIGGER}")


def register_runner(runner):
    STATE["runner"] = runner
