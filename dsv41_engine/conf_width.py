"""K3: the batch-1 verify width chosen on the device from the current drafts.

Every batch-1 step verifies DSpark's 7 drafts (a fixed 1x7 graph; DynK off). Right after the drafter, a few capturable
ops read the drafter's q cache for the 7 new drafts, take each position's confidence c_j = max_x q_j(x) (temperature
1), and keep the longest prefix whose survival prod(c_1..c_j) >= LAMBDA (at least 1): the request's width W. In the
next step (which verifies those drafts):
  - the MoE router treats verify rows beyond W as padding (no hot or cold expert: their MoE work is skipped), through
    fastpath_boundary.CW_MASK inside the whole-step decode graphs;
  - the rejection sampler sees -1 for drafts beyond W (vLLM's placeholder that ends verification), so at most W drafts
    are accepted.
Rows beyond W still run attention and the dense layers; their KV lands at positions the step rolls back. The output
distribution is unchanged (verifying a prefix of the drafts is always exact); only the work per step and the accepted
length change. Offline replay (k3/replay.py, greedy prose at K=7): +4..+9% tokens per ms over a K=5 graph at
LAMBDA 0.2-0.3, depending on what a masked row saves.

Enable with DSV41_CONF_WIDTH=<lambda> (0 = off) and a batch-1 schedule of 7 drafts (DSV41_DYNK=0). install() after
suffix / dynk (it wraps the speculator's propose).

DSV41_CONF_WIDTH_MODE=graphs (K3b): no masked rows. The width also lands in a pinned host word at the end of each
step (a D2H copy inside the captured step), and the runner replays a graph captured for exactly W drafts (rows
1..W+1 of the target forward, the sampler and rejection over W drafts, the drafter as always): runner.py `_k3b_*`.
Served K3 with masked rows (c4m22bcw) saved only ~0.16 ms per masked row; a W-draft graph saves the whole row.

DSV41_CONF_WIDTH_B=<lambda>: the same masked rows at batch >= 2, a width per request (the graphs stay B x (K + 1) rows):
the survival rule with this lambda; right after prepare_inputs each row compares its position with its request's width,
inside the graph, so a reordered batch keeps the right rows. Off at 0; independent of the batch-1 mode.

DSV41_CONF_HEAD=1: the checkpoint's confidence head (`mtp.2.confidence_head`, sigmoid over [head hidden; Markov embed of
the previous token], trained on the acceptance probability) next to the max-probability signal: STATE["head"] [slot, K],
logged by the runner's DSV41_CONF_LOG. Far better calibrated than max-probability, yet a width policy gains nothing from
it (k3log2: prose +5.8% vs +5.9%), so the rules stay on max-probability."""
import importlib
import os
import sys

import torch

from vllm.triton_utils import tl, triton

LAMBDA = float(os.environ.get("DSV41_CONF_WIDTH", "0") or 0)
KD = int(os.environ.get("DSV41_DSPARK_K", "7"))
MODE = os.environ.get("DSV41_CONF_WIDTH_MODE", "mask")   # "mask" (rows beyond W masked) or "graphs" (K3b)
# K3b's rule: widen while expected tokens per ms, (1 + sum_j<=W prod_i<=j SHRINK c_i) / T(W), improves, with T the served
# batch-1 step period per width (replay of the K=7 log at these periods: prose +5.9%, T=1 +5.3% over K=5; the survival
# rule +5.7% / +4.3%). SHRINK corrects the drafter's overconfidence (position 1: c in [0.6, 0.8) accepted 54%)
SHRINK = float(os.environ.get("DSV41_CONF_WIDTH_SHRINK", "0.93"))
PERIODS = [float(x) for x in os.environ.get("DSV41_CONF_WIDTH_T", "5.80,6.28,6.44,6.71,6.98,7.30,7.63").split(",")]
LAMBDA_B = float(os.environ.get("DSV41_CONF_WIDTH_B", "0") or 0)
HEAD = int(os.environ.get("DSV41_CONF_HEAD", "0") or 0)
STATE = {"width": None, "mask": None, "steps": 0, "width_host": None, "width_np": None, "head": None, "masked": False}
_LOG = lambda m: sys.stderr.write(f"DSV41_CONF_WIDTH {m}\n")
_installed = False


NS = 16        # vocabulary splits of the confidence reduction


@triton.jit
def _conf_part(dl, slots, pm, ps, stride_r, stride_k, V, KD: tl.constexpr, NSPLIT: tl.constexpr, BLOCK: tl.constexpr):
    """program (request, draft position, vocabulary split): the split's max logit and sum of exp"""
    i = tl.program_id(0)
    j = tl.program_id(1)
    s = tl.program_id(2)
    slot = tl.maximum(tl.load(slots + i), 0).to(tl.int64)  # dummy / profiling batches carry -1
    base = dl + slot * stride_r + j * stride_k
    chunk = tl.cdiv(tl.cdiv(V, NSPLIT), BLOCK) * BLOCK
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for off in range(s * chunk, tl.minimum((s + 1) * chunk, V), BLOCK):
        col = off + tl.arange(0, BLOCK)
        x = tl.load(base + col, mask=col < V, other=float("-inf")).to(tl.float32)
        nm = tl.maximum(m, x)
        acc = acc * tl.exp(tl.where(m == nm, 0.0, m - nm)) + tl.where(x == float("-inf"), 0.0, tl.exp(x - nm))
        m = nm
    gm = tl.max(m, 0)
    tl.store(pm + (i * KD + j) * NSPLIT + s, gm)
    tl.store(ps + (i * KD + j) * NSPLIT + s, tl.sum(acc * tl.where(m == float("-inf"), 0.0, tl.exp(m - gm)), 0))


@triton.jit
def _mul(a, b):
    return a * b


@triton.jit
def _conf_policy(pm, ps, slots, width, periods, seq, word_host, lam, shrink, KD: tl.constexpr, KP: tl.constexpr,
                 NSPLIT: tl.constexpr, RATE: tl.constexpr):
    """program per request: c_j = max_x q_j(x) from the splits, then the width. RATE (batch 1): widen while tokens per
    ms improve, publish (sequence << 8 | width) to the pinned word; else the survival rule"""
    i = tl.program_id(0)
    j = tl.arange(0, KP)
    sp = tl.arange(0, NSPLIT)
    inside = j < KD
    off = (i * KD + j)[:, None] * NSPLIT + sp[None, :]
    m = tl.load(pm + off, mask=inside[:, None], other=0.0)
    l = tl.load(ps + off, mask=inside[:, None], other=1.0)
    gm = tl.max(m, 1)
    c = 1.0 / tl.sum(l * tl.exp(m - gm[:, None]), 1)
    slot = tl.maximum(tl.load(slots + i), 0)
    if RATE:
        surv = tl.associative_scan(tl.where(inside, c * shrink, 1.0), 0, _mul)
        rate = (1.0 + tl.cumsum(tl.where(inside, surv, 0.0), 0)) / tl.load(periods + j, mask=inside, other=1.0)
        w = 1
        go = 1
        for t in tl.static_range(1, KD):
            cur = tl.sum(tl.where(j == t, rate, 0.0), 0)
            prev = tl.sum(tl.where(j == t - 1, rate, 0.0), 0)
            go = go * (cur > prev).to(tl.int32)
            w = w + go
        n = tl.load(seq) + 256
        tl.store(seq, n)
        tl.store(word_host, n + w)
    else:
        surv = tl.associative_scan(tl.where(inside, c, 1.0), 0, _mul)
        w = tl.maximum(tl.sum(tl.where(inside & (surv >= lam), 1, 0), 0), 1)
    tl.store(width + slot, w)


class _HostPtr:
    """the pinned word as a Triton pointer argument (UVA: its host address is valid on the device)"""
    dtype = torch.int64

    def __init__(self, t):
        self.ptr = t.data_ptr()

    def data_ptr(self):
        return self.ptr


def _buffers(device, max_reqs):
    if STATE["width"] is not None:
        return
    STATE["width"] = torch.full((max_reqs,), KD, dtype=torch.int32, device=device)
    fb = sys.modules.get("fastpath_boundary")
    if MODE != "graphs" or LAMBDA_B > 0:
        STATE["mask"] = torch.zeros(max_reqs * (KD + 1), dtype=torch.bool, device=device)
        if fb is not None:
            fb.CW_MASK = STATE["mask"]
        _LOG(f"rows beyond a request's width masked at batch {'>= 2' if MODE == 'graphs' else '>= 1'}, lambda "
             f"{LAMBDA_B if MODE == 'graphs' else LAMBDA} ({'router mask on' if fb is not None else 'acceptance cap only'})")
    STATE["pm"] = torch.zeros(max_reqs * KD * NS, dtype=torch.float32, device=device)
    STATE["ps"] = torch.zeros(max_reqs * KD * NS, dtype=torch.float32, device=device)
    if MODE == "graphs":
        # one 64-bit word for the host to spin on: sequence << 8 | width (read as one load), stored by _conf_policy
        STATE["seq"] = torch.zeros(1, dtype=torch.int64, device=device)
        STATE["width_host"] = torch.full((1,), KD, dtype=torch.int64).pin_memory()
        STATE["width_np"] = STATE["width_host"].numpy()
        STATE["periods"] = torch.tensor(PERIODS[:KD], dtype=torch.float32, device=device)
        _LOG(f"lambda {LAMBDA}: batch-1 width from the draft confidences, per-width step graphs (runner K3b)")


def _after_propose(spec, input_batch):
    dl = spec.draft_logits
    _buffers(dl.device, dl.shape[0])
    n = input_batch.num_reqs
    lam = LAMBDA_B if n > 1 else LAMBDA
    if lam <= 0:
        return
    # The width of draft j may depend only on drafts before j. c_j is known before draft j is sampled, so both rules stop
    # at the first position they refuse; a global argmax would let the admission of draft j depend on c_{j+1}, which depends
    # on the sampled draft j, and that biases the output at T > 0 (DSpark appendix A; k3/width_bias.py)
    rate = n == 1 and MODE == "graphs"
    slots = input_batch.idx_mapping
    _conf_part[(n, KD, NS)](dl, slots, STATE["pm"], STATE["ps"], dl.stride(0), dl.stride(1), dl.shape[-1], KD=KD,
                            NSPLIT=NS, BLOCK=2048, num_warps=4)
    _conf_policy[(n,)](STATE["pm"], STATE["ps"], slots, STATE["width"], STATE.get("periods", STATE["pm"]),
                       STATE.get("seq", STATE["pm"]), _HostPtr(STATE["width_host"]) if rate else STATE["pm"], lam, SHRINK,
                       KD=KD, KP=triton.next_power_of_2(KD), NSPLIT=NS, RATE=rate, num_warps=1)


def _after_prepare(runner, ib):
    """verify rows beyond each request's width, as this step's own batch orders them (inside the graph): rows 1..K are
    drafts 1..K, row 0 the anchor. Only whole-step graphs mask; the rejection wrapper follows STATE["masked"]."""
    fb = sys.modules.get("fastpath_boundary")
    lam = LAMBDA_B if ib.num_reqs > 1 else (0.0 if MODE == "graphs" else LAMBDA)
    # only where the mask skips work: the masked router (router_mega) takes T <= ROUTER_MEGA_MAX_T; above it (batch 7-8
    # at 3 drafts) a mask would only cap acceptance (Fable r11 2.2: C8 +2.7%)
    on = (fb is not None and fb.WS_DECODE[0] and lam > 0 and ib.num_draft_tokens > 0
          and ib.num_tokens <= getattr(fb, "ROUTER_MEGA_MAX_T", 0))
    STATE["masked"] = on
    if fb is not None:
        fb.CW_ROWS = (ib.num_tokens,) if on else ()
    if on:
        _buffers(runner.device, runner.max_num_reqs)
        w = STATE["width"].index_select(0, ib.expanded_idx_mapping.long().clamp_min(0))
        torch.gt(ib.expanded_local_pos, w, out=STATE["mask"][:ib.num_tokens])


def _record_head(spec, num_reqs, head_hidden):
    k = spec.num_speculative_steps
    n = num_reqs * k
    anchor = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]].to(spec.draft_tokens.dtype)
    prev = torch.cat([anchor[:, None], spec.draft_tokens[:num_reqs, :k - 1]], 1).reshape(-1)
    c = spec.model.compute_confidence(head_hidden[spec.sample_indices[:n]], spec.model.markov_embed(prev))
    slot = spec.sample_idx_mapping[:n].view(num_reqs, k)[:, 0].long().clamp_min(0)
    STATE["head"].index_copy_(0, slot, c.view(num_reqs, k).float())


def _install_head(cls):
    load, sample = cls.load_draft_model, cls._sample_sequential

    def load_draft_model(self, *args, **kwargs):
        model = load(self, *args, **kwargs)
        if model.model.confidence_head is None:
            raise RuntimeError("DSV41_CONF_HEAD: the checkpoint has no confidence head")
        STATE["head"] = torch.ones(self.max_num_reqs, self.num_speculative_steps, dtype=torch.float32,
                                   device=self.device)
        _LOG(f"confidence head on ({self.num_speculative_steps} positions)")
        return model

    def _sample_sequential(self, num_reqs, head_hidden):
        out = sample(self, num_reqs, head_hidden)
        _record_head(self, num_reqs, head_hidden)
        return out

    cls.load_draft_model, cls._sample_sequential = load_draft_model, _sample_sequential


def install():
    global _installed
    if _installed or (LAMBDA <= 0 and LAMBDA_B <= 0 and not HEAD):
        return
    smod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    cls = smod.DSparkSpeculator
    if HEAD:
        _install_head(cls)
    if LAMBDA <= 0 and LAMBDA_B <= 0:
        _installed = True
        return
    prev = cls.propose

    def propose(self, input_batch, *args, **kwargs):
        out = prev(self, input_batch, *args, **kwargs)
        dummy = kwargs.get("dummy_run", False) or kwargs.get("is_profile", False)
        if not dummy and self.draft_logits is not None:
            _after_propose(self, input_batch)
        return out

    cls.propose = propose
    if MODE == "graphs" and LAMBDA_B <= 0:  # no masked rows: the runner replays a graph of exactly W drafts
        _installed = True
        _LOG(f"installed (K={KD}, lambda {LAMBDA}, per-width graphs)")
        return

    mr = importlib.import_module("vllm.v1.worker.gpu.model_runner")
    prepare = mr.GPUModelRunner.prepare_inputs

    def prepare_inputs(self, *args, **kwargs):
        ib = prepare(self, *args, **kwargs)
        _after_prepare(self, ib)
        return ib

    mr.GPUModelRunner.prepare_inputs = prepare_inputs
    rmod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.rejection_sampler")
    orig_gather = rmod.gather_draft_sampled

    def gather_draft_sampled(input_ids, positions, logits_indices, expanded_idx_mapping, expanded_local_pos,
                             prefill_len):
        draft_sampled, pos = orig_gather(input_ids, positions, logits_indices, expanded_idx_mapping,
                                         expanded_local_pos, prefill_len)
        if STATE["masked"]:
            beyond = expanded_local_pos > STATE["width"][expanded_idx_mapping.long().clamp_min(0)]  # padded rows: -1
            draft_sampled.masked_fill_(beyond, -1)
        return draft_sampled, pos

    rmod.gather_draft_sampled = gather_draft_sampled
    _installed = True
    _LOG(f"installed (K={KD}, lambda {LAMBDA}, batch >= 2 lambda {LAMBDA_B})")
