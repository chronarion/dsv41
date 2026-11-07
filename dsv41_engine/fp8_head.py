"""Lossy profile option: the verify LM head with FP8 weights (F9 / S10).

The head weight [V, H] bf16 is quantized once per row to e4m3 with scale = row amax / 448 (the quantization whose
KLD was measured: +0.0005 vs the official head); activations stay bf16 and the dot accumulates in
fp32, so this is a weight-only FP8 GEMV that reads half the bytes (0.66 GB instead of 1.32 GB per step). A Triton
kernel: each program streams a block of vocabulary rows through K and multiplies by the <= 16 activation rows.
Enable with DSV41_HEAD_FP8=1; the runner swaps model.compute_logits for decode-sized batches.

DSV41_HEAD_FP8=dg instead runs the verify head on the drafter's own quantized copy of the SAME lm_head (ds41's
tail_drafthead_mxfp8: DGHead, MXFP8 weights with ue8m0 scales per 32, two-term MXFP8 activations, DeepGEMM's sm100
block-scaled GEMM at ~6.1 TB/s): no extra HBM, finer weight scales than the per-row FP8 above."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_head_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, V, H, stride_xm, stride_wv, stride_om,
                     BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_v = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < H), other=0.0)
        w = tl.load(w_ptr + offs_v[:, None] * stride_wv + offs_k[None, :],
                    mask=(offs_v[:, None] < V) & (offs_k[None, :] < H), other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + offs_v, mask=offs_v < V, other=0.0)
    acc = acc * s[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_v[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_v[None, :] < V))


class FP8Head:
    def __init__(self, weight: torch.Tensor):
        w = weight.detach().float()
        scale = w.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
        self.w8 = (w / scale).to(torch.float8_e4m3fn).contiguous()
        self.scale = scale.squeeze(-1).contiguous()
        self.V, self.H = self.w8.shape
        del w

    def __call__(self, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
        M = x.shape[0]
        x = x.to(torch.bfloat16).contiguous()
        out = torch.empty(M, self.V, dtype=out_dtype, device=x.device)
        BLOCK_M = 16 if M <= 16 else 32
        assert M <= BLOCK_M, M
        BLOCK_V, BLOCK_K = 64, 256
        grid = (triton.cdiv(self.V, BLOCK_V),)
        _fp8_head_kernel[grid](x, self.w8, self.scale, out, M, self.V, self.H, x.stride(0), self.w8.stride(0),
                               out.stride(0), BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M, num_warps=4,
                               num_stages=4)
        return out


def drafter_dg_head(runner):
    """the drafter's DGHead (an MXFP8 copy of the lm_head it shares with the target), or None"""
    for path in (("speculator", "model"), ("speculator", "draft_model"), ("drafter", "model")):
        obj = runner
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        head = getattr(obj, "_w5_head", None) if obj is not None else None
        if head is not None:
            if getattr(head, "fmt", "mxfp8") != "mxfp8":
                # the drafter runs an MXFP4 copy (DSV41_DRAFTHEAD_FMT=mxfp4): the verify head gets its own MXFP8 one
                own = getattr(runner, "_dsv41_verify_dghead", None)
                if own is None:
                    own = type(head)(_target_lm_head(runner), "mxfp8", two_term=True)
                    own.warmup(32)
                    runner._dsv41_verify_dghead = own
                return own
            return head
    return None


def _target_lm_head(runner):
    lm = getattr(runner.model, "language_model", runner.model)
    return lm.lm_head.weight
