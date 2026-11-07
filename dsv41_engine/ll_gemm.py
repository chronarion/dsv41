"""G3 / S29: MXFP8 dense linears at M <= 8 on FlashInfer's split-K low-latency CuTe kernel, with pinned tactics.

vLLM 0.32's FlashInferCutedslMxfp8LinearKernel always asks mm_mxfp8 for backend="cute-dsl", whose persistent kernel
launches one CTA per 128-wide N tile (40 CTAs for wo_b, 14 for wq_a + wkv on a 152-SM GB300), so decode-sized GEMMs
stream weights at 2-4.5 TB/s. "cutedsl_low_latency" (M <= 8) splits K over more CTAs and takes the same operands
(column-major [K, N] weight, 128x4-swizzled UE8M0 scales). ds41 measured, L2-cold at M=6 under graphs: wq_a + wkv
4.64 -> 3.39 us, wq_b 9.28 -> 7.53, wo_b 11.63 -> 9.43 (its hook/ll_gemm.py, MIT).

Tactics: ds41's pinned FlashInfer autotune file (`fiset-w6pf-post1-c7483c4a.json`, vendored under
kernels/vendor/ds41/fastpath/hook/fi-tune), written by FlashInfer 0.7.0.post1 on a GB300, the FlashInfer vLLM 0.32
ships. FlashInfer ignores a file whose environment metadata differs; the FlashInfer version and the GPU must match
here, the other fields (CUDA, cuBLAS, cuDNN, CuTe-DSL stack) are logged and wildcarded ("*"), since the low-latency
tactics are tile / stage / cluster / split-K choices of the same kernel source. Without the file an untuned shape
takes FlashInfer's default tactic inside graph capture.

Numerics: FP8 x FP8 with FP32 accumulation like the persistent kernel; split-K changes only the summation order.
Enable with DSV41_LL_GEMM=1 (harness: before the LLM is built; served: the runner installs it at load).

I6 / J2a (DSV41_LL_WIDE=1): 8 < M <= 32 on the same low-latency kernel class compiled with a 16- or 32-row activation
tile (vLLM #63921, Apache-2.0: FlashInfer's runner offers only the 8-row tile), a fixed tactic per (K, N, tile)
(DSV41_LL_WIDE_TACTICS overrides), weights streamed before the PDL wait like the 8-row kernel. Compiled at install
time on the current device, never inside a capture. Same numerics class (FP32 accumulation, split-K order)."""
import json
import os
import sys
import tempfile

LL_MAX_M = 8
# (K, N) measured faster on cutedsl_low_latency at M <= 8 (ds41): wq_a + wkv, wq_b, wo_b
LL_SHAPES = {(5120, 1792), (1280, 32768), (8192, 5120)}
WIDE_MAX_M = 32 if os.environ.get("DSV41_LL_WIDE", "0") == "1" else 0
# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (K, N) and activation tile; starting points from vLLM #63921's GB300
# TP4 table (wq_a + wkv is the same shape; wq_b has 4x its N, wo_b 4x its K here), to be swept in the chain
WIDE_TACTICS = {  # standalone L2-cold sweep 2026-11-05 (widell/test_wide_ll.py): us at M=16 / M=24 vs persistent
    (5120, 1792): {16: (128, 9, 1, 4), 32: (128, 8, 1, 4)},    # 3.89 / 4.60 vs 6.75 / 6.96
    (1280, 32768): {16: (128, 5, 1, 1), 32: (128, 5, 1, 1)},   # 8.11 / 8.27 vs 9.60 / 9.49 (bitwise: split 1)
    (8192, 5120): {16: (128, 10, 1, 2), 32: (128, 8, 1, 2)},   # 9.52 / 10.33 vs 11.25 / 11.44
}
_wide_kernels = {}
TARGET = "vllm.model_executor.kernels.linear.mxfp8.flashinfer"
TUNE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "kernels/vendor/ds41/fastpath/hook/fi-tune/fiset-w6pf-post1-c7483c4a.json")
_LOG = lambda m: sys.stderr.write(f"DSV41_LL_GEMM {m}\n")
_installed = False


def _load_tactics(path):
    from flashinfer.autotuner import AutoTuner
    from flashinfer.autotuner.autotuner import _collect_metadata

    data = json.load(open(path))
    saved = data.get("_metadata") or {}
    cur = _collect_metadata()
    for k in ("flashinfer_version", "gpu"):
        if saved.get(k) not in (cur.get(k), "*"):
            raise RuntimeError(f"pinned tactics {path}: {k} saved={saved.get(k)} current={cur.get(k)}; refusing")
    loose = {k: (v, cur.get(k)) for k, v in saved.items() if k in cur and v not in (cur[k], "*")}
    for k in loose:
        saved[k] = "*"
    data["_metadata"] = saved
    # DSV41_LL_TACTICS='{"8192x5120": {"6": [128, 10, 1, 4], "8": [128, 10, 1, 4]}}' (H2): override or add the
    # low-latency tactic (cta_k, ab_stages, sfb_stages, split_k) for a (K, N) shape at an M bucket; each new key is
    # the file's key for that shape at another M with only M replaced
    over = json.loads(os.environ.get("DSV41_LL_TACTICS") or "{}")
    for shape, by_m in over.items():
        K, N = (int(x) for x in shape.split("x"))
        tmpl = next(k for k in data if "LowLatency" in k and f"({K}, {N})" in k)
        m0 = tmpl.split("((", 1)[1].split(",", 1)[0]
        for m, tactic in by_m.items():
            key = tmpl.replace(f"(({m0}, {K})", f"(({int(m)}, {K})", 1)
            data[key] = ["CutedslLowLatencyBlockscaledGemmRunner", list(tactic)]
            _LOG(f"tactic override ({K}, {N}) M={m}: {tactic}")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
    ok = AutoTuner.get().load_configs(f.name)
    os.unlink(f.name)
    n = sum(1 for k in data if "LowLatency" in k)
    _LOG(f"tactics {os.path.basename(path)}: loaded={ok}, {n} low-latency entries; wildcarded {loose or 'nothing'}")
    if not ok:
        raise RuntimeError(f"FlashInfer refused the pinned tactics {path}")


def _gmem_ptr(dtype, address, align):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _wide_kernel(tactic, tile, device_index):
    """FlashInfer's LowLatencyBlockscaledGemmKernel with a `tile`-row activation tile (vLLM #63921's _wide_kernel)"""
    key = (tuple(tactic), tile, device_index)
    compiled = _wide_kernels.get(key)
    if compiled is None:
        import cutlass
        import cutlass.cute as cute
        import torch
        from cutlass.cute.runtime import make_fake_stream
        from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import LowLatencyBlockscaledGemmKernel

        cta_k, ab_stages, sfb_stages, split_k = tactic
        gemm = LowLatencyBlockscaledGemmKernel(acc_dtype=cutlass.Float32, mma_tiler_mnk=(128, tile, cta_k),
                                               num_ab_stage=ab_stages, num_sfb_tmem_stage=sfb_stages, sf_vec_size=32,
                                               use_pdl=True, split_k=split_k)
        f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
        with torch.cuda.device(device_index):
            compiled = cute.compile(gemm, _gmem_ptr(f8, 16, 16), _gmem_ptr(sf, 32, 32), _gmem_ptr(f8, 16, 16),
                                    _gmem_ptr(sf, 16, 16), _gmem_ptr(bf16, 16, 16), None, None,
                                    tuple(cutlass.Int32(v) for v in (128, tile, cta_k, 1)), make_fake_stream())
        _wide_kernels[key] = compiled
    return compiled


def _wide_launch(compiled, weight, weight_scale, a, a_scale, out):
    """out [m, N] bf16 = a [m, K] (MXFP8, 128x4-swizzled scales) x weight (column-major [K, N]); the kernel computes
    out.T with the weight as its A operand (its M is the weight's N)"""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import torch

    k, n = weight.shape
    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    compiled(_gmem_ptr(f8, weight.data_ptr(), 16), _gmem_ptr(sf, weight_scale.data_ptr(), 32),
             _gmem_ptr(f8, a.data_ptr(), 16), _gmem_ptr(sf, a_scale.data_ptr(), 16), _gmem_ptr(bf16, out.data_ptr(), 16),
             None, None, tuple(cutlass.Int32(v) for v in (n, a.shape[0], k, 1)),
             cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream))


def _wide_tactics():
    tac = {k: dict(v) for k, v in WIDE_TACTICS.items()}
    # DSV41_LL_WIDE_TACTICS='{"8192x5120": {"16": [128, 8, 1, 4], "32": [128, 10, 1, 2]}}' (sweeps)
    for shape, by_tile in json.loads(os.environ.get("DSV41_LL_WIDE_TACTICS") or "{}").items():
        K, N = (int(x) for x in shape.split("x"))
        for tile, tactic in by_tile.items():
            tac.setdefault((K, N), {})[int(tile)] = tuple(tactic)
    return tac


def precompile_wide(device_index=None):
    """compile every (shape, tile) tactic of the wide path on one device, outside any capture"""
    import torch

    if not WIDE_MAX_M:
        return
    dev = torch.cuda.current_device() if device_index is None else device_index
    tac = _wide_tactics()
    for (K, N), by_tile in sorted(tac.items()):
        for tile, tactic in sorted(by_tile.items()):
            _wide_kernel(tactic, tile, dev)
    _LOG(f"wide path 8 < M <= {WIDE_MAX_M}: compiled {sum(len(v) for v in tac.values())} kernels on device {dev}: "
         f"{ {f'{K}x{N}': v for (K, N), v in sorted(tac.items())} }")


def _patch(module):
    cls = module.FlashInferCutedslMxfp8LinearKernel
    if getattr(cls, "_dsv41_ll_gemm", False):
        return
    from vllm.model_executor.layers.fusion.quant_activation import as_quantized_activation
    from vllm.utils import flashinfer as vllm_flashinfer
    import torch

    wide_tac = _wide_tactics()
    orig_apply = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        weight = layer.weight  # [K, N], column-major
        K, N = weight.shape
        qa = as_quantized_activation(x, self.input_quant_key())
        m = (qa.data if qa is not None else x).numel() // K
        if (K, N) not in LL_SHAPES or bias is not None or m > max(LL_MAX_M, WIDE_MAX_M):
            return orig_apply(self, layer, x, bias)
        if m > LL_MAX_M:  # I6 / J2a: the wide low-latency kernel (16- or 32-row tile)
            tile = (m + 15) // 16 * 16
            tactic = wide_tac.get((K, N), {}).get(tile)
            if tactic is None:
                return orig_apply(self, layer, x, bias)
            if qa is not None:
                if not (qa.scale.is_contiguous() and qa.data.is_contiguous()) or qa.orig_dtype != torch.bfloat16:
                    return orig_apply(self, layer, x, bias)
                a, a_scale, input_shape = qa.data.view(-1, K), qa.scale, qa.orig_shape
            else:
                if x.dtype != torch.bfloat16:
                    return orig_apply(self, layer, x, bias)
                input_shape = x.shape
                a, a_scale = module.mxfp8_e4m3_quantize(x.reshape(-1, K), is_sf_swizzled_layout=True)
            out = torch.empty(m, N, dtype=torch.bfloat16, device=a.device)
            _wide_launch(_wide_kernel(tactic, tile, a.device.index), weight, layer.weight_scale, a, a_scale, out)
            return out.view(*input_shape[:-1], N)
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            out_dtype, input_shape = x.dtype, x.shape
            # the module's own name, looked up per call: a component may wrap it (W4 boundary MXFP8_OUT)
            input_mxfp8, input_scale = module.mxfp8_e4m3_quantize(x.view(-1, K), is_sf_swizzled_layout=True)
        out = vllm_flashinfer.mm_mxfp8(input_mxfp8.view(-1, K), weight, input_scale, layer.weight_scale,
                                       out_dtype=out_dtype, backend="cutedsl_low_latency")
        return out.view(*input_shape[:-1], N)

    cls.apply_weights = apply_weights
    cls._dsv41_ll_gemm = True
    _LOG(f"ACTIVE {TARGET}.FlashInferCutedslMxfp8LinearKernel: M <= {LL_MAX_M}, (K, N) in {sorted(LL_SHAPES)} -> "
         "cutedsl_low_latency" + (f"; {LL_MAX_M} < M <= {WIDE_MAX_M} -> the wide low-latency kernel" if WIDE_MAX_M else ""))


def install(tune_file=None):
    """load the pinned tactics into FlashInfer's autotuner and route the three shapes at M <= 8 to the LL kernel"""
    global _installed
    if _installed:
        return
    import importlib
    _load_tactics(tune_file or os.environ.get("DSV41_LL_TUNE_FILE") or TUNE_FILE)
    _patch(importlib.import_module(TARGET))
    precompile_wide()
    _installed = True
