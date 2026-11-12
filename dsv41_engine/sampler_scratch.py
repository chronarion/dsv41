"""vLLM's top-k / top-p scratch buffer, kept alive once a CUDA graph has used it (a vLLM 0.32 bug, stock included).

apply_top_k_top_p_triton keeps one scratch tensor per (device, dtype, vocab) in _TRITON_BUFFER_CACHE and replaces it
when a batch needs more rows. A graph captured before the replacement still writes into the old tensor on every
replay, after it was freed: silent while the block stays free, wrong data once the allocator hands it to another
tensor, an illegal-address fault once a cache release unmaps it. Replaced buffers are retired here instead of freed
(the same rule FlashInfer uses for its own scratch). They add at most about one extra full-size buffer of memory."""
import importlib
import sys

_LOG = lambda m: sys.stderr.write(f"DSV41_SAMPLER_SCRATCH {m}\n")


class _RetainingDict(dict):
    def __init__(self, *a, retain=True, **kw):
        super().__init__(*a, **kw)
        self.retired, self.retain = [], retain

    def __setitem__(self, key, value):
        old = self.get(key)
        if old is not None and old is not value:
            if self.retain:
                self.retired.append(old)
            _LOG(f"grew {tuple(old.shape)} @ {old.data_ptr():#x} -> {tuple(value.shape)} @ {value.data_ptr():#x}, "
                 f"{'kept' if self.retain else 'FREED (log mode)'} the old buffer")
        elif old is None:
            _LOG(f"created {tuple(value.shape)} @ {value.data_ptr():#x} for {key[1:]}")
        super().__setitem__(key, value)

    def clear(self):   # reset_buffer_cache(): only safe with no graph alive, so keep the retired ones too
        self.retired.extend(v for v in self.values())
        super().clear()


def install(retain=True):
    mod = importlib.import_module("vllm.v1.sample.ops.topk_topp_triton")
    cache = getattr(mod, "_TRITON_BUFFER_CACHE", None)
    if cache is None:
        _LOG("not found (vLLM changed): nothing patched")
        return
    if not isinstance(cache, _RetainingDict):
        mod._TRITON_BUFFER_CACHE = _RetainingDict(cache, retain=retain)
    _LOG(("installed" if retain else "log only (old buffers are freed as in stock vLLM)") + f"; {len(cache)} buffers already cached")
