"""Sparse prefill index rows, cleared before they are filled (a vLLM 0.32 bug, stock included).

combine_topk_swa_indices stores only a row's valid entries when the caller hands it a workspace `out` (without one it starts
from a -1 fill), and the prefill attention kernel reads past those entries. A row shorter than the full width (the first
window of a chunk, every kept row of the decoder replay) then attends stale indices. With short prompts alone these fall
outside the small gathered KV and are skipped; with a long request in the batch they land in workspace bytes that hold
anything, including NaN."""
import importlib
import sys

_LOG = lambda m: sys.stderr.write(f"DSV41_PREFILL_INDICES {m}\n")


def install():
    ops = importlib.import_module("vllm.models.deepseek_v41.common.ops.cache_utils")
    orig = ops.combine_topk_swa_indices

    def combine_topk_swa_indices(*args, out=None, **kwargs):
        if out is not None:
            out[0].fill_(-1)
        return orig(*args, out=out, **kwargs)

    ops.combine_topk_swa_indices = combine_topk_swa_indices
    for name, mod in list(sys.modules.items()):   # the attention backends import it by name
        if name.startswith("vllm.models.deepseek_v41") and getattr(mod, "combine_topk_swa_indices", None) is orig:
            mod.combine_topk_swa_indices = combine_topk_swa_indices
    _LOG("installed")
