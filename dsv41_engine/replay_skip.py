"""A4 / S60: skip the decoder-side replay of prefill chunks whose replay output nothing reads.

vLLM 0.32 runs V4.1's decoder side (layer 20's rest and layers 21-39, SWA-only KV) as a bounded replay on each
prefill chunk's last `window` (128) rows (`DeepseekV41ModelState._prepare_replay_batch`). Those rows' window KV is read
only by the next 128 positions: the final chunk's replay reads window KV from max(replay_start, seq_len - window), decode
reads the window [L - 127, L], and the DSpark layers are window-only. So when a request still has >= window prompt
tokens after this chunk, this chunk's replay (20 layers on 128 rows, 20 small RTX calls) is never read. Here its kept
rows drop to 0. When no request of the batch keeps a row (a long prompt's intermediate chunk at c=1), the replay
layers do not run at all: the first replay layer still writes every row's KV (`write_batch_kv`, which decode reads),
and the layers' outputs are zeros of the shapes recorded from the first real replay (nothing reads them). Until a shape
is recorded, one row is kept (first measured version: a 1-row eager replay cost more than the 128-row graph it replaced).

Enable with DSV41_REPLAY_SKIP=1 (install() before the model runner is built). Requests that keep all rows
(`_keeps_rows_np`) and requests without a known prompt length are left alone."""
import importlib
import inspect
import os
import sys
import textwrap

import numpy as np

STATS = {"skipped": 0, "steps": 0, "logged": 0}
_LOG = lambda m: sys.stderr.write(f"DSV41_REPLAY_SKIP {m}\n")
_installed = False
_ANCHOR = """        kept_lens = np.where(
            self._keeps_rows_np[input_batch.idx_mapping_np[:num_reqs]],
            query_lens,
            np.minimum(query_lens, layers.window),
        )
"""


def trim_unread(state, input_batch, num_reqs, query_lens, kept_lens, window):
    rs = getattr(state, "_dsv41_req_states", None)
    if rs is None:
        return kept_lens
    idx = input_batch.idx_mapping_np[:num_reqs]
    seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
    prompt = rs.prefill_len.np[idx]
    skip = (input_batch.is_prefilling_np[:num_reqs] & ~state._keeps_rows_np[idx] & (kept_lens > 0)
            & (prompt - seq_lens >= window))
    if not skip.any():
        return kept_lens
    kept = kept_lens.copy()
    kept[skip] = 0
    if int(kept.sum()) > 0 or getattr(state.decoder_replay_layers, "_dsv41_out_meta", None) is None:
        # The replay batch runs (another request keeps rows, or no output shapes are recorded yet): give each skipped
        # request one row, never zero. A zero-row request sorts first as a "decode" in vLLM's metadata split, and when
        # no real decode follows, the indexer gets num_decodes > 0 with zero decode tokens and divides by zero
        # (sparse_attn_indexer: num_decode_tokens % decode_lens.shape[0]): several long prompts prefilling at once.
        # One row is the shape vLLM itself produces for a 1-token final chunk; its replay output is unread.
        kept[skip] = 1
    STATS["skipped"] += int(skip.sum())
    STATS["steps"] += 1
    if STATS["steps"] >= 2 * max(1, STATS["logged"]):
        STATS["logged"] = STATS["steps"]
        _LOG(f"replays skipped: {STATS['skipped']} request-chunks over {STATS['steps']} steps "
             f"({STATS.get('empty', 0)} steps ran no replay layer)")
    return kept


class _Skip:
    """replay_batch marker: run no replay layer, return zeros (rows = None)"""
    rows = None


SKIP = _Skip()


def skip_all(layers, kept_lens):
    """True (and the marker set) when no request keeps a replay row and the output shapes are known"""
    if int(kept_lens.sum()) != 0 or getattr(layers, "_dsv41_out_meta", None) is None:
        return False
    layers.replay_batch = SKIP
    STATS["empty"] = STATS.get("empty", 0) + 1
    return True


def _patch_layers(cls):
    import torch

    orig_run = cls._run

    def _run(self, hidden_states, *states):
        if self.replay_batch is SKIP:
            n = hidden_states.shape[0]
            return tuple(torch.zeros((n, *shape), dtype=dt, device=hidden_states.device)
                         for shape, dt in self._dsv41_out_meta)
        out = orig_run(self, hidden_states, *states)
        if getattr(self, "_dsv41_out_meta", None) is None:
            self._dsv41_out_meta = [(tuple(o.shape[1:]), o.dtype) for o in out]
        return out

    cls._run = _run


def install():
    """patch DeepseekV41ModelState: remember the request states, trim the unread replays"""
    global _installed
    if _installed:
        return
    mod = importlib.import_module("vllm.models.deepseek_v41.nvidia.model_state")
    cls = mod.DeepseekV41ModelState
    raw = inspect.getsource(cls._prepare_replay_batch)  # class-level source: the anchor carries its 8 spaces
    if raw.count(_ANCHOR) != 1:
        raise RuntimeError("DSV41_REPLAY_SKIP: vLLM's _prepare_replay_batch changed; the patch anchor is missing")
    raw = raw.replace(_ANCHOR, _ANCHOR + "        kept_lens = _dsv41_trim_unread(self, input_batch, num_reqs, query_lens, "
                                         "kept_lens, layers.window)\n"
                                         "        if _dsv41_skip_all(layers, kept_lens):\n"
                                         "            return\n", 1)
    # exec'd outside the class body: zero-argument super() has no __class__ cell
    raw = raw.replace("super().", "super(DeepseekV41ModelState, self).")
    src = textwrap.dedent(raw)
    ns = dict(vars(mod))
    ns["_dsv41_trim_unread"] = trim_unread
    ns["_dsv41_skip_all"] = skip_all
    exec(compile(src, f"<dsv41 replay_skip: {mod.__file__}>", "exec"), ns)
    cls._prepare_replay_batch = ns["_prepare_replay_batch"]

    orig_prepare_inputs = cls.prepare_inputs

    def prepare_inputs(self, input_batch, req_states, *args, **kwargs):
        self._dsv41_req_states = req_states
        return orig_prepare_inputs(self, input_batch, req_states, *args, **kwargs)

    cls.prepare_inputs = prepare_inputs
    _patch_layers(importlib.import_module("vllm.models.deepseek_v41.decoder_replay_layers").DecoderReplayLayers)
    _installed = True
    _LOG(f"installed: a prefill chunk's decoder replay is skipped while >= the replay window of prompt tokens remain")
