"""dsv41 (Fable round 13, rank 1): warm slots that follow the text at prefill.

The full-row prefill layers (0-19) keep 48 cold experts as a second copy in MegaMoE rows n_hot.. ("warm"), so prefill
computes them on the GB300 instead of sending their rows to the RTX. A fixed warm set only helps text like the corpus
it was ranked on; one text's cold traffic is concentrated on a few experts. Here the warm slots follow the text:

- at load, the raw weights of every cold expert of these layers are kept in CPU memory;
- at the first prefill call of a layer, they are transformed into MegaMoE's layout once (the same transform as the
  module's own rows), kept in pinned host memory, and the raw copies are freed. Self-check: the transformed copies of
  the boot's warm experts must equal the module's rows bit for bit, else the layer stays static;
- every prefill call counts the rows per cold expert (under the decode row map, on the GPU) and copies the counts to
  the host; the next call of the layer swaps the least-used warm slots for the most-used cold experts (with
  hysteresis and a cap per call): the expert's weights are copied into the slot and the prefill row map is updated,
  both on the stream, before this call's routing.

Decode is untouched: it routes by the decode row map, which never points at warm rows, and the RTX keeps every cold
expert. The weight tensors keep their addresses, so no graph is invalidated.
  DSV41_WARM_ADAPT=1, DSV41_WARM_ADAPT_MIN_T (rows, default 2048), DSV41_WARM_ADAPT_SWAPS (per call, default 24)"""
import os
import sys

import numpy as np
import torch

MIN_T = int(os.environ.get("DSV41_WARM_ADAPT_MIN_T", "2048"))
MAX_SWAPS = int(os.environ.get("DSV41_WARM_ADAPT_SWAPS", "24"))
_LOG = lambda m: sys.stderr.write(f"DSV41_WARM_ADAPT {m}\n")
_KEYS = ("w13", "w13s", "w2", "w2s")
_stats = {"swaps": 0, "calls": 0, "layers_ready": 0, "logged": 0}


class Adapt:
    def __init__(self, st):
        self.st = st
        self.raw = {}            # key -> CPU uint8 [n_cold, ...] (until setup)
        self.host = None         # 4 pinned host tensors [n_cold, ...] in the module's transformed layout
        self.ready = False
        self.failed = False
        self.slot = list(st.warm)                      # expert id held by each warm slot
        self.cidx = {e: c for c, e in enumerate(st.cold)}
        self.cnt_host = None
        self.ev = None
        self.pending = False

    # -------------------------------------------------------------------------------------------- load time
    def stash(self, key, param, c, shard_id, intermediate_size, loaded_weight):
        buf = self.raw.get(key)
        if buf is None:
            buf = self.raw[key] = torch.empty((self.st.n_cold,) + tuple(param.shape[1:]), dtype=param.dtype)
        dst = buf[c]
        if shard_id in ("w1", "w3"):
            dst = dst.narrow(0, 0 if shard_id == "w1" else intermediate_size, intermediate_size)
        dst.copy_(loaded_weight)

    # ---------------------------------------------------------------------- first prefill call: transform once
    def setup(self, mod):
        if self.ready or self.failed:
            return
        try:
            self._setup(mod)
        except Exception as e:  # noqa: BLE001  (fall back to the static warm set)
            self.failed = True
            self.raw = {}
            _LOG(f"layer {self.st.idx}: setup failed ({e!r}); static warm set")

    def _setup(self, mod):
        st = self.st
        if set(self.raw) != {"w13", "w13s", "w2", "w2s"}:
            raise RuntimeError(f"raw weights incomplete: {sorted(self.raw)}")
        dev_t = [mod._transformed_l1_weights[0], mod._transformed_l1_weights[1],
                 mod._transformed_l2_weights[0], mod._transformed_l2_weights[1]]
        dev = dev_t[0].device
        backend = mod._ensure_backend() if mod._backend is None else mod._backend
        self.host = []
        for t in dev_t:
            h = torch.empty((st.n_cold,) + tuple(t.shape[1:]), dtype=t.dtype)
            err = torch.cuda.cudart().cudaHostRegister(h.data_ptr(), h.numel() * h.element_size(), 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister failed ({err})")
            self.host.append(h)
        B = 4
        for c0 in range(0, st.n_cold, B):
            c1 = min(c0 + B, st.n_cold)
            out = backend.transform_weights(
                w13_weight=self.raw["w13"][c0:c1].to(dev), w13_weight_scale=self.raw["w13s"][c0:c1].to(dev),
                w2_weight=self.raw["w2"][c0:c1].to(dev), w2_weight_scale=self.raw["w2s"][c0:c1].to(dev),
                num_local_experts=c1 - c0, hidden_size=mod.hidden_size, intermediate_size=mod.intermediate_size)
            outs = [out[0][0], out[0][1], out[1][0], out[1][1]]
            for h, o, t in zip(self.host, outs, dev_t):
                if tuple(o.shape[1:]) != tuple(t.shape[1:]) or o.dtype != t.dtype:
                    raise RuntimeError(f"transform shape {tuple(o.shape)} {o.dtype} vs module {tuple(t.shape)} {t.dtype}")
                for i in range(c1 - c0):
                    h[c0 + i].copy_(o[i])
            del out, outs
        torch.cuda.synchronize(dev)
        self.raw = {}
        # self-check: the boot's warm experts, transformed here, must equal the module's own rows
        for w, e in enumerate(self.slot):
            r, c = st.n_hot + w, self.cidx[e]
            for k, (h, t) in enumerate(zip(self.host, dev_t)):
                if not torch.equal(t[r], h[c].to(dev)):
                    raise RuntimeError(f"self-check: expert {e} ({_KEYS[k]}) differs from MegaMoE row {r}")
        self.dev_t = dev_t
        self.cnt_host = torch.zeros(st.n_cold + 1, dtype=torch.int64, pin_memory=True)
        self.cnt_dev = torch.zeros(st.n_cold + 1, dtype=torch.int64, device=dev)
        self.ev = torch.cuda.Event()
        # pinned staging for the row-map updates: a pageable host-to-device copy waits for the stream to drain
        self.stage_i = torch.empty(2 * MAX_SWAPS, dtype=torch.long, pin_memory=True)
        self.stage_v = torch.empty(2 * MAX_SWAPS, dtype=torch.int32, pin_memory=True)
        self.stage_w = torch.empty(2 * MAX_SWAPS, dtype=torch.int32, pin_memory=True)
        self.ready = True
        _stats["layers_ready"] += 1
        mb = sum(h.numel() * h.element_size() for h in self.host) >> 20
        _LOG(f"layer {st.idx}: {st.n_cold} cold experts in pinned host memory ({mb} MiB), self-check of "
             f"{len(self.slot)} warm experts passed ({_stats['layers_ready']} layers ready)")

    # ------------------------------------------------------------------------------------------ each call
    def before(self, mod, T):
        """swap slots from the previous call's counts (no host wait: only if its copy has landed)"""
        if not self.ready or not self.pending or not self.ev.query():
            return
        st = self.st
        self.pending = False
        cnt = self.cnt_host.numpy()[:st.n_cold].copy()
        n_warm = len(self.slot)
        cur = {self.cidx[e]: w for w, e in enumerate(self.slot)}       # cold idx -> slot
        top = np.argsort(-cnt, kind="stable")[:n_warm]
        topset = set(int(x) for x in top)
        adds = [int(c) for c in top if int(c) not in cur]
        rems = sorted((c for c in cur if c not in topset), key=lambda c: cnt[c])
        idx, val = [], []
        for a, r in zip(adds, rems):
            if cnt[a] <= cnt[r] * 1.25 + 16 or len(idx) // 2 >= MAX_SWAPS:
                break
            w = cur.pop(r)
            row = st.n_hot + w
            for h, t in zip(self.host, self.dev_t):
                t[row].copy_(h[a], non_blocking=True)
            self.slot[w] = st.cold[a]
            idx += [st.cold[r], st.cold[a]]
            val += [-(r + 1), row]
        if idx:
            # the staging buffers are reused only after self.ev (recorded after this call's routing) has passed
            dev, n = st.row_map_static.device, len(idx)
            self.stage_i.numpy()[:n] = idx
            self.stage_v.numpy()[:n] = val
            self.stage_w.numpy()[:n] = [-1 if v < 0 else v for v in val]
            it = self.stage_i[:n].to(dev, non_blocking=True)
            st.row_map_static.index_put_((it,), self.stage_v[:n].to(dev, non_blocking=True))
            if st.warm_row_map is not None:
                st.warm_row_map.index_put_((it,), self.stage_w[:n].to(dev, non_blocking=True))
            st.warm = list(self.slot)
            _stats["swaps"] += len(idx) // 2
        _stats["calls"] += 1
        if st.idx == 0 and _stats["calls"] >= 20 * (2 ** _stats["logged"]):
            _stats["logged"] += 1
            _LOG(f"{_stats['calls']} adaptive calls, {_stats['swaps']} slot swaps so far")

    def after(self, topk_ids, valid):
        """count this call's rows per cold expert (decode row map) and start the copy to the host"""
        if not self.ready:
            return
        st = self.st
        rmd = st.row_map[topk_ids.clamp_min(0).long()]
        c = torch.where(valid & (rmd < 0), -rmd - 1, st.n_cold).flatten().long()
        self.cnt_dev.zero_()   # scatter_add_: one kernel (bincount and index_put_(accumulate) block the host)
        self.cnt_dev.scatter_add_(0, c, torch.ones_like(c))
        self.cnt_host.copy_(self.cnt_dev, non_blocking=True)
        self.ev.record()
        self.pending = True
