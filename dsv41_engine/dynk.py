"""S27 / G16: dynamic DSpark draft length per decode step, scheduler side (ds41's DynK estimator, MIT, from its
fastpath_w9_suffix.py), and the one schedule() patch the suffix drafter shares.

DSpark drafts K_HI = 7 tokens every step (the checkpoint's block is 5; on M6 positions 6 and 7 were accepted on
29-34% / 21-22% of T=1 GSM8K steps). A static 7 won on fixed / coding (+4.5% / +7.8%) and lost on prose (c=1 -4.0%,
c=4 -7.4%), so each decode step verifies K_LO = 5 or K_HI = 7 drafts, whichever gives more expected tokens per unit
of step time:

  K_HI  if  1 + m + f * g  >  ratio * (1 + m)

with m = E[min(accepted, K_LO)], f = P(accepted >= K_LO), g = E[min(accepted, K_HI) - K_LO | accepted >= K_LO] from
the request's decayed step history shrunk toward global estimates, and ratio = T(K_HI) / T(K_LO) (M6 GPU periods:
1.089 at batch 1, 1.081 at batch 4; ds41's recipe uses 1.09). g is observed only on K_HI steps: every 16th
consecutive K_LO choice is turned into K_HI, and a full K_LO accept adds an optimistic pseudo-sample of the prior.

Batches of 2..MAX_BATCH requests verify one width for all (the whole-step graphs are keyed by a uniform K): K_HI when
the summed gain beats the summed cost, sum(f * g) > (ratio - 1) * sum(1 + m), with the same exploration per batch.

Not ported: ds41's early decision (shortens the drafter's Markov loop on K_LO steps; the drafter here runs inside
the replayed whole-step graph) and its K_LO block variant (off in its recipe).
Enable with DSV41_DYNK=1 and a speculative config that schedules K_HI drafts at batch sizes 1..MAX_BATCH."""
import importlib
import os
import statistics
import sys
import json
import time
from collections import deque

K_LO = int(os.environ.get("DSV41_DYNK_LO", "5"))
K_HI = int(os.environ.get("DSV41_DYNK_HI", "7"))
COST = os.environ.get("DSV41_DYNK_COST", "1.09")        # step-time ratio T(K_HI) / T(K_LO), or "auto" (batch-1 median)
MAX_BATCH = int(os.environ.get("DSV41_DYNK_MAX_BATCH", "4"))
DECAY, N0, PRIOR = 0.9, 4.0, (2.6, 0.3, 1.15)            # ds41's recipe
EXPLORE, EXPLORE_LO, OPTIMISM = 16, 64, 0.2
LOG_EVERY = 4096
_LOG = lambda m: sys.stderr.write(f"DSV41_DYNK {m}\n")

# the suffix drafter registers here: wide(req) -> bool (its match length says copy), its width, and the cap for steps
# it does not widen (DSpark's block)
STATE = {"wide": None, "k_wide": None, "cap": None, "last_single": None, "counts": {}, "updates": 0, "logged": 0}
DYN = None
_installed = False


class _Req:
    """One request's evidence: per-step decayed sums (weight w, min(a, lo) m, [a >= lo] f), per-sample decayed tail
    sums (weight tw, sum tg of min(a, hi) - lo over hi steps that reached lo, plus optimistic prior pseudo-samples) and
    the runs of consecutive lo / hi choices (for exploration)."""

    __slots__ = ("w", "m", "f", "tw", "tg", "lo_run", "hi_run")

    def __init__(self, w=0.0, m=0.0, f=0.0, tw=0.0, tg=0.0):
        self.w, self.m, self.f, self.tw, self.tg, self.lo_run, self.hi_run = w, m, f, tw, tg, 0, 0


class DynK:
    """Choice between lo and hi drafts per decode step (see the module docstring); ds41's estimator plus a batch rule."""

    def __init__(self, lo, hi, cost, decay=DECAY, n0=N0, prior=PRIOR, explore=EXPLORE, tail_decay=0.9, tail_n0=4.0,
                 glob_tail_decay=0.98, glob_tail_n0=4.0, optimism=OPTIMISM, explore_lo=EXPLORE_LO, cost_window=64):
        self.lo, self.hi, self.decay, self.n0 = lo, hi, decay, n0
        self.explore, self.tail_decay, self.tail_n0, self.explore_lo = explore, tail_decay, tail_n0, explore_lo
        self.glob_tail_decay, self.glob_tail_n0, self.optimism = glob_tail_decay, glob_tail_n0, optimism
        self.auto = cost == "auto"
        self.fixed_ratio = 1.09 if self.auto else float(cost)
        pm, pf, self.g_prior = prior
        self.glob = _Req(8.0, pm * 8.0, pf * 8.0)
        self.reqs = {}
        self.batch = _Req()  # exploration runs of multi-request steps
        self.step_us = {lo: 0.0, hi: 0.0}
        self.step_n = {lo: 0, hi: 0}
        self.step_win = {lo: deque(maxlen=cost_window), hi: deque(maxlen=cost_window)}
        self.rejected = 0
        self.chosen = {}  # (batch size, k) -> steps
        self.explored = [0, 0]
        self.last_t = None

    def _tail(self, r, sample):
        """A hi step that reached lo is a tail sample; a full lo accept, where a hi step would have produced one, adds
        an optimistic pseudo-sample of the prior, so a tail estimate only hi steps can correct drifts back toward the
        prior instead of locking lo in."""
        for e, d, w in ((r, self.tail_decay, 1.0), (self.glob, self.glob_tail_decay, 0.25)):
            if sample is not None:
                e.tw = d * e.tw + 1.0
                e.tg = d * e.tg + sample
            else:
                e.tw += self.optimism * w
                e.tg += self.optimism * w * self.g_prior

    def observe(self, req_id, k, a):
        """One finished step of one request (k scheduled drafts, a accepted)."""
        if k not in (self.lo, self.hi):
            return
        a = max(0, min(a, k))
        r = self.reqs.get(req_id)
        if r is None:
            r = self.reqs[req_id] = _Req()
        full = 1.0 if a >= self.lo else 0.0
        for e, d in ((r, self.decay), (self.glob, 0.998)):
            e.w = d * e.w + 1.0
            e.m = d * e.m + min(a, self.lo)
            e.f = d * e.f + full
        if full:
            self._tail(r, float(min(a, self.hi) - self.lo) if k == self.hi else None)

    def observe_interval(self, k, now):
        """Output arrival of a step; k = its drafts if it was a batch-1 decode step (else None)."""
        last, self.last_t = self.last_t, (now if k is not None else None)
        if k not in self.step_us or last is None:
            return
        us = (now - last) * 1e6
        win = self.step_win[k]
        if us > 200_000 or (len(win) >= 16 and us > 1.5 * self.step_us[k]):
            self.rejected += 1
            return  # an idle gap or a host stall, not a step
        win.append(us)
        self.step_n[k] += 1
        self.step_us[k] = statistics.median(win)

    def ratio(self):
        if self.auto and min(self.step_n.values()) >= 16 and self.step_us[self.lo] > 0:
            return min(max(self.step_us[self.hi] / self.step_us[self.lo], 1.0), 1.5)
        return self.fixed_ratio

    def glob_g(self):
        gl, n = self.glob, self.glob_tail_n0
        return (gl.tg + n * self.g_prior) / (gl.tw + n)

    def estimates(self, req_id):
        """(m, f, g): the request's evidence shrunk toward the global estimates (the global g toward the prior)."""
        gl = self.glob
        gm, gf, gg = gl.m / gl.w, gl.f / gl.w, self.glob_g()
        r = self.reqs.get(req_id)
        if r is None:
            return gm, gf, gg
        n0, nt = self.n0, self.tail_n0
        return (r.m + n0 * gm) / (r.w + n0), (r.f + n0 * gf) / (r.w + n0), (r.tg + nt * gg) / (r.tw + nt)

    def _explore(self, r, k):
        """every explore-th consecutive lo choice becomes hi (fresh tail samples), every explore_lo-th hi becomes lo"""
        if k == self.lo:
            r.lo_run += 1
            if self.explore and r.lo_run >= self.explore:
                k = self.hi
                self.explored[0] += 1
        else:
            r.hi_run += 1
            if self.explore_lo and r.hi_run >= self.explore_lo:
                k = self.lo
                self.explored[1] += 1
        if k == self.hi:
            r.lo_run = 0
        else:
            r.hi_run = 0
        return k

    def choose(self, req_ids):
        """Drafts to verify in the next decode step of these requests (one width for all)."""
        base = gain = 0.0
        for rid in req_ids:
            m, f, g = self.estimates(rid)
            base += 1.0 + m
            gain += f * g
        k = self.hi if base + gain > self.ratio() * base else self.lo
        if len(req_ids) == 1:
            r = self.reqs.get(req_ids[0])
            if r is not None:
                k = self._explore(r, k)
        else:
            k = self._explore(self.batch, k)
        key = (len(req_ids), k)
        self.chosen[key] = self.chosen.get(key, 0) + 1
        return k

    def prune(self, live):
        for rid in [r for r in self.reqs if r not in live]:
            del self.reqs[rid]

    def summary(self):
        gl = self.glob
        ch = ", ".join(f"b{b} k{k} x{n}" for (b, k), n in sorted(self.chosen.items()))
        return (f"chose {ch} (explored {self.explored[0]} up, {self.explored[1]} down; cost samples rejected "
                f"{self.rejected}); global m {gl.m / gl.w:.2f} f {gl.f / gl.w:.3f} g {self.glob_g():.2f}; batch-1 "
                f"step us {self.lo}: {self.step_us[self.lo]:.0f} (n {self.step_n[self.lo]}), {self.hi}: "
                f"{self.step_us[self.hi]:.0f} (n {self.step_n[self.hi]}); ratio {self.ratio():.3f}"
                f"{' (auto)' if self.auto and min(self.step_n.values()) >= 16 else ' (fixed)'}")


def _decode_reqs(sched):
    """The running requests when this schedule() is a pure decode step (nothing waiting, every request has drafts,
    none is a prefill chunk or structured), else None."""
    if not sched.running or len(sched.waiting) or len(getattr(sched, "skipped_waiting", ())):
        return None
    for req in sched.running:
        if not req.spec_token_ids or req.is_prefill_chunk or getattr(req, "use_structured_output", False):
            return None
    return sched.running


def _batch1_step_k(scheduler_output):
    """Drafts verified by a finished step if it was a one-request decode step (else None)."""
    nst = scheduler_output.num_scheduled_tokens
    if len(nst) != 1:
        return None
    (rid, n), = nst.items()
    spec = scheduler_output.scheduled_spec_decode_tokens.get(rid)
    if not spec or n != len(spec) + 1:
        return None
    return len(spec)


_KFILE = os.environ.get("DSV41_K_FILE")   # experiments: {"lo-hi": K, ...} drafts per pure-decode step by batch size,
_KF = {"t": 0.0, "map": []}                 # re-read once a second (sweeps without a reboot); off when unset


def _kfile_k(n):
    now = time.time()
    if now - _KF["t"] > 1.0:
        _KF["t"] = now
        try:
            with open(_KFILE) as f:
                _KF["map"] = [(int(r.split("-")[0]), int(r.split("-")[-1]), int(k)) for r, k in json.load(f).items()]
        except Exception:  # noqa: BLE001  (a file being rewritten: keep the last map)
            pass
    return next((k for lo, hi, k in _KF["map"] if lo <= n <= hi), None)


def _patch_scheduler(module):
    cls = module.Scheduler
    if getattr(cls, "_dsv41_dynk", False):
        return
    orig_schedule, orig_stats, orig_update = cls.schedule, cls.make_spec_decoding_stats, cls.update_from_output

    def schedule(self, *args, **kwargs):
        reqs, k = _decode_reqs(self), None
        if reqs is not None:
            # wide copy verify (suffix) only when the previous step ran this request alone: its propose() then ran the
            # chain, so the draft row past DSpark's block and its one-hot q rows belong to this request
            if (len(reqs) == 1 and STATE["wide"] is not None and STATE["last_single"] == reqs[0].request_id
                    and STATE["wide"](reqs[0])):
                k = STATE["k_wide"]
            elif DYN is not None and len(reqs) <= MAX_BATCH:
                k = DYN.choose([r.request_id for r in reqs])
            elif STATE["cap"] is not None and len(reqs) == 1:
                k = STATE["cap"]
            if k is None and _KFILE and len(reqs) >= 2:
                k = _kfile_k(len(reqs))
        if k is not None:
            for req in reqs:
                if len(req.spec_token_ids) != k:
                    req.spec_token_ids = [-1] * k
        elif STATE["cap"] is not None:  # the suffix's wide K never reaches other steps
            for req in self.running:
                if len(req.spec_token_ids) > STATE["cap"]:
                    req.spec_token_ids = [-1] * STATE["cap"]
        c = STATE["counts"]
        c[k] = c.get(k, 0) + 1
        out = orig_schedule(self, *args, **kwargs)
        nst = out.num_scheduled_tokens
        STATE["last_single"] = next(iter(nst)) if len(nst) == 1 else None
        return out

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        if DYN is not None:
            DYN.observe(request_id, int(num_draft_tokens), int(num_accepted_tokens))
        return orig_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens,
                          request_id)

    def update_from_output(self, scheduler_output, model_runner_output):
        if DYN is not None:
            DYN.observe_interval(_batch1_step_k(scheduler_output), time.perf_counter())
        out = orig_update(self, scheduler_output, model_runner_output)
        STATE["updates"] += 1
        if DYN is not None:
            if STATE["updates"] % 1024 == 0 and len(DYN.reqs) > 64:
                DYN.prune(self.requests)
            n = sum(DYN.chosen.values())
            if n // LOG_EVERY != STATE["logged"]:
                STATE["logged"] = n // LOG_EVERY
                _LOG(f"{DYN.summary()}; schedule widths {dict(sorted(STATE['counts'].items(), key=str))}")
        return out

    cls.schedule = schedule
    cls.make_spec_decoding_stats = make_spec_decoding_stats
    cls.update_from_output = update_from_output
    cls._dsv41_dynk = True


def register_wide(wide, k_wide, cap):
    """the suffix drafter: wide(req) says the request's next step verifies k_wide copied drafts; cap bounds the drafts
    of every step it does not widen"""
    STATE.update(wide=wide, k_wide=k_wide, cap=cap)


def install(dynamic=None):
    """patch the scheduler once (before the engine core builds it); dynamic K when DSV41_DYNK=1 (or dynamic=True)"""
    global _installed, DYN
    if dynamic is None:
        dynamic = os.environ.get("DSV41_DYNK") == "1"
    if dynamic and DYN is None:
        DYN = DynK(K_LO, K_HI, COST)
        _LOG(f"on: decode steps of 1..{MAX_BATCH} requests verify {K_LO} or {K_HI} drafts (cost "
             f"{'auto, fallback ' if DYN.auto else ''}{DYN.fixed_ratio}, decay {DECAY}, n0 {N0}, prior {PRIOR}, "
             f"explore up every {EXPLORE} / down every {EXPLORE_LO}, optimism {OPTIMISM})")
    if not _installed:
        _patch_scheduler(importlib.import_module("vllm.v1.core.sched.scheduler"))
        _installed = True
