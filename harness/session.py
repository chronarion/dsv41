"""In-process vLLM 0.32 session for the decode-step executor (C2/C3): the M0 configuration of
the launcher (MegaMoE with ds41's hot/cold rowmap, the cold experts on the RTX sidecar through peer3),
loaded with VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine core, the worker and the V2 model runner live in this process.

Run inside the vLLM 0.32 image with ds41's sitecustomize and hook mounted (as m0_run.sh does) and the sidecar up:
    python3 session.py OUT [--probe]
--probe: one greedy generation, the KV-cache layout (tensors, groups, per-layer views), and per-phase host/device time of
decode steps through vLLM's own path."""
import argparse, json, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/model"
CAPTURE_SIZES = [1, 2, 4, 6, 8, 12, 16, 18, 20, 24]   # <= max_num_seqs x (K+1): a larger piecewise dummy batch
# (8 tokens per request) takes the sparse-indexer prefill path, whose DeepGEMM metadata call refuses capture
SPEC = {"method": "dspark", "num_speculative_tokens": 5, "num_speculative_tokens_per_batch_size": [[1, 4, 5], [5, 8, 3], [9, 32, 3]],
        "draft_sample_method": "probabilistic", "rejection_sample_method": "block"}
if os.environ.get("DSV41_SPEC"):   # e.g. the suffix's K=31 batch-1 spec, or DSpark block 7
    SPEC = json.loads(os.environ["DSV41_SPEC"])


def engine_kwargs(max_model_len=131072, max_num_seqs=4, prefix_caching=False, async_scheduling=None):
    """The M0 server's engine arguments (m0_run.sh), sized for the harness. DSV41_PROFILE=stock gives the stock
    compose instead (a stock vLLM compose: 80 GiB of experts offloaded to host memory, default MoE backend,
    dense indexer logits, auto KV dtype), which needs no RTX sidecar and no ds41 hooks."""
    if os.environ.get("DSV41_PROFILE") == "stock":
        kw = dict(model=MODEL, trust_remote_code=True, tokenizer_mode="deepseek_v41", tensor_parallel_size=1,
                  engram_config={"use_thp": True}, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  gpu_memory_utilization=0.92, block_size=128, cpu_offload_gb=80, cpu_offload_params={"experts"},
                  attention_config={"indexer_kv_dtype": "mxfp4"}, kernel_config={"enable_flashinfer_autotune": False},
                  enable_prefix_caching=prefix_caching, speculative_config=dict(SPEC),
                  cudagraph_capture_sizes=CAPTURE_SIZES)
        if async_scheduling is not None:
            kw["async_scheduling"] = async_scheduling
        return kw
    kw = dict(model=MODEL, trust_remote_code=True, tokenizer_mode="deepseek_v41", tensor_parallel_size=1,
              moe_backend="deep_gemm_mega_moe", enable_expert_parallel=True,
              engram_config={"cpu_offload": True, "use_thp": True}, max_model_len=max_model_len,
              max_num_seqs=max_num_seqs, max_num_batched_tokens=8192, gpu_memory_utilization=0.95,
              kv_cache_dtype="nvfp4_ds_mla", block_size=128,
              attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
              kernel_config={"enable_flashinfer_autotune": False}, enable_prefix_caching=prefix_caching,
              speculative_config=dict(SPEC), cudagraph_capture_sizes=CAPTURE_SIZES)
    if async_scheduling is not None:
        kw["async_scheduling"] = async_scheduling
    return kw


def find_runner(obj, depth=0, path="llm"):
    """walk the in-process engine to the GPU model runner"""
    if depth > 7 or obj is None:
        return None
    if type(obj).__name__.endswith("ModelRunner") and hasattr(obj, "model"):
        return path, obj
    for name in ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner"):
        if hasattr(obj, name):
            r = find_runner(getattr(obj, name), depth + 1, f"{path}.{name}")
            if r:
                return r
    return None


def load(**kw):
    from vllm import LLM
    t0 = time.time()
    if os.environ.get("DSV41_LL_GEMM") == "1":
        # G3: low-latency MXFP8 GEMM tactics for M <= 8 (engine/dsv41_engine/ll_gemm.py), before any capture
        sys.path.insert(0, "/engine")
        from dsv41_engine.ll_gemm import install
        install()
    if os.environ.get("DSV41_DFLASH_PREP") == "1":
        sys.path.insert(0, "/engine")
        from dsv41_engine.dflash_prep import install as prep_install
        prep_install()
    if os.environ.get("DSV41_SUFFIX") == "1" or os.environ.get("DSV41_DYNK") == "1":
        # G6 / S27: suffix drafting and the per-step draft width (engine/dsv41_engine/{suffix,dynk}.py), before the
        # speculator and the scheduler are built
        sys.path.insert(0, "/engine")
        if os.environ.get("DSV41_SUFFIX") == "1":
            from dsv41_engine.suffix import install as sfx_install
            sfx_install()
        else:
            from dsv41_engine.dynk import install as dynk_install
            dynk_install()
    sys.path.insert(0, "/engine")
    from dsv41_engine.prefill_indices import install
    install()
    llm = LLM(**engine_kwargs(**kw))
    path, runner = find_runner(llm)
    print(f"loaded in {time.time() - t0:.0f} s; runner {path} ({type(runner).__name__})", flush=True)
    if os.environ.get("DSV41_SUFFIX") == "1":
        from dsv41_engine.suffix import register_runner
        register_runner(runner)
    if os.environ.get("DSV41_HEAD_FP8") in ("1", "dg") and not hasattr(runner, "_ws_install_fp8_head"):
        # the lossy FP8 verify head (engine/dsv41_engine/fp8_head.py) on the stock runner the harness drives;
        # dg: the drafter's MXFP8 DeepGEMM copy of the shared lm_head
        sys.path.insert(0, "/engine")
        from dsv41_engine.fp8_head import FP8Head, drafter_dg_head
        lm = getattr(runner.model, "language_model", runner.model)
        if os.environ.get("DSV41_HEAD_FP8") == "dg":
            head = drafter_dg_head(runner)
            assert head is not None, "DSV41_HEAD_FP8=dg needs the drafter's DGHead (fastpath tail_drafthead_mxfp8)"
            head.V, head.H = head.v, head.k
        else:
            head = FP8Head(lm.lm_head.weight)
        vocab = lm.logits_processor.org_vocab_size
        orig = lm.compute_logits

        def compute_logits(hidden_states):
            if hidden_states.shape[0] <= 32:
                return head(hidden_states)[:, :vocab]
            return orig(hidden_states)
        lm.compute_logits = compute_logits
        print(f"FP8 verify head on: [{head.V}, {head.H}] {os.environ.get('DSV41_HEAD_FP8')}, vocab {vocab}", flush=True)
    return llm, runner


def kv_layout(runner):
    """KV-cache tensors, groups and per-layer views (open question 1 of engine/docs/decode_step_spec.md)"""
    import torch
    rep = {}
    cfg = runner.kv_cache_config
    rep["num_blocks"] = getattr(cfg, "num_blocks", None)
    rep["kv_cache_tensors"] = [dict(size=t.size, shared_by=list(t.shared_by)) for t in cfg.kv_cache_tensors]
    rep["groups"] = []
    for gi, g in enumerate(cfg.kv_cache_groups):
        spec = g.kv_cache_spec
        rep["groups"].append(dict(index=gi, layers=list(g.layer_names), spec=type(spec).__name__,
                                  block_size=getattr(spec, "block_size", None),
                                  page_size_bytes=getattr(spec, "page_size_bytes", None),
                                  dtype=str(getattr(spec, "dtype", None))))
    views = {}
    fwd = runner.compilation_config.static_forward_context if hasattr(runner, "compilation_config") else {}
    for name, mod in fwd.items():
        kv = getattr(mod, "kv_cache", None)
        if isinstance(kv, (list, tuple)) and kv:
            kv = kv[0]
        if isinstance(kv, torch.Tensor):
            views[name] = dict(shape=list(kv.shape), stride=list(kv.stride()), dtype=str(kv.dtype),
                               data_ptr=kv.data_ptr(), storage_offset=kv.storage_offset())
    rep["layer_views"] = views
    caches = getattr(runner, "kv_caches", None)
    if isinstance(caches, (list, tuple)):
        rep["runner_kv_caches"] = [dict(shape=list(c.shape), dtype=str(c.dtype), data_ptr=c.data_ptr())
                                   for c in caches if hasattr(c, "shape")]
    return rep


def time_phases(runner):
    """wrap execute_model / sample_tokens / the speculator's propose with host timers and CUDA events"""
    import torch
    rec = []
    cur = {}

    def wrap(obj, name, key):
        orig = getattr(obj, name)

        def f(*a, **k):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            t0 = time.perf_counter()
            r = orig(*a, **k)
            t1 = time.perf_counter()
            e1.record()
            cur.setdefault("ev", []).append((key, e0, e1, t1 - t0))
            if key == "sample_tokens":
                rec.append(cur.pop("ev"))
            return r
        setattr(obj, name, f)
        return orig

    wrap(runner, "execute_model", "execute_model")
    wrap(runner, "sample_tokens", "sample_tokens")

    def summary():
        torch.cuda.synchronize()
        out = []
        for evs in rec:
            out.append({k: dict(host_ms=round(h * 1e3, 3), span_ms=round(e0.elapsed_time(e1), 3)) for k, e0, e1, h in evs})
        return out
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    llm, runner = load()
    from vllm import SamplingParams
    rep = dict(runner_class=type(runner).__module__ + "." + type(runner).__name__)
    rep["kv"] = kv_layout(runner)
    json.dump(rep, open(os.path.join(a.out, "session.json"), "w"), indent=1)
    if a.probe:
        summary = time_phases(runner)
        prompt = "Write a Python function that returns the n-th Fibonacci number, with a short explanation."
        t0 = time.time()
        o = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=256))
        dt = time.time() - t0
        rep["gen"] = dict(seconds=round(dt, 2), tokens=len(o[0].outputs[0].token_ids), text=o[0].outputs[0].text[:300])
        rep["steps"] = summary()
        json.dump(rep, open(os.path.join(a.out, "session.json"), "w"), indent=1)
        n = len(rep["steps"])
        print(f"generated {rep['gen']['tokens']} tokens in {dt:.2f} s; {n} steps timed", flush=True)
        print(rep["gen"]["text"][:200].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
