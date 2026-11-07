"""C2 step 2 probe: load DeepSeek-V4.1-Flash-1017 with vLLM 0.32 in this process (no engine-core subprocess) and record
the interop surface our decode-step executor has to use: the model runner object path, layer 0's parameters (names,
shapes, dtypes, devices), the KV-cache tensors (per group: shape, dtype, stride), and one decode step's timing through
vLLM's own path. Writes OUT/probe.json. Run in the vLLM 0.32 image with the GB300 (same memory knobs as the baseline):
   VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 -I inproc_probe.py OUT"""
import json, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams  # noqa: E402

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
SNAP = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
t0 = time.time()
llm = LLM(model=SNAP, tokenizer_mode="deepseek_v41", max_model_len=65536, max_num_seqs=4, gpu_memory_utilization=0.92,
          block_size=128, cpu_offload_gb=80, cpu_offload_params={"experts"}, attention_config={"indexer_kv_dtype": "mxfp4"},
          engram_config={"use_thp": True}, kernel_config={"enable_flashinfer_autotune": False},
          speculative_config={"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"})
rep = {"load_s": round(time.time() - t0, 1)}


def find_runner(obj, depth=0, path="llm"):
    """walk attributes to the GPU model runner (the attribute chain differs between vLLM versions)"""
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


path, runner = find_runner(llm)
rep["runner_path"], rep["runner_class"] = path, type(runner).__module__ + "." + type(runner).__name__
model = runner.model
rep["model_class"] = type(model).__module__ + "." + type(model).__name__
layer0 = {}
for n, p in model.named_parameters():
    if ".layers.0." in n or n.startswith("model.layers.0.") or "layers.0." in n.split("mtp")[0]:
        layer0[n] = dict(shape=list(p.shape), dtype=str(p.dtype), device=str(p.device))
rep["layer0_params"] = layer0
rep["top_params"] = {n: dict(shape=list(p.shape), dtype=str(p.dtype)) for n, p in model.named_parameters() if "layers." not in n}
kv = {}
for name in ("kv_caches", "attn_groups", "kv_cache_config"):
    if hasattr(runner, name):
        kv[name] = str(type(getattr(runner, name)))
caches = getattr(runner, "kv_caches", None)
if isinstance(caches, (list, tuple)):
    rep["kv_caches"] = [dict(shape=list(c.shape), dtype=str(c.dtype), stride=list(c.stride())) for c in caches if hasattr(c, "shape")]
elif isinstance(caches, dict):
    rep["kv_caches"] = {k: dict(shape=list(c.shape), dtype=str(c.dtype), stride=list(c.stride())) for k, c in caches.items() if hasattr(c, "shape")}
rep["runner_attrs"] = sorted(a for a in dir(runner) if not a.startswith("__"))[:400]
t1 = time.time()
o = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=64))
rep["gen64_s"] = round(time.time() - t1, 2); rep["gen_text"] = o[0].outputs[0].text[:200]
json.dump(rep, open(os.path.join(out, "probe.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("load_s", "runner_path", "runner_class", "model_class", "gen64_s", "gen_text")}, indent=1))
print(len(layer0), "layer-0 params;", "kv caches:", len(rep.get("kv_caches", [])))
