#!/bin/bash
# dsv41: serve DeepSeek-V4.1-Flash-1017 on a DGX Station GB300 as an OpenAI-compatible endpoint (vLLM 0.32 + the dsv41
# engine: whole-step CUDA graphs, ds41's kernels, DSpark speculation with confidence-chosen widths and copy drafting).
#   dsv41.sh up [PROFILE]    start the RTX sidecar, then the server (PROFILE = profiles/PROFILE.env, default: default)
#   dsv41.sh down | status | logs
# Needs: build.sh run once (RUNTIME), the checkpoint at MODEL, Docker with the NVIDIA CDI devices of both GPUs
# (GB300 = GB, RTX PRO 6000 = RTX: CDI UUIDs), the vLLM 0.32 and 0.31 aarch64 images.
# The cold experts (the rowmap's 129 of 384 per layer) run on the RTX sidecar (ds41's peer3, unchanged, on the 0.31
# image); the rest of the model on the GB300.
set -u
E=$(cd "$(dirname "$0")" && pwd)
RUNTIME=${RUNTIME:-$E/runtime}
MODEL=${MODEL:-/models/DeepSeek-V4.1-Flash-1017}
STATE=${STATE:-$E/state}          # JIT caches, sidecar logs, peer3's GB300 build
IMG=${IMG:-vllm/vllm-openai:v0.32.0-aarch64}
SIMG=${SIMG:-vllm/vllm-openai:v0.31.0}
gpu_uuid() { nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' -v n="$1" 'index($1, n) { print $2; exit }'; }
GB=${GB:-$(gpu_uuid GB300)}
RTX=${RTX:-$(gpu_uuid "RTX PRO 6000")}
PORT=${PORT:-8106}
MAXLEN=${MAXLEN:-1048576}
NAME=${NAME:-dsv41}
ROWMAP=${ROWMAP:-rowmap-mix-v1-h256.json}   # 256 hot experts per layer (MegaMoE fast dispatch); h255 OOMs the RTX
log() { echo "[dsv41 $(date +%T)] $*"; }

down() {
  for n in $NAME $NAME-sidecar; do docker stop -t 30 $n >/dev/null 2>&1; docker rm -f $n >/dev/null 2>&1; done
  docker run --rm -v /dev/shm:/dev/shm --entrypoint /bin/rm $SIMG -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3
  log "down"
}

sidecar() {
  mkdir -p $STATE/{sidecar-logs,b12x}
  # runtime/ is build output: after `git pull` it can be older than the checkout (#6: a new row map in the profile)
  if [ ! -f "$RUNTIME/hook/$ROWMAP" ]; then
    log "runtime/hook has no $ROWMAP: runtime/ is older than the checkout; run ./build.sh"; return 1
  fi
  diff -rq "$E/kernels/vendor/ds41/fastpath/hook" "$RUNTIME/hook" >/dev/null 2>&1 \
    || log "warning: runtime/hook differs from the checkout (kernels/vendor/ds41/fastpath/hook); run ./build.sh after updating"
  log "starting the RTX sidecar"
  docker run -d --name $NAME-sidecar --restart no --device nvidia.com/gpu=$RTX --ipc host --network none \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 \
    -v $MODEL:/model:ro -v $STATE/sidecar-logs:/var/lib/ds41/sidecar -v $STATE/b12x:/var/lib/ds41/b12x-cache \
    -v $RUNTIME/hook:/opt/ds41/hook:ro -v $RUNTIME/peer3:/opt/ds41/peer3:ro -v $RUNTIME/b12x:/opt/ds41/b12x:ro \
    -e PYTHONPATH=/opt/ds41/b12x -e PYTHONUNBUFFERED=1 -e B12X_COMPILE_CACHE_DIR=/var/lib/ds41/b12x-cache/compile \
    -e TRITON_CACHE_DIR=/var/lib/ds41/b12x-cache/triton -e PEER3_DIR=/opt/ds41/peer3 -e PEER3_PREBUILT=/opt/ds41/peer3/prebuilt \
    -e CUDA_MODULE_LOADING=EAGER -e PEER3_PREFETCH_K=4 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint /usr/bin/numactl $SIMG --membind=0 \
    python3 /opt/ds41/peer3/sidecar/peer_server3.py --rowmap /opt/ds41/hook/$ROWMAP --model /model >/dev/null || return 1
  for i in $(seq 1 360); do
    docker logs $NAME-sidecar 2>&1 | grep -q '^serving' && break
    [ "$(docker inspect -f '{{.State.Status}}' $NAME-sidecar 2>/dev/null)" = running ] || { log "sidecar died"; docker logs $NAME-sidecar 2>&1 | tail -15; return 1; }
    sleep 10
  done
  docker logs $NAME-sidecar 2>&1 | grep -E '^prepared|^serving' | tail -2
}

main() {
  mkdir -p $STATE/{vllm,flashinfer,peer3-build} $STATE/jit/{tilelang,triton,nv,dj}
  log "starting the GB300 server: profile $PROFILE, port $PORT, max model length $MAXLEN"
  docker run -d --name $NAME --restart no --device nvidia.com/gpu=$GB --ipc host --network host \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --security-opt label=disable \
    -v $MODEL:/model:ro -v $STATE/vllm:/root/.cache/vllm -v $STATE/flashinfer:/root/.cache/flashinfer \
    -v $STATE/jit/tilelang:/root/.tilelang -v $STATE/jit/triton:/root/.triton -v $STATE/jit/nv:/root/.nv -v $STATE/jit/dj:/root/.dj \
    -v $STATE/peer3-build:/peer3-build -v $E:/engine:ro -e PYTHONPATH=/engine \
    -v $E/patches/vllm/gumbel.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/gumbel.py:ro \
    -v $E/patches/vllm/rejection_sampler_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:ro \
    -v $E/patches/vllm/deepseek_v41_encoding.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41_encoding.py:ro \
    -v $E/patches/vllm/sampling_params.py:/usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py:ro \
    -v $E/patches/vllm/deepseek_v41.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41.py:ro \
    -v $E/patches/vllm/detokenizer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py:ro \
    $(env | grep -o "^DSV41_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") \
    -v $RUNTIME/hook:/w:ro -v $RUNTIME/fastpath/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \
    -v $RUNTIME/fastpath:/fastpath:ro \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO -e CUDA_LOG_FILE=stderr -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:512 \
    -e MEGA_ROWMAP=/w/$ROWMAP -e DS41_FASTPATH=$FASTPATH -e DS41_GPU_RESERVE_MIB=0 \
    -e PEER3_BUILD_DIR=/peer3-build -e VLLM_EXP_PEER3_SHM=/dev/shm/vllm_peer_tier3 \
    --entrypoint /usr/bin/numactl $IMG --membind=0 \
    vllm serve --model /model --served-model-name deepseek-v4.1-flash deepseek DeepSeek-V4.1-Flash dsv41 --trust-remote-code \
    --tensor-parallel-size 1 --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
    --worker-cls dsv41_engine.worker.DSV41Worker \
    --engram-config '{"cpu_offload": true, "use_thp": true}' --max-model-len $MAXLEN --max-num-seqs 32 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --kv-cache-dtype nvfp4_ds_mla --block-size 128 \
    --attention-config '{"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": true}' \
    --kernel-config '{"enable_flashinfer_autotune": false}' \
    --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
    --long-prefill-token-threshold 6144 --enable-prefix-caching \
    --speculative-config "$SPEC" \
    --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128 144 160 192 224 256 ${EXTRA_CAPTURE_SIZES:-} \
    ${EXTRA_CAPTURE_SIZES:+--compilation-config '{"decoder_replay_trim_threshold": 257, "decoder_replay_cudagraph_capture_sizes": [128, 256]}'} \
    ${KV_OFFLOAD_GB:+--kv-offloading-backend native --kv-offloading-size $KV_OFFLOAD_GB} \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
    ${DSV41_TORCH_PROFILE:+--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/root/.cache/vllm/prof", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": false}'} \
    --host 0.0.0.0 --port $PORT >/dev/null || return 1
  for i in $(seq 1 240); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
      log "ready on :$PORT after $((i*15)) s"
      # whole-step graphs for the request shapes clients send (bench/warm.py); DSV41_WARM=0 skips it
      if [ "${DSV41_WARM:-1}" = 1 ]; then
        python3 -I "$E/bench/warm.py" "http://127.0.0.1:$PORT" >/dev/null 2>&1 && log "warmed" || log "warm-up failed (serving anyway)"
      fi
      return 0
    fi
    [ "$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)" = running ] || { log "server died"; docker logs $NAME 2>&1 | tail -30; return 1; }
    sleep 15
  done
  log "not ready after 60 min"; return 1
}

PROFILE=${2:-default}
case "${1:-}" in
  up|main)
    P=$E/profiles/$PROFILE.env
    [ -f "$P" ] || { echo "no profile $P"; exit 2; }
    [ -d "$RUNTIME/fastpath" ] || { echo "no runtime at $RUNTIME: run build.sh first"; exit 2; }
    [ -f "$MODEL/model.safetensors.index.json" ] || { echo "no checkpoint at MODEL=$MODEL"; exit 2; }
    set -a; . "$P"; set +a
    if [ "$1" = up ]; then down; sidecar && main; else main; fi ;;
  sidecar) sidecar ;;
  down) down ;;
  status) docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'; curl -s http://127.0.0.1:$PORT/v1/models | head -c 300; echo ;;
  logs) docker logs --tail 50 $NAME ;;
  *) echo "usage: $0 up [PROFILE] | sidecar | main [PROFILE] | down | status | logs"; exit 2 ;;
esac
