#!/bin/bash
# Build the vendored ds41 kernels (MIT; vendor/ds41) for sm_103a inside the vLLM 0.32 image, into build/.
# Same nvcc lines as their kernels/*/build.sh (which target the v0.31 image); our executor runs on 0.32.
# The torch-extension kernels (w1 attention, w3 preattn, peer3) are built separately against the image's torch.
#   build_vendor.sh            -> build/libw4boundary.so lib_seam_quant_cs16.so libw6mk.so libw6tp.so libw6combine.so
#                                 libw8oproj.so libw8topkg.so
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
V=$K/vendor/ds41; O=$K/build; mkdir -p $O
IMG=${IMG:-vllm/vllm-openai:v0.32.0-aarch64}
NV="nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC"
docker run --rm --network none --user "$(id -u):$(id -g)" -v $V:/v:ro -v $O:/o --entrypoint bash $IMG -c "
set -e
cd /v/w4-boundary
$NV -lineinfo -I include src/csrc/boundary.cu src/csrc/mix.cu src/csrc/router.cu src/csrc/router_mega.cu src/csrc/mega.cu \
  src/csrc/draft_router.cu src/csrc/head.cu -o /o/libw4boundary.so
$NV -lineinfo -I seam-quant/include -I include -I include/w4 seam-quant/seam_quant_cs16.cu -o /o/lib_seam_quant_cs16.so
cd /v/w6-moe
$NV markov/w6mk.cu -o /o/libw6mk.so
$NV topp/w6tp.cu -o /o/libw6tp.so
$NV combine/w6combine.cu -o /o/libw6combine.so
cd /v/w8-persist
$NV --expt-relaxed-constexpr -DW8_SPIN_TIMEOUT_NS=10000000000ull -lineinfo -I include -I oproj -I /v/w4-boundary/include \
  oproj/w8oproj_lib.cu -o /o/libw8oproj.so
$NV --expt-relaxed-constexpr -DW8TG_SORT=0 -I topkg topkg/w8topk_global.cu -o /o/libw8topkg.so
"
ls -la $O
