#!/bin/bash
# Rebuild only libw4boundary.so (build_vendor.sh's nvcc line) into build/: for kernel sets that change only the W4
# boundary kernels (seam / mix / router / mega / draft router / head).
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
"
ls -la $O/libw4boundary.so
