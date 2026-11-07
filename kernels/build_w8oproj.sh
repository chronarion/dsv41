#!/bin/bash
# Rebuild only libw8oproj.so (build_vendor.sh's nvcc line) into build/: for kernel sets that change only the W8 woa.
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
V=$K/vendor/ds41; O=$K/build; mkdir -p $O
IMG=${IMG:-vllm/vllm-openai:v0.32.0-aarch64}
NV="nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC"
docker run --rm --network none --user "$(id -u):$(id -g)" -v $V:/v:ro -v $O:/o --entrypoint bash $IMG -c "
set -e
cd /v/w8-persist
$NV --expt-relaxed-constexpr -DW8_SPIN_TIMEOUT_NS=10000000000ull -lineinfo -I include -I oproj -I /v/w4-boundary/include \
  oproj/w8oproj_lib.cu -o /o/libw8oproj.so
"
ls -la $O/libw8oproj.so
