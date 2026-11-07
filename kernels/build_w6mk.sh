#!/bin/bash
# Rebuild only libw6mk.so (ds41's fused Markov loop, build_vendor.sh's nvcc line) into build/.
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
V=$K/vendor/ds41; O=$K/build; mkdir -p $O
IMG=${IMG:-vllm/vllm-openai:v0.32.0-aarch64}
NV="nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared"
docker run --rm --network none --user "$(id -u):$(id -g)" -v $V:/v:ro -v $O:/o --entrypoint bash $IMG -c "
set -e
cd /v/w6-moe
$NV markov/w6mk.cu -o /o/libw6mk.so
"
ls -la $O/libw6mk.so
