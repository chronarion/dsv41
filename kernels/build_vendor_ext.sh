#!/bin/bash
# Build the vendored torch-extension kernels (ds41 MIT: W1 attention, W3 preattn) against the vLLM 0.32 image's torch,
# into build/ (same entry points as their kernels/w1-attn and w3-preattn build.sh).
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd); V=$K/vendor/ds41; O=$K/build; mkdir -p $O
IMG=${IMG:-vllm/vllm-openai:v0.32.0-aarch64}
for X in w1 w3; do
  T=$(mktemp -d); mkdir -p $T/build
  if [ $X = w1 ]; then cp -a $V/fastpath/w1attn $V/w1-attn/csrc $V/w1-attn/include $T/
    CMD="import sys; sys.path.insert(0,'/w'); import w1attn.ops as o; print(o.build().__file__)"
  else cp -a $V/fastpath/w3preattn $V/w3-preattn/csrc $V/w3-preattn/include $V/w3-preattn/src $T/
    CMD="import sys; sys.path.insert(0,'/w/src'); from w3.fused import ext; print(ext().__file__)"; fi
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/w -v $T:/w -w /w -e TORCH_CUDA_ARCH_LIST=10.3a \
    --entrypoint python3 $IMG -c "$CMD" 2>&1 | tail -3
  find $T/build -name '*.so' -exec cp {} $O/ \;
  rm -rf $T
done
ls -la $O
