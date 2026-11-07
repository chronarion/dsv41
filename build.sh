#!/bin/bash
# Build the dsv41 engine's kernels (vendored ds41 sources with our edits, kernels/vendor; VENDORED.md lists the edits)
# inside the vLLM 0.32 image and assemble the runtime directory that dsv41.sh mounts:
#   OUT/fastpath     ds41's fastpath components (vLLM 0.32 ports) + the built .so files + peer3 (GB300 side)
#   OUT/hook         the MegaMoE hot/cold hook, rowmaps, pinned FlashInfer tactics
#   OUT/peer3        the RTX sidecar (runs on the vLLM 0.31 image), OUT/b12x its kernels (Apache-2.0)
#   build.sh [OUT]      (OUT default: runtime; built into OUT.tmp, then swapped in)
# The torch extensions (W1 attention, W3 preattn) build against the image's torch; peer3's GB300 extension is
# JIT-built by the server on first boot into the launcher's P3B cache directory.
set -euo pipefail
E=$(cd "$(dirname "$0")" && pwd)
V=$E/kernels/vendor; B=$E/kernels/build; OUT=${1:-$E/runtime}
"$E/kernels/build_vendor.sh"
"$E/kernels/build_vendor_ext.sh"
T=$OUT.tmp
rm -rf "$T"; mkdir -p "$T"
cp -a "$V/ds41/fastpath" "$T/fastpath"
rm -rf "$T/fastpath/hook"
for f in libw4boundary.so lib_seam_quant_cs16.so libw6mk.so libw6tp.so libw6combine.so libw8oproj.so; do
  cp "$B/$f" "$T/fastpath/"
done
cp "$B/w1_attn_ext.so" "$T/fastpath/w1attn/"
cp "$B/w3_ext.so" "$B/libw8topkg.so" "$T/fastpath/w3preattn/"
cp -a "$V/ds41/peer3" "$T/fastpath/peer3"
cp -a "$V/ds41/fastpath/hook" "$T/hook"
cp -a "$V/ds41/peer3" "$T/peer3"
cp -a "$V/b12x" "$T/b12x"
cp "$V/ds41/LICENSE" "$V/ds41/NOTICE" "$V/ds41/THIRD_PARTY.md" "$V/ds41/VENDORED.md" "$T/"
find "$T" -name __pycache__ -prune -exec rm -rf {} +
rm -rf "$OUT"; mv "$T" "$OUT"
echo "runtime assembled in $OUT:"; du -sh "$OUT"/*
