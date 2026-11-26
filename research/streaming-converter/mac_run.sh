#!/usr/bin/env bash
# One streaming conversion on the M4 16 GB, through run_convert.py (quiet gate, Monitor, kill-switch, disk sampler),
# reading the Source by HTTP byte range from the NAS mirror, then per-tensor hashes of the output, which is deleted.
#
# Usage: mac_run.sh REPO NAME MIRROR_URL MIX PLANNED_S
#   MIX: prefix under $MF_MIX of <MIX>.types.json and the imatrix (<MIX>.imatrix.gguf)
# Environment: MF_LLAMA, MF_GGML_LIB, MF_MIX, MF_CONV (scratch dir for the output), MF_PY
set -euo pipefail
repo=$1 name=$2 mirror=$3 mix=$4 planned=$5
here=$(cd "$(dirname "$0")" && pwd)
out=$MF_CONV/$name
rm -rf "$out"
export MF_SOURCE_MIRROR="$repo=$mirror"
"$MF_PY" "$here/run_convert.py" "$name" "$planned" "$out" -- "$MF_PY" "$here/stream_convert.py" "$repo" "$out/$name.gguf" \
  --types "$MF_MIX/$mix.types.json" --imatrix "$MF_MIX/$mix.imatrix.gguf" --ftype MOSTLY_IQ3_XXS --threads 6 \
  --stats "$out/stats.json" --workdir "$out/work" || true
cp "$out/stats.json" "$here/results/$name.stats.json"
cp "$out/$name.types.json" "$here/results/$name.types-used.json"
"$MF_PY" "$here/tensor_hashes.py" hash "$out/$name.gguf" "$here/results/$name.mac.hashes.json"
rm -f "$out/$name.gguf"
