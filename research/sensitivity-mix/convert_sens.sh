#!/usr/bin/env bash
# Inside the sarto-ref image on the NAS: quantize a Source's F32 GGUF to a per-tensor type map with llama-quantize,
# naming every tensor (table-mix/convert_mix.sh, unchanged), then, for Inkling, swap the NVFP4 tensors the map
# keeps for the Source's own bytes and `.scale` (splice_nvfp4.py, checked against Unsloth's file).
#
# Usage: convert_sens.sh NAME TYPES_JSON IMATRIX OUT_GGUF [SOURCE_SAFETENSORS CHECK_GGUF]   (paths under /scratch)
set -euo pipefail
name=$1 types=$2 imatrix=$3 out=$4 src=${5:-} check=${6:-}
tt=${out%.gguf}.tensor-types.txt
python3 -c 'import json,re,sys; [print("^"+re.escape(k)+"$="+v.lower()) for k,v in json.load(open(sys.argv[1])).items()]' "$types" > "$tt"
q=$out
[ -n "$src" ] && q=${out%.gguf}.requant.gguf
echo "$(date +%s) quantize-start"
/llama.cpp/build/bin/llama-quantize --imatrix "$imatrix" --tensor-type-file "$tt" \
  "/scratch/ref/$name/F32.gguf" "$q" IQ3_XXS 5 > "${out%.gguf}.quantize.log" 2>&1
echo "$(date +%s) quantize-end $(stat -c %s "$q")"
if [ -n "$src" ]; then
  PYTHONPATH=/llama.cpp/gguf-py python3 "$(dirname "$0")/splice_nvfp4.py" "$q" "$src" "$out" --check "$check" > "${out%.gguf}.splice.json"
  rm -f "$q"
  echo "$(date +%s) splice-end $(stat -c %s "$out")"
fi
sha256sum "$out" | cut -d' ' -f1 > "$out.sha256"
