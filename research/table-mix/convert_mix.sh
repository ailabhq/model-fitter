#!/usr/bin/env bash
# Inside the sarto-ref image on the NAS: quantize a Source's F32 GGUF to a per-tensor type map with llama-quantize
# (the reference path of "Can a per-tensor streaming converter build both reference models' GGUFs ...", which the
# streaming converter matched tensor for tensor), naming every tensor so no ftype rule applies.
#
# Usage: convert_mix.sh NAME TYPES_JSON IMATRIX OUT_GGUF     (paths under /scratch, the NAS scratch dir)
set -euo pipefail
name=$1 types=$2 imatrix=$3 out=$4
tt=${out%.gguf}.tensor-types.txt
python -c 'import json,re,sys; [print("^"+re.escape(k)+"$="+v.lower()) for k,v in json.load(open(sys.argv[1])).items()]' "$types" > "$tt"
echo "$(date +%s) quantize-start"
/llama.cpp/build/bin/llama-quantize --imatrix "$imatrix" --tensor-type-file "$tt" \
  "/scratch/ref/$name/F32.gguf" "$out" IQ3_XXS 5 > "${out%.gguf}.quantize.log" 2>&1
sha256sum "$out" | cut -d' ' -f1 > "$out.sha256"
echo "$(date +%s) quantize-end $(stat -c %s "$out")"
