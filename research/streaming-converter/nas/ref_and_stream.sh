#!/usr/bin/env bash
# Inside the sarto-ref image on the NAS: llama.cpp's own conversion of one Source (convert_hf_to_gguf.py --outtype
# f32, then llama-quantize with the imatrix and the per-tensor types), and stream_convert.py on the same machine
# reading the same local Source as a mirror. Compares the two tensor for tensor and hashes both.
#
# Usage: ref_and_stream.sh REPO NAME TYPES_JSON IMATRIX     (paths under /scratch, the NAS scratch dir)
set -euo pipefail
repo=$1 name=$2 types=$3 imatrix=$4
src=/scratch/src/$name out=/scratch/ref/$name h=/scratch/harness
mkdir -p "$out"
stamp() { echo "$(date +%s) $*"; }

python -c 'import json,re,sys; [print("^"+re.escape(k)+"$="+v.lower()) for k,v in json.load(open(sys.argv[1])).items()]' "$types" > "$out/tensor-types.txt"

if [ ! -f "$out/F32.done" ]; then
  stamp convert-start
  python "$MF_LLAMA/convert_hf_to_gguf.py" "$src" --outtype f32 --outfile "$out/F32.gguf" > "$out/convert.log" 2>&1
  touch "$out/F32.done"
  stamp convert-end
fi

stamp quantize-start
/llama.cpp/build/bin/llama-quantize --imatrix "$imatrix" --tensor-type-file "$out/tensor-types.txt" \
  "$out/F32.gguf" "$out/ref.gguf" IQ3_XXS 4 > "$out/quantize.log" 2>&1
stamp quantize-end

stamp stream-start
MF_SOURCE_MIRROR="$repo=$src" python "$h/stream_convert.py" "$repo" "$out/ours.gguf" --types "$types" --imatrix "$imatrix" \
  --ftype MOSTLY_IQ3_XXS --threads 4 --stats "$out/ours.stats.json" --workdir "$out/stream" > "$out/stream.log" 2>&1
stamp stream-end

python "$h/compare.py" "$out/ours.gguf" "$out/ref.gguf" "$out/compare.json"
python "$h/tensor_hashes.py" hash "$out/ref.gguf" "$out/ref.hashes.json"
python "$h/tensor_hashes.py" hash "$out/ours.gguf" "$out/ours.hashes.json"
stamp done
