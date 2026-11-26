#!/usr/bin/env bash
# From one model's group probes to its scored mix: sₜ (sens.py), the weighted solve (solve_weighted.py), the
# conversion on the NAS (convert_sens.sh in the sarto-ref image, with the NVFP4 copy for Inkling), then scoring on
# this Mac against the draft Reference pack at -b 128 (score_fit.py) and the 8K benchmark (bench_gmix.py).
# Each step is skipped when its output exists.
#
# Usage: run_gmix.sh qwen|inkling PROBE_DIR [TAG]
#   PROBE_DIR holds groups.jsonl (and, for the remedy, PROBE_DIR2=... with the second type's groups.jsonl).
# Needs: $NAS_HOST (ssh) running the sarto-ref image over the scratch dir $SCRATCH_NAS, the same dir mounted here at
# $SCRATCH_MAC, $MF_ROOT as in bench.py, and
# $MF_TREE: a tree holding research/{baseline,draft-packs} from their research branches.
set -euo pipefail
m=$1 probes=$2 tag=${3:-gmix}
HERE=$(cd "$(dirname "$0")" && pwd)
MF_ROOT=${MF_ROOT:-$HOME/mf-bench}
MF_TREE=${MF_TREE:-$MF_ROOT/sens/tree}
SCRATCH_MAC=${SCRATCH_MAC:?mount of the NAS scratch dir}
SCRATCH_NAS=${SCRATCH_NAS:?the scratch dir on the NAS}
NAS_HOST=${NAS_HOST:?ssh host of the NAS}
PY=$MF_ROOT/omlx-venv/bin/python
R=$HERE/results
short=$([ "$m" = qwen ] && echo qwen || echo ink)
mkdir -p "$R/$m-groups" "$R/mixes" "$MF_ROOT/models/mix52"

case $m in
qwen)
  name=qwen3.8-27b table=$MF_ROOT/sens/bundle52/tables/qwen3.8-27b.tsv.gz pack=qwen imatrix=/scratch/mix/qwen-adapted.imatrix.gguf
  solve=(--ceiling 10934860704 --header-bytes 10990016 --other-bytes 10582016 --fixed-bytes 351008768
         --fixed-match '^blk\.64\.' --fixed "$MF_ROOT/conv/mix/qwen-adapted.types.json")
  splice=() ;;
inkling)
  name=inkling-2-mini table=$MF_ROOT/sens/bundle52/tables/inkling-2-mini.tsv.gz pack=inkling imatrix=/scratch/mix/ink-imatrix_unsloth.gguf
  nv='^blk\.(1?[0-9])\.ffn_(gate_up|down)_exps'
  solve=(--ceiling 11663310784 --header-bytes 11977760 --other-bytes 147523584 --source-cap "$nv=4.5" --copy-nvfp4 "$nv")
  splice=(/scratch/src/inkling-2-mini/model-00001-of-00001.safetensors /scratch/src/mix47/Inkling-2-Mini-UD-IQ3_XXS.gguf) ;;
esac

second=()
if [ -n "${PROBE_DIR2:-}" ]; then second=(--second "$PROBE_DIR2/groups.jsonl"); fi
cp "$probes/groups.jsonl" "$R/$m-groups/groups.jsonl"
[ -z "${PROBE_DIR2:-}" ] || cp "$PROBE_DIR2/groups.jsonl" "$R/$m-groups/groups2.jsonl"
python3 "$HERE/sens.py" "$probes/groups.jsonl" "${second[@]}" --out "$R/$m-groups" > "$R/$m-groups/summary.md"
s2=()
[ -z "${PROBE_DIR2:-}" ] || s2=(--sens2 "$R/$m-groups/sens2.jsonl")
types=$R/mixes/$short-$tag.types.json
python3 "$HERE/solve_weighted.py" "$table" "${solve[@]}" --sens "$R/$m-groups/sens.jsonl" "${s2[@]}" --sens-col kld_share \
  --out "$types" | tee "$R/mixes/$short-$tag.solve.json"

gguf=$short-$tag.gguf
if [ ! -f "$SCRATCH_MAC/sens/$gguf.sha256" ]; then
  cp "$HERE/splice_nvfp4.py" "$types" "$SCRATCH_MAC/sens/"
  ssh "$NAS_HOST" "docker run --rm -v $SCRATCH_NAS:/scratch sarto-ref:22b99122e bash /scratch/sens/convert_sens.sh \
    $name /scratch/sens/$(basename "$types") $imatrix /scratch/sens/$gguf ${splice[*]}"
fi
[ -f "$MF_ROOT/models/mix52/$gguf" ] || cp "$SCRATCH_MAC/sens/$gguf" "$MF_ROOT/models/mix52/$gguf"
[ ! -f "$SCRATCH_MAC/sens/${gguf%.gguf}.splice.json" ] || cp "$SCRATCH_MAC/sens/${gguf%.gguf}.splice.json" "$R/mixes/"
echo "{\"file_bytes\": $(stat -f %z "$MF_ROOT/models/mix52/$gguf"), \"sha256\": \"$(cat "$SCRATCH_MAC/sens/$gguf.sha256")\"}" > "$R/mixes/$short-$tag.file.json"

export MF_OUT=$R MF_ROOT
label=$short-$tag-llamacpp-fit-b128
# score_fit.py with bench.py's start gate at a 6 GiB floor (MF_GATE_AVAILABLE_GIB), as bench_mix.py: the desktop
# leaves ~6 GiB, and the 8 GiB default stalled this gate for over an hour.
[ -f "$R/$label.json" ] || "$PY" -c 'import os, runpy, sys
sys.path.insert(0, os.path.join(sys.argv[1], "research/baseline/harness"))
import bench
bench.QUIET["available_gib_min"] = float(os.environ.get("MF_GATE_AVAILABLE_GIB", 6))
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")' "$MF_TREE" "$MF_TREE/research/draft-packs/score/score_fit.py" llamacpp "$label" "$MF_ROOT/packs/$pack" \
  "$MF_ROOT/models/mix52/$gguf" -b 128 -ub 128 2>&1 | tail -n 3
if [ "$tag" = gmix ] && [ ! -f "$R/$short-gmix-llamacpp-fit.json" ]; then
  mkdir -p "$MF_TREE/research/sensitivity-mix" && cp "$HERE/bench_gmix.py" "$MF_TREE/research/sensitivity-mix/"
  "$PY" "$MF_TREE/research/sensitivity-mix/bench_gmix.py" "$short-gmix-llamacpp-fit" 2>&1 | tail -n 2
fi
rm -f "$MF_ROOT/models/mix52/$gguf"  # 11 GB: the copy on the NAS stays
