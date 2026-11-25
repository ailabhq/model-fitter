#!/usr/bin/env bash
# Score the four Usable builds one after another, each behind score_fit.py's quiet gate.
# With ~17 GB of free disk, each build is downloaded only after the previous one is deleted.
# Usage: run_all.sh PACKS_DIR [WAIT_PID]   (WAIT_PID: another session's benchmark to let finish first)
set -euo pipefail
PACKS=$1
HERE=$(cd "$(dirname "$0")" && pwd)
M=${MF_ROOT:-$HOME/mf-bench}/models
PY=${MF_ROOT:-$HOME/mf-bench}/omlx-venv/bin/python
export MF_OUT=${MF_OUT:-$HERE/../results}

if [ -n "${2:-}" ]; then
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
fi

fetch() { hf download "$@" 2>&1 | grep -v -i hint | tail -1; }
score() { "$PY" "$HERE/score_fit.py" "$@" 2>&1 | tail -1; }

if [ ! -f "$MF_OUT/qwen-iq3xxs-llamacpp-fit.json" ]; then
  [ -f "$M/qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf" ] || fetch unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ3_XXS.gguf \
    --revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 --local-dir "$M/qwen"
  score llamacpp qwen-iq3xxs-llamacpp-fit "$PACKS/qwen" "$M/qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf"
fi
rm -rf "$M/qwen"

INK_REV=8078d149286eec7193795413b9e0643da6d11fa1
for q in UD-IQ3_XXS:iq3xxs UD-Q2_K_XL:q2kxl; do
  quant=${q%%:*} tag=${q##*:}
  [ -f "$MF_OUT/ink-$tag-llamacpp-nvfp4-fit.json" ] && continue
  fetch unsloth/Inkling-2-Mini-GGUF "Mainline_Support/Inkling-2-Mini-$quant.gguf_file" --revision $INK_REV --local-dir "$M/ink"
  score llamacpp "ink-$tag-llamacpp-nvfp4-fit" "$PACKS/inkling" "$M/ink/Mainline_Support/Inkling-2-Mini-$quant.gguf_file"
  rm -rf "$M/ink"
done

if [ ! -f "$MF_OUT/ink-oq25e-mlx-p128.json" ]; then
  fetch mlx-community/Inkling-2-Mini-oQ2.5e-mtp --revision d4a4f3f555fdf6f06102de1865d07e8ac5effe6d --local-dir "$M/mlx/Inkling-2-Mini-oQ2.5e-mtp"
  score mlx ink-oq25e-mlx-p128 "$PACKS/inkling" "$M/mlx/Inkling-2-Mini-oQ2.5e-mtp" 128
  rm -rf "$M/mlx/Inkling-2-Mini-oQ2.5e-mtp"
fi
echo ALL_SCORED
