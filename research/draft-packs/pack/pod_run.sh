#!/usr/bin/env bash
# Build and cross-check both draft Reference packs on one rented CUDA GPU (H100 80 GB).
# Run from the directory holding pack/ and prompt-set-v0/. Writes packs/<name>/ and logs/.
# No credentials: both Sources are public and ungated.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
QWEN=(Qwen/Qwen3.8-27B 37561816e66b5b9153b1b64c10968c18c1e6f4a0 qwen)
INK=(thinkingmachines/Inkling-2-Mini c9d028302cd0f035fdf8e7e91f089a4ce94d2195 inkling)
mkdir -p logs packs

command -v uv >/dev/null || pip install -q uv
[ -x tf/bin/python ] || { uv venv -q -p 3.12 tf && VIRTUAL_ENV=tf uv pip install -q --torch-backend=auto torch 'transformers==5.24.0' numpy accelerate; }
[ -x vl/bin/python ] || { uv venv -q -p 3.12 vl && VIRTUAL_ENV=vl uv pip install -q --pre -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly; }
tf/bin/python -c 'import torch, transformers; print("tf", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
vl/bin/python -c 'import vllm; print("vllm", vllm.__version__)'

for src in QWEN INK; do
  declare -n s=$src
  name=${s[2]}
  if [ ! -f packs/$name/manifest.json ]; then
    tf/bin/python pack/build_pack.py "${s[0]}" "${s[1]}" prompt-set-v0 packs/$name 2>&1 | tee logs/build-$name.log
  fi
  if [ ! -f packs/$name/cross_check.json ]; then
    (cd pack && ../vl/bin/python cross_check.py ../packs/$name --engine '{"gpu_memory_utilization": 0.85}') 2>&1 | tee logs/check-$name.log || true
  fi
done
echo ALL_DONE
