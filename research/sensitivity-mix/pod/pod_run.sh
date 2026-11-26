#!/usr/bin/env bash
# Sensitivity probes for both reference models on one rented CUDA GPU.
# Run from a directory holding pod/, calib/prompts.jsonl, tables/*.tsv.gz and imatrix/*.gguf.
# Steps: environments, libggml-base at the pin, the Sources' calibration responses (vLLM), then the probes
# (transformers). Each step is skipped when its output exists, so the script can be rerun after a stop.
# Usage: pod_run.sh [STAGE ...]   stages: setup calib-qwen calib-inkling probe-qwen probe-inkling
#                                         gprobe-qwen gprobe-inkling gprobe2-qwen gprobe2-inkling
# (default: setup calib-qwen calib-inkling probe-qwen probe-inkling). setup skips vLLM when both calibration
# responses already exist. gprobe-*: group probes (group_probe.py); gprobe2-*: the second probe type, swap only.
# Extra probe.py flags per model: PROBE_ARGS_qwen, PROBE_ARGS_inkling. No credentials: both Sources are public.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
PIN=22b99122e1dfe1d8a5175bfa2eaa8389699c0b92
declare -A REPO=([qwen]=Qwen/Qwen3.8-27B [inkling]=thinkingmachines/Inkling-2-Mini)
declare -A REV=([qwen]=37561816e66b5b9153b1b64c10968c18c1e6f4a0 [inkling]=c9d028302cd0f035fdf8e7e91f089a4ce94d2195)
declare -A TABLE=([qwen]=tables/qwen3.8-27b.tsv.gz [inkling]=tables/inkling-2-mini.tsv.gz)
declare -A IMAT=([qwen]=imatrix/qwen-adapted.imatrix.gguf [inkling]=imatrix/ink-imatrix_unsloth.gguf)
stages=${*:-setup calib-qwen calib-inkling probe-qwen probe-inkling}
mkdir -p logs out calib

for st in $stages; do
  case $st in
  setup)
    command -v uv >/dev/null || pip install -q uv
    # environments, the pinned libggml-base and both Sources, side by side
    ( [ -x tf/bin/python ] || { uv venv -q -p 3.12 tf && VIRTUAL_ENV=tf uv pip install -q --torch-backend=auto torch \
        'transformers==5.24.0' numpy accelerate safetensors sentencepiece tiktoken huggingface_hub && \
        { VIRTUAL_ENV=tf uv pip install -q flash-linear-attention || echo "flash-linear-attention not installed"; }; } ) > logs/setup-tf.log 2>&1 &
    ( [ -x vl/bin/python ] || { [ -f calib/qwen.jsonl ] && [ -f calib/inkling.jsonl ]; } || { uv venv -q -p 3.12 vl && \
        VIRTUAL_ENV=vl uv pip install -q --pre -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly; } ) > logs/setup-vl.log 2>&1 &
    ( [ -f llama.cpp/build/bin/libggml-base.so ] || { mkdir -p llama.cpp && cd llama.cpp && git init -q \
        && git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp $PIN && git checkout -q FETCH_HEAD \
        && cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
           -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF && cmake --build build --target ggml-base -j; } ) > logs/setup-ggml.log 2>&1 &
    ( { [ -x dl/bin/python ] || { uv venv -q -p 3.12 dl && VIRTUAL_ENV=dl uv pip install -q -U huggingface_hub hf_xet; }; } && for m in qwen inkling; do dl/bin/python -c \
        "from huggingface_hub import snapshot_download as s; print(s('${REPO[$m]}', revision='${REV[$m]}', max_workers=16))"; done ) > logs/setup-dl.log 2>&1 &
    wait
    tail -n 3 logs/setup-*.log
    tf/bin/python -c 'import torch, transformers; print("tf", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
    tf/bin/python -c 'import fla; print("fla", fla.__version__)' || true
    [ ! -x vl/bin/python ] || vl/bin/python -c 'import vllm; print("vllm", vllm.__version__)'
    ls llama.cpp/build/bin/libggml-base.so
    ;;
  calib-*)
    m=${st#calib-}
    [ -f calib/$m.jsonl ] || vl/bin/python pod/gen_calib.py ${REPO[$m]} ${REV[$m]} calib/prompts.jsonl calib/$m.jsonl \
      --engine '{"gpu_memory_utilization": 0.85, "max_num_seqs": 224}' 2>&1 | tee logs/calib-$m.log
    ;;
  probe-*)
    m=${st#probe-}
    args_var=PROBE_ARGS_$m
    (cd pod && MF_GGML_LIB=$PWD/../llama.cpp/build/bin/libggml-base.so PYTHONPATH=$PWD/../llama.cpp/gguf-py \
      ../tf/bin/python probe.py $m ${REPO[$m]} ${REV[$m]} ../calib/$m.jsonl ../${TABLE[$m]} ../${IMAT[$m]} ../out/$m-${PROBE_TAG:-iq3} \
      ${!args_var:-}) 2>&1 | tee -a logs/probe-$m-${PROBE_TAG:-iq3}.log
    ;;
  gprobe-*|gprobe2-*)
    m=${st#*-}
    if [ "${st%%-*}" = gprobe ]; then types=iq3_xxs,q3_K,iq4_nl tag=g-iq3 extra=; else types=${TYPES2:-iq2_xxs,q2_K,mxfp4} tag=g2 extra=--swap-only; fi
    (cd pod && MF_GGML_LIB=$PWD/../llama.cpp/build/bin/libggml-base.so PYTHONPATH=$PWD/../llama.cpp/gguf-py \
      ../tf/bin/python group_probe.py $m ${REPO[$m]} ${REV[$m]} ../calib/$m.jsonl ../${TABLE[$m]} ../${IMAT[$m]} ../out/$m-$tag \
      --types $types $extra ${GPROBE_ARGS:-}) 2>&1 | tee -a logs/gprobe-$m-$tag.log
    ;;
  esac
done
echo ALL_DONE
