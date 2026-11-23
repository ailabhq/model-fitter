#!/bin/sh
# placement.sh MODEL_PATH OUT_FILE [llama-server args...]
# Unmeasured load at debug verbosity, only to record where --fit puts layers and buffers.
set -eu
model=$1 out=$2; shift 2
bin="${MF_ROOT:-$HOME/mf-bench}/llama.cpp/build/bin/llama-server"
"$bin" -m "$model" -c 8704 -np 1 -fa on --port 8091 --no-webui -lv 4 "$@" > "$out.raw" 2>&1 &
pid=$!
i=0
until curl -sf localhost:8091/health >/dev/null 2>&1 || [ $i -ge 600 ] || ! kill -0 $pid 2>/dev/null; do sleep 1; i=$((i+1)); done
kill -INT $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
grep -iE 'offload|buffer size|fit|projected|working set|n_gpu_layers|repack|KV self size|recommendedMax|exceed' "$out.raw" \
  | sed -e "s#$HOME#~#g" > "$out"
rm -f "$out.raw"
