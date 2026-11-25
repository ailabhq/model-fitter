#!/usr/bin/env python3
"""MLX without a server: mlx-lm's generation loop over oMLX's Inkling 2 model code.

Stock mlx-vlm (0.7.11, 0.8.0) has no looped Inkling 2 (`inkling2_loop`), so this
registers oMLX's `inkling2` module with mlx-lm and drives `mlx_lm.stream_generate`
directly, where prefill step, KV quantization and MLX's buffer cache are plain knobs.

Usage: mlx_direct.py MODEL_DIR SETTINGS_JSON
  SETTINGS_JSON: {"prefill_step": int, "kv_bits": int|null, "kv_start": int, "cache_limit_mb": int|null}
Protocol: prints {"ready": ...} after load, then answers one JSON request per stdin line
({"ids": [...], "n": int}) with one JSON line.
"""
import json
import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from omlx.patches.inkling2 import apply_inkling2_patch

path, opts = sys.argv[1], json.loads(sys.argv[2])
if opts.get("cache_limit_mb") is not None:
    mx.set_cache_limit(opts["cache_limit_mb"] << 20)
apply_inkling2_patch()

t = time.perf_counter()
model, tok = mlx_lm.load(path)
mx.eval(model.parameters())
load_s = time.perf_counter() - t
eos = sorted(tok.eos_token_ids)


def no_eos(_tokens, logits):  # fixed-length generation, like ignore_eos on the servers
    logits[..., eos] = -float("inf")
    return logits


GiB = 1 << 30
print(json.dumps({"ready": True, "load_s": round(load_s, 2), "active_gib": round(mx.get_active_memory() / GiB, 2),
                  "max_recommended_gib": round(mx.device_info()["max_recommended_working_set_size"] / GiB, 2)}), flush=True)

for line in sys.stdin:
    req = json.loads(line)
    mx.reset_peak_memory()
    t, first, last = time.perf_counter(), None, None
    text = ""
    try:
        for r in stream_generate(model, tok, mx.array(req["ids"]), max_tokens=req["n"], sampler=make_sampler(0.0),
                                 logits_processors=[no_eos], prefill_step_size=opts["prefill_step"],
                                 kv_bits=opts.get("kv_bits"), kv_group_size=64, quantized_kv_start=opts.get("kv_start", 0)):
            first = first or time.perf_counter()
            text += r.text
            last = r
        out = {"wall_s": round(time.perf_counter() - t, 2), "prompt_n": last.prompt_tokens,
               "ttft_s": round(first - t, 2), "prompt_tps": round(last.prompt_tps, 2), "prompt_s": round(last.prompt_tokens / last.prompt_tps, 2),
               "gen_n": last.generation_tokens, "gen_tps": round(last.generation_tps, 2),
               "peak_mlx_gib": round(mx.get_peak_memory() / GiB, 2), "sample": text[:300]}
    except Exception as e:  # e.g. Metal out-of-memory; reported, the worker stays up
        out = {"error": f"{type(e).__name__}: {e}"}
    mx.clear_cache()
    print(json.dumps(out), flush=True)
