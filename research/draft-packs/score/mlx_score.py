#!/usr/bin/env python3
"""MLX side of mf-score: same IN/OUT files as llama.cpp's llama-mf-score, over oMLX's Inkling 2 model code.

Usage: mlx_score.py MODEL_DIR IN OUT PREFILL_STEP

Prefill runs in chunks of PREFILL_STEP tokens into one prompt cache per item, as mlx-lm's generation does,
and keeps the logits of every chunk. Prints one JSON line per item.
"""
import json
import struct
import sys
import time

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from omlx.patches.inkling2 import apply_inkling2_patch

path, in_path, out_path, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
apply_inkling2_patch()
t = time.perf_counter()
model, tok = mlx_lm.load(path)
mx.eval(model.parameters())
GiB = 1 << 30
print(json.dumps({"ready": True, "load_s": round(time.perf_counter() - t, 2), "active_gib": round(mx.get_active_memory() / GiB, 2),
                  "prefill_step": step}), flush=True)

fi, fo = open(in_path, "rb"), open(out_path, "wb")
rd = lambda n: np.frombuffer(fi.read(4 * n), np.int32)
(n_items,) = rd(1)
total_s = total_pos = 0
mx.reset_peak_memory()
for it in range(n_items):
    n, k = (int(x) for x in rd(2))
    ids = rd(n)
    ref = rd((n - 1) * k).reshape(n - 1, k)
    tl = int(rd(1)[0])
    text = fi.read(tl).decode()
    if tl % 4:
        fi.read(4 - tl % 4)

    t = time.perf_counter()
    cache = make_prompt_cache(model)
    lq, am = [], []
    for s in range(0, n - 1, step):
        e = min(n - 1, s + step)  # positions s..e-1 predict tokens s+1..e
        logits = model(mx.array(ids[s:e])[None], cache=cache)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        g = mx.take_along_axis(lp, mx.array(ref[s:e]), axis=-1)
        a = mx.argmax(lp, axis=-1)
        mx.eval(g, a, [c.state for c in cache])
        lq.append(np.array(g))
        am.append(np.array(a).astype(np.int32))
        mx.clear_cache()
    dt = time.perf_counter() - t
    total_s += dt
    total_pos += n
    fo.write(np.concatenate(lq).astype(np.float32).tobytes())
    fo.write(np.concatenate(am).tobytes())
    toks = tok._tokenizer(text, add_special_tokens=False)["input_ids"]
    fo.write(struct.pack("<i", len(toks)) + np.array(toks, np.int32).tobytes())
    fo.flush()
    print(json.dumps({"item": it, "tokens": n, "s": round(dt, 2), "tok_s": round(n / dt, 1)}), flush=True)
fo.close()
print(json.dumps({"done": True, "positions": total_pos, "score_s": round(total_s, 1), "tok_s": round(total_pos / total_s, 1),
                  "peak_mlx_gib": round(mx.get_peak_memory() / GiB, 2)}), flush=True)
