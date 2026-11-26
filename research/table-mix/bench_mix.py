#!/usr/bin/env python3
"""bench.py's 8K benchmark (llama.cpp `--fit`, the Baseline's arguments) for the table-allocated mixes and the
Unsloth builds they are compared with, all read from $MF_ROOT/models/mix47/.

Usage: bench_mix.py CONFIG [CONFIG ...]   (`bench_mix.py list`)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline" / "harness"))
import bench  # noqa: E402

# The 6 GiB floor of the streaming-converter runs: the desktop now leaves ~7 GiB, and 8 GiB stalled the gate for hours.
bench.QUIET["available_gib_min"] = float(os.environ.get("MF_GATE_AVAILABLE_GIB", 6))

MIXES = {}
for name, file in (("qwen-tablemix", "qwen-tablemix.gguf"), ("qwen-unsloth-iq3xxs", "Qwen3.8-27B-UD-IQ3_XXS.gguf"),
                   ("ink-tablemix", "ink-tablemix.gguf"), ("ink-unsloth-iq3xxs", "Inkling-2-Mini-UD-IQ3_XXS.gguf")):
    MIXES[f"{name}-llamacpp-fit"] = dict(runtime="llamacpp", model={"repo": "local", "file": file, "local": f"mix47/{file}"}, args=[])
bench.CONFIGS |= MIXES

if __name__ == "__main__":
    if sys.argv[1:] == ["list"]:
        sys.exit(print("\n".join(MIXES)))
    for n in sys.argv[1:]:
        for attempt in range(bench.QUIET["retries"] + 1):
            r = bench.run(n)
            if not r.get("contaminated"):
                break
            print(json.dumps({"config": n, "contaminated": r["contaminated"], "retry": attempt < bench.QUIET["retries"]}), flush=True)
        m = r["memory"]
        print(json.dumps({**{k: r.get(k) for k in ("config", "load_s", "gen_tps_at_8k", "responsive", "usable", "aborted", "error")},
                          **{k: m[k] for k in ("peak_proc_rss_gib", "peak_wired_gib", "swapouts_gib", "pressure", "window_ms")}}), flush=True)
