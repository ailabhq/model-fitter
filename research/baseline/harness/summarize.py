#!/usr/bin/env python3
"""summarize.py RESULTS_DIR — print one markdown table row per run."""
import json
import sys
from pathlib import Path

cols = ["config", "size GB", "wired MB", "load s (cold)", "prefill tok/s @8K", "gen tok/s @8K", "gen tok/s short",
        "peak RSS GiB", "peak wired GiB", "swap-outs GiB", "max pressure", "timer p99 ms", "spawn max ms", "responsive", "Usable"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for f in sorted(Path(sys.argv[1]).glob("*.json")):
    if f.name.endswith(".samples.json"):
        continue
    r = json.loads(f.read_text())
    m = r["memory"]
    g = lambda k, f: (r.get(k) or {}).get(f, "–")
    row = [r["config"], f"{r['model_bytes'] / 1e9:.1f}" if r.get("model_bytes") else "–", r.get("iogpu_wired_limit_mb", 0) or "default",
           r.get("load_s", "–"), g("req_8k_cold", "prompt_tps"), r.get("gen_tps_at_8k") or "–", g("req_short", "gen_tps"),
           m["peak_proc_rss_gib"], m["peak_wired_gib"], m["swapouts_gib"], {1: "normal", 2: "warn", 4: "critical"}.get(m["max_pressure_level"], m["max_pressure_level"]),
           m["timer_late_ms"]["p99"], m["spawn_ms"]["max"], "yes" if r["responsive"] else "no",
           ("**yes**" if r["usable"] else "no") + (f" (error: {r['error'][:60]})" if r.get("error") else "")]
    print("| " + " | ".join(str(x) for x in row) + " |")
