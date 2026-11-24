#!/usr/bin/env python3
"""summarize.py RESULTS_DIR — one markdown table row per llama.cpp run, judged by the Responsive rule."""
import json
import sys
from pathlib import Path

cols = ["Config", "Load s", "8K prefill s", "Prefill tok/s @8K", "Gen tok/s @8K", "Gen tok/s short", "Peak wired GiB",
        "Swap-outs GiB", "Page-ins GiB", "Critical s (share, longest)", "Window max ms", "Timer p99 / max ms", "Launch max ms",
        "Other procs, cores (mean / max)", "Responsive", "Usable", "Outcome"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for f in sorted(Path(sys.argv[1]).glob("*.json"), key=lambda p: p.stat().st_mtime):
    if f.name.endswith(".samples.json"):
        continue
    r = json.loads(f.read_text())
    m, p = r["memory"], r["memory"]["pressure"]
    g = lambda k, f: (r.get(k) or {}).get(f) or "–"
    d = lambda k, f: (m.get(k) or {}).get(f, "–")
    outcome = (f"killed: {r['aborted']}" if r.get("aborted") else r["error"][:90] if r.get("error") else "ok")
    row = [r["config"], r.get("load_s", "–"), g("req_8k_cold", "prompt_s"), g("req_8k_cold", "prompt_tps"), r.get("gen_tps_at_8k") or "–",
           g("req_short", "gen_tps"), m["peak_wired_gib"], m["swapouts_gib"], m["pageins_gib"],
           f"{p['critical_s']} ({p['critical_share']:.1%}, {p['critical_longest_s']})", d("window_ms", "max"),
           f"{d('timer_late_ms', 'p99')} / {d('timer_late_ms', 'max')}", d("spawn_ms", "max"),
           f"{d('foreign_cores', 'mean')} / {d('foreign_cores', 'max')}",
           "–" if r.get("error") else "yes" if r["responsive"] else "**no**", "**yes**" if r["usable"] else "no", outcome.replace("|", "/")]
    print("| " + " | ".join(str(x) for x in row) + " |")
