#!/usr/bin/env python3
"""Additivity of candidate and desktop load: tables for research/additivity.md from results/rep*/load*g/*.json.

For candidates i, j run under the same load L, the counterfactual prediction of j from i's run is
    obs_j(L) ~ obs_i(L) + delta_ij
with delta_ij either the planned footprint difference (memory metrics, from the placement files) or the
gap measured on the quiet machine (load 0, mean over replicates). The interaction ratio is
    rho_ij = Var_L(obs_j - obs_i) / Var_L(obs_j)
over every (replicate, load) where both ran: below 1, i's run under the load predicts j better than j's
mean over loads does.
"""
import json
import re
import statistics
import sys
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MiB = 1 << 20

METRICS = {"peak_wired_gib": "Peak wired GiB", "swapouts_gib": "Swap-outs GiB", "critical_share": "Critical share",
           "gen_tps_at_8k": "Gen tok/s @8K"}
FOOTPRINT_METRICS = ("peak_wired_gib", "swapouts_gib")


def short(name: str) -> str:
    return name.removeprefix("qwen-iq3xxs-")


def planned() -> dict:
    """Planned footprint per candidate (GiB): every buffer llama.cpp reports, and the GPU (MTL0) part."""
    out = {}
    for f in sorted(RES.glob("placement-*.txt")):
        total = gpu = 0.0
        for line in f.read_text().splitlines():
            m = re.search(r"\b(CPU\S*|MTL\d\S*)\s+(?:model |KV |RS |compute |output )buffer size =\s+([\d.]+) MiB", line)
            if m:
                total += float(m[2])
                gpu += float(m[2]) if m[1].startswith("MTL") else 0.0
        out[f.stem.removeprefix("placement-")] = {"total": total * MiB / (1 << 30), "gpu": gpu * MiB / (1 << 30)}
    return out


def load_runs() -> list:
    runs = []
    for f in sorted(RES.glob("rep*/load*g/*.json")):
        if f.name.endswith(".samples.json"):
            continue
        r = json.loads(f.read_text())
        m = r["memory"]
        runs.append({"rep": f.parts[-3].removeprefix("rep"), "load": float(f.parts[-2].removeprefix("load").removesuffix("g")),
                     "config": r["config"], "peak_wired_gib": m["peak_wired_gib"], "swapouts_gib": m["swapouts_gib"],
                     "critical_share": m["pressure"]["critical_share"], "gen_tps_at_8k": r.get("gen_tps_at_8k"),
                     "responsive": r["responsive"], "usable": r["usable"], "aborted": r.get("aborted"), "error": r.get("error"),
                     "pageins_gib": m["pageins_gib"], "window_max": (m["window_ms"] or {}).get("max"),
                     "holder": r.get("holder", {}), "start_gate": r.get("start_gate", {}), "baseline": m["baseline"],
                     "ram_gib": r["machine"]["mem_gib"],
                     "prefill_tps": (r.get("req_8k_cold") or {}).get("prompt_tps"),
                     "complete": not r.get("aborted") and not r.get("error")})
    return runs


def outcome(r: dict) -> str:
    if r["usable"]:
        return "Usable"
    if r["aborted"]:
        return "killed"
    if r["error"]:
        return "error"
    return "not Responsive" if not r["responsive"] else "too slow"


def var(xs: list) -> float:
    return statistics.pvariance(xs) if len(xs) > 1 else float("nan")


def fmt(x, nd=2) -> str:
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    fp, runs = planned(), load_runs()
    if not runs:
        sys.exit("no results")
    by = {(r["rep"], r["load"], r["config"]): r for r in runs}
    configs = sorted({r["config"] for r in runs}, key=lambda c: (fp[c]["gpu"], c) if c in fp else (99, c))
    cells = sorted({(r["rep"], r["load"]) for r in runs})
    loads = sorted({r["load"] for r in runs})

    print("## Planned footprints\n\n| Candidate | Planned total GiB | On the GPU GiB |\n|---|---|---|")
    for c in configs:
        print(f"| {short(c)} | {fp[c]['total']:.2f} | {fp[c]['gpu']:.2f} |")

    def headroom(r):
        """RAM minus what the desktop held without a way to drop it (wired + anonymous + compressor) just before
        the load, minus the candidate's planned total. psutil's "available" counts idle anonymous pages, the
        holder's among them, so it doesn't measure the load."""
        b = r["baseline"]
        return round(r["ram_gib"] - b["wired"] - b["anonymous"] - b["compressor"] - fp[r["config"]]["total"], 2)

    print("\n## All runs\n\n| Rep | Load GiB | Candidate | Outcome | Gen tok/s @8K | Prefill tok/s | Peak wired GiB | Swap-outs GiB "
          "| Page-ins GiB | Critical share | Window max ms | Holder RSS first / min GiB | Desktop committed GiB | Headroom GiB | Why |")
    print("|" + "---|" * 15)
    for r in sorted(runs, key=lambda r: (r["rep"], r["load"], configs.index(r["config"]))):
        h, why = r["holder"], (r["aborted"] or r["error"] or "")
        hold = f"{h['rss_first_gib']} / {h['rss_min_gib']}" if h.get("gib") else "–"
        print(f"| {r['rep']} | {r['load']} | {short(r['config'])} | {outcome(r)} | {fmt(r['gen_tps_at_8k'])} | {fmt(r['prefill_tps'])} "
              f"| {r['peak_wired_gib']} | {r['swapouts_gib']} | {r['pageins_gib']} | {r['critical_share']} | {fmt(r['window_max'], 0)} "
              f"| {hold} | {r['ram_gib'] - fp[r['config']]['total'] - headroom(r):.2f} | {fmt(headroom(r))} | {why[:90]} |")

    print("\n## Outcome by load (is failure monotone in the planned footprint?)\n")
    print("| Rep | Load GiB | " + " | ".join(short(c) for c in configs) + " | Failures upward-closed |")
    print("|" + "---|" * (len(configs) + 3))
    gpu_ladder = [c for c in configs if "-gpu-" in c]
    for rep, load in cells:
        row = [by.get((rep, load, c)) for c in configs]
        oks = [by[(rep, load, c)]["usable"] for c in gpu_ladder if (rep, load, c) in by]
        closed = all(not (not a and b) for a, b in zip(oks, oks[1:]))
        print(f"| {rep} | {load} | " + " | ".join(outcome(r) if r else "–" for r in row) + f" | {'yes' if closed else '**no**'} |")

    print("\n## Outcome against headroom (RAM − desktop committed − planned total), all runs\n")
    print("| Headroom GiB | Candidate | Rep | Load GiB | Outcome | Window max ms | Critical share |\n|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=headroom):
        print(f"| {headroom(r):.2f} | {short(r['config'])} | {r['rep']} | {r['load']:g} | {outcome(r)} | {fmt(r['window_max'], 0)} "
              f"| {r['critical_share']} |")

    print("\n## Interaction ratio rho = Var(obs_j - obs_i) / Var(obs_j) over (rep, load), completed runs only\n")
    print("| Metric | i → j | n | Var gap | Var obs_j | rho | MAE: obs_i + footprint Δ | MAE: obs_i + quiet gap | MAE: j's quiet run |")
    print("|" + "---|" * 9)
    for metric, label in METRICS.items():
        for i, j in permutations(configs, 2):
            pairs = [(by[(rp, ld, i)][metric], by[(rp, ld, j)][metric], ld, rp) for rp, ld in cells
                     if (rp, ld, i) in by and (rp, ld, j) in by and by[(rp, ld, i)]["complete"] and by[(rp, ld, j)]["complete"]
                     and by[(rp, ld, i)][metric] is not None and by[(rp, ld, j)][metric] is not None]
            if len(pairs) < 3:
                continue
            gaps, ojs = [b - a for a, b, _, _ in pairs], [b for _, b, _, _ in pairs]
            vg, vo = var(gaps), var(ojs)
            rho = vg / vo if vo > 1e-12 else float("nan")
            quiet = [(a, b) for a, b, ld, _ in pairs if ld == 0]
            qgap = statistics.mean(b - a for a, b in quiet) if quiet else None
            qj = statistics.mean(b for _, b in quiet) if quiet else None
            loaded = [(a, b) for a, b, ld, _ in pairs if ld > 0]
            mae = lambda pred: statistics.mean(abs(pred(a) - b) for a, b in loaded) if loaded else None
            fd = fp[j]["gpu"] - fp[i]["gpu"] if metric == "peak_wired_gib" else fp[j]["total"] - fp[i]["total"]
            m_fp = mae(lambda a: a + fd) if metric in FOOTPRINT_METRICS else None
            m_q = mae(lambda a: a + qgap) if qgap is not None else None
            m_j = mae(lambda a: qj) if qj is not None else None
            print(f"| {label} | {short(i)} → {short(j)} | {len(pairs)} | {vg:.4f} | {vo:.4f} | {fmt(rho)} | {fmt(m_fp)} | {fmt(m_q)} | {fmt(m_j)} |")

    print("\n## Replicate noise (same load, same candidate: |rep1 - rep2|)\n")
    print("| Metric | mean | max |\n|---|---|---|")
    for metric, label in METRICS.items():
        d = [abs(by[("1", ld, c)][metric] - by[("2", ld, c)][metric]) for ld in loads for c in configs
             if ("1", ld, c) in by and ("2", ld, c) in by and by[("1", ld, c)][metric] is not None and by[("2", ld, c)][metric] is not None]
        if d:
            print(f"| {label} | {statistics.mean(d):.3f} | {max(d):.3f} |")


if __name__ == "__main__":
    main()
