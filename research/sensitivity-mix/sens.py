#!/usr/bin/env python3
"""sₜ per GGUF tensor from group probes (pod/group_probe.py), as decided in "How does sarto measure a tensor's
sensitivity when one BF16 swap reads mostly noise?".

Per group, on the Source's response positions:
- excess = Σᵢ(swapᵢ − controlᵢ) / Σᵢ nᵢ over sequences i (the position-weighted mean of the paired difference),
  clipped at 0; its standard error is the ratio estimator's: sqrt(m/(m−1) · Σᵢ(dᵢ − excess·nᵢ)²) / Σᵢ nᵢ.
- resolved when excess ≥ 2 SE: sₜ = excess / Σ members' eₜ(probe type); else sₜ = (excess + 2 SE) / Σ eₜ, flagged.
- every member shares sₜ. `kld_share` = sₜ·eₜ is the part of the group's excess the member carries, the column the
  solver reads (`solve_weighted.py --sens sens.jsonl --sens-col kld_share`).
- `--second`: swap-only probes at a second type over the same groups, against the first run's controls. A
  per-group slope α = log(excess₂/excess₁) / log(Σe₂/Σe₁) gives KLD = c·eₜ^α with c shared by the group;
  sens2.jsonl then holds each member's c·eₜ₂^α (and sens.jsonl c·eₜ₁^α), so the solver's two-point fit is α.
- The repeated-seed controls: control₂ − control₁ per sequence, with the same paired standard error.

Usage: sens.py GROUPS_JSONL [--second GROUPS2_JSONL] --out DIR
Writes DIR/sens.jsonl (and sens2.jsonl), DIR/groups.json, and prints a Markdown summary.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def paired(a: list, b: list, n: list) -> tuple:
    a, b, n = (np.asarray(x, dtype=np.float64) for x in (a, b, n))
    d = a - b
    mean = d.sum() / n.sum()
    m = len(d)
    se = math.sqrt(m / (m - 1) * ((d - mean * n) ** 2).sum()) / n.sum() if m > 1 else float("inf")
    return float(mean), float(se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups")
    ap.add_argument("--second")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(x) for x in open(a.groups)]
    second = {r["group"]: r for r in map(json.loads, open(a.second))} if a.second else {}

    summary, rows, rows2, repeats = [], [], [], []
    for r in recs:
        n = np.asarray(r["n_resp"], dtype=np.float64)
        excess, se = paired(r["swap"], r["control"], r["n_resp"])
        raw = excess
        excess = max(excess, 0.0)
        resolved = excess >= 2 * se
        sum_e = sum(r["table_errors"])
        s = (excess if resolved else excess + 2 * se) / sum_e if sum_e > 0 else 0.0
        g = {"group": r["group"], "role": r["role"], "block": r["block"], "members": len(r["members"]), "nelem": r["nelem"],
             "probe_type": sorted(set(r["probe_types"])), "sum_e": sum_e,
             "swap": float(sum(r["swap"]) / n.sum()), "control": float(sum(r["control"]) / n.sum()),
             "excess_raw": raw, "excess": excess, "se": se, "z": raw / se if se > 0 else float("inf"),
             "resolved": resolved, "s": s, "swap_top1": float(sum(r["swap_top1"]) / n.sum()),
             "control_top1": float(sum(r["control_top1"]) / n.sum())}
        shares = [s * e for e in r["table_errors"]]
        if r["group"] in second:
            r2 = second[r["group"]]
            assert r2["members"] == r["members"] and r2["n_resp"] == r["n_resp"], r["group"]
            ex2, se2 = paired(r2["swap"], r["control"], r["n_resp"])
            sum_e2 = sum(r2["table_errors"])
            g |= {"probe_type2": sorted(set(r2["probe_types"])), "sum_e2": sum_e2, "excess2_raw": ex2, "se2": se2,
                  "resolved2": ex2 >= 2 * se2}
            k1 = excess if resolved else excess + 2 * se
            k2 = max(ex2, 0.0) if ex2 >= 2 * se2 else max(ex2, 0.0) + 2 * se2
            if k1 > 0 and k2 > 0 and sum_e2 > 0 and sum_e2 != sum_e:
                alpha = math.log(k2 / k1) / math.log(sum_e2 / sum_e)
                c = k1 / sum(e ** alpha for e in r["table_errors"])
                shares = [c * e ** alpha for e in r["table_errors"]]
                shares2 = [c * e ** alpha for e in r2["table_errors"]]
                g |= {"alpha": alpha}
            else:
                shares2 = [s * e for e in r2["table_errors"]]
                g |= {"alpha": None}
            for u, pt, e, k in zip(r2["members"], r2["probe_types"], r2["table_errors"], shares2):
                rows2.append({"unit": u, "group": r["group"], "probe_type": pt, "table_error": e, "kld_share": k})
        if "control2" in r:
            c, cse = paired(r["control2"], r["control"], r["n_resp"])
            repeats.append({"group": r["group"], "control2_minus_control": c, "se": cse, "z": c / cse if cse > 0 else None,
                            "control": g["control"], "control2": float(sum(r["control2"]) / n.sum()), "swap_se": se})
        for u, pt, e, k in zip(r["members"], r["probe_types"], r["table_errors"], shares):
            rows.append({"unit": u, "group": r["group"], "probe_type": pt, "table_error": e, "s": s, "kld_share": k,
                         "swap_kld": g["swap"], "control_kld": g["control"], "excess_kld": excess, "paired_se": se,
                         "resolved": resolved})
        summary.append(g)

    (a.out / "sens.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    if rows2:
        (a.out / "sens2.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows2))
    (a.out / "groups.json").write_text(json.dumps({"groups": summary, "repeat_controls": repeats}, indent=1))

    print(f"groups {len(summary)}, unresolved {sum(not g['resolved'] for g in summary)}: "
          + ", ".join(g["group"] for g in summary if not g["resolved"]))
    print("\n| group | members | Σe | swap | control | excess | SE | z | sₜ |" + (" α |" if second else ""))
    print("|---|---|---|---|---|---|---|---|---|" + ("---|" if second else ""))
    for g in summary:
        print(f"| {g['group']} | {g['members']} | {g['sum_e']:.4g} | {g['swap']:.6f} | {g['control']:.6f} | "
              f"{g['excess_raw']:+.6f} | {g['se']:.1e} | {g['z']:.1f} | {g['s']:.3e}{'' if g['resolved'] else ' (bound)'} |"
              + (f" {g['alpha']:.2f} |" if second and g.get("alpha") is not None else (" – |" if second else "")))
    if repeats:
        print("\n| repeat control | control₂ − control | SE | z | swap SE |\n|---|---|---|---|---|")
        for x in repeats:
            print(f"| {x['group']} | {x['control2_minus_control']:+.2e} | {x['se']:.1e} | {x['z']:.2f} | {x['swap_se']:.1e} |")


if __name__ == "__main__":
    main()
