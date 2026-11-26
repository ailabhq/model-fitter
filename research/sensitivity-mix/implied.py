#!/usr/bin/env python3
"""Measured sₜ against the weights a Community build's types imply, by role and eighth of depth.

The implied weight of tensor t is the lower bound of "How does sarto's allocator weigh one tensor's quantization
error…": at the multiplier λ of the unweighted solve, the build's type for t beats every cheaper option only if
wₜ ≥ λ · Δbytes / Δerror, the largest such bound over the cheaper options (0 when nothing is cheaper).
Both columns are reduced to a geometric mean per role × eighth, then each is scaled so the `ffn_gate`/`ffn_up`
groups (Qwen) or the routed-expert groups (Inkling) average 1, which is how the ratios in that ticket are stated.

Usage: implied.py TABLE_TSV_GZ BUILD_TYPES_JSON SENS_JSONL --ceiling B --header-bytes N --other-bytes N
                  [--fixed-bytes N --fixed-match RX] [--source-cap RX=BPW] [--ref-roles RX]
"""
import argparse
import json
import math
import re
from collections import defaultdict

import solve_weighted as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("types")
    ap.add_argument("sens")
    ap.add_argument("--ceiling", type=int, required=True)
    ap.add_argument("--header-bytes", type=int, required=True)
    ap.add_argument("--other-bytes", type=int, required=True)
    ap.add_argument("--fixed-bytes", type=int, default=0)
    ap.add_argument("--fixed-match", default=r"^$")
    ap.add_argument("--source-cap", action="append", default=[])
    ap.add_argument("--ref-roles", default=r"^ffn_(gate|up)\.weight$|^ffn_gate_up_exps|^ffn_down_exps")
    a = ap.parse_args()
    table = S.load_table(a.table)
    build = {k: v.lower() for k, v in json.load(open(a.types)).items()}
    fixed = re.compile(a.fixed_match)
    caps = [(re.compile(k), float(v)) for k, v in (c.split("=", 1) for c in a.source_cap)]
    hulls = {}
    for n, e in table.items():
        if fixed.search(n):
            continue
        opts = list(e["opts"].values())
        for rx, bpw in caps:
            if rx.search(n):
                opts = [o for o in opts if o[0] <= e["nelem"] * bpw / 8 + 0.5]
        hulls[n] = S.hull(opts)
    budget = a.ceiling - a.header_bytes - a.other_bytes - a.fixed_bytes
    total = lambda lam: sum(S.pad(min(h, key=lambda o: o[1] + lam * S.pad(o[0]))[0]) for h in hulls.values())
    lo, hi = 0.0, 1.0
    while total(hi) > budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(mid) > budget else (lo, mid)
    lam = hi

    implied = {}
    for n in hulls:
        t = {k.lower(): k for k in table[n]["opts"]}.get(build.get(n, ""))
        if t is None:
            continue
        bu, eu, _ = table[n]["opts"][t]
        bounds = [lam * (S.pad(bu) - S.pad(b)) / (e - eu) for b, e, _ in table[n]["opts"].values() if b < bu and e > eu]
        implied[n] = max(bounds, default=0.0)

    sens = {r["unit"]: r for r in map(json.loads, open(a.sens))}
    cell = defaultdict(lambda: {"s": [], "w": [], "resolved": True})
    for n, r in sens.items():
        key = r["group"]
        cell[key]["s"].append(r["s"])
        cell[key]["resolved"] &= r["resolved"]
        if n in implied and implied[n] > 0:
            cell[key]["w"].append(implied[n])
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs and all(x > 0 for x in xs) else None
    rows = {k: {"s": gm(v["s"]), "w": gm(v["w"]), "resolved": v["resolved"]} for k, v in cell.items()}
    ref = re.compile(a.ref_roles)
    ref_s = gm([r["s"] for k, r in rows.items() if ref.search(k.split("@")[0]) and r["s"]])
    ref_w = gm([r["w"] for k, r in rows.items() if ref.search(k.split("@")[0]) and r["w"]])
    out = {"lambda": lam, "ref_roles": a.ref_roles, "groups": {}}
    print("| group | sₜ (rel.) | implied (rel.) | sₜ / implied |\n|---|---|---|---|")
    for k in sorted(rows, key=lambda k: (k.split("@")[0], int(k.split("@")[1]) if "@" in k else -1)):
        r = rows[k]
        s = r["s"] / ref_s if r["s"] and ref_s else None
        w = r["w"] / ref_w if r["w"] and ref_w else None
        out["groups"][k] = {"s_rel": s, "implied_rel": w, "resolved": r["resolved"]}
        f = lambda x: "–" if x is None else f"{x:.3g}"
        print(f"| {k}{'' if r['resolved'] else ' (bound)'} | {f(s)} | {f(w)} | {f(s / w if s and w else None)} |")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
