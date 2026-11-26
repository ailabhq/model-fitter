#!/usr/bin/env python3
"""Allocate one type per tensor under a whole-file byte ceiling, from per-tensor error tables.

The rules from "How does sarto's converter choose a type for each tensor to hit a target file size?":
each tensor's options are cut to the lower convex hull of (bytes, error), a Lagrangian solve picks a point
on every hull, then a greedy fill spends what is left. No boosts, no caps, ties break by tensor name.
A tensor never gets more bytes than its Source type (`--source-cap NAME_REGEX=TYPE`). Tensors outside the
tables keep what the conversion gives them; `--fixed TYPES_JSON --fixed-match REGEX` pins some tensors
(MTP) to a template's types outside the allocation.

Usage:
  solve.py TABLE_TSV --ceiling BYTES --header-bytes N --other-bytes N [--error A|B]
           [--source-cap REGEX=BYTES_PER_ELEMENT ...] [--fixed TYPES_JSON --fixed-match REGEX --fixed-bytes N]
           --out TYPES_JSON
"""
import argparse
import json
import re
from collections import defaultdict

ALIGN = 32


def pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def load_table(path: str, col: str) -> dict:
    """name -> {"nelem", "imatrix", "opts": [(bytes, error, type)]}."""
    t: dict = {}
    for line in open(path):
        name, typ, nbytes, err_a, err_b, _rows, nelem, imat = line.rstrip("\n").split("\t")
        e = t.setdefault(name, {"nelem": int(nelem), "imatrix": imat == "1", "opts": {}})
        e["opts"][typ] = (int(nbytes), float(err_a if col == "A" else err_b), typ)
    for e in t.values():
        e["opts"] = sorted(e["opts"].values())
    return t


def hull(opts: list) -> list:
    """Lower convex hull of (bytes, error), cheapest first, error strictly falling."""
    best: dict = {}
    for b, e, ty in opts:
        if b not in best or (e, ty) < (best[b][1], best[b][2]):
            best[b] = (b, e, ty)
    pareto, min_e = [], float("inf")
    for b in sorted(best):
        if best[b][1] < min_e:
            min_e = best[b][1]
            pareto.append(best[b])
    out: list = []
    for p in pareto:
        while len(out) >= 2:
            (b0, e0, _), (b1, e1, _) = out[-2], out[-1]
            if (b1 - b0) * (p[1] - e0) - (p[0] - b0) * (e1 - e0) <= 0:
                out.pop()
            else:
                break
        out.append(p)
    return out


def solve(hulls: dict, budget: int) -> dict:
    names = sorted(hulls)

    def pick(lam: float) -> dict:
        ch = {}
        for n in names:
            h = hulls[n]
            ch[n] = min(range(len(h)), key=lambda i: (h[i][1] + lam * pad(h[i][0]), i))
        return ch

    total = lambda ch: sum(pad(hulls[n][i][0]) for n, i in ch.items())
    lo, hi = 0.0, 1.0
    while total(pick(hi)) > budget:
        hi *= 2
        if hi > 1e30:
            raise SystemExit(f"ceiling below the smallest mix: {total(pick(hi))} > {budget}")
    for _ in range(200):
        mid = (lo + hi) / 2
        if total(pick(mid)) > budget:
            lo = mid
        else:
            hi = mid
    ch = pick(hi)
    spent = total(ch)
    # Greedy fill: the upgrade with the largest error drop per byte that still fits, until none fits.
    while True:
        best = None
        for n in names:
            i = ch[n]
            h = hulls[n]
            if i + 1 >= len(h):
                continue
            db = pad(h[i + 1][0]) - pad(h[i][0])
            if spent + db > budget:
                continue
            gain = (h[i][1] - h[i + 1][1]) / db
            if best is None or gain > best[0]:
                best = (gain, n, db)
        if best is None:
            return ch
        _, n, db = best
        ch[n] += 1
        spent += db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--ceiling", type=int, required=True, help="whole-file byte ceiling")
    ap.add_argument("--header-bytes", type=int, required=True, help="GGUF header and metadata bytes")
    ap.add_argument("--other-bytes", type=int, required=True, help="padded bytes of tensors outside the tables")
    ap.add_argument("--error", choices=["A", "B"], default="A")
    ap.add_argument("--source-cap", action="append", default=[], metavar="REGEX=BITS_PER_WEIGHT")
    ap.add_argument("--fixed", help="types JSON of a template (e.g. Unsloth's MTP types)")
    ap.add_argument("--fixed-match", default=r"^$")
    ap.add_argument("--fixed-bytes", type=int, default=0, help="padded bytes of the fixed tensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    table = load_table(a.table, a.error)
    fixed_re = re.compile(a.fixed_match)
    caps = [(re.compile(k), float(v)) for k, v in (c.split("=", 1) for c in a.source_cap)]
    hulls, capped = {}, defaultdict(list)
    for n, e in table.items():
        if fixed_re.search(n):
            continue
        opts = e["opts"]
        for rx, bpw in caps:
            if rx.search(n):
                limit = e["nelem"] * bpw / 8
                opts = [o for o in opts if o[0] <= limit + 0.5]
                capped[n] = [o[2] for o in opts]
        hulls[n] = hull(opts)
    budget = a.ceiling - a.header_bytes - a.other_bytes - a.fixed_bytes
    ch = solve(hulls, budget)
    types = {n: hulls[n][i][2] for n, i in sorted(ch.items())}
    if a.fixed:
        tpl = json.load(open(a.fixed))
        types.update({n: t for n, t in tpl.items() if fixed_re.search(n)})
    spent = sum(pad(hulls[n][i][0]) for n, i in ch.items())
    est = {"ceiling": a.ceiling, "budget_tensor_bytes": budget, "allocated_tensor_bytes": spent,
           "estimated_file_bytes": a.header_bytes + a.other_bytes + a.fixed_bytes + spent,
           "below_ceiling_bytes": budget - spent, "error_column": a.error,
           "total_table_error": sum(hulls[n][i][1] for n, i in ch.items()), "tensors": len(types)}
    json.dump(types, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(est))


if __name__ == "__main__":
    main()
