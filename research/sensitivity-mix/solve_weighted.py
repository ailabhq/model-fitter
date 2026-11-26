#!/usr/bin/env python3
"""The table-mix solver, unchanged, minimizing Σ sₜ·eₜ(type) instead of Σ eₜ(type).

Hull, Lagrangian and greedy fill are those of table-mix/solve.py ("How does sarto's converter choose a type for
each tensor…"): each tensor's options are cut to the lower convex hull of (bytes, weighted error), a Lagrangian
solve picks a point on every hull, then a greedy fill spends what is left; ties break by name, no boosts or caps.

What changes (from "How does sarto's allocator weigh one tensor's quantization error against another's…"):
- `--sens PROBES_JSONL`: sₜ = KLDₜ / eₜ(probe type) per GGUF tensor, from probe.py. Every option of tensor t
  is weighted by its sₜ. `--sens-col` picks the KLD column (kld_all, decided; kld_response for comparison).
- `--sens2 PROBES_JSONL`: a second probe type. Then each tensor's error follows a fitted power law
  KLD = sₜ·eₜ^αₜ through both probes (the remedy if the one-type check fails); options are KLD(eₜ(type)).
- `--copy-nvfp4 REGEX`: a tensor the Source stores as NVFP4 gets a copy option at error 0 and the NVFP4 bytes,
  plus `--copy-overhead` bytes for its `.scale` tensor ("Can sarto's converter keep a Source's NVFP4 tensors
  bit-exact…"). It replaces the table's requantized NVFP4 option.

Usage: solve_weighted.py TABLE_TSV_GZ --ceiling B --header-bytes N --other-bytes N [--fixed-bytes N --fixed-match RX]
       [--sens P.jsonl [--sens2 P2.jsonl]] [--source-cap RX=BPW ...] [--copy-nvfp4 RX] --out TYPES_JSON
"""
import argparse
import gzip
import json
import math
import re

ALIGN = 32


def pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def load_table(path: str, col: str = "A") -> dict:
    t: dict = {}
    for line in (gzip.open(path, "rt") if path.endswith(".gz") else open(path)):
        name, typ, nbytes, err_a, err_b, _rows, nelem, imat = line.rstrip("\n").split("\t")
        e = t.setdefault(name, {"nelem": int(nelem), "imatrix": imat == "1", "opts": {}})
        e["opts"][typ] = (int(nbytes), float(err_a if col == "A" else err_b), typ)
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


def load_probes(path: str, col: str) -> dict:
    """name -> (probe type, table error at that type, measured KLD)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["unit"]] = (r["probe_type"], r["table_error"], r[col])
    return out


def sensitivities(table: dict, p1: dict, p2: dict | None) -> dict:
    """name -> f(e): the KLD that table error e is expected to cost for that tensor."""
    f = {}
    for n, (t1, e1, k1) in p1.items():
        if n not in table:
            continue
        s = k1 / e1 if e1 > 0 else 0.0
        if p2 is None or n not in p2:
            f[n] = {"s": s, "alpha": 1.0, "fn": (lambda e, s=s: s * e)}
            continue
        t2, e2, k2 = p2[n]
        if e2 > 0 and k2 > 0 and k1 > 0 and e1 > 0 and e1 != e2:
            alpha = math.log(k2 / k1) / math.log(e2 / e1)
            alpha = min(max(alpha, 0.5), 3.0)  # bounded: two points can be noisy for tiny tensors
            c = k1 / e1 ** alpha
            f[n] = {"s": s, "alpha": alpha, "fn": (lambda e, c=c, a=alpha: c * e ** a)}
        else:
            f[n] = {"s": s, "alpha": 1.0, "fn": (lambda e, s=s: s * e)}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--ceiling", type=int, required=True)
    ap.add_argument("--header-bytes", type=int, required=True)
    ap.add_argument("--other-bytes", type=int, required=True)
    ap.add_argument("--fixed-bytes", type=int, default=0)
    ap.add_argument("--fixed-match", default=r"^$")
    ap.add_argument("--fixed", help="types JSON of a template, for the fixed tensors")
    ap.add_argument("--sens")
    ap.add_argument("--sens2")
    ap.add_argument("--sens-col", default="kld_all")
    ap.add_argument("--source-cap", action="append", default=[], metavar="REGEX=BITS_PER_WEIGHT")
    ap.add_argument("--copy-nvfp4", default=r"^$")
    ap.add_argument("--copy-overhead", type=int, default=1024, help="bytes reserved per copied NVFP4 tensor (.scale)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    table = load_table(a.table)
    fixed_re, copy_re = re.compile(a.fixed_match), re.compile(a.copy_nvfp4)
    caps = [(re.compile(k), float(v)) for k, v in (c.split("=", 1) for c in a.source_cap)]
    sens = None
    if a.sens:
        sens = sensitivities(table, load_probes(a.sens, a.sens_col), load_probes(a.sens2, a.sens_col) if a.sens2 else None)
    hulls, copied = {}, []
    for n, e in table.items():
        if fixed_re.search(n):
            continue
        opts = list(e["opts"].values())
        for rx, bpw in caps:
            if rx.search(n):
                limit = e["nelem"] * bpw / 8
                opts = [o for o in opts if o[0] <= limit + 0.5]
        if copy_re.search(n) and any(o[2] == "nvfp4" for o in opts):
            b = e["opts"]["nvfp4"][0]
            opts = [o for o in opts if o[2] != "nvfp4"] + [(b + a.copy_overhead, 0.0, "nvfp4")]
            copied.append(n)
        if sens is not None:
            if n not in sens:
                raise SystemExit(f"no probe for {n}")
            fn = sens[n]["fn"]
            opts = [(b, fn(err), ty) for b, err, ty in opts]
        hulls[n] = hull(opts)
    budget = a.ceiling - a.header_bytes - a.other_bytes - a.fixed_bytes
    ch = solve(hulls, budget)
    types = {n: hulls[n][i][2] for n, i in sorted(ch.items())}
    if a.fixed:
        tpl = json.load(open(a.fixed))
        types.update({n: t for n, t in tpl.items() if fixed_re.search(n)})
    spent = sum(pad(hulls[n][i][0]) for n, i in ch.items())
    raw_err = sum(table[n]["opts"][hulls[n][i][2]][1] if not (n in copied and hulls[n][i][2] == "nvfp4") else 0.0
                  for n, i in ch.items())
    est = {"ceiling": a.ceiling, "budget_tensor_bytes": budget, "allocated_tensor_bytes": spent,
           "estimated_file_bytes": a.header_bytes + a.other_bytes + a.fixed_bytes + spent,
           "below_ceiling_bytes": budget - spent, "objective": sum(hulls[n][i][1] for n, i in ch.items()),
           "total_table_error": raw_err, "tensors": len(types), "copied_nvfp4": sum(types[n] == "nvfp4" for n in copied),
           "weighted": bool(sens), "power_law": bool(a.sens2)}
    json.dump(types, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(est))


if __name__ == "__main__":
    main()
