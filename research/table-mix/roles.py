#!/usr/bin/env python3
"""Per-role bytes, bits per weight and table error of two type maps of one Source (the table mix and Unsloth's).

A role is the tensor name without its layer number. Bytes come from the error table for tabled tensors and from
the reference conversion's header otherwise. Table error is the sum of the chosen types' table errors (column A
and B); a type missing from a tensor's table counts as missing.

Usage: roles.py TABLE_TSV REF_SIZES_JSON OURS_TYPES_JSON UNSLOTH_TYPES_JSON OUT_JSON
"""
import json
import re
import sys
from collections import defaultdict


def main():
    table_p, sizes_p, ours_p, uns_p, out_p = sys.argv[1:]
    table = defaultdict(dict)
    for line in open(table_p):
        name, typ, nbytes, ea, eb, _r, _n, _i = line.rstrip("\n").split("\t")
        table[name][typ.upper()] = (int(nbytes), float(ea), float(eb))
    sizes = json.load(open(sizes_p))["tensors"]
    maps = {"ours": json.load(open(ours_p)), "unsloth": json.load(open(uns_p))}
    out = {}
    for who, types in maps.items():
        roles = defaultdict(lambda: {"bytes": 0, "nelem": 0, "err_a": 0.0, "err_b": 0.0, "types": defaultdict(int), "layers": {}})
        for name, s in sizes.items():
            t = types.get(name, s["type"]).upper()
            role = re.sub(r"^blk\.\d+\.", "blk.N.", name)
            r = roles[role]
            if name in table and t in table[name]:
                nb, ea, eb = table[name][t]
                r["err_a"] += ea
                r["err_b"] += eb
            elif name in table:
                nb = s["nbytes"] if t == s["type"].upper() else None
                r["err_missing"] = r.get("err_missing", 0) + 1
            else:
                nb = s["nbytes"] if t == s["type"].upper() else None
            if nb is None:
                raise SystemExit(f"{who}: no size for {name} as {t}")
            r["bytes"] += nb
            r["nelem"] += s["nelem"]
            r["types"][t] += 1
            if m := re.match(r"^blk\.(\d+)\.", name):
                r["layers"][int(m.group(1))] = round(nb * 8 / s["nelem"], 3)
        total = sum(r["bytes"] for r in roles.values())
        out[who] = {"total_tensor_bytes": total, "total_err_a": sum(r["err_a"] for r in roles.values()),
                    "total_err_b": sum(r["err_b"] for r in roles.values()),
                    "roles": {k: {**v, "gib": v["bytes"] / 2**30, "share": v["bytes"] / total, "bpw": v["bytes"] * 8 / v["nelem"],
                                  "types": dict(v["types"]), "layers": dict(sorted(v["layers"].items()))}
                              for k, v in sorted(roles.items(), key=lambda kv: -kv[1]["bytes"])}}
    json.dump(out, open(out_p, "w"), indent=1)
    print(f"| Role | Unsloth GiB | Unsloth bpw | Table mix GiB | Table mix bpw | Unsloth err A | Table mix err A |")
    print("|---|---|---|---|---|---|---|")
    for role, u in out["unsloth"]["roles"].items():
        o = out["ours"]["roles"][role]
        if u["bytes"] < 2**20 and o["bytes"] < 2**20:
            continue
        print(f"| `{role}` | {u['gib']:.3f} | {u['bpw']:.2f} | {o['gib']:.3f} | {o['bpw']:.2f} | {u['err_a']:.4g} | {o['err_a']:.4g} |")
    print(f"| total | {out['unsloth']['total_tensor_bytes'] / 2**30:.3f} | | {out['ours']['total_tensor_bytes'] / 2**30:.3f} | |"
          f" {out['unsloth']['total_err_a']:.4g} | {out['ours']['total_err_a']:.4g} |")


if __name__ == "__main__":
    main()
