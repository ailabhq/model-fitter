#!/usr/bin/env python3
"""Byte layout of a GGUF: data offset and each tensor's padded size, from its header alone.

Usage: layout.py GGUF [--table TABLE_TSV_GZ] [--fixed-match REGEX]
Prints JSON: file_bytes, data_offset, and the padded bytes of tensors in the table, fixed (e.g. MTP), and other.
"""
import argparse
import gzip
import json
import os
import re
import struct

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def header(path: str) -> dict:
    f = open(path, "rb")

    def u(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def s():
        return f.read(u("<Q")).decode("utf-8", "replace")

    def v(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = u("<I"), u("<Q")
            return [v(et) for _ in range(n)]
        return u(SCALAR[t])

    assert f.read(4) == b"GGUF"
    _ver, nt, nkv = u("<I"), u("<Q"), u("<Q")
    kv = {}
    for _ in range(nkv):
        k = s()
        kv[k] = v(u("<I"))
    ts = []
    for _ in range(nt):
        name = s()
        shape = [u("<Q") for _ in range(u("<I"))]
        ts.append({"name": name, "shape": shape, "type": u("<I"), "offset": u("<Q")})
    align = kv.get("general.alignment", 32)
    pos = f.tell()
    return {"tensors": ts, "data_offset": (pos + align - 1) // align * align, "file_bytes": os.path.getsize(path)}


def layout(path: str) -> dict:
    h = header(path)
    ts = sorted(h["tensors"], key=lambda t: t["offset"])
    end = h["file_bytes"] - h["data_offset"]
    sizes = {t["name"]: (ts[i + 1]["offset"] if i + 1 < len(ts) else end) - t["offset"] for i, t in enumerate(ts)}
    return {**h, "sizes": sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--table")
    ap.add_argument("--fixed-match", default=r"^$")
    a = ap.parse_args()
    lay = layout(a.gguf)
    names = {line.split("\t", 1)[0] for line in gzip.open(a.table, "rt")} if a.table else set()
    fx = re.compile(a.fixed_match)
    out = {"file_bytes": lay["file_bytes"], "data_offset": lay["data_offset"], "tensors": len(lay["sizes"]),
           "table_bytes": 0, "fixed_bytes": 0, "other_bytes": 0, "other": []}
    for n, b in lay["sizes"].items():
        if fx.search(n):
            out["fixed_bytes"] += b
        elif n in names:
            out["table_bytes"] += b
        else:
            out["other_bytes"] += b
            out["other"].append(n)
    out["other"] = len(out["other"])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
