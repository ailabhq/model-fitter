#!/usr/bin/env python3
"""Tensor-for-tensor comparison of two GGUFs: names, order, types, shapes and data bytes; plus metadata differences.

Usage: compare.py OURS.gguf REFERENCE.gguf OUT.json
For each tensor whose bytes differ: how many rows differ, and (for types gguf-py can dequantize) the largest
absolute difference of the dequantized values.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ["MF_LLAMA"]) / "gguf-py"))
import gguf  # noqa: E402

SKIP_KV = ("GGUF.",)


def kv(r):
    out = {}
    for f in r.fields.values():
        if f.name.startswith(SKIP_KV):
            continue
        try:
            v = f.contents()
        except Exception:
            v = None
        if isinstance(v, list) and len(v) > 64:
            v = hashlib.sha256(json.dumps(v, default=str).encode()).hexdigest()[:16] + f" ({len(v)} items)"
        out[f.name] = v
    return out


def main():
    a, b = gguf.GGUFReader(sys.argv[1]), gguf.GGUFReader(sys.argv[2])
    ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
    res = {"tensors_ours": len(ta), "tensors_ref": len(tb), "only_ours": sorted(ta.keys() - tb.keys()),
           "only_ref": sorted(tb.keys() - ta.keys()),
           "same_order": [t.name for t in a.tensors if t.name in tb] == [t.name for t in b.tensors if t.name in ta],
           "identical": 0, "type_differs": [], "shape_differs": [], "data_differs": []}
    for name, x in ta.items():
        y = tb.get(name)
        if y is None:
            continue
        if x.tensor_type != y.tensor_type:
            res["type_differs"].append({"name": name, "ours": x.tensor_type.name, "ref": y.tensor_type.name})
            continue
        if list(x.shape) != list(y.shape):
            res["shape_differs"].append({"name": name, "ours": list(map(int, x.shape)), "ref": list(map(int, y.shape))})
            continue
        da, db = np.asarray(x.data).view(np.uint8).reshape(-1), np.asarray(y.data).view(np.uint8).reshape(-1)
        if da.size == db.size and np.array_equal(da, db):
            res["identical"] += 1
            continue
        nrows = int(np.prod(x.shape[1:])) if len(x.shape) > 1 else 1
        rows_a, rows_b = da.reshape(nrows, -1), db.reshape(nrows, -1)
        diff_rows = int((rows_a != rows_b).any(axis=1).sum())
        entry = {"name": name, "type": x.tensor_type.name, "rows": nrows, "rows_differ": diff_rows}
        try:
            fa = gguf.quants.dequantize(rows_a, x.tensor_type).astype(np.float64)
            fb = gguf.quants.dequantize(rows_b, y.tensor_type).astype(np.float64)
            entry["max_abs_diff"] = float(np.abs(fa - fb).max())
            entry["max_abs_ref"] = float(np.abs(fb).max())
        except Exception as e:
            entry["dequant"] = f"n/a ({type(e).__name__})"
        res["data_differs"].append(entry)
    ka, kb = kv(a), kv(b)
    res["kv_only_ours"] = sorted(ka.keys() - kb.keys())
    res["kv_only_ref"] = {k: kb[k] for k in sorted(kb.keys() - ka.keys())}
    res["kv_differs"] = {k: {"ours": ka[k], "ref": kb[k]} for k in sorted(ka.keys() & kb.keys()) if ka[k] != kb[k]}
    res["all_tensors_identical"] = (res["identical"] == len(ta) == len(tb))
    Path(sys.argv[3]).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items() if not k.startswith("kv_")}, default=str))


if __name__ == "__main__":
    main()
