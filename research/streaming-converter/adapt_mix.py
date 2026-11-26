#!/usr/bin/env python3
"""Carry a published mix (per-tensor types + imatrix) over to the pinned converter's tensor layout.

The pinned llama.cpp fuses tensors that share an input, which an older converter wrote separately (Qwen3.5:
attn_qkv + attn_gate -> attn_qkv_gate, ssm_beta + ssm_alpha -> ssm_beta_alpha). A fused tensor takes the type of
its largest part, and the imatrix entry of its parts, which must be equal: the imatrix of a matrix is the mean
square of its input, and the parts share one input.

Usage: adapt_mix.py PLAN.json TYPES.json IMATRIX.gguf OUT_PREFIX
  PLAN.json is stream_convert.py --plan-only's output; writes OUT_PREFIX.types.json and OUT_PREFIX.imatrix.gguf
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ["MF_LLAMA"]) / "gguf-py"))
import gguf  # noqa: E402

FUSED = {"attn_qkv_gate": ["attn_qkv", "attn_gate"], "ssm_beta_alpha": ["ssm_beta", "ssm_alpha"]}


def main():
    plan, types = json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text())
    r = gguf.GGUFReader(sys.argv[3])
    im = {t.name: np.array(t.data) for t in r.tensors}
    out_types, added, report = {}, {}, {"fused": 0, "kept": 0, "unmapped": [], "dropped_from_mix": []}
    for name in plan["tensors"]:
        if name in types:
            out_types[name] = types[name]
            report["kept"] += 1
            continue
        m = re.fullmatch(r"(blk\.\d+)\.(\w+)\.weight", name)
        parts = FUSED.get(m[2]) if m else None
        if not parts or any(f"{m[1]}.{p}.weight" not in types for p in parts):
            report["unmapped"].append(name)
            continue
        pn = [f"{m[1]}.{p}.weight" for p in parts]
        out_types[name] = types[pn[0]]  # the first part is the largest (qkv, beta|alpha are equal-sized)
        sums = [im.get(p + ".in_sum2") for p in pn]
        if all(s is not None for s in sums):
            if not all(np.array_equal(sums[0], s) for s in sums[1:]):
                raise SystemExit(f"{name}: parts {pn} have different imatrix entries; they don't share an input")
            added[name + ".in_sum2"] = sums[0]
            added[name + ".counts"] = im[pn[0] + ".counts"]
        report["fused"] += 1
    report["dropped_from_mix"] = sorted(set(types) - set(plan["tensors"]))
    Path(sys.argv[4] + ".types.json").write_text(json.dumps(out_types, indent=0))

    w = gguf.GGUFWriter(sys.argv[4] + ".imatrix.gguf", arch="imatrix")
    for f in r.fields.values():
        if f.name.startswith(("GGUF.", "general.architecture")):
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1])
        else:
            w.add_key_value(f.name, f.contents(), vt)
    for t in r.tensors:
        w.add_tensor(t.name, np.array(t.data))
    for k, v in added.items():
        w.add_tensor(k, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    report["imatrix_entries_added"] = len(added) // 2
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 8 else f"{len(v)}: {v[:4]}...") for k, v in report.items()}))


if __name__ == "__main__":
    main()
