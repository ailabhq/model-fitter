#!/usr/bin/env python3
"""Per-tensor sha256 of a GGUF's data, for comparing GGUFs that live on different machines.

Usage: tensor_hashes.py hash FILE.gguf OUT.json      ({name: [type, shape (ggml order), sha256]}, in file order)
       tensor_hashes.py diff A.json B.json            (prints identical / differing / missing counts and names)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def hash_file(path, out):
    sys.path.insert(0, str(Path(os.environ["MF_LLAMA"]) / "gguf-py"))
    import gguf
    r = gguf.GGUFReader(path)
    res = {}
    for t in r.tensors:
        h = hashlib.sha256()
        data = np.asarray(t.data).view(np.uint8).reshape(-1)
        for o in range(0, data.size, 1 << 28):
            h.update(data[o:o + (1 << 28)].tobytes())
        res[t.name] = [t.tensor_type.name, [int(x) for x in t.shape], h.hexdigest()]
    Path(out).write_text(json.dumps(res, indent=0))
    print(json.dumps({"tensors": len(res)}))


def diff(a, b):
    A, B = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    same = [n for n in A if n in B and A[n] == B[n]]
    differ = [{"name": n, "a": A[n][:2], "b": B[n][:2], "same_type_shape": A[n][:2] == B[n][:2]} for n in A if n in B and A[n] != B[n]]
    res = {"identical": len(same), "differ": len(differ), "only_a": sorted(A.keys() - B.keys()), "only_b": sorted(B.keys() - A.keys()),
           "same_order": [n for n in A if n in B] == [n for n in B if n in A], "differing": differ}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"hash": hash_file, "diff": diff}[sys.argv[1]](*sys.argv[2:])
