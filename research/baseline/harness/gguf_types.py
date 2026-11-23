#!/usr/bin/env python3
"""gguf_types.py REPO FILE... — tensor types of remote GGUFs, reading only the header via HTTP ranges."""
import os
import struct
import sys
from collections import defaultdict

import requests

sys.path.insert(0, os.path.join(os.environ.get("MF_ROOT", os.path.expanduser("~/mf-bench")), "llama.cpp", "gguf-py"))
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: E402


class Remote:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, b"", 0
        self.h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}

    def read(self, n):
        while self.pos + n > len(self.buf):
            start = len(self.buf)
            r = requests.get(self.url, headers=self.h | {"Range": f"bytes={start}-{start + (8 << 20) - 1}"}, timeout=60)
            self.buf += r.content
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def s(self):
        return self.read(self.u("Q")).decode()


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def skip_value(f, t):
    if t == 8:
        f.s()
    elif t == 9:
        et, n = f.u("I"), f.u("Q")
        for _ in range(n):
            skip_value(f, et)
    else:
        f.u(SCALAR[t])


def types(repo, file):
    f = Remote(f"https://huggingface.co/{repo}/resolve/main/{file}")
    assert f.read(4) == b"GGUF"
    f.u("I")
    n_tensors, n_kv = f.u("Q"), f.u("Q")
    for _ in range(n_kv):
        f.s()
        skip_value(f, f.u("I"))
    by = defaultdict(lambda: [0, 0, None])
    for _ in range(n_tensors):
        name = f.s()
        dims = [f.u("Q") for _ in range(f.u("I"))]
        qt = GGMLQuantizationType(f.u("I"))
        f.u("Q")
        n = 1
        for d in dims:
            n *= d
        block, size = GGML_QUANT_SIZES[qt]
        by[qt.name][0] += 1
        by[qt.name][1] += n // block * size
        by[qt.name][2] = by[qt.name][2] or name
    return by


if __name__ == "__main__":
    repo = sys.argv[1]
    for file in sys.argv[2:]:
        print(f"== {file}")
        for k, (n, b, ex) in sorted(types(repo, file).items(), key=lambda x: -x[1][1]):
            print(f"  {k:9} {n:4} tensors {b / 2**20:8.0f} MiB  e.g. {ex}")
