"""Minimal GGUF header reader: KV pairs and tensor infos, from a local file or an HTTP byte range.

Only the header is read, so a remote GGUF's per-tensor types cost a few MB, not the whole file.
"""
import struct

import requests

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Buf:
    def __init__(self, fetch):
        self.fetch, self.data, self.pos = fetch, b"", 0

    def take(self, n):
        while self.pos + n > len(self.data):
            self.data += self.fetch(len(self.data), max(1 << 22, n))
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q")).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.unpack("<I"), self.unpack("<Q")
            return [self.value(et) for _ in range(n)]
        return self.unpack(SCALAR[t])


def read_header(src: str, token: str | None = None) -> dict:
    """{'kv': {key: value}, 'tensors': [{'name', 'shape' (ggml order), 'type', 'offset'}], 'header_bytes'}."""
    if src.startswith("http"):
        s = requests.Session()
        if token:
            s.headers["Authorization"] = f"Bearer {token}"

        def fetch(off, n):
            r = s.get(src, headers={"Range": f"bytes={off}-{off + n - 1}"}, allow_redirects=True)
            r.raise_for_status()
            return r.content
    else:
        f = open(src, "rb")

        def fetch(off, n):
            f.seek(off)
            return f.read(n)
    b = _Buf(fetch)
    assert b.take(4) == b"GGUF"
    version, n_tensors, n_kv = b.unpack("<I"), b.unpack("<Q"), b.unpack("<Q")
    kv = {}
    for _ in range(n_kv):
        k = b.string()
        kv[k] = b.value(b.unpack("<I"))
    tensors = []
    for _ in range(n_tensors):
        name = b.string()
        nd = b.unpack("<I")
        shape = [b.unpack("<Q") for _ in range(nd)]
        tensors.append({"name": name, "shape": shape, "type": b.unpack("<I"), "offset": b.unpack("<Q")})
    return {"version": version, "kv": kv, "tensors": tensors, "header_bytes": b.pos}
