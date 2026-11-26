#!/usr/bin/env python3
"""Replace a converted mix's NVFP4 expert tensors by the Source's own NVFP4 bytes, plus their `.scale`.

This is the copy of "Can sarto's converter keep a Source's NVFP4 tensors bit-exact…": the Source's packed E2M1
values and E4M3 block scales are reshuffled into ggml's block_nvfp4 (copy_check.py's `to_ggml`), w13's
interleaved gate/up rows are put in the pinned converter's gate-then-up order, and fp32(amax / denominator)
is written as `<tensor>.scale` [n_expert]. Every other tensor and every metadata key is copied unchanged.

`llama-quantize` first writes these tensors as NVFP4 requantized from the F32 conversion (same size); this
swaps in the exact bytes. With --check, every copied tensor that the given GGUF (Unsloth's) also stores as
NVFP4 must match it byte for byte, `.scale` included.

Usage: splice_nvfp4.py MIX.gguf SOURCE.safetensors OUT.gguf [--check UNSLOTH.gguf]
"""
import argparse
import json
import re
import struct
import sys

import numpy as np

import gguf

NVFP4 = gguf.GGMLQuantizationType.NVFP4


def to_ggml(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """packed uint8 [n, k/2] (element 2i low nibble), scales E4M3 bytes [n, k/16] -> block_nvfp4 rows [n, k/64*36]."""
    n, nb = scales.shape
    assert packed.shape[1] * 2 == nb * 16 and nb % 4 == 0
    if (scales & 0x80).any() or (scales == 0x7F).any():
        raise ValueError("sign bit or NaN in a block scale")
    nib = np.stack([packed & 15, packed >> 4], axis=-1).reshape(n, nb, 16)
    qs = (nib[:, :, :8] | (nib[:, :, 8:] << 4)).astype(np.uint8)
    return np.concatenate([scales.reshape(n, nb // 4, 4), qs.reshape(n, nb // 4, 32)], axis=-1).reshape(n, -1)


class Safetensors:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.meta = json.loads(f.read(n))
        self.base, self.mm = 8 + n, np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str, dtype) -> np.ndarray:
        m = self.meta[name]
        a, b = m["data_offsets"]
        return np.asarray(self.mm[self.base + a:self.base + b]).view(dtype).reshape(m["shape"])


def copy_tensor(src: Safetensors, gguf_name: str) -> tuple:
    """(block_nvfp4 bytes [E, rows, row_bytes], scale f32 [E]) for blk.L.ffn_{down,gate_up}_exps.weight."""
    m = re.fullmatch(r"blk\.(\d+)\.ffn_(down|gate_up)_exps\.weight", gguf_name)
    base = f"model.llm.layers.{m[1]}.mlp.experts." + ("w2_weight" if m[2] == "down" else "w13_weight")
    packed, scales = src.get(base, np.uint8), src.get(base + "_scale", np.uint8)
    amax, den = src.get(base + "_amax", np.float32), src.get(base + "_outer_scale_denominator", np.float32)
    E, rows, _ = packed.shape
    out = []
    for e in range(E):
        p, s = packed[e], scales[e]
        if m[2] == "gate_up":  # Source rows interleave gate and up; the converter writes gate rows, then up rows
            p, s = np.concatenate([p[0::2], p[1::2]]), np.concatenate([s[0::2], s[1::2]])
        out.append(to_ggml(p, s))
    scale = (amax.astype(np.float32) / np.float32(den.reshape(-1)[0])).astype(np.float32)
    return np.stack(out), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mix")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--check")
    a = ap.parse_args()
    r = gguf.GGUFReader(a.mix)
    src = Safetensors(a.source)
    keep = [t.name for t in r.tensors if t.tensor_type == NVFP4 and re.fullmatch(r"blk\.\d+\.ffn_(down|gate_up)_exps\.weight", t.name)]
    chk = {t.name: t for t in gguf.GGUFReader(a.check).tensors} if a.check else {}
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(a.out, arch=arch, endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    copies, report = {}, {"copied": [], "checked_against": a.check is not None, "matched": [], "not_in_check": []}
    for t in r.tensors:
        if t.name in keep:
            data, scale = copy_tensor(src, t.name)
            assert data.nbytes == t.n_bytes and data.shape == tuple(t.data.shape), (t.name, data.shape, t.data.shape)
            copies[t.name] = (data, scale)
            report["copied"].append(t.name)
            c = chk.get(t.name)
            if c is not None and c.tensor_type == NVFP4:
                cs = chk[t.name.removesuffix(".weight") + ".scale"]
                if not (np.array_equal(np.asarray(c.data), data) and np.array_equal(np.asarray(cs.data).view(np.uint32), scale.view(np.uint32))):
                    sys.exit(f"{t.name}: copy differs from {a.check}")
                report["matched"].append(t.name)
            else:
                report["not_in_check"].append(t.name)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        if t.name in copies:
            s = copies[t.name][1]
            w.add_tensor_info(t.name.removesuffix(".weight") + ".scale", s.shape, s.dtype, s.nbytes, gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        if t.name in copies:
            w.write_tensor_data(copies[t.name][0])
            w.write_tensor_data(copies[t.name][1])
        else:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    report["copied_n"], report["matched_n"] = len(report["copied"]), len(report["matched"])
    print(json.dumps(report))


if __name__ == "__main__":
    main()
