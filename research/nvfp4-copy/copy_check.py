"""Copy one Inkling-2-Mini NVFP4 expert into ggml's NVFP4 layout, byte for byte, and check it.

For each (layer, tensor, expert) asked for, by HTTP range reads only (about 3-6 MB per expert):
  1. read the Source's packed E2M1 bytes, E4M3 block scales, amax and outer_scale_denominator;
  2. repack them into block_nvfp4 rows (this script's numpy mapping, and the pinned converter's own
     ModelBase._nvfp4_pack, which the ModelOpt path uses);
  3. compare with the same expert's bytes and .scale in Unsloth's mainline GGUF;
  4. dequantize the repacked rows with ggml's C dequantize_row_nvfp4 (libggml-base built at the pin) and with
     gguf-py's NVFP4 at the pin, multiply by the .scale, and compare bit for bit with the pinned Inkling
     converter's reference dequantization (conversion/inkling2.py _dequant_expert_chunk formula).

Environment: LLAMA_CPP = a llama.cpp checkout at 22b99122e; LIBGGML_BASE = its libggml-base dylib/so.
"""
import ctypes
import json
import os
import struct
import sys

import numpy as np
import requests

# Source revision c9d02830; its safetensors file is the same blob (sha256 65e60f51...) as at 7a67ae23
SRC = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/c9d028302cd0f035fdf8e7e91f089a4ce94d2195/model-00001-of-00001.safetensors"
GGUF_REV = "8078d149286eec7193795413b9e0643da6d11fa1"
GGUF = f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{GGUF_REV}/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"
HIDDEN, INTER, EXPERTS = 2304, 1152, 128
NVFP4 = 40

sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP"], "gguf-py"))
sys.path.insert(0, os.environ["LLAMA_CPP"])
sys.path.insert(0, os.path.dirname(__file__))
from gguf.quants import dequantize as gguf_dequantize  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf_header import read_header  # noqa: E402

http = requests.Session()


def get(url, off, n):
    r = http.get(url, headers={"Range": f"bytes={off}-{off + n - 1}"}, allow_redirects=True)
    r.raise_for_status()
    assert len(r.content) == n
    return r.content


def safetensors_header(url):
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    return 8 + n, json.loads(get(url, 8, n))


# --- reference: the pinned Inkling converter's dequantization (conversion/inkling2.py L18-65 at 22b99122e) ---
def _e4m3_table():
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3 = _e4m3_table()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def reference_dequant(packed, scales, a, d):
    values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    values[:, 0::2] = E2M1[packed & 15]
    values[:, 1::2] = E2M1[packed >> 4]
    values.reshape(values.shape[0], -1, 16)[:] *= E4M3[scales][:, :, None]
    values *= np.float32(a) / np.float32(d)
    return values


# --- the copy: Source rows -> block_nvfp4 rows ---
def to_ggml(packed, scales):
    """packed uint8 [n, k/2] (element 2i low nibble, 2i+1 high), scales E4M3 bytes [n, k/16] -> [n, k/64*36]."""
    n, nb = scales.shape
    assert packed.shape[1] * 2 == nb * 16 and nb % 4 == 0
    assert not (scales & 0x80).any() and not (scales == 0x7F).any(), "sign or NaN in a block scale"
    nib = np.stack([packed & 15, packed >> 4], axis=-1).reshape(n, nb, 16)
    qs = (nib[:, :, :8] | (nib[:, :, 8:] << 4)).astype(np.uint8)  # ggml: byte j holds elements j and j + 8
    return np.concatenate([scales.reshape(n, nb // 4, 4), qs.reshape(n, nb // 4, 32)], axis=-1).reshape(n, -1)


def pinned_pack(packed, scales):
    """The pin's own ModelOpt NVFP4 repack, conversion/base.py ModelBase._nvfp4_pack, lifted out by its AST
    (importing conversion.base pulls in transformers)."""
    import ast
    import textwrap
    import torch
    src = open(os.path.join(os.environ["LLAMA_CPP"], "conversion", "base.py")).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_nvfp4_pack")
    ns = {"np": np, "torch": torch, "Tensor": torch.Tensor}
    exec(textwrap.dedent("\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno])), ns)
    raw, _ = ns["_nvfp4_pack"](torch.from_numpy(packed), torch.from_numpy(scales).view(torch.float8_e4m3fn))
    return raw


def ggml_c_dequant(raw, k):
    lib = ctypes.CDLL(os.environ["LIBGGML_BASE"])
    lib.dequantize_row_nvfp4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    raw = np.ascontiguousarray(raw)
    out = np.empty(raw.shape[0] * k, dtype=np.float32)
    lib.dequantize_row_nvfp4(raw.ctypes.data, out.ctypes.data, out.size)
    return out.reshape(raw.shape[0], k)


def bits_equal(x, y):
    return x.shape == y.shape and np.array_equal(x.view(np.uint32), y.view(np.uint32))


def dequant_report(x, ref):
    """Bit-for-bit comparison; ggml decodes the E2M1 code 0b1000 (-0) as +0, so count those apart."""
    diff = x.view(np.uint32) != ref.view(np.uint32)
    neg0 = diff & (ref == 0) & (x == 0)
    return f"{int(diff.sum() - neg0.sum())} values differ, {int(neg0.sum())} are -0 in the reference and +0 here"


def main():
    jobs = [(0, "w2", 0), (0, "w2", 77), (0, "w13", 0), (0, "w13", 77), (19, "w2", 5), (19, "w13", 127)]
    base, st = safetensors_header(SRC)
    gg = read_header(GGUF)
    gt = {t["name"]: t for t in gg["tensors"]}
    align = gg["kv"].get("general.alignment", 32)
    data0 = (gg["header_bytes"] + align - 1) // align * align
    print(f"source: {SRC}\ngguf:   {GGUF}")
    for layer, kind, e in jobs:
        name = f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"
        w, s, am, dn = (st[name + x] for x in ("", "_scale", "_amax", "_outer_scale_denominator"))
        assert w["dtype"] == "U8" and s["dtype"] == "F8_E4M3" and am["dtype"] == "F32" and dn["dtype"] == "F32"
        rows, cols = (2 * INTER, HIDDEN) if kind == "w13" else (HIDDEN, INTER)
        assert w["shape"] == [EXPERTS, rows, cols // 2] and s["shape"] == [EXPERTS, rows, cols // 16]
        assert am["shape"] == [EXPERTS] and dn["shape"] == [1]  # one global scale per expert, shared by gate and up

        def expert(t, row_bytes):
            o = base + t["data_offsets"][0] + e * rows * row_bytes
            return np.frombuffer(get(SRC, o, rows * row_bytes), np.uint8).reshape(rows, row_bytes)

        packed, scales = expert(w, cols // 2), expert(s, cols // 16)
        amax = np.frombuffer(get(SRC, base + am["data_offsets"][0], 4 * EXPERTS), np.float32)
        den = np.frombuffer(get(SRC, base + dn["data_offsets"][0], 4), np.float32)
        scale = (amax / den).astype(np.float32)  # fp32 division, as the pinned converter does it

        # gate_up: the Source interleaves gate and up rows; GGUF holds all gate rows, then all up rows
        order = np.concatenate([np.arange(0, rows, 2), np.arange(1, rows, 2)]) if kind == "w13" else np.arange(rows)
        mine = to_ggml(packed[order], scales[order])
        pin = pinned_pack(packed[order], scales[order])

        gname = f"blk.{layer}." + ("ffn_gate_up_exps" if kind == "w13" else "ffn_down_exps")
        t, ts = gt[gname + ".weight"], gt[gname + ".scale"]
        assert t["type"] == NVFP4 and t["shape"] == [cols, rows, EXPERTS], (t["type"], t["shape"])
        ex_bytes = rows * cols // 64 * 36
        theirs = np.frombuffer(get(GGUF, data0 + t["offset"] + e * ex_bytes, ex_bytes), np.uint8).reshape(rows, -1)
        their_scale = np.frombuffer(get(GGUF, data0 + ts["offset"], 4 * EXPERTS), np.float32)

        ref = reference_dequant(packed, scales, amax[e], den[0])[order]
        c = ggml_c_dequant(mine, cols) * their_scale[e]
        py = gguf_dequantize(mine, GGMLQuantizationType.NVFP4).astype(np.float32) * their_scale[e]
        nz = scales[scales != 0]
        print(f"\nlayer {layer} {kind} expert {e}: {rows}x{cols}, source {packed.nbytes + scales.nbytes} B -> ggml {mine.nbytes} B")
        print(f"  E4M3 block scales: {scales.size}, zero {int((scales == 0).sum())}, subnormal {int(((nz >> 3) == 0).sum())}, "
              f"sign bit {int((scales & 0x80).astype(bool).sum())}, max byte 0x{int(scales.max()):02x}")
        print(f"  global scale amax/denominator = {amax[e]:.6g}/{den[0]:.6g} = {scale[e]:.9g}")
        print(f"  this mapping == pinned _nvfp4_pack:           {np.array_equal(mine, pin)}")
        print(f"  this mapping == Unsloth GGUF bytes:           {np.array_equal(mine, theirs)}")
        print(f"  fp32(amax/denominator) == Unsloth .scale:     {bits_equal(scale, their_scale)}")
        print(f"  ggml C dequant x .scale vs reference:         {dequant_report(c, ref)}")
        print(f"  gguf-py dequant x .scale vs reference:        {dequant_report(py, ref)}")
        folded = gguf_dequantize(to_ggml(*refold(packed[order], scales[order], scale[e])), GGMLQuantizationType.NVFP4)
        print(f"  fold scale into E4M3 block scales instead:    {fold_report(folded.astype(np.float32), ref)}")
    # every NVFP4 tensor's .scale, in the all-NVFP4 file and in UD-IQ3_XXS (where gate_up is an IQ type)
    iq3 = GGUF.replace("UD-Q4_K_XL", "UD-IQ3_XXS")
    hi = read_header(iq3)
    hit = {t["name"]: t for t in hi["tensors"]}
    d3 = (hi["header_bytes"] + align - 1) // align * align
    same = same3 = 0
    gs = []
    for layer in range(20):
        for kind, g in (("w2", "ffn_down_exps"), ("w13", "ffn_gate_up_exps")):
            name = f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"
            amax = np.frombuffer(get(SRC, base + st[name + "_amax"]["data_offsets"][0], 4 * EXPERTS), np.float32)
            den = np.frombuffer(get(SRC, base + st[name + "_outer_scale_denominator"]["data_offsets"][0], 4), np.float32)
            scale = (amax / den).astype(np.float32)
            gs.append(scale)
            key = f"blk.{layer}.{g}.scale"
            same += bits_equal(scale, np.frombuffer(get(GGUF, data0 + gt[key]["offset"], 4 * EXPERTS), np.float32))
            same3 += bits_equal(scale, np.frombuffer(get(iq3, d3 + hit[key]["offset"], 4 * EXPERTS), np.float32))
    gs = np.concatenate(gs)
    print(f"\nall 40 NVFP4 tensors (layers 0-19, down and gate_up): fp32(amax/denominator) == .scale for all 128 experts in "
          f"{same}/40 (UD-Q4_K_XL) and {same3}/40 (UD-IQ3_XXS); per-expert scale range {gs.min():.3g} .. {gs.max():.3g}")


def refold(packed, scales, g):
    """Try to drop the .scale tensor: multiply each E4M3 block scale by g and round back to UE4M3 (ggml's rounding)."""
    from gguf.quants import NVFP4 as Q
    s = Q.fp32_to_ue4m3((E4M3[scales] * g).astype(np.float32)).astype(np.uint8)
    return packed, s


def fold_report(x, ref):
    exact = np.mean(x.view(np.uint32) == ref.view(np.uint32))
    rel = np.sqrt(np.mean((x - ref) ** 2) / np.mean(ref ** 2))
    return f"{exact:.1%} of values exact, relative RMS error {rel:.3g}"


if __name__ == "__main__":
    main()
