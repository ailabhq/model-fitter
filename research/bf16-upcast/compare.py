"""Compare Inkling 2 Mini's Source expert values with ggml-org's BF16 upcast.

HTTP range reads only. No weight file is written.

The reference dequantization is conversion/inkling2.py `_dequant_expert_chunk` at llama.cpp
22b99122e (the pin): NVFP4 is E2M1 x E4M3 x fp32(amax/denominator), MXFP8 is E4M3 x 2^(UE8M0-127).
The BF16 rounding is gguf-py `BF16.quantize_blocks` at the same commit, which is
`ggml_compute_fp32_to_bf16` (round to nearest even).

Run: python3 compare.py
     python3 compare.py --inventory-only
"""
import argparse
import json
import struct
import sys
import threading
import time
from collections import Counter

import numpy as np
import requests

from gguf_header import read_header

SRC_REV = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
GGUF_REV = "0bdc580634ceac4c4d921f452777bf44228de9af"
SRC = (
    "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/"
    f"{SRC_REV}/model-00001-of-00001.safetensors"
)
BF16_URL = (
    "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/"
    f"{GGUF_REV}/Inkling-2-Mini-BF16.gguf"
)
NVFP4_URL = (
    "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/"
    f"{GGUF_REV}/Inkling-2-Mini-NVFP4.gguf"
)

HIDDEN, INTER, EXPERTS = 2304, 1152, 128
# ggml type id -> (name, block elements, block bytes). Sizes from ggml.c at 22b99122e.
TYPE = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    30: ("BF16", 1, 2),
    40: ("NVFP4", 64, 36),
}
BF16_TYPE, F32_TYPE, Q8_0, NVFP4 = 30, 0, 8, 40

# Complete experts: every value. Chosen to touch both projections, both ends of the
# expert index, and the first and last layer of each format.
FULL = {
    "nvfp4": [(0, "w2", 0), (0, "w13", 77), (19, "w2", 127), (19, "w13", 3)],
    "mxfp8": [(20, "w2", 0), (20, "w13", 127), (23, "w2", 50), (23, "w13", 1)],
}
# One row from every layer, three experts, two or three rows. Independent of FULL.
GRID_EXPERTS = (0, 63, 127)
GRID_ROWS = {"w2": (0, 1000), "w13": (0, 1000, 2000)}

_local = threading.local()
_bytes = 0
_lock = threading.Lock()


def session():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "sarto-bf16-upcast"
        _local.s = s
    return s


def get(url, off, n):
    global _bytes
    last = None
    for attempt in range(5):
        try:
            r = session().get(url, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=180)
            r.raise_for_status()
            if len(r.content) != n:
                raise IOError(f"short read {len(r.content)} != {n}")
            with _lock:
                _bytes += n
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 ** attempt)
    raise last


def file_size(url):
    r = session().get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    total = r.headers["Content-Range"].split("/")[-1]
    return int(total)


def safetensors_header(url):
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    return 8 + n, json.loads(get(url, 8, n))


def e4m3_table():
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3 = e4m3_table()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def f32_to_bf16_bits(x):
    """Round-to-nearest-even BF16 bits. Same arithmetic as gguf-py BF16.quantize_blocks at 22b99122e."""
    n = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    quiet = (n & np.uint32(0xFFFF0000)) | np.uint32(64 << 16)
    n = np.where((n & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000), quiet, n)
    n64 = n.astype(np.uint64)
    rounded = (n64 + (np.uint64(0x7FFF) + ((n64 >> np.uint64(16)) & np.uint64(1)))) >> np.uint64(16)
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits):
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16_bit(value):
    bits = f32_to_bf16_bits(np.asarray(value, dtype=np.float32)).reshape(-1)
    return int(bits[0])


def self_test_rounding():
    assert _bf16_bit(1.0) == 0x3F80 and _bf16_bit(0.0) == 0 and _bf16_bit(-0.0) == 0x8000
    back = bf16_bits_to_f32(np.array([_bf16_bit(1.0)], dtype=np.uint16))
    assert int(back.view(np.uint32)[0]) == 0x3F800000


def dequant_nvfp4(packed, scales, a, d):
    values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    values[:, 0::2] = E2M1[packed & 15]
    values[:, 1::2] = E2M1[packed >> 4]
    values.reshape(values.shape[0], -1, 16)[:] *= E4M3[scales][:, :, None]
    unscaled = values.copy()
    values *= np.float32(a) / np.float32(d)
    return values, unscaled


def dequant_mxfp8(codes, scales):
    values = E4M3[codes].copy()
    block = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
    values.reshape(values.shape[0], -1, 32)[:] *= block[:, :, None]
    return values


def to_ggml_nvfp4(packed, scales):
    """Source rows -> block_nvfp4 rows. Same shuffle as master `_repack_nvfp4_rows`."""
    n, nb = scales.shape
    nib = np.stack([packed & 15, packed >> 4], axis=-1).reshape(n, nb, 16)
    qs = (nib[:, :, :8] | (nib[:, :, 8:] << 4)).astype(np.uint8)
    return np.concatenate([scales.reshape(n, nb // 4, 4), qs.reshape(n, nb // 4, 32)], axis=-1).reshape(n, -1)


def logical_shape(kind):
    rows, cols = (2 * INTER, HIDDEN) if kind == "w13" else (HIDDEN, INTER)
    return rows, cols


def source_row(kind, logical_row):
    if kind == "w2":
        return logical_row
    half = INTER
    if logical_row < half:
        return 2 * logical_row
    return 2 * (logical_row - half) + 1


def deinterleave(kind, values):
    if kind == "w2":
        return values
    return np.concatenate([values[0::2], values[1::2]], axis=0)


def tensor_nbytes(t):
    name, blck, size = TYPE[t["type"]]
    n = 1
    for d in t["shape"]:
        n *= d
    if n % blck:
        raise ValueError(f"{t['name']}: {n} elements not divisible by {name} block {blck}")
    return n // blck * size


def data_origin(header):
    align = int(header["kv"].get("general.alignment", 32))
    return (header["header_bytes"] + align - 1) // align * align


def gguf_name(layer, kind):
    stem = "ffn_gate_up_exps" if kind == "w13" else "ffn_down_exps"
    return f"blk.{layer}.{stem}.weight"


def scale_name(layer, kind):
    return gguf_name(layer, kind).removesuffix(".weight") + ".scale"


class Acc:
    def __init__(self):
        self.n = 0
        self.n_numeric = 0
        self.n_neg0 = 0
        self.n_rne = 0
        self.n_unscaled_rne = 0
        self.n_unscaled = 0
        self.sum_abs = 0.0
        self.sum_rel = 0.0
        self.n_rel = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.max_ulp = 0.0
        self.max_abs_pair = (0.0, 0.0)
        self.max_rel_pair = (0.0, 0.0)
        self.n_ref_zero = 0
        self.n_stored_inf = 0
        self.n_ref_inf = 0
        # exact, <=2^-10, <=2^-9, <=2^-8, above 2^-8
        self.buckets = [0, 0, 0, 0, 0]

    def add(self, ref, stored_f32, unscaled=None):
        ref = np.ascontiguousarray(ref, dtype=np.float32)
        stored = np.ascontiguousarray(stored_f32, dtype=np.float32)
        both_zero = (stored == 0) & (ref == 0)
        numeric = (stored != ref) & ~both_zero
        neg0 = both_zero & (stored.view(np.uint32) != ref.view(np.uint32))
        rne = f32_to_bf16_bits(ref)
        stored_bits = (stored.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        ad = np.abs(stored.astype(np.float64) - ref.astype(np.float64))
        nonzero = ref != 0
        rel = np.zeros(ref.shape, dtype=np.float64)
        rel[nonzero] = ad[nonzero] / np.abs(ref.astype(np.float64)[nonzero])
        ax = np.abs(ref.astype(np.float64))
        exp = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))
        ulp = np.where(ax < 2.0 ** -126, 2.0 ** -133, np.ldexp(1.0, exp.astype(np.int32) - 7))
        ulp_err = np.zeros(ref.shape, dtype=np.float64)
        ulp_err[nonzero] = ad[nonzero] / ulp[nonzero]

        self.n += ref.size
        self.n_numeric += int(numeric.sum())
        self.n_neg0 += int(neg0.sum())
        self.n_rne += int((stored_bits == rne).sum())
        self.sum_abs += float(ad.sum())
        self.sum_rel += float(rel[nonzero].sum()) if nonzero.any() else 0.0
        self.n_rel += int(nonzero.sum())
        self.n_ref_zero += int((~nonzero).sum())
        self.n_stored_inf += int(np.isinf(stored).sum())
        self.n_ref_inf += int(np.isinf(ref).sum())
        if ref.size:
            i = int(np.argmax(ad))
            if ad.flat[i] > self.max_abs:
                self.max_abs = float(ad.flat[i])
                self.max_abs_pair = (float(ref.flat[i]), float(stored.flat[i]))
            j = int(np.argmax(rel))
            if rel.flat[j] > self.max_rel:
                self.max_rel = float(rel.flat[j])
                self.max_rel_pair = (float(ref.flat[j]), float(stored.flat[j]))
            k = int(np.argmax(ulp_err))
            self.max_ulp = max(self.max_ulp, float(ulp_err.flat[k]))
        exact = ~numeric
        self.buckets[0] += int(exact.sum())
        rest = numeric
        edges = (2.0 ** -10, 2.0 ** -9, 2.0 ** -8)
        prev = np.zeros(ref.shape, dtype=bool)
        for b, edge in enumerate(edges, start=1):
            hit = rest & (rel <= edge) & ~prev
            self.buckets[b] += int(hit.sum())
            prev = prev | hit
        self.buckets[4] += int((rest & ~prev).sum())
        if unscaled is not None:
            ubits = f32_to_bf16_bits(np.ascontiguousarray(unscaled, dtype=np.float32))
            self.n_unscaled_rne += int((stored_bits == ubits).sum())
            self.n_unscaled += ref.size

    def line(self, label):
        mean_abs = self.sum_abs / self.n if self.n else float("nan")
        mean_rel = self.sum_rel / self.n_rel if self.n_rel else float("nan")
        return (
            f"{label}: n={self.n} numeric_diff={self.n_numeric} neg0_only={self.n_neg0} "
            f"rne_match={self.n_rne}/{self.n} max_abs={self.max_abs:.6g} mean_abs={mean_abs:.6g} "
            f"max_rel={self.max_rel:.6g} mean_rel={mean_rel:.6g} max_ulp={self.max_ulp:.6g} "
            f"ref_zero={self.n_ref_zero} stored_inf={self.n_stored_inf} ref_inf={self.n_ref_inf} "
            f"buckets(exact,le2^-10,le2^-9,le2^-8,above)=({', '.join(str(b) for b in self.buckets)}) "
            f"max_abs_pair(ref,stored)=({self.max_abs_pair[0]:.8g}, {self.max_abs_pair[1]:.8g}) "
            f"max_rel_pair(ref,stored)=({self.max_rel_pair[0]:.8g}, {self.max_rel_pair[1]:.8g})"
            + (
                f" unscaled_rne={self.n_unscaled_rne}/{self.n_unscaled}"
                if self.n_unscaled
                else ""
            )
        )


def load_source_rows(base, st, layer, kind, expert, logical_rows):
    rows, cols = logical_shape(kind)
    fmt = "nvfp4" if layer < 20 else "mxfp8"
    weight = st[f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"]
    scale = st[f"model.llm.layers.{layer}.mlp.experts.{kind}_weight_scale"]
    src_rows = [source_row(kind, r) for r in logical_rows]
    # Source stores the interleaved order contiguously per expert. Read each source row.
    if fmt == "nvfp4":
        w_row, s_row = cols // 2, cols // 16
    else:
        w_row, s_row = cols, cols // 32
    w_off = base + weight["data_offsets"][0] + expert * rows * w_row
    s_off = base + scale["data_offsets"][0] + expert * rows * s_row
    packed = np.vstack([
        np.frombuffer(get(SRC, w_off + sr * w_row, w_row), np.uint8) for sr in src_rows
    ])
    scales = np.vstack([
        np.frombuffer(get(SRC, s_off + sr * s_row, s_row), np.uint8) for sr in src_rows
    ])
    return fmt, packed, scales


def load_source_expert(base, st, layer, kind, expert):
    rows, cols = logical_shape(kind)
    fmt = "nvfp4" if layer < 20 else "mxfp8"
    weight = st[f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"]
    scale = st[f"model.llm.layers.{layer}.mlp.experts.{kind}_weight_scale"]
    if fmt == "nvfp4":
        w_row, s_row = cols // 2, cols // 16
    else:
        w_row, s_row = cols, cols // 32
    w_off = base + weight["data_offsets"][0] + expert * rows * w_row
    s_off = base + scale["data_offsets"][0] + expert * rows * s_row
    packed = np.frombuffer(get(SRC, w_off, rows * w_row), np.uint8).reshape(rows, w_row)
    scales = np.frombuffer(get(SRC, s_off, rows * s_row), np.uint8).reshape(rows, s_row)
    return fmt, packed, scales


def amax_den(base, st, layer, kind):
    name = f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"
    am = st[name + "_amax"]
    dn = st[name + "_outer_scale_denominator"]
    amax = np.frombuffer(get(SRC, base + am["data_offsets"][0], 4 * EXPERTS), np.float32).copy()
    den = np.frombuffer(get(SRC, base + dn["data_offsets"][0], 4), np.float32).copy()
    return amax, den


def reference(fmt, packed, scales, a, d):
    if fmt == "nvfp4":
        return dequant_nvfp4(packed, scales, a, d)
    return dequant_mxfp8(packed, scales), None


def read_gguf_bf16_rows(url, origin, tensor, expert, row0, nrows):
    cols, rows, experts = tensor["shape"]
    assert experts == EXPERTS and tensor["type"] == BF16_TYPE
    off = origin + tensor["offset"] + (expert * rows + row0) * cols * 2
    bits = np.frombuffer(get(url, off, nrows * cols * 2), np.uint16).copy().reshape(nrows, cols)
    return bf16_bits_to_f32(bits)


def read_gguf_q8_rows(url, origin, tensor, expert, row0, nrows):
    cols, rows, experts = tensor["shape"]
    assert experts == EXPERTS and tensor["type"] == Q8_0 and cols % 32 == 0
    row_bytes = cols // 32 * 34
    off = origin + tensor["offset"] + (expert * rows + row0) * row_bytes
    raw = np.frombuffer(get(url, off, nrows * row_bytes), np.uint8).reshape(nrows, cols // 32, 34)
    d = np.frombuffer(np.ascontiguousarray(raw[:, :, :2]).tobytes(), np.float16).astype(np.float32)
    qs = np.frombuffer(np.ascontiguousarray(raw[:, :, 2:]).tobytes(), np.int8).astype(np.float32)
    return (qs.reshape(nrows, cols // 32, 32) * d.reshape(nrows, cols // 32, 1)).reshape(nrows, cols)


def inventory(header, nbytes_file, label):
    origin = data_origin(header)
    tensors = header["tensors"]
    by_type = Counter()
    type_bytes = Counter()
    missing = [t for t in tensors if t["type"] not in TYPE]
    print(f"\n== {label} ==")
    print(f"version {header['version']} tensors {len(tensors)} header_bytes {header['header_bytes']} data0 {origin} file_bytes {nbytes_file}")
    kv = header["kv"]
    for key in sorted(kv):
        val = kv[key]
        if isinstance(val, (str, list)) and (not isinstance(val, str) or len(val) > 200):
            shown = f"<{type(val).__name__} len {len(val)}>"
        else:
            shown = repr(val)
        if key.startswith("tokenizer.") or key.startswith("general.") or "file_type" in key or key.endswith(".expert_count") or "block_count" in key or "embedding_length" in key:
            if key.startswith("tokenizer.") and key not in ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id"):
                continue
            print(f"  kv {key} = {shown}")
    if missing:
        print("UNKNOWN TYPES", Counter(t["type"] for t in missing))
    ordered = sorted(tensors, key=lambda t: t["offset"])
    gap = 0
    logical = 0
    for i, t in enumerate(ordered):
        n = tensor_nbytes(t) if t["type"] in TYPE else None
        if n is None:
            continue
        logical += n
        nxt = ordered[i + 1]["offset"] if i + 1 < len(ordered) else nbytes_file - origin
        gap += nxt - t["offset"] - n
        by_type[TYPE[t["type"]][0]] += 1
        type_bytes[TYPE[t["type"]][0]] += n
    print("counts", dict(by_type))
    print("logical_bytes", dict(type_bytes), "sum", logical, "inter_tensor_gap", gap)
    end = ordered[-1]["offset"] + tensor_nbytes(ordered[-1]) if ordered[-1]["type"] in TYPE else None
    print(f"data span end {None if end is None else origin + end} file {nbytes_file} tail {None if end is None else nbytes_file - (origin + end)}")
    return origin


def expert_groups(tensors):
    groups = {"nvfp4_exps": 0, "mxfp8_exps": 0, "other_bf16": 0, "other": 0}
    detail = Counter()
    for t in tensors:
        if t["type"] not in TYPE:
            continue
        n = tensor_nbytes(t)
        name = t["name"]
        is_exp = name.endswith(".ffn_gate_up_exps.weight") or name.endswith(".ffn_down_exps.weight")
        layer = None
        if name.startswith("blk."):
            try:
                layer = int(name.split(".")[1])
            except ValueError:
                layer = None
        kind = TYPE[t["type"]][0]
        if is_exp and layer is not None and layer < 20:
            groups["nvfp4_exps"] += n
        elif is_exp and layer is not None and layer >= 20:
            groups["mxfp8_exps"] += n
        elif kind == "BF16":
            groups["other_bf16"] += n
        else:
            groups["other"] += n
        if is_exp:
            detail[(layer, kind)] += 1
    return groups, detail


def positive_control(base, st, bf16, origin):
    """A Source BF16 tensor copied into the GGUF must match byte for byte. This is the layout check."""
    src_name = "model.llm.embed.weight"
    info = st[src_name]
    assert info["dtype"] == "BF16" and info["shape"] == [201024, HIDDEN]
    g = next(t for t in bf16["tensors"] if t["name"] in ("token_embd.weight", "token_embd"))
    print(f"\npositive control: source {src_name} {info['dtype']} {info['shape']} vs gguf {g['name']} type {g['type']} shape {g['shape']}")
    assert g["type"] == BF16_TYPE and g["shape"] == [HIDDEN, 201024], g
    nrows = 8
    n = nrows * HIDDEN * 2
    src = get(SRC, base + info["data_offsets"][0], n)
    src2 = get(SRC, base + info["data_offsets"][0], n)
    dst = get(BF16_URL, origin + g["offset"], n)
    print(f"  double-read source equal: {src == src2}")
    print(f"  source bytes == gguf bytes ({n} B, first {nrows} rows): {src == dst}")
    if src != src2 or src != dst:
        print("POSITIVE CONTROL FAILED")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    assert sys.byteorder == "little"
    self_test_rounding()
    print(f"source {SRC}")
    print(f"bf16   {BF16_URL}")
    print(f"nvfp4  {NVFP4_URL}")
    print("rounding self-test: 1.0 -> 0x3f80, +0 -> 0, -0 -> 0x8000")

    bf16_size = file_size(BF16_URL)
    nv_size = file_size(NVFP4_URL)
    base, st = safetensors_header(SRC)
    src_size = file_size(SRC)
    print(f"source safetensors bytes {src_size} header_end {base} tensors {len(st) - (1 if '__metadata__' in st else 0)}")
    bf16 = read_header(BF16_URL)
    nv = read_header(NVFP4_URL)
    origin_b = inventory(bf16, bf16_size, "ggml-org Inkling-2-Mini-BF16.gguf")
    origin_n = inventory(nv, nv_size, "ggml-org Inkling-2-Mini-NVFP4.gguf")
    gb, db = expert_groups(bf16["tensors"])
    gn, dn = expert_groups(nv["tensors"])
    print("\nBF16 file logical bytes by role", gb)
    print("NVFP4 file logical bytes by role", gn)
    bf16_payload = gb["nvfp4_exps"] + gb["mxfp8_exps"] + gb["other_bf16"]
    scales = 40 * EXPERTS * 4
    sizes = {
        "bf16_file": bf16_size,
        "f32_all": bf16_size + bf16_payload,
        "f32_nvfp4_experts_only": bf16_size + gb["nvfp4_exps"],
        "path_i2": bf16_size - gb["nvfp4_exps"] + gn["nvfp4_exps"] + scales,
        "published_nvfp4_file": nv_size,
    }
    print(
        "candidate sizes (bytes). f32_all doubles every BF16 payload. "
        "f32_nvfp4_experts_only doubles only the layer 0-19 expert payloads. "
        "path_i2 swaps those payloads for the published file's NVFP4 bytes plus 40 F32 scales of 512 B."
    )
    for name, n in sizes.items():
        print(f"  {name}: {n} bytes, {n / 1e9:.6f} GB, {n / 1024 ** 3:.6f} GiB")
    mis = sum(1 for t in bf16["tensors"] if tensor_nbytes(t) % 32)
    print(f"bf16 tensor logical sizes not divisible by 32: {mis}/{len(bf16['tensors'])}")
    # One GGUF tensor record: uint64 name length, name bytes, uint32 n_dims, uint64 dim, uint32 type, uint64 offset.
    info = sum(32 + len(scale_name(layer, kind)) for layer in range(20) for kind in ("w2", "w13"))
    align = int(bf16["kv"].get("general.alignment", 32))
    new_header = bf16["header_bytes"] + info
    new_data0 = (new_header + align - 1) // align * align
    path_file = sizes["path_i2"] - origin_b + new_data0
    print(
        f"path_i2 with 40 scale records: record_bytes {info}, header {bf16['header_bytes']} -> {new_header}, "
        f"data0 {origin_b} -> {new_data0}, file {path_file} bytes, {path_file / 1e9:.6f} GB, {path_file / 1024 ** 3:.6f} GiB"
    )
    print("BF16 expert tensor types", sorted(db.items())[:8], "...", sorted(db.items())[-4:])
    print("NVFP4-file expert tensor types", sorted(dn.items()))
    names_b = {t["name"]: t for t in bf16["tensors"]}
    names_n = {t["name"]: t for t in nv["tensors"]}
    print("bf16 has ffn scale tensors", sum(1 for n in names_b if n.endswith("ffn_down_exps.scale") or n.endswith("ffn_gate_up_exps.scale")))
    print("nvfp4 file has ffn scale tensors", sum(1 for n in names_n if n.endswith("ffn_down_exps.scale") or n.endswith("ffn_gate_up_exps.scale")))
    if args.inventory_only:
        print(f"bytes read {_bytes}")
        return

    positive_control(base, st, bf16, origin_b)

    # Every NVFP4 per-expert scale, the whole population (40 tensors x 128), not a sample.
    scale_match = 0
    scale_n = 0
    scale_vals = []
    for layer in range(20):
        for kind in ("w2", "w13"):
            amax, den = amax_den(base, st, layer, kind)
            pred = (amax / den[0]).astype(np.float32)
            scale_vals.append(pred)
            key = scale_name(layer, kind)
            got = names_n.get(key)
            if got is None:
                print(f"MISSING scale {key}")
                continue
            raw = np.frombuffer(get(NVFP4_URL, origin_n + got["offset"], pred.nbytes), np.float32).copy()
            scale_match += int(np.array_equal(pred.view(np.uint32), raw.view(np.uint32)))
            scale_n += 1
    gs = np.concatenate(scale_vals)
    print(
        f"\nall NVFP4 .scale tensors: fp32(amax/denominator) bit-exact vs published NVFP4 file "
        f"{scale_match}/{scale_n}; range {gs.min():.6g} .. {gs.max():.6g}"
    )

    full_acc = {"nvfp4": Acc(), "mxfp8": Acc()}
    for fmt, jobs in FULL.items():
        for layer, kind, expert in jobs:
            rows, cols = logical_shape(kind)
            sfmt, packed, scales = load_source_expert(base, st, layer, kind, expert)
            assert sfmt == fmt
            if fmt == "nvfp4":
                amax, den = amax_den(base, st, layer, kind)
                values, unscaled = reference(fmt, packed, scales, amax[expert], den[0])
                values, unscaled = deinterleave(kind, values), deinterleave(kind, unscaled)
                repacked = to_ggml_nvfp4(deinterleave(kind, packed), deinterleave(kind, scales))
                nt = names_n[gguf_name(layer, kind)]
                assert nt["type"] == NVFP4 and nt["shape"] == [cols, rows, EXPERTS], nt
                ex_bytes = rows * cols // 64 * 36
                theirs = np.frombuffer(
                    get(NVFP4_URL, origin_n + nt["offset"] + expert * ex_bytes, ex_bytes), np.uint8
                ).reshape(rows, -1)
                byte_equal = bool(np.array_equal(repacked, theirs))
            else:
                values, unscaled = deinterleave(kind, dequant_mxfp8(packed, scales)), None
                byte_equal = None
                sc = scales.reshape(-1)
                print(
                    f"  mxfp8 scale bytes layer {layer} {kind} e{expert}: "
                    f"min {int(sc.min())} max {int(sc.max())} n255 {int((sc == 255).sum())}"
                )
            bt = names_b[gguf_name(layer, kind)]
            assert bt["type"] == BF16_TYPE and bt["shape"] == [cols, rows, EXPERTS], bt
            stored = read_gguf_bf16_rows(BF16_URL, origin_b, bt, expert, 0, rows)
            full_acc[fmt].add(values, stored, unscaled)
            rne = f32_to_bf16_bits(values)
            stored_bits = (np.ascontiguousarray(stored).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            match = int((stored_bits == rne).sum())
            numeric = int(((stored != values) & ~((stored == 0) & (values == 0))).sum())
            extra = f" nvfp4_bytes_equal_published={byte_equal}" if byte_equal is not None else ""
            print(
                f"full {fmt} L{layer} {kind} e{expert}: values {values.size} numeric_diff {numeric} "
                f"rne {match}/{values.size}{extra}"
            )
            if match < values.size * 0.5 and kind == "w13":
                alt_bits = f32_to_bf16_bits(dequant_mxfp8(packed, scales) if fmt == "mxfp8" else reference(fmt, packed, scales, amax[expert], den[0])[0])
                print(f"  no-deinterleave rne match {int((stored_bits == alt_bits).sum())}/{values.size}")

    print("\n" + full_acc["nvfp4"].line("FULL nvfp4 vs BF16 GGUF"))
    print(full_acc["mxfp8"].line("FULL mxfp8 vs BF16 GGUF"))

    grid_acc = {"nvfp4": Acc(), "mxfp8": Acc()}
    layer_match = {}
    for layer in range(24):
        fmt = "nvfp4" if layer < 20 else "mxfp8"
        for kind in ("w2", "w13"):
            acc = Acc()
            amax = den = None
            if fmt == "nvfp4":
                amax, den = amax_den(base, st, layer, kind)
            bt = names_b[gguf_name(layer, kind)]
            for expert in GRID_EXPERTS:
                logical_rows = list(GRID_ROWS[kind])
                sfmt, packed, scales = load_source_rows(base, st, layer, kind, expert, logical_rows)
                if fmt == "nvfp4":
                    values, unscaled = reference(fmt, packed, scales, amax[expert], den[0])
                else:
                    values, unscaled = dequant_mxfp8(packed, scales), None
                # load_source_rows already returns logical (deinterleaved) row order.
                stored_parts = [
                    read_gguf_bf16_rows(BF16_URL, origin_b, bt, expert, r, 1) for r in logical_rows
                ]
                stored = np.concatenate(stored_parts, axis=0)
                acc.add(values, stored, unscaled)
                grid_acc[fmt].add(values, stored, unscaled)
            layer_match[(layer, kind)] = (acc.n_rne, acc.n, acc.n_numeric)
            print(acc.line(f"grid L{layer} {kind}"))
    print("\n" + grid_acc["nvfp4"].line("GRID nvfp4 vs BF16 GGUF"))
    print(grid_acc["mxfp8"].line("GRID mxfp8 vs BF16 GGUF"))
    worst = min(layer_match.items(), key=lambda kv: kv[1][0] / kv[1][1])
    print(f"worst grid layer rne match: {worst[0]} {worst[1][0]}/{worst[1][1]} numeric_diff {worst[1][2]}")

    # The published NVFP4 file stores MXFP8 experts as whatever type the header says.
    pub = Acc()
    for layer, kind, expert in FULL["mxfp8"]:
        rows, cols = logical_shape(kind)
        _fmt, packed, scales = load_source_expert(base, st, layer, kind, expert)
        values = deinterleave(kind, dequant_mxfp8(packed, scales))
        nt = names_n[gguf_name(layer, kind)]
        if nt["type"] == Q8_0:
            stored = read_gguf_q8_rows(NVFP4_URL, origin_n, nt, expert, 0, rows)
        elif nt["type"] == BF16_TYPE:
            stored = read_gguf_bf16_rows(NVFP4_URL, origin_n, nt, expert, 0, rows)
        else:
            print(f"published MXFP8 L{layer} {kind} type {nt['type']} not compared")
            continue
        before = pub.n_numeric
        pub.add(values, stored)
        print(
            f"published-file {TYPE[nt['type']][0]} L{layer} {kind} e{expert}: "
            f"numeric_diff {pub.n_numeric - before}/{values.size}"
        )
    print(pub.line("FULL mxfp8 vs published NVFP4-file experts"))
    print(f"\nbytes read {_bytes}")


if __name__ == "__main__":
    main()
