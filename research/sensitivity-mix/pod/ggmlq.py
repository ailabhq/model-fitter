"""ggml's own quantize-then-dequantize, through libggml-base built at the pinned llama.cpp.

Rows are quantized with ggml_quantize_chunk, the function llama-quantize calls, in the same row chunks and with
the same imatrix arithmetic (common/imatrix-loader.cpp: in_sum2 / round(counts), 1 where the count is 0), then
dequantized with the type's own to_float. The probe needs only the dequantized values.
"""
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TYPES = {"q8_0": 8, "q5_0": 6, "q5_1": 7, "q4_0": 2, "q4_1": 3, "q2_K": 10, "q3_K": 11, "q4_K": 12, "q5_K": 13,
         "q6_K": 14, "iq2_xxs": 16, "iq2_xs": 17, "iq3_xxs": 18, "iq1_s": 19, "iq4_nl": 20, "iq3_s": 21,
         "iq2_s": 22, "iq4_xs": 23, "iq1_m": 29, "tq2_0": 35, "mxfp4": 39, "nvfp4": 40, "q1_0": 41, "q2_0": 42}
MIN_CHUNK = 32 * 512  # llama-quant.cpp's min_chunk_size


class TypeTraits(ctypes.Structure):  # ggml.h at the pin
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)


class GGML:
    def __init__(self, lib: str | None = None, threads: int | None = None):
        g = self.g = ctypes.CDLL(lib or os.environ["MF_GGML_LIB"])
        g.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_void_p]
        g.ggml_quantize_chunk.restype = ctypes.c_size_t
        g.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        g.ggml_row_size.restype = ctypes.c_size_t
        g.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        g.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        g.ggml_quantize_init.argtypes = [ctypes.c_int]
        g.ggml_get_type_traits.argtypes = [ctypes.c_int]
        g.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
        g.ggml_validate_row_data.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        g.ggml_validate_row_data.restype = ctypes.c_bool
        self.threads = threads or os.cpu_count()
        self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix="ggml")
        self._inited = set()

    def requires_imatrix(self, qtype: str) -> bool:
        return bool(self.g.ggml_quantize_requires_imatrix(TYPES[qtype]))

    def roundtrip(self, qtype: str, f32: np.ndarray, imat: np.ndarray | None, rows_per_expert: int | None = None) -> np.ndarray:
        """Dequantized ggml quantization of every row of f32 [rows, n]. imat: [n] or [n_expert, n]."""
        t = TYPES[qtype]
        if t not in self._inited:
            self.g.ggml_quantize_init(t)
            self._inited.add(t)
        f32 = np.ascontiguousarray(f32, dtype=np.float32)
        nrows, n = f32.shape
        rpe = rows_per_expert or nrows
        if imat is not None:
            imat = np.ascontiguousarray(imat, dtype=np.float32).reshape(-1, n)
            assert imat.shape[0] in (1, nrows // rpe), (imat.shape, nrows, rpe)
        elif self.requires_imatrix(qtype):
            raise ValueError(f"{qtype} needs an imatrix")
        row_size = self.g.ggml_row_size(t, n)
        q = np.empty(nrows * row_size, dtype=np.uint8)
        out = np.empty_like(f32)
        per_chunk = max(1, (n if n >= MIN_CHUNK else n * ((MIN_CHUNK + n - 1) // n)) // n)
        jobs, r = [], 0
        while r < nrows:
            k = min(nrows - r, per_chunk, rpe - r % rpe)
            jobs.append((r, k, None if imat is None else imat[(r // rpe) if imat.shape[0] > 1 else 0]))
            r += k
        to_float = TO_FLOAT(self.g.ggml_get_type_traits(t).contents.to_float)
        src, dst, deq = f32.ctypes.data, q.ctypes.data, out.ctypes.data

        def one(job):
            r0, k, im = job
            p = dst + r0 * row_size
            size = self.g.ggml_quantize_chunk(t, src + r0 * n * 4, p, 0, k, n, None if im is None else im.ctypes.data)
            if size != k * row_size or not self.g.ggml_validate_row_data(t, p, size):
                raise RuntimeError(f"quantized data validation failed ({qtype})")
            to_float(p, deq + r0 * n * 4, k * n)

        list(self.pool.map(one, jobs))
        return out


def load_imatrix(path: str) -> dict:
    """llama-quantize's imatrix values per tensor: [n_expert or 1, n]."""
    import gguf
    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}
    out = {}
    for name in t:
        if not name.endswith(".in_sum2"):
            continue
        base = name.removesuffix(".in_sum2")
        sums = np.array(t[name].data, dtype=np.float32).reshape(-1)
        counts = np.floor(np.array(t[base + ".counts"].data, dtype=np.float64).reshape(-1) + 0.5).astype(np.float32)
        v = sums.reshape(len(counts), -1).copy()
        for j, c in enumerate(counts):
            v[j] = v[j] / c if c > 0 else 1.0
        out[base] = v
    return out
