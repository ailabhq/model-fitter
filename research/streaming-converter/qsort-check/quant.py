"""Quantize one tensor's F32 data with the pinned ggml and compare it with the x86 reference bytes."""
import ctypes, os, sys, hashlib
import numpy as np, gguf

name, qtype_name, ne0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
ggml = ctypes.CDLL(os.environ["MF_GGML_LIB"])
ggml.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
ggml.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
ggml.ggml_row_size.restype = ctypes.c_size_t

r = gguf.GGUFReader(os.path.expanduser("~/mf-bench/conv/mix/qwen-adapted.imatrix.gguf"))
t = {x.name: x for x in r.tensors}
sums = np.array(t[name + ".in_sum2"].data, dtype=np.float32).reshape(-1)
count = np.floor(float(np.array(t[name + ".counts"].data).reshape(-1)[0]) + 0.5)
imat = np.ascontiguousarray(sums / np.float32(count), dtype=np.float32)

f32 = np.fromfile(name + ".f32", dtype=np.float32)
rows = f32.size // ne0
qtype = int(gguf.GGMLQuantizationType[qtype_name])
out = np.empty(rows * ggml.ggml_row_size(qtype, ne0), dtype=np.uint8)
ggml.ggml_quantize_chunk(qtype, f32.ctypes.data, out.ctypes.data, 0, rows, ne0, imat.ctypes.data)
ref = np.fromfile(name + ".ref", dtype=np.uint8)
bs = ggml.ggml_row_size(qtype, ne0) // (ne0 // 256)  # bytes per 256-wide block
diff_blocks = int((out.reshape(-1, bs) != ref.reshape(-1, bs)).any(axis=1).sum())
print(name, qtype_name, "identical" if out.tobytes() == ref.tobytes() else f"differs in {diff_blocks}/{out.size // bs} blocks",
      "qsort=" + ("stable" if os.environ.get("DYLD_INSERT_LIBRARIES") else "macOS libc"))
