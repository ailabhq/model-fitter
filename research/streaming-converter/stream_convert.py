#!/usr/bin/env python3
"""Throwaway per-tensor streaming converter: Hugging Face Source -> quantized GGUF, without the Source on disk.

The pinned llama.cpp's own converter (its `conversion` package and gguf-py) maps names and applies every
per-architecture transform, exactly as `convert_hf_to_gguf.py --remote --outtype f32` would. Instead of writing
F32, each tensor is quantized on the way out with ggml_quantize_chunk from the pinned llama.cpp (the function
llama-quantize calls), with the same imatrix arithmetic, so the output should equal
`convert_hf_to_gguf.py --outtype f32` + `llama-quantize --imatrix I --tensor-type-file T`.

Source bytes come by HTTP byte range only. Tensors that reach the writer as a plain dtype upcast of a stored
tensor are read and quantized in row slabs; tensors the converter transforms are materialized whole; Inkling's
packed NVFP4/MXFP8 experts are read one expert at a time and dequantized by llama.cpp's own chunk function.
The disk holds the output GGUF and the config/tokenizer files, nothing else.

Usage: stream_convert.py HF_REPO OUT.gguf --types types.json --imatrix imatrix.gguf --ftype MOSTLY_IQ3_XXS
                         [--default-type Q8_0] [--threads 6] [--stats stats.json]
Environment: MF_LLAMA (pinned llama.cpp source tree), MF_GGML_LIB (its libggml-base.dylib), HF_TOKEN.
"""
import argparse
import ctypes
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch

LLAMA = Path(os.environ["MF_LLAMA"])
sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py")]
import gguf  # noqa: E402
from conversion import ModelType, get_model_architecture, get_model_class  # noqa: E402
from conversion import base as conv_base  # noqa: E402

log = logging.getLogger("stream_convert")
QT = gguf.GGMLQuantizationType
STATS = {"bytes_downloaded": 0, "range_requests": 0, "retries": 0, "streamed": 0, "whole": 0, "chunked": 0,
         "whole_max_f32_bytes": 0, "whole_names": [], "kept": 0, "types": {}, "missing_imatrix": [], "defaulted": []}
_lock = threading.Lock()

# ---------------------------------------------------------------- byte-range fetching

PIECE = 32 << 20
_session = requests.Session()
if os.environ.get("HF_TOKEN"):
    _session.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
_net = ThreadPoolExecutor(8, thread_name_prefix="net")
# MF_SOURCE_MIRROR="REPO=DIR_OR_URL[;...]": read REPO's files from a mirror (a directory, or an HTTP server with
# byte ranges, e.g. the NAS) instead of from Hugging Face. The bytes and offsets are the same; only the transport changes.
MIRROR = {f"https://huggingface.co/{r}/resolve/main/": d for r, _, d in
          (s.partition("=") for s in os.environ.get("MF_SOURCE_MIRROR", "").split(";") if s)}
_files: dict = {}


def _mirror_read(path: str, start: int, size: int) -> bytes:
    with _lock:
        fd = _files.get(path)
        if fd is None:
            fd = _files[path] = os.open(path, os.O_RDONLY)
    data = os.pread(fd, size, start)
    if len(data) != size:
        raise IOError(f"{path}: short read at {start}+{size}")
    with _lock:
        STATS["bytes_downloaded"] += size
        STATS["range_requests"] += 1
    return data


def _get(url: str, start: int, size: int) -> bytes:
    for prefix, d in MIRROR.items():
        if url.startswith(prefix):
            if not d.startswith("http"):
                return _mirror_read(os.path.join(d, url[len(prefix):]), start, size)
            url = d.rstrip("/") + "/" + url[len(prefix):]
    for attempt in range(8):
        try:
            hdr = {"Range": f"bytes={start}-{start + size - 1}"}
            if not url.startswith("https://huggingface.co/"):
                hdr["Authorization"] = None  # the Hugging Face token never goes to a mirror
            r = _session.get(url, headers=hdr, timeout=120)
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != size:
                raise IOError(f"range {start}+{size}: status {r.status_code}, {len(r.content)} bytes")
            with _lock:
                STATS["bytes_downloaded"] += size
                STATS["range_requests"] += 1
            return r.content
        except (requests.RequestException, IOError):
            if attempt == 7:
                raise
            with _lock:
                STATS["retries"] += 1
            time.sleep(2 ** attempt)


def fetch(url: str, start: int, size: int) -> bytearray:
    """Exactly bytes [start, start + size) of url, in parallel pieces."""
    out = bytearray(size)
    pieces = [(o, min(PIECE, size - o)) for o in range(0, size, PIECE)]
    for (o, n), data in zip(pieces, _net.map(lambda p: _get(url, start + p[0], p[1]), pieces)):
        out[o:o + n] = data
    return out


# gguf-py fetches a whole tensor in one request (and one byte too many); route tensor ranges through fetch().
# Whole-file reads (size -1: the shard index JSON) stay on gguf-py's own path.
_whole = gguf.utility.SafetensorRemote.get_data_by_range
gguf.utility.SafetensorRemote.get_data_by_range = classmethod(
    lambda cls, url, start, size=-1: fetch(url, start, size) if size >= 0 else _whole(url, start, size))

# A lazy tensor answers a method that returns a non-tensor by materializing itself. In remote mode that is a full
# download: the pinned Inkling converter asks `element_size()` of every tensor while indexing, which fetched the
# whole Source once before converting it. These methods only depend on dtype and shape: answer them from the meta tensor.
for _m in ("element_size", "dim", "ndimension", "numel", "nelement", "is_floating_point", "size"):
    setattr(conv_base.LazyTorchTensor, _m, (lambda m: lambda self, *a, **k: getattr(self._meta, m)(*a, **k))(_m))
# ---------------------------------------------------------------- ggml quantization

ggml = ctypes.CDLL(os.environ["MF_GGML_LIB"])
ggml.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
ggml.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
ggml.ggml_row_size.restype = ctypes.c_size_t
ggml.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
ggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
ggml.ggml_quantize_init.argtypes = [ctypes.c_int]
ggml.ggml_validate_row_data.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
ggml.ggml_validate_row_data.restype = ctypes.c_bool

MIN_CHUNK = 32 * 512  # llama-quant.cpp's min_chunk_size
SLAB_F32_BYTES = 256 << 20


def load_imatrix(path: str) -> dict:
    """llama-quantize's values: in_sum2 / lround(counts) per matrix, 1 where the count is 0 (common/imatrix-loader.cpp)."""
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


class Quantizer:
    def __init__(self, threads: int):
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="quant")
        self.threads = threads

    def rows(self, qtype: int, f32: np.ndarray, first_row: int, rows_per_expert: int, imat: np.ndarray | None) -> np.ndarray:
        """Quantize rows [first_row, first_row + len(f32)) of a tensor, never crossing an expert boundary."""
        f32 = np.ascontiguousarray(f32, dtype=np.float32)
        nrows, n = f32.shape
        row_size = ggml.ggml_row_size(qtype, n)
        out = np.empty(nrows * row_size, dtype=np.uint8)
        per_chunk = max(1, (n if n >= MIN_CHUNK else n * ((MIN_CHUNK + n - 1) // n)) // n)
        jobs, r = [], 0
        while r < nrows:
            g = first_row + r
            k = min(nrows - r, per_chunk, rows_per_expert - g % rows_per_expert)
            jobs.append((r, k, None if imat is None else imat[g // rows_per_expert]))
            r += k
        src, dst = f32.ctypes.data, out.ctypes.data

        def one(job):
            r0, k, im = job
            p = dst + r0 * row_size
            size = ggml.ggml_quantize_chunk(qtype, src + r0 * n * 4, p, 0, k, n, None if im is None else im.ctypes.data)
            if size != k * row_size or not ggml.ggml_validate_row_data(qtype, p, size):
                raise RuntimeError(f"quantized data validation failed (type {qtype})")
            return size

        assert sum(self.pool.map(one, jobs)) == out.nbytes
        return out


# ---------------------------------------------------------------- source access paths


def _method(func) -> str:
    """Name of the tensor method a lazy node calls (LazyMeta wraps `t.to(...)` etc. in a lambda over `name`)."""
    if func in (torch.Tensor.to, torch.Tensor.float, torch.Tensor.contiguous):
        return func.__name__
    q = getattr(func, "__qualname__", "")
    if q.endswith("__getattr__.<locals>.<lambda>") and func.__code__.co_freevars == ("name",):
        return func.__closure__[0].cell_contents
    if q.endswith("numpy.<locals>.<lambda>"):
        return "numpy"
    if q.endswith("astype.<locals>.<lambda>"):
        return "astype"
    return ""


def upcast_root(t):
    """(RemoteTensor, torch dtype) if t is only an exact float upcast of one stored remote tensor, else None."""
    node, LB = t, gguf.LazyBase
    while isinstance(node, LB):
        if node._data is not None:
            return None
        a = node._args
        if len(a) == 1 and isinstance(a[0], gguf.utility.RemoteTensor):
            dt = node._meta.dtype
            return (a[0], dt) if dt in (torch.bfloat16, torch.float16, torch.float32) else None
        if not a or not isinstance(a[0], LB) or tuple(a[0]._meta.shape) != tuple(node._meta.shape):
            return None
        m, rest = _method(node._func), list(a[1:]) + list(node._kwargs.values())
        ok = ((m in ("numpy", "float", "contiguous") and not rest)
              or (m == "to" and rest == [torch.float32])
              or (m == "astype" and rest and np.dtype(rest[0]) == np.float32 and all(x is False for x in rest[1:])))
        if not ok:
            return None
        node = a[0]
    return None


class QTensor:
    """A tensor the GGUFWriter writes as quantized bytes, produced while it writes."""

    def __init__(self, name, src, qtype: int, imat, q: Quantizer):
        self.name, self.src, self.qtype, self.imat, self.q = name, src, qtype, imat, q
        shape = tuple(src.shape)
        self.n, self.rows_per_expert = shape[-1], shape[-2] if len(shape) >= 2 else 1
        self.nrows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        row = ggml.ggml_row_size(qtype, self.n)
        self.shape, self.dtype, self.nbytes = (*shape[:-1], row), np.dtype(np.uint8), self.nrows * row

    def _write(self, fout, f32, first):
        self.q.rows(self.qtype, f32.reshape(-1, self.n), first, self.rows_per_expert, self.imat).tofile(fout)
        return f32.size // self.n

    def tofile(self, fout):
        src, done = self.src, 0
        if isinstance(src, gguf.LazyChunkedTensor):
            STATS["chunked"] += 1
            for load in src._chunks:
                done += self._write(fout, load(), done)
        elif (root := upcast_root(src)) is not None:
            STATS["streamed"] += 1
            rt, dt = root
            item = torch.empty(0, dtype=dt).element_size()
            slab = max(1, SLAB_F32_BYTES // (self.n * 4))
            for r0 in range(0, self.nrows, slab):
                k = min(slab, self.nrows - r0)
                raw = fetch(rt.url, rt.offset_start + r0 * self.n * item, k * self.n * item)
                f32 = torch.frombuffer(raw, dtype=torch.uint8).view(dt).to(torch.float32).numpy()
                done += self._write(fout, f32, r0)
        else:
            STATS["whole"] += 1
            STATS["whole_names"].append(self.name)
            f32 = np.asarray(gguf.LazyNumpyTensor.to_eager(src) if isinstance(src, gguf.LazyBase) else src, dtype=np.float32).reshape(-1, self.n)
            STATS["whole_max_f32_bytes"] = max(STATS["whole_max_f32_bytes"], f32.nbytes)
            slab = max(1, SLAB_F32_BYTES // (self.n * 4))
            for r0 in range(0, self.nrows, slab):
                done += self._write(fout, f32[r0:r0 + slab], r0)
            del f32
        assert done == self.nrows, (self.name, done, self.nrows)
        self.src = None


# ---------------------------------------------------------------- Inkling packed experts by byte range


class _ExpertSlab:
    """Stands in for a packed remote expert tensor inside llama.cpp's _dequant_expert_chunk: [expert, rows]
    reads only that expert, once, from a one-expert cache."""
    cache: dict = {}

    def __init__(self, rt):
        self.rt = rt

    def view(self, dtype):
        return self

    def __getitem__(self, idx):
        expert, rows = idx
        key = (self.rt.url, self.rt.offset_start)
        hit = self.cache.get(key)
        if hit is None or hit[0] != expert:
            per = int(np.prod(self.rt.shape[1:]))  # every packed weight/scale dtype here is one byte
            raw = fetch(self.rt.url, self.rt.offset_start + expert * per, per)
            hit = self.cache[key] = (expert, torch.frombuffer(raw, dtype=torch.uint8).reshape(self.rt.shape[1:]))
        return hit[1][rows]


def patch_inkling():
    from conversion import inkling2
    orig = inkling2._dequant_expert_chunk
    small: dict = {}

    def remote_of(gen):
        return gen.__defaults__[0]  # index_tensors binds the RemoteTensor as the generator's default arg

    def eager_small(gen):
        rt = remote_of(gen)
        key = rt.url + str(rt.offset_start)
        if key not in small:
            small[key] = conv_base.LazyTorchTensor.to_eager(gen())
        return small[key]

    def chunk(weight, scale, amax, denominator, name, expert, rows):
        w, s = _ExpertSlab(remote_of(weight)), _ExpertSlab(remote_of(scale))
        a = (lambda: eager_small(amax)) if amax is not None else None
        d = (lambda: eager_small(denominator)) if denominator is not None else None
        return orig(lambda: w, lambda: s, a, d, name, expert, rows)

    inkling2._dequant_expert_chunk = chunk


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out", type=Path)
    ap.add_argument("--types", required=True, help="JSON {tensor name: ggml type name} (the target mix)")
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--ftype", required=True, help="LlamaFileType name recorded in general.file_type")
    ap.add_argument("--default-type", default="Q8_0", help="type for 2D+ tensors the mix doesn't name")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("work"))
    ap.add_argument("--plan-only", action="store_true",
                    help="map every tensor and size the output, read no tensor data, write OUT.plan.json and stop")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    t0 = time.time()

    from huggingface_hub import snapshot_download
    cfg_dir = Path(snapshot_download(repo_id=args.repo, allow_patterns=["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model", "*.jinja"],
                                     local_dir=args.workdir / "config"))
    STATS["config_bytes"] = sum(f.stat().st_size for f in cfg_dir.rglob("*") if f.is_file())

    types = {k: QT[v.upper()] for k, v in json.loads(Path(args.types).read_text()).items()}
    imatrix = load_imatrix(args.imatrix)
    default = QT[args.default_type.upper()]
    q = Quantizer(args.threads)
    patch_inkling()

    with torch.inference_mode():
        hparams = conv_base.ModelBase.load_hparams(cfg_dir, False)
        cls = get_model_class(get_model_architecture(hparams, ModelType.TEXT), mmproj=False)
        model = cls(cfg_dir, gguf.LlamaFileType.ALL_F32, args.out, remote_hf_model_id=args.repo)
        add = model.gguf_writer.add_tensor
        used = {}

        def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
            assert raw_dtype in (None, QT.F32), (name, raw_dtype)
            nd = len(tensor.shape)
            qt = types.get(name)
            if qt is None:
                qt = QT.F32 if nd <= 1 else default
                STATS["defaulted"].append(name)
            used[name] = qt.name
            STATS["types"][qt.name] = STATS["types"].get(qt.name, 0) + 1
            if qt == QT.F32:
                STATS["kept"] += 1
                return add(name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype)
            im = imatrix.get(name)
            n_mat = int(np.prod(tensor.shape[:-2])) if nd >= 3 else 1
            if im is not None and im.size != tensor.shape[-1] * n_mat:
                if not name.startswith("token_embd"):
                    raise ValueError(f"imatrix size {im.size} is different from tensor size {tensor.shape[-1] * n_mat} for {name}")
                im = None
            if im is None:
                STATS["missing_imatrix"].append(name)
                if ggml.ggml_quantize_requires_imatrix(int(qt)) and not args.plan_only:
                    raise ValueError(f"Missing importance matrix for tensor {name} in a very low-bit quantization")
            ggml.ggml_quantize_init(int(qt))
            return add(name, QTensor(name, tensor, int(qt), im, q), raw_dtype=qt)

        model.gguf_writer.add_tensor = add_tensor
        model.prepare_tensors()
        planned = sum(ti.nbytes for shard in model.gguf_writer.tensors for ti in shard.values())
        STATS["planned_tensor_bytes"] = planned
        if args.plan_only:
            plan = {"tensors": used, "planned_tensor_bytes": planned, "defaulted": STATS["defaulted"],
                    "missing_imatrix": STATS["missing_imatrix"], "types": STATS["types"]}
            (args.out.parent / (args.out.stem + ".plan.json")).write_text(json.dumps(plan, indent=0))
            log.info("plan: %s", json.dumps({k: v for k, v in plan.items() if k != "tensors"}))
            return
        free = os.statvfs(args.out.parent).f_bavail * os.statvfs(args.out.parent).f_frsize
        if planned + (1 << 30) > free:
            raise SystemExit(f"output needs {planned / 2**30:.2f} GiB + 1 GiB margin, {free / 2**30:.2f} GiB free")
        model.ftype = gguf.LlamaFileType[args.ftype]
        model.prepare_metadata(vocab_only=False)
        model.gguf_writer.write_header_to_file(path=args.out)
        model.gguf_writer.write_kv_data_to_file()
        STATS["t_header_s"] = round(time.time() - t0, 1)
        model.gguf_writer.write_tensors_to_file(progress=False)
        model.gguf_writer.close()

    STATS["wall_s"] = round(time.time() - t0, 1)
    STATS["out_bytes"] = args.out.stat().st_size
    (args.out.parent / (args.out.stem + ".types.json")).write_text(json.dumps(used, indent=0))
    if args.stats:
        args.stats.write_text(json.dumps(STATS, indent=2))
    log.info("done: %s", json.dumps({k: v for k, v in STATS.items() if k not in ("whole_names", "defaulted", "missing_imatrix")}))


if __name__ == "__main__":
    main()
