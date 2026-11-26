#!/usr/bin/env python3
"""Small-model check of stream_convert.py against llama.cpp's own pipeline, before the 27B runs.

Usage: smoke.py HF_REPO WORKDIR
Downloads the (small) Source, converts it with convert_hf_to_gguf.py --outtype f32, gives every 2D/3D tensor a type
from a cycle of k- and i-quants and a synthetic imatrix (random per-expert importances), then runs
stream_convert.py (byte ranges from Hugging Face) and llama-quantize with the same types and imatrix, and compares.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LLAMA = Path(os.environ["MF_LLAMA"])
sys.path.insert(0, str(LLAMA / "gguf-py"))
import gguf  # noqa: E402

CYCLE = ["IQ3_XXS", "Q4_K", "IQ2_S", "Q6_K", "IQ4_XS", "Q8_0", "IQ3_S", "Q3_K", "IQ2_XXS", "Q5_K", "BF16", "F16"]
PY = sys.executable


def run(*cmd, **kw):
    print("+", " ".join(map(str, cmd))[:200], flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def main():
    repo, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    src, f32 = Path(os.environ.get("SMOKE_SRC", work / "src")), work / "ref-F32.gguf"
    if "SMOKE_SRC" not in os.environ:
        run("hf", "download", repo, "--local-dir", src, "--quiet", stdout=subprocess.DEVNULL)
    run(PY, LLAMA / "convert_hf_to_gguf.py", src, "--outtype", "f32", "--outfile", f32, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    r = gguf.GGUFReader(f32)
    types, rng, i = {}, np.random.default_rng(0), 0
    w = gguf.GGUFWriter(work / "imatrix.gguf", arch="imatrix")
    w.add_type("imatrix")
    w.add_array("imatrix.datasets", ["synthetic"])
    w.add_uint32("imatrix.chunk_count", 1)
    w.add_uint32("imatrix.chunk_size", 512)
    for t in r.tensors:
        ne = [int(x) for x in t.shape]
        if len(ne) < 2 or t.name.endswith("_norm.weight") or "ffn_gate_inp" in t.name:  # llama-quantize never quantizes these
            types[t.name] = "F32"
            continue
        qt = CYCLE[i % len(CYCLE)]
        i += 1
        if ne[0] % gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[qt]][0]:
            qt = "Q8_0" if ne[0] % 32 == 0 else "F16"
        types[t.name] = qt
        n_mat = int(np.prod(ne[2:])) if len(ne) > 2 else 1
        w.add_tensor(t.name + ".in_sum2", (rng.random((n_mat, ne[0]), dtype=np.float32) * 100 + 1e-3))
        w.add_tensor(t.name + ".counts", np.full((n_mat, 1), 37.0, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    (work / "types.json").write_text(json.dumps(types))
    (work / "tensor-types.txt").write_text("".join(f"^{k.replace('.', chr(92) + '.')}$={v.lower()}\n" for k, v in types.items()))

    run(os.environ.get("MF_QUANTIZE", LLAMA / "build-quant/bin/llama-quantize"), "--imatrix", work / "imatrix.gguf", "--tensor-type-file", work / "tensor-types.txt",
        f32, work / "ref.gguf", "IQ3_XXS", "6", stdout=open(work / "quantize.log", "w"), stderr=subprocess.STDOUT)
    run(PY, HERE / "stream_convert.py", repo, work / "ours.gguf", "--types", work / "types.json", "--imatrix", work / "imatrix.gguf",
        "--ftype", "MOSTLY_IQ3_XXS", "--stats", work / "ours.stats.json", "--workdir", work / "stream",
        stdout=open(work / "stream.log", "w"), stderr=subprocess.STDOUT)
    run(PY, HERE / "compare.py", work / "ours.gguf", work / "ref.gguf", work / "compare.json")


if __name__ == "__main__":
    main()
