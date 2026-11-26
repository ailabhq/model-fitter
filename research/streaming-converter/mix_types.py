#!/usr/bin/env python3
"""Per-tensor types of a published GGUF (local path or HF URL), from its header only.

Usage: mix_types.py SRC OUT.json    (writes {tensor name: ggml type name}; prints totals)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ["MF_LLAMA"]) / "gguf-py"))
import gguf  # noqa: E402
from gguf_header import read_header  # noqa: E402

h = read_header(sys.argv[1], os.environ.get("HF_TOKEN"))
types = {t["name"]: gguf.GGMLQuantizationType(t["type"]).name for t in h["tensors"]}
Path(sys.argv[2]).write_text(json.dumps(types, indent=0))
print(json.dumps({"tensors": len(types), "header_bytes": h["header_bytes"], "types": Counter(types.values()),
                  "arch": h["kv"].get("general.architecture"), "file_type": h["kv"].get("general.file_type"),
                  "imatrix": {k: v for k, v in h["kv"].items() if k.startswith("quantize.")}}, default=str))
