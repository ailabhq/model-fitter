"""Unsloth's Inkling-2-Mini UD-IQ3_XXS header: metadata, NVFP4 tensors and their .scale tensors (header bytes only)."""
import collections
import sys

from gguf_header import read_header

REV = "8078d149286eec7193795413b9e0643da6d11fa1"
FILE = sys.argv[1] if len(sys.argv) > 1 else "Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file"
URL = f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{REV}/{FILE}"
TYPES = {0: "F32", 1: "F16", 30: "BF16", 40: "NVFP4"}

h = read_header(URL)
print(f"{FILE}: GGUF v{h['version']}, {len(h['tensors'])} tensors, header {h['header_bytes']} bytes")
for k, v in h["kv"].items():
    if k.startswith("tokenizer.ggml.") and isinstance(v, list):
        continue
    if isinstance(v, list) and len(v) > 8:
        v = f"[{len(v)} items] {v[:8]}..."
    print(f"  {k} = {v}")
by = collections.Counter()
for t in h["tensors"]:
    by[TYPES.get(t["type"], t["type"])] += 1
print("tensor types:", dict(by))
for t in h["tensors"]:
    if t["type"] == 40 or t["name"].endswith((".scale", ".input_scale")):
        print(f"  {t['name']:40s} {TYPES.get(t['type'], t['type'])!s:6s} {t['shape']} @ {t['offset']}")
