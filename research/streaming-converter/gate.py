#!/usr/bin/env python3
"""Quiet gate: block until the Mac is quiet, then exit 0 (prints what it waited for as JSON).

Uses bench.py's start_gate (normal memory pressure, CPU <= 20 %, no other runtime or download, held
for 60 s) with the 6 GiB available-memory floor the last additivity runs used: the 8 GiB floor stalled
the research gate for hours once the desktop grew. Other sessions' scorers and conversions count as
runtimes. Usage: gate.py [--timeout S]
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

bench.QUIET["available_gib_min"] = float(os.environ.get("MF_GATE_AVAILABLE_GIB", 6))
_is_runtime = bench.is_runtime


def is_runtime(argv: list) -> bool:
    names = [os.path.basename(a) for a in argv[:3]]
    return (_is_runtime(argv) or "llama-mf-score" in names or "llama-quantize" in names
            or any(n in ("convert_hf_to_gguf.py", "stream_convert.py", "mlx_score.py", "score_fit.py") for n in names))


bench.is_runtime = is_runtime

if __name__ == "__main__":
    timeout = int(sys.argv[2]) if sys.argv[1:2] == ["--timeout"] else 4 * 3600
    print(json.dumps({"gate": bench.start_gate(timeout), "criteria": bench.QUIET}), flush=True)
