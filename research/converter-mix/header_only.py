"""Run stream_convert.py's exact header path, but write no tensor data.

Writes OUT (header + KV + tensor infos + alignment padding up to the data section) and OUT.sparse.gguf
(the same bytes, truncated up to the full file size, so the data region is a sparse hole).
Usage: same args as stream_convert.py (without --plan-only).
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stream_convert as sc  # noqa: E402

GW = sc.gguf.GGUFWriter
info = {}


def header_only(self, *, progress=False):
    self.write_ti_data_to_file()
    for fout in self.fout:
        self.write_padding(fout, fout.tell())
    self.flush()
    info["data_offset"] = self.fout[0].tell()
    info["data_bytes"] = sum(GW.ggml_pad(ti.nbytes, self.data_alignment) for t in self.tensors for ti in t.values())
    info["n_tensors"] = sum(len(t) for t in self.tensors)
    self.state = sc.gguf.gguf_writer.WriterState.WEIGHTS


GW.write_tensors_to_file = header_only
sc.main()
out = Path(sys.argv[2])
full = info["data_offset"] + info["data_bytes"]
sparse = out.with_suffix(".sparse.gguf")
shutil.copyfile(out, sparse)
os.truncate(sparse, full)
info.update(header_file_bytes=out.stat().st_size, full_bytes=full)
print("HEADER_ONLY " + json.dumps(info))
