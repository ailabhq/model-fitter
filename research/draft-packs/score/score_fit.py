#!/usr/bin/env python3
"""Score a Fit on this Mac against a Reference pack: prefill only, in the runtime the Fit runs on.

Usage:
  score_fit.py llamacpp LABEL PACK_DIR GGUF [extra llama.cpp args ...]
  score_fit.py mlx      LABEL PACK_DIR MODEL_DIR PREFILL_STEP

Waits for a quiet machine (bench.py's start gate), runs the scorer under bench.py's Monitor (Responsive
probes, kill-switch, contamination check; a contaminated run is retried), then writes
$MF_OUT/LABEL.json: the Quality loss report (whole corpus, per domain, 8K slice, provisional band), the
tokenizer check, scoring wall time and memory. Results carry no paths, usernames or hostnames.
"""
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pack"))
sys.path.insert(0, str(HERE.parent.parent / "baseline" / "harness"))
import bench  # noqa: E402
from metrics import item_rows, kld_topk, load_pack, report, top1_agree  # noqa: E402

CTX = 8704
LLAMA_ARGS = ["-c", str(CTX), "-np", "1", "-fa", "on", "-b", "512"]


def write_in(pack: dict, path: Path):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(pack["items"])))
        for i, it in enumerate(pack["items"]):
            ids, rows = item_rows(pack, i)
            k = pack["topk_ids"].shape[1]
            f.write(struct.pack("<ii", len(ids), k))
            f.write(ids.astype(np.int32).tobytes())
            f.write(pack["topk_ids"][rows].astype(np.int32).tobytes())
            b = it["text"].encode()
            f.write(struct.pack("<i", len(b)) + b + b"\0" * ((4 - len(b) % 4) % 4))


def read_out(pack: dict, path: Path) -> dict:
    k = pack["topk_ids"].shape[1]
    raw = path.read_bytes()
    off, out = 0, {}
    for i, it in enumerate(pack["items"]):
        n = pack["manifest"]["items"][i]["n_tokens"] - 1
        lq = np.frombuffer(raw, np.float32, n * k, off).reshape(n, k)
        off += 4 * n * k
        am = np.frombuffer(raw, np.int32, n, off)
        off += 4 * n
        (nt,) = struct.unpack_from("<i", raw, off)
        off += 4
        toks = np.frombuffer(raw, np.int32, nt, off)
        off += 4 * nt
        out[it["id"]] = (lq, am, toks)
    return out


def cmd_for(kind: str, args: list, in_p: Path, out_p: Path) -> list:
    if kind == "llamacpp":
        gguf, extra = args[0], args[1:]
        return [str(bench.LLAMA / "llama-mf-score"), "--mf-in", str(in_p), "--mf-out", str(out_p), "-m", gguf, *LLAMA_ARGS, *extra]
    model_dir, step = args
    return [str(bench.OMLX_BIN / "python"), str(HERE / "mlx_score.py"), model_dir, str(in_p), str(out_p), step]


def run_once(kind: str, label: str, pack: dict, args: list, out_dir: Path) -> dict:
    work = Path(os.environ.get("MF_WORK", bench.ROOT / "work" / "score"))
    work.mkdir(parents=True, exist_ok=True)
    in_p, out_p, log_p = work / f"{label}.in", work / f"{label}.out", out_dir / f"{label}.log"
    write_in(pack, in_p)
    cmd = cmd_for(kind, args, in_p, out_p)
    res = {"config": label, "runtime": kind, "pack": {k: pack["manifest"][k] for k in ("source", "prompt_set", "k", "positions")},
           "args": [bench.clean(a) for a in cmd[5:]] if kind == "llamacpp" else ["prefill_step", cmd[-1]],
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "quiet_criteria": bench.QUIET}
    if kind == "llamacpp":
        res["runtime_version"] = subprocess.run([str(bench.LLAMA / "llama-server"), "--version"], capture_output=True, text=True).stderr.strip().splitlines()[0]
    res["start_gate"] = bench.start_gate()
    baseline = bench.snapshot()
    lines = []
    with open(log_p, "w") as log, bench.Monitor() as mon:
        mon.phase = "score"
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
        mon.pids_fn = lambda: [p.pid]
        for line in p.stdout:
            line = line.strip()
            if line.startswith("{"):
                lines.append(json.loads(line))
                print(line, flush=True)
        p.wait()
        res["wall_s"] = round(time.perf_counter() - t0, 1)
        mon.phase = "idle"
    res["returncode"] = p.returncode
    res["memory"] = mon.summary(baseline)
    res["contaminated"] = mon.contaminated
    res["aborted"] = mon.tripped
    res["responsive"] = bench.responsive(res["memory"], mon.tripped)
    res["worker"] = {"ready": next((l for l in lines if l.get("ready")), None), "done": next((l for l in lines if l.get("done")), None),
                     "items": [l for l in lines if "item" in l]}
    log_p.write_text(bench.clean(log_p.read_text(errors="ignore")))
    if p.returncode != 0 or mon.contaminated or mon.tripped:
        return res

    out = read_out(pack, out_p)
    (out_dir / f"{label}.scores.bin").write_bytes(out_p.read_bytes())  # raw per-position output, for re-analysis
    per, tokcheck = {}, {}
    for i, it in enumerate(pack["items"]):
        _, rows = item_rows(pack, i)
        lq, am, toks = out[it["id"]]
        per[it["id"]] = (kld_topk(pack["topk_lp"][rows], lq), top1_agree(pack["topk_ids"][rows], pack["topk_lp"][rows], am))
        tokcheck[it["id"]] = bool(np.array_equal(toks, np.array(it["text_ids"], np.int32)))
    # Row j predicts token j + 1, so rows from n_prompt - 1 on predict the Source's own response tokens.
    cut = {m["id"]: m["n_prompt"] - 1 for m in pack["manifest"]["items"]}
    for key, part in (("quality", lambda v, c: v), ("quality_response_only", lambda v, c: v[c:]), ("quality_prompt_only", lambda v, c: v[:c])):
        sub = {k: (part(v[0], cut[k]), part(v[1], cut[k])) for k, v in per.items()}
        res[key] = report(pack, sub)
        res[key]["items"] = {k: {"mean_kld": float(v[0].mean()), "top1": float(v[1].mean())} for k, v in sub.items()}
    res["tokenizer_check"] = {"pass": all(tokcheck.values()), "mismatched_items": [k for k, v in tokcheck.items() if not v]}
    return res


def main():
    kind, label, pack_dir, *args = sys.argv[1:]
    pack = load_pack(Path(pack_dir))
    out_dir = Path(os.environ.get("MF_OUT", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(bench.QUIET["retries"] + 1):
        res = run_once(kind, label, pack, args, out_dir)
        if not res["contaminated"]:
            break
        cdir = out_dir / "contaminated"
        cdir.mkdir(exist_ok=True)
        (cdir / f"{label}.{int(time.time())}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"contaminated": res["contaminated"], "retry": attempt < bench.QUIET["retries"]}), flush=True)
    res["machine"] = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                      "mem_gib": round(bench.psutil.virtual_memory().total / bench.GiB)}
    (out_dir / f"{label}.json").write_text(json.dumps(res, indent=1))
    q, r = res.get("quality") or {}, res.get("quality_response_only") or {}
    print(json.dumps({"config": label, "band": q.get("band"), "whole": q.get("whole"), "8k": (q.get("domains") or {}).get("long_8k"),
                      "band_response_only": r.get("band"), "response_only": r.get("whole"),
                      "8k_response_only": (r.get("domains") or {}).get("long_8k"),
                      "tokenizer_check": res.get("tokenizer_check"), "wall_s": res.get("wall_s"), "responsive": res["responsive"],
                      "aborted": res["aborted"], "returncode": res["returncode"]}), flush=True)


if __name__ == "__main__":
    main()
