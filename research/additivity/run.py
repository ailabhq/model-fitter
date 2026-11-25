#!/usr/bin/env python3
"""Candidates x synthetic desktop load, through bench.py's gate and Monitor.

Usage: run.py REP LOAD_GIB[,LOAD_GIB...] CONFIG[,CONFIG...]

For each load (in the order given), for each config: start a holder process that keeps LOAD_GIB of
anonymous, incompressible, touched memory, run the config with bench.run (cold load, quiet start gate,
Monitor, kill-switch), then stop the holder. A fresh holder per run keeps the load the same for every
candidate: an idle holder that a previous run pushed to swap would be a lighter load for the next one.
Results go to results/rep<REP>/load<G>g/.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "baseline" / "harness"))
import bench  # noqa: E402

GiB = 1 << 30

# Every 16 KiB page gets distinct random bytes, so the compressor can't shrink the holder: under
# pressure it is swapped out, like a real app's data.
HOLDER = r"""
import os, signal, sys
n, step = int(float(sys.argv[1]) * (1 << 30)), 1 << 20
blk = os.urandom(step)
buf = bytearray(n)
mv = memoryview(buf)
for off in range(0, n, step):
    mv[off:off + step] = blk
print("ready", flush=True)
signal.pause()
"""

# The desktop drifts by GiBs over hours, so the analysis measures the load as committed memory per run; the gate's
# floor only keeps a run from starting on a machine that is already full. Results record the value used.
GATE_AVAILABLE_GIB = float(os.environ.get("MF_GATE_AVAILABLE_GIB", bench.QUIET["available_gib_min"]))
_is_runtime = bench.is_runtime


def is_runtime(argv: list) -> bool:
    """bench.py's runtimes plus the Quality-loss scorers, which another session may run on this Mac."""
    return _is_runtime(argv) or any(os.path.basename(a) == "llama-mf-score" or a.endswith(("mlx_score.py", "score_fit.py"))
                                    for a in argv[:3])


bench.is_runtime = is_runtime


class Holder:
    def __init__(self, gib: int):
        self.gib, self.proc, self.rss = gib, None, []
        self._stop = threading.Event()

    def __enter__(self):
        # The gate asks the same of the desktop as without a holder: the holder's own memory is the load.
        bench.QUIET["available_gib_min"] = GATE_AVAILABLE_GIB - self.gib
        if self.gib:
            self.proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(self.gib)], stdout=subprocess.PIPE, text=True)
            if self.proc.stdout.readline().strip() != "ready":
                raise RuntimeError("holder failed to start")
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def _sample(self):
        p = psutil.Process(self.proc.pid)
        while not self._stop.is_set():
            try:
                self.rss.append(p.memory_info().rss)
            except psutil.Error:
                return
            self._stop.wait(2)

    def __exit__(self, *a):
        self._stop.set()
        if self.proc:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
        bench.QUIET["available_gib_min"] = GATE_AVAILABLE_GIB

    def summary(self) -> dict:
        if not self.gib:
            return {"gib": 0}
        return {"gib": self.gib, "rss_first_gib": round(self.rss[0] / GiB, 2), "rss_min_gib": round(min(self.rss) / GiB, 2),
                "rss_last_gib": round(self.rss[-1] / GiB, 2)}


def one(name: str, gib: float, out: Path) -> dict:
    bench.OUT = out
    for attempt in range(bench.QUIET["retries"] + 1):
        with Holder(gib) as h:
            r = bench.run(name)
        if not r.get("contaminated"):
            break
        print(json.dumps({"config": name, "load_gib": gib, "contaminated": r["contaminated"]}), flush=True)
    else:
        return r
    r["holder"] = h.summary()
    (out / f"{name}.json").write_text(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    rep, loads, names = sys.argv[1], [float(g) for g in sys.argv[2].split(",")], sys.argv[3].split(",")
    for g in loads:
        for n in names:
            r = one(n, g, HERE / "results" / f"rep{rep}" / f"load{g:g}g")
            m = r["memory"]
            print(json.dumps({"rep": rep, "load_gib": g, "config": n, "gen_tps_at_8k": r.get("gen_tps_at_8k"),
                              "peak_wired_gib": m["peak_wired_gib"], "swapouts_gib": m["swapouts_gib"],
                              "critical_share": m["pressure"]["critical_share"], "responsive": r["responsive"],
                              "aborted": r.get("aborted"), "error": r.get("error"), "holder": r.get("holder"),
                              "contaminated": r.get("contaminated")}), flush=True)
