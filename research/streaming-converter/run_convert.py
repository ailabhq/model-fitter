#!/usr/bin/env python3
"""Run one conversion under bench.py's quiet gate, Monitor and kill-switch, and measure disk, memory and time.

Usage: run_convert.py LABEL PLANNED_S OUTDIR -- COMMAND...

Waits for a quiet Mac (gate.py's criteria), then runs COMMAND with the Monitor sampling memory and the Responsive
probes, and the kill-switch applying the Responsive rule (critical-pressure budget scaled to PLANNED_S). A disk
sampler records the bytes under OUTDIR and the volume's free space every second. A run stopped because another
workload shared the machine is retried; a Stall (one >= 500 ms pause, pressure never critical) is run once more,
as decided in "Does one >= 500 ms pause with memory pressure never critical fail Responsive, or is the run retried?".
Results go to results/LABEL[.N].json, cleaned of paths, usernames and hostnames.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate  # noqa: E402,F401  (sets the gate floor and runtime list on bench)
import bench  # noqa: E402


# A conversion runs for hours beside the user's normal desktop: that coexistence is what Responsive judges, so
# other processes' CPU is recorded (foreign_cores) but never stops the run, unlike bench.py's 5-minute benchmarks.
bench.QUIET["busy_checks"] = 10 ** 9

GiB = 1 << 30


def tree_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_blocks * 512
            except OSError:
                pass
    return total


class Disk:
    def __init__(self, outdir: Path):
        self.outdir, self.samples, self._stop = outdir, [], threading.Event()

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            self.samples.append((round(time.time() - t0, 1), tree_bytes(self.outdir), shutil.disk_usage(self.outdir).free))
            self._stop.wait(1.0)

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join(5)

    def summary(self):
        return {"peak_outdir_gib": round(max(s[1] for s in self.samples) / GiB, 3),
                "free_before_gib": round(self.samples[0][2] / GiB, 2),
                "min_free_gib": round(min(s[2] for s in self.samples) / GiB, 2)}


def once(label: str, planned_s: int, outdir: Path, cmd: list) -> dict:
    bench.PLANNED_S = planned_s
    res = {"label": label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "planned_s": planned_s,
           "responsive_criteria": bench.RESPONSIVE, "quiet_criteria": bench.QUIET}
    res["start_gate"] = bench.start_gate()
    outdir.mkdir(parents=True, exist_ok=True)
    baseline = bench.snapshot()
    log = open(outdir / f"{label}.log", "w")
    t0 = time.time()
    with Disk(outdir) as disk, bench.Monitor() as mon:
        mon.phase = "convert"
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        mon.pids_fn = lambda: [proc.pid] if proc.poll() is None else []
        res["exit_code"] = proc.wait()
        mon.phase = "idle"
        time.sleep(2)
    res["wall_s"] = round(time.time() - t0, 1)
    log.close()
    res["disk"] = disk.summary()
    res["memory"] = mon.summary(baseline)
    res["contaminated"], res["aborted"] = mon.contaminated, mon.tripped
    res["responsive"] = bench.responsive(res["memory"], mon.tripped)
    res["memory_failure"] = res["memory"]["max_pressure_level"] >= 4
    res["machine"] = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                      "mem_gib": round(bench.psutil.virtual_memory().total / GiB)}
    return res


def main():
    i = sys.argv.index("--")
    label, planned_s, outdir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]).expanduser()
    cmd = sys.argv[i + 1:]
    (HERE / "results").mkdir(exist_ok=True)
    runs, stalls, contaminated = [], 0, 0
    while True:
        r = once(label, planned_s, outdir, cmd)
        r["run"] = len(runs) + 1
        runs.append(r)
        name = f"{label}.{r['run']}" if len(runs) > 1 or r["contaminated"] or r["aborted"] else label
        (HERE / "results" / f"{name}.json").write_text(bench.clean(json.dumps(r, indent=2)))
        print(json.dumps({k: r.get(k) for k in ("label", "run", "exit_code", "wall_s", "responsive", "aborted", "contaminated", "disk")}), flush=True)
        if r["contaminated"] and contaminated < bench.QUIET["retries"]:
            contaminated += 1
            continue
        if r["aborted"] and not r["memory_failure"] and not r["contaminated"] and stalls == 0:
            stalls += 1  # a Stall: run the candidate once more
            continue
        break
    sys.exit(0 if runs[-1]["exit_code"] == 0 and runs[-1]["responsive"] else 1)


if __name__ == "__main__":
    main()
