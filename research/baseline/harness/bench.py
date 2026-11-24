#!/usr/bin/env python3
"""Baseline benchmark: off-the-shelf quantized builds on one Apple Silicon Mac.

Usage: bench.py CONFIG [CONFIG ...]     (see CONFIGS below; `bench.py list` prints them)

Environment:
  MF_ROOT     scratch dir holding models/, llama.cpp/, omlx-venv/ (never inside the repo)
  MF_OUT      where result JSON goes (default: ./results)
  MF_TEXT     prompt corpus dir (default: $MF_ROOT/llama.cpp/docs; the Baseline used llama.cpp b6d5ce4)

For each config: start the runtime, time the load, send an ~8000-token prompt and
generate 256 tokens (twice, the second time from the prompt cache where the runtime
has one), then a short prompt for a shallow-context reference. A sampler records system
memory and the Responsive probes (window server, sleep timer, process launch, memory
pressure) the whole time, and the kill-switch applies the Responsive rule live.
Results carry no paths, usernames or hostnames.
"""
import ctypes
import json
import os
import platform
import re
import signal
import socket
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(os.environ.get("MF_ROOT", Path.home() / "mf-bench"))
OUT = Path(os.environ.get("MF_OUT", "results"))
MODELS = ROOT / "models"
LLAMA = ROOT / "llama.cpp" / "build" / "bin"
OMLX_BIN = ROOT / "omlx-venv" / "bin"
TEXT_DIR = Path(os.environ.get("MF_TEXT", ROOT / "llama.cpp" / "docs"))
PROMPT_TOKENS = 8000
GEN_TOKENS = 256
CTX = 8704
PORT = 8090

QWEN_GGUF = "unsloth/Qwen3.8-27B-GGUF"
INK_GGUF = "unsloth/Inkling-2-Mini-GGUF"


def gguf(repo, file, local):
    return {"repo": repo, "file": file, "local": local}


CONFIGS = {
    # Qwen3.8-27B, dense
    "qwen-iq3xxs-llamacpp-fit": dict(runtime="llamacpp", model=gguf(QWEN_GGUF, "Qwen3.8-27B-UD-IQ3_XXS.gguf", "qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf"), args=[]),
    "qwen-iq3xxs-llamacpp-allgpu": dict(runtime="llamacpp", model=gguf(QWEN_GGUF, "Qwen3.8-27B-UD-IQ3_XXS.gguf", "qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf"), args=["--fit", "off", "-ngl", "999"]),
    "qwen-iq3xxs-ollama": dict(runtime="ollama", model=gguf(QWEN_GGUF, "Qwen3.8-27B-UD-IQ3_XXS.gguf", "qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf"), tag="mf-qwen38-27b-iq3xxs"),
    "qwen-q3kxl-llamacpp-fit": dict(runtime="llamacpp", model=gguf(QWEN_GGUF, "Qwen3.8-27B-UD-Q3_K_XL.gguf", "qwen/Qwen3.8-27B-UD-Q3_K_XL.gguf"), args=[]),
    "qwen-q3kxl-llamacpp-allgpu": dict(runtime="llamacpp", model=gguf(QWEN_GGUF, "Qwen3.8-27B-UD-Q3_K_XL.gguf", "qwen/Qwen3.8-27B-UD-Q3_K_XL.gguf"), args=["--fit", "off", "-ngl", "999"]),
    # Inkling 2 Mini, looped MoE (mainline-layout GGUFs)
    "ink-iq3xxs-llamacpp-fit": dict(runtime="llamacpp", model=gguf(INK_GGUF, "Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file", "ink/Inkling-2-Mini-UD-IQ3_XXS.gguf"), args=[]),
    "ink-iq3xxs-llamacpp-allgpu": dict(runtime="llamacpp", model=gguf(INK_GGUF, "Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file", "ink/Inkling-2-Mini-UD-IQ3_XXS.gguf"), args=["--fit", "off", "-ngl", "999"]),
    "ink-q3kxl-llamacpp-fit": dict(runtime="llamacpp", model=gguf(INK_GGUF, "Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file", "ink/Inkling-2-Mini-UD-Q3_K_XL.gguf"), args=[]),
    "ink-q3kxl-llamacpp-allgpu": dict(runtime="llamacpp", model=gguf(INK_GGUF, "Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file", "ink/Inkling-2-Mini-UD-Q3_K_XL.gguf"), args=["--fit", "off", "-ngl", "999"]),
    "ink-iq4xs-llamacpp-fit": dict(runtime="llamacpp", model=gguf(INK_GGUF, "Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file", "ink/Inkling-2-Mini-UD-IQ4_XS.gguf"), args=[]),
    "ink-iq4xs-llamacpp-allgpu": dict(runtime="llamacpp", model=gguf(INK_GGUF, "Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file", "ink/Inkling-2-Mini-UD-IQ4_XS.gguf"), args=["--fit", "off", "-ngl", "999"]),
    # Inkling 2 Mini, oMLX mixed-precision MLX checkpoints
    "ink-oq3e-omlx": dict(runtime="omlx", model={"repo": "mlx-community/Inkling-2-Mini-oQ3e-mtp", "local": "mlx/Inkling-2-Mini-oQ3e-mtp"}),
    "ink-oq25e-omlx": dict(runtime="omlx", model={"repo": "mlx-community/Inkling-2-Mini-oQ2.5e-mtp", "local": "mlx/Inkling-2-Mini-oQ2.5e-mtp"}),
    "ink-oq25e-omlx-guard-aggressive": dict(runtime="omlx", model={"repo": "mlx-community/Inkling-2-Mini-oQ2.5e-mtp", "local": "mlx/Inkling-2-Mini-oQ2.5e-mtp"}, guard="aggressive"),
    "ink-oq25e-omlx-guard-off": dict(runtime="omlx", model={"repo": "mlx-community/Inkling-2-Mini-oQ2.5e-mtp", "local": "mlx/Inkling-2-Mini-oQ2.5e-mtp"}, guard="off"),
    "ink-oq3e-omlx-guard-aggressive": dict(runtime="omlx", model={"repo": "mlx-community/Inkling-2-Mini-oQ3e-mtp", "local": "mlx/Inkling-2-Mini-oQ3e-mtp"}, guard="aggressive"),
    "ink-oq3e-omlx-guard-off": dict(runtime="omlx", model={"repo": "mlx-community/Inkling-2-Mini-oQ3e-mtp", "local": "mlx/Inkling-2-Mini-oQ3e-mtp"}, guard="off"),
}

# Inkling oQ2.5e at 8K on MLX ("Can MLX hold Inkling oQ2.5e at 8K context ...", map ticket 16).
INK25 = {"repo": "mlx-community/Inkling-2-Mini-oQ2.5e-mtp", "revision": "d4a4f3f555fdf6f06102de1865d07e8ac5effe6d",
         "local": "mlx/Inkling-2-Mini-oQ2.5e-mtp"}


def direct(step: int, kv_bits=None, cache_mb=None) -> tuple:
    """mlx-lm loop: prefill step, KV bits (quantized from the first token), MLX buffer-cache limit."""
    name = f"ink-oq25e-mlx-p{step}" + (f"-kv{kv_bits}" if kv_bits else "") + (f"-cache{cache_mb}" if cache_mb is not None else "")
    return name, dict(runtime="mlxdirect", model=INK25, direct={"prefill_step": step, "kv_bits": kv_bits, "kv_start": 0, "cache_limit_mb": cache_mb})


CONFIGS |= dict([direct(2048), direct(512), direct(256), direct(128), direct(256, kv_bits=4), direct(256, cache_mb=0)])
CONFIGS |= {
    "ink-oq25e-omlx-guard-off-p256": dict(runtime="omlx", model=INK25, guard="off", prefill_step=256),
    "ink-oq25e-omlx-guard-12gb": dict(runtime="omlx", model=INK25, guard_gb=12),
    "ink-oq25e-omlx-guard-13gb": dict(runtime="omlx", model=INK25, guard_gb=13),
    "ink-oq25e-omlx-guard-12gb-nomtp": dict(runtime="omlx", model=INK25, guard_gb=12, model_settings={"mtp_enabled": False}),
    # oMLX loads 11.78 GB (vision tower, DSpark drafter) against 10.9 GB for the bare text model
    "ink-oq25e-omlx-guard-off-p256-nomtp": dict(runtime="omlx", model=INK25, guard="off", prefill_step=256,
                                                model_settings={"mtp_enabled": False}),
    "ink-oq25e-omlx-guard-off-p256-nomtp-llm": dict(runtime="omlx", model=INK25, guard="off", prefill_step=256,
                                                    model_settings={"mtp_enabled": False, "model_type_override": "llm"}),
    "ink-oq25e-omlx-guard-off-p256-nomtp-kv4": dict(runtime="omlx", model=INK25, guard="off", prefill_step=256,
                                                    model_settings={"mtp_enabled": False, "turboquant_kv_enabled": True,
                                                                    "turboquant_kv_scheme": "affine4", "turboquant_kv_bits": 4}),
}

# Inkling on llama.cpp with Metal NVFP4 (master at or after 51831d47d), map ticket 18. Same GGUFs as the
# Baseline, pinned; distinct names so the Baseline's b6d5ce4 results stay as they are.
INK_REV = "8078d149286eec7193795413b9e0643da6d11fa1"


def ink_gguf(quant: str) -> dict:
    return {**gguf(INK_GGUF, f"Mainline_Support/Inkling-2-Mini-{quant}.gguf_file", f"ink/Inkling-2-Mini-{quant}.gguf"), "revision": INK_REV}


# -ub sets the physical batch, which sizes the Metal compute buffer (1047 MiB at 512, 523 at 256, 280 at 128 for
# IQ3_XXS at 8704 context): the llama.cpp counterpart of MLX's prefill step.
for _q, _tag in (("UD-IQ3_XXS", "iq3xxs"), ("UD-Q2_K_XL", "q2kxl"), ("UD-IQ2_XXS", "iq2xxs")):
    for _ub in (None, 256, 128):
        _sfx, _ubarg = (f"-ub{_ub}", ["-ub", str(_ub)]) if _ub else ("", [])
        CONFIGS[f"ink-{_tag}-llamacpp-nvfp4-fit{_sfx}"] = dict(runtime="llamacpp", model=ink_gguf(_q), args=_ubarg)
        CONFIGS[f"ink-{_tag}-llamacpp-nvfp4-allgpu{_sfx}"] = dict(runtime="llamacpp", model=ink_gguf(_q), args=["--fit", "off", "-ngl", "999", *_ubarg])

# ---------------------------------------------------------------- sanitizing

_HOME = str(Path.home())
_USER = os.environ.get("USER", "")
_HOST = socket.gethostname().split(".")[0]


def clean(s: str) -> str:
    s = s.replace(_HOME, "~").replace(str(ROOT), "$MF_ROOT")
    for word in {_USER, _HOST} - {""}:
        s = re.sub(re.escape(word), "<redacted>", s, flags=re.I)
    return s


# ---------------------------------------------------------------- system sampling

_libc = ctypes.CDLL("/usr/lib/libc.dylib")


def sysctl_int(name: str) -> int:
    v = ctypes.c_int64(0)
    size = ctypes.c_size_t(8)
    if _libc.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(size), None, 0) != 0:
        return -1
    return v.value if size.value == 8 else ctypes.c_int32.from_buffer(v).value


def vm_stat() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    d = {}
    for line in out.splitlines()[1:]:
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            d[k.strip()] = int(v) * page
    return d


GiB = 1 << 30


def tree_rss(pids) -> int:
    total = 0
    for pid in pids:
        try:
            p = psutil.Process(pid)
            procs = [p] + p.children(recursive=True)
        except psutil.NoSuchProcess:
            continue
        for q in procs:
            try:
                total += q.memory_info().rss
            except psutil.Error:
                pass
    return total


def snapshot() -> dict:
    vm = vm_stat()
    return {
        "wired": vm.get("Pages wired down", 0), "compressor": vm.get("Pages occupied by compressor", 0),
        "free": vm.get("Pages free", 0), "file_backed": vm.get("File-backed pages", 0),
        "anonymous": vm.get("Anonymous pages", 0),
        "swapouts": vm.get("Swapouts", 0), "pageouts": vm.get("Pageouts", 0), "pageins": vm.get("Pageins", 0),
        "swap_used": psutil.swap_memory().used, "pressure": sysctl_int("kern.memorystatus_vm_pressure_level"),
    }


_CG = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
_CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_CG.CGWindowListCopyWindowInfo.restype = ctypes.c_void_p
_CG.CGWindowListCopyWindowInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_CF.CFRelease.argtypes = [ctypes.c_void_p]


def window_roundtrip() -> bool:
    """One WindowServer round-trip (on-screen window list). False without a desktop session."""
    info = _CG.CGWindowListCopyWindowInfo(1, 0)  # kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    if not info:
        return False
    _CF.CFRelease(info)
    return True


# The Responsive rule ("What measurable test decides that the machine stays responsive for a
# Usable Fit?", map ticket 15). Hang/microhang are Apple's 500/250 ms; 100 ms p99 is where delay
# becomes noticeable. The critical-pressure limits are provisional.
RESPONSIVE = {"hang_ms": 500, "microhang_ms": 250, "p99_ms_max": 100,
              "critical_share_max": 0.05, "critical_stretch_s_max": 10}
# The kill-switch is the same rule, applied as soon as failure is certain. Critical time is
# budgeted against the planned benchmark length (load + two 8K requests + short request).
PLANNED_S = 300
WIRED_MAX_MB = 13312


class Monitor:
    """Samples memory every 0.5 s and runs the Responsive probes; SIGKILLs the runtime on failure."""

    PROBES = {"window": 0.5, "timer": 0.0, "spawn": 1.0}  # probe -> pause between checks (s)

    def __init__(self):
        self.samples = []
        self.probe = {k: [] for k in self.PROBES}  # probe -> [(t, ms)]; timer records lateness
        self.inflight = {k: None for k in self.PROBES}
        self.window_checked = window_roundtrip()
        self.foreign = {}  # process name -> peak RSS of processes > 1 GiB outside the runtime under test
        self.foreign_cpu = {}  # process name -> peak cores of processes using > 0.25 core outside the runtime under test
        self.foreign_busy = []  # (t, cores used by all other processes) every 5 s
        self._cpu_prev = None
        self.contaminated = None  # why the run was stopped for sharing the machine, if it was
        self.pids_fn = lambda: []
        self.phase = "idle"
        self.tripped = None  # reason the kill-switch fired, if it did
        self._stop = threading.Event()
        self.t0 = time.time()

    def _trip(self, reason: str):
        """Kill-switch: SIGKILL the runtime (the thrash run ignored SIGTERM)."""
        if self.tripped:
            return
        self.tripped = f"{reason} at t={time.time() - self.t0:.0f}s in phase {self.phase}"
        self._kill()

    def _kill(self):
        for pid in self.pids_fn():
            try:
                p = psutil.Process(pid)
                for q in p.children(recursive=True) + [p]:
                    q.kill()
            except psutil.Error:
                pass

    def _foreign(self):
        """Other processes: peak RSS of those > 1 GiB, and their CPU. Sustained foreign CPU means another
        workload shares the machine, so the run measures it too: stop and mark the run contaminated."""
        ours = {0}  # kernel_task does our paging
        for pid in self.pids_fn() + [os.getpid()]:
            try:
                p = psutil.Process(pid)
                ours |= {pid} | {c.pid for c in p.children(recursive=True)}
            except psutil.NoSuchProcess:
                pass
        now, cpu, busy = time.time(), {}, 0.0
        for p in psutil.process_iter(["name", "memory_info", "cpu_times"]):
            if p.pid in ours:
                continue
            mi, ct = p.info["memory_info"], p.info["cpu_times"]
            if mi and mi.rss > GiB:
                self.foreign[p.info["name"]] = max(self.foreign.get(p.info["name"], 0), mi.rss)
            if ct:
                cpu[p.pid] = ct.user + ct.system
                if self._cpu_prev and p.pid in self._cpu_prev[1]:
                    used = (cpu[p.pid] - self._cpu_prev[1][p.pid]) / (now - self._cpu_prev[0])
                    busy += used
                    if used > 0.25:
                        name = p.info["name"] or "?"
                        self.foreign_cpu[name] = max(self.foreign_cpu.get(name, 0), used)
        if self._cpu_prev:
            self.foreign_busy.append((round(now - self.t0, 1), round(busy, 2)))
            recent = self.foreign_busy[-QUIET["busy_checks"]:]
            if len(recent) == QUIET["busy_checks"] and all(b > QUIET["foreign_cores_max"] for _, b in recent):
                self.contaminated = f"other processes used {busy:.1f} cores at t={now - self.t0:.0f}s in phase {self.phase}"
                self._kill()
        self._cpu_prev = (now, cpu)

    def _sample(self):
        n, since, critical_s, last = 0, None, 0.0, time.time()
        while not self._stop.is_set():
            s = snapshot()
            now = time.time()
            self.samples.append({"t": round(now - self.t0, 2), "phase": self.phase,
                                 **s, "proc_rss": tree_rss(self.pids_fn())})
            if s["pressure"] >= 4 and self.phase != "idle":
                critical_s += now - last
                since = since or now
                if now - since >= RESPONSIVE["critical_stretch_s_max"]:
                    self._trip(f"memory pressure critical for {now - since:.0f} s in one stretch")
                if critical_s > RESPONSIVE["critical_share_max"] * PLANNED_S:
                    self._trip(f"memory pressure critical for {critical_s:.0f} s in total")
            else:
                since = None
            last = now
            if n % 10 == 0:
                self._foreign()
            n += 1
            self._stop.wait(0.5)

    def _check(self, name: str):
        if name == "window":
            window_roundtrip()
        elif name == "timer":
            time.sleep(0.02)
        else:
            subprocess.run(["/usr/bin/true"])

    def _late_ms(self, name: str, took_s: float) -> float:
        return (took_s - (0.02 if name == "timer" else 0.0)) * 1e3

    def _probe(self, name: str):
        while not self._stop.is_set():
            t = time.perf_counter()
            self.inflight[name] = t
            self._check(name)
            self.inflight[name] = None
            ms = self._late_ms(name, time.perf_counter() - t)
            self.probe[name].append((round(time.time() - self.t0, 2), ms))
            if ms >= RESPONSIVE["hang_ms"]:
                self._trip(f"{name} check took {ms:.0f} ms")
            if self.PROBES[name]:
                self._stop.wait(self.PROBES[name])

    def _watchdog(self):  # a check that never returns is a hang too
        while not self._stop.is_set():
            for name, t in list(self.inflight.items()):
                if t is not None and self._late_ms(name, time.perf_counter() - t) >= RESPONSIVE["hang_ms"]:
                    self._trip(f"{name} check stuck for >= {RESPONSIVE['hang_ms']} ms")
            self._stop.wait(0.05)

    def __enter__(self):
        probes = [n for n in self.PROBES if n != "window" or self.window_checked]
        self.threads = [threading.Thread(target=self._sample, daemon=True), threading.Thread(target=self._watchdog, daemon=True)]
        self.threads += [threading.Thread(target=self._probe, args=(n,), daemon=True) for n in probes]
        for th in self.threads:
            th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        for th in self.threads:
            th.join(timeout=5)

    def _pressure_time(self, s: list) -> dict:
        """Seconds at critical / warning over the benchmark, and the longest critical stretch."""
        crit = warn = stretch = longest = 0.0
        for a, b in zip(s, s[1:]):
            dt = b["t"] - a["t"]
            if a["pressure"] >= 4:
                crit, stretch = crit + dt, stretch + dt
                longest = max(longest, stretch)
            else:
                stretch = 0.0
                warn += dt if a["pressure"] == 2 else 0.0
        span = (s[-1]["t"] - s[0]["t"]) if len(s) > 1 else 0.0
        return {"benchmark_s": round(span, 1), "critical_s": round(crit, 1), "critical_share": round(crit / span, 3) if span else 0.0,
                "critical_longest_s": round(longest, 1), "warning_s": round(warn, 1)}

    def summary(self, baseline: dict) -> dict:
        s = [x for x in self.samples if x["phase"] != "idle"] or self.samples
        first, last = self.samples[0], self.samples[-1]
        pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

        def dist(name):
            xs = sorted(ms for _, ms in self.probe[name])
            if not xs:
                return None
            return {"n": len(xs), "p50": round(pct(xs, .5), 1), "p99": round(pct(xs, .99), 1), "max": round(xs[-1], 1),
                    "microhangs": sum(RESPONSIVE["microhang_ms"] <= x < RESPONSIVE["hang_ms"] for x in xs)}

        return {
            "baseline": {k: round(baseline[k] / GiB, 2) for k in ("wired", "compressor", "free", "swap_used", "anonymous", "file_backed")},
            "peak_proc_rss_gib": round(max(x["proc_rss"] for x in s) / GiB, 2),
            "peak_wired_gib": round(max(x["wired"] for x in s) / GiB, 2),
            "peak_compressor_gib": round(max(x["compressor"] for x in s) / GiB, 2),
            "min_free_gib": round(min(x["free"] for x in s) / GiB, 2),
            "peak_swap_used_gib": round(max(x["swap_used"] for x in s) / GiB, 2),
            "swapouts_gib": round((last["swapouts"] - first["swapouts"]) / GiB, 2),
            "pageouts_gib": round((last["pageouts"] - first["pageouts"]) / GiB, 2),
            "pageins_gib": round((last["pageins"] - first["pageins"]) / GiB, 2),
            "max_pressure_level": max(x["pressure"] for x in s),  # 1 normal, 2 warn, 4 critical
            "pressure": self._pressure_time(s),
            "window_checked": self.window_checked,
            "window_ms": dist("window"),
            "timer_late_ms": dist("timer"),
            "spawn_ms": dist("spawn"),
            "foreign_procs_over_1gib": {clean(k): round(v / GiB, 2) for k, v in self.foreign.items()},
            "foreign_procs_over_quarter_core": {clean(k): round(v, 2) for k, v in self.foreign_cpu.items()},
            "foreign_cores": {"max": max((b for _, b in self.foreign_busy), default=None),
                              "mean": round(statistics.mean(b for _, b in self.foreign_busy), 2) if self.foreign_busy else None},
        }


def responsive(m: dict, tripped) -> bool:
    r, p = RESPONSIVE, m["pressure"]
    for name in ("window_ms", "timer_late_ms"):
        d = m[name]
        if d and (d["max"] >= r["hang_ms"] or d["p99"] > r["p99_ms_max"]):
            return False
    return (not tripped and m["spawn_ms"]["max"] < r["hang_ms"]
            and p["critical_share"] <= r["critical_share_max"] and p["critical_longest_s"] < r["critical_stretch_s_max"])


# ---------------------------------------------------------------- prompt text

def corpus() -> str:
    """Deterministic English/Markdown text: llama.cpp's docs (MIT), sorted by path."""
    parts = [p.read_text(errors="ignore") for p in sorted(TEXT_DIR.rglob("*.md"))]
    return "\n\n".join(parts)


# ---------------------------------------------------------------- runtimes

def port_free():
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} busy: a previous runtime is still running")


def wait_http(url: str, proc, timeout: float, ok=lambda r: r.status_code == 200) -> float:
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"runtime exited with {proc.returncode}")
        try:
            if ok(requests.get(url, timeout=2)):
                return time.perf_counter() - t
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError(url)


class LlamaCpp:
    name = "llama.cpp"

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.path = MODELS / cfg["model"]["local"]
        self.url = f"http://127.0.0.1:{PORT}"

    def version(self):
        out = subprocess.run([LLAMA / "llama-server", "--version"], capture_output=True, text=True)
        return clean((out.stdout + out.stderr).strip().splitlines()[-2])

    def start(self) -> float:
        cmd = [str(LLAMA / "llama-server"), "-m", str(self.path), "-c", str(CTX), "-np", "1", "-fa", "on",
               "--port", str(PORT), "--no-webui", *self.cfg["args"]]
        self.cmdline = clean(" ".join(cmd))
        port_free()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        return wait_http(self.url + "/health", self.proc, 1800)

    def pids(self):
        return [self.proc.pid] if hasattr(self, "proc") else []

    def prompt(self, text):
        toks = requests.post(self.url + "/tokenize", json={"content": text}).json()["tokens"][:PROMPT_TOKENS]
        return requests.post(self.url + "/detokenize", json={"tokens": toks}).json()["content"]

    def generate(self, prompt, n, cache):
        t = time.perf_counter()
        r = requests.post(self.url + "/completion", json={
            "prompt": prompt, "n_predict": n, "temperature": 0, "ignore_eos": True, "cache_prompt": cache}, timeout=3600).json()
        if "timings" not in r:
            raise RuntimeError(f"server error: {r.get('error', r)}")
        tm = r["timings"]
        return {"wall_s": round(time.perf_counter() - t, 2), "prompt_n": tm["prompt_n"], "cached_n": r.get("tokens_cached"),
                "prompt_tps": round(tm["prompt_per_second"], 2), "prompt_s": round(tm["prompt_ms"] / 1e3, 2),
                "gen_n": tm["predicted_n"], "gen_tps": round(tm["predicted_per_second"], 2), "sample": r["content"][:300]}

    def stop(self):
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Ollama:
    name = "ollama"
    url = "http://127.0.0.1:11434"

    def __init__(self, cfg, log):
        self.cfg, self.log, self.tag = cfg, log, cfg["tag"]

    def version(self):
        return requests.get(self.url + "/api/version").json()["version"]

    def start(self) -> float:
        subprocess.run(["ollama", "stop", self.tag], capture_output=True)
        t = time.perf_counter()
        r = requests.post(self.url + "/api/generate", json={"model": self.tag, "prompt": "", "keep_alive": "30m",
                                                           "options": {"num_ctx": CTX}}, timeout=1800).json()
        self.load_duration = r.get("load_duration", 0) / 1e9
        self.cmdline = f"ollama /api/generate model={self.tag} options.num_ctx={CTX} raw=true"
        return time.perf_counter() - t

    def pids(self):
        return [p.pid for p in psutil.process_iter(["name"]) if p.info["name"] and "ollama" in p.info["name"].lower()]

    def prompt(self, text):  # same GGUF, same tokenizer: reuse the llama.cpp-trimmed prompt
        return (OUT / f"prompt-{Path(self.cfg['model']['local']).stem}.txt").read_text()

    def generate(self, prompt, n, cache):
        t = time.perf_counter()
        r = requests.post(self.url + "/api/generate", json={
            "model": self.tag, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
            "options": {"num_ctx": CTX, "num_predict": n, "temperature": 0}}, timeout=3600).json()
        pe = r.get("prompt_eval_duration", 0) / 1e9
        return {"wall_s": round(time.perf_counter() - t, 2), "prompt_n": r.get("prompt_eval_count"),
                "prompt_tps": round(r.get("prompt_eval_count", 0) / pe, 2) if pe else None, "prompt_s": round(pe, 2),
                "gen_n": r["eval_count"], "gen_tps": round(r["eval_count"] / (r["eval_duration"] / 1e9), 2), "sample": r["response"][:300]}

    def stop(self):
        subprocess.run(["ollama", "stop", self.tag], capture_output=True)


class OMLX:
    name = "omlx"

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.path = MODELS / cfg["model"]["local"]
        self.url = f"http://127.0.0.1:{PORT}"
        self.home = ROOT / "omlx-home"  # isolated oMLX base path: settings and prefix cache start empty each run

    def version(self):
        out = subprocess.run([OMLX_BIN / "python", "-c", "import importlib.metadata as m;print('omlx', m.version('omlx'), 'mlx', m.version('mlx'))"],
                             capture_output=True, text=True)
        return out.stdout.strip()

    def _prepare_home(self):
        shutil.rmtree(self.home, ignore_errors=True)
        self.home.mkdir(parents=True)
        if "model_settings" in self.cfg:
            (self.home / "model_settings.json").write_text(json.dumps(
                {"version": 3, "models": {self.path.name: self.cfg["model_settings"]}}))

    def start(self) -> float:
        self._prepare_home()
        guard = ["--memory-guard", self.cfg["guard"]] if "guard" in self.cfg else []
        guard += ["--memory-guard-gb", str(self.cfg["guard_gb"])] if "guard_gb" in self.cfg else []
        args = ["serve", "--model-dir", str(self.path.parent), "--port", str(PORT), "--base-path", str(self.home), *guard]
        if "prefill_step" in self.cfg:  # oMLX has no flag for it: patch the scheduler config it builds
            cmd = [str(OMLX_BIN / "python"), "-c", OMLX_PREFILL_PATCH.format(step=self.cfg["prefill_step"]), *args]
        else:
            cmd = [str(OMLX_BIN / "omlx"), *args]
        self.cmdline = clean(" ".join(cmd))
        port_free()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        wait_http(self.url + "/v1/models", self.proc, 600)
        # the server is up; the model loads on first use, so time a 1-token request
        t = time.perf_counter()
        for _ in self._complete("Hi", 1).iter_lines():  # consume, so the load has really finished
            pass
        return time.perf_counter() - t

    def pids(self):
        return [self.proc.pid] if hasattr(self, "proc") else []

    def _tok(self):
        from tokenizers import Tokenizer  # available in the conda base
        return Tokenizer.from_file(str(self.path / "tokenizer.json"))

    def prompt(self, text):
        tok = self._tok()
        return tok.decode(tok.encode(text, add_special_tokens=False).ids[:PROMPT_TOKENS])

    def _complete(self, prompt, n):
        r = requests.post(self.url + "/v1/completions", json={
            "model": self.path.name, "prompt": prompt, "max_tokens": n, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}, stream=True, timeout=3600)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r

    def generate(self, prompt, n, cache):
        t = time.perf_counter()
        usage, text, stamps = {}, "", []
        # chunk_size=1: the default 512-byte reads hold SSE events back and make the stream look like one burst
        for line in self._complete(prompt, n).iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("error"):
                raise RuntimeError(f"stream error: {d['error']}")
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                stamps.append(time.perf_counter())
                text += d["choices"][0]["text"]
        rnd = lambda v: round(v, 2) if isinstance(v, (int, float)) else None
        # Client-side token arrival times; the server's own timings are kept alongside.
        client_tps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else None
        return {"wall_s": round(time.perf_counter() - t, 2), "prompt_n": usage.get("prompt_tokens"),
                "ttft_s": rnd(stamps[0] - t) if stamps else None, "prompt_tps": rnd(usage.get("prompt_tokens") / (stamps[0] - t)) if stamps and usage.get("prompt_tokens") else None,
                "gen_n": usage.get("completion_tokens"), "text_events": len(stamps), "gen_tps": rnd(client_tps),
                "server_usage": {k: rnd(usage.get(k)) for k in ("time_to_first_token", "prompt_tokens_per_second", "prompt_eval_duration", "generation_tokens_per_second")},
                "sample": text[:300]}

    def derive_decode(self, prompt, res):
        """oMLX 0.8.3 sends Inkling's text as one event at the end of the request and its usage
        timings count the whole request as prefill, so decode speed comes from the same prompt,
        prefix-cached, asked for 1 token: (n - 1) / (wall(n) - wall(1)). Cold prefill speed is then
        the cold request minus that decode time."""
        for key, p, n in (("req_8k_cached", prompt, GEN_TOKENS), ("req_short", prompt[:300], 128)):
            r = res.get(key)
            if not r or r.get("gen_tps"):
                continue
            one = self.generate(p, 1, cache=True)
            r["wall_1tok_s"], r["gen_tps_method"] = one["wall_s"], "(n-1)/(wall(n)-wall(1)), prefix-cached"
            r["gen_tps"] = round((n - 1) / (r["wall_s"] - one["wall_s"]), 2)
        cold, cached = res.get("req_8k_cold"), res.get("req_8k_cached")
        if cold and cached and cached.get("gen_tps"):
            prompt_s = cold["wall_s"] - (GEN_TOKENS - 1) / cached["gen_tps"]
            cold["prompt_s"], cold["prompt_tps"] = round(prompt_s, 2), round(cold["prompt_n"] / prompt_s, 2)
            cold["ttft_s"], cold["prompt_method"] = round(prompt_s, 2), "wall minus decode time from req_8k_cached"

    def stop(self):
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


OMLX_PREFILL_PATCH = """
import sys
from omlx.settings import GlobalSettings
_orig = GlobalSettings.to_scheduler_config
def _patched(self):
    c = _orig(self)
    c.prefill_step_size = {step}
    print("mf: scheduler prefill_step_size =", c.prefill_step_size, flush=True)
    return c
GlobalSettings.to_scheduler_config = _patched
from omlx.cli import main
sys.argv = ["omlx", *sys.argv[1:]]
sys.exit(main())
"""


class MLXDirect:
    """mlx-lm's generation loop in a worker process (mlx_direct.py), no server or scheduler."""
    name = "mlx-direct"

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.path = MODELS / cfg["model"]["local"]

    def version(self):
        out = subprocess.run([OMLX_BIN / "python", "-c", "import importlib.metadata as m;"
                              "print('mlx-lm', m.version('mlx-lm'), 'mlx', m.version('mlx'), 'model code: omlx', m.version('omlx'))"],
                             capture_output=True, text=True)
        return out.stdout.strip()

    def start(self) -> float:
        cmd = [str(OMLX_BIN / "python"), str(Path(__file__).with_name("mlx_direct.py")), str(self.path), json.dumps(self.cfg["direct"])]
        self.cmdline = clean(" ".join(cmd))
        t = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.ready = self._read()
        return time.perf_counter() - t

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited with {self.proc.wait()}")
        return json.loads(line)

    def pids(self):
        return [self.proc.pid] if hasattr(self, "proc") else []

    def prompt(self, text):
        return OMLX.prompt(self, text)

    def _tok(self):
        return OMLX._tok(self)

    def generate(self, prompt, n, cache):
        ids = self._tok().encode(prompt, add_special_tokens=False).ids
        self.proc.stdin.write(json.dumps({"ids": ids, "n": n}) + "\n")
        self.proc.stdin.flush()
        out = self._read()
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def stop(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


RUNTIMES = {"llamacpp": LlamaCpp, "ollama": Ollama, "omlx": OMLX, "mlxdirect": MLXDirect}

# ---------------------------------------------------------------- one config


def interesting(log_text: str) -> list:
    pat = re.compile(r"offload|fit|Mapped model buffer|model buffer size|KV buffer|compute buffer|working set|recommendedMax|"
                     r"warn|error|exceed|repack|n_ctx|wired", re.I)
    return [clean(l) for l in log_text.splitlines() if pat.search(l)][:80]


START = {"pressure_level": 1}
# A quiet machine: the usual desktop, no other heavy session. Before the run, CPU and available memory must
# stay quiet for hold_s; during it, other processes using more than foreign_cores_max cores for busy_checks
# checks in a row (5 s apart) stop the run as contaminated, and it is retried once the machine is quiet again.
QUIET = {"cpu_percent_max": 20, "available_gib_min": 8, "hold_s": 60, "foreign_cores_max": 1.5, "busy_checks": 3, "retries": 3}


def is_runtime(argv: list) -> bool:
    """Another model runtime (matched on its argv, not on text inside a prompt argument)."""
    exe = os.path.basename(argv[0]) if argv else ""
    return (exe == "llama-server" or (exe == "ollama" and "runner" in argv[1:2])
            or ("serve" in argv and "--model-dir" in argv) or any(a.endswith("mlx_direct.py") for a in argv))


def start_gate(timeout=4 * 3600) -> dict:
    """Wait for normal memory pressure (the Responsive rule's only start condition), then for the machine to
    stay quiet for QUIET["hold_s"]: no other runtime or download, low CPU, enough available memory. The
    usual desktop stays open; what else holds memory or CPU is recorded by the Monitor."""
    t = time.perf_counter()
    quiet_since = None
    while True:
        s = snapshot()
        argvs = [p.info["cmdline"] or [] for p in psutil.process_iter(["cmdline"]) if p.pid != os.getpid()]
        downloading = any("download" in argv and any(os.path.basename(a) in ("hf", "huggingface-cli") for a in argv[:2]) for argv in argvs)
        runtime = any(is_runtime(argv) for argv in argvs)
        cpu = psutil.cpu_percent(interval=5)
        avail = psutil.virtual_memory().available / GiB
        quiet = (s["pressure"] == START["pressure_level"] and cpu <= QUIET["cpu_percent_max"]
                 and avail >= QUIET["available_gib_min"] and not downloading and not runtime)
        quiet_since = (quiet_since or time.perf_counter()) if quiet else None
        if quiet_since and time.perf_counter() - quiet_since >= QUIET["hold_s"]:
            return {"waited_s": round(time.perf_counter() - t, 1), "cpu_percent": cpu, "available_gib": round(avail, 2),
                    "compressor_gib": round(s["compressor"] / GiB, 2), "free_gib": round(s["free"] / GiB, 2),
                    "swap_used_gib": round(s["swap_used"] / GiB, 2)}
        if time.perf_counter() - t > timeout:
            raise TimeoutError(f"cannot start: pressure={s['pressure']} cpu={cpu}% available={avail:.1f} GiB "
                               f"downloading={downloading} other runtime={runtime}")
        time.sleep(5)


def set_wired(mb: int):
    subprocess.run(["sudo", "-n", "/usr/sbin/sysctl", f"iogpu.wired_limit_mb={mb}"], check=True, capture_output=True)


def run(name: str, wired: int = 0) -> dict:
    cfg = CONFIGS[name]
    label = f"{name}@wired{wired}" if wired else name
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"{label}.log"
    set_wired(wired)
    try:
        res = _run(cfg, name, label, log_path)
    finally:
        set_wired(0)
    return res


def _run(cfg, name, label, log_path) -> dict:
    with open(log_path, "w") as log:
        rt = RUNTIMES[cfg["runtime"]](cfg, log)
        size = None
        if "local" in cfg["model"]:
            p = MODELS / cfg["model"]["local"]
            size = (p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))
        res = {"config": label, "runtime": rt.name, "runtime_version": rt.version(), "model": {k: v for k, v in cfg["model"].items() if k != "local"},
               "model_bytes": size, "ctx": CTX, "prompt_tokens_target": PROMPT_TOKENS, "gen_tokens": GEN_TOKENS,
               "iogpu_wired_limit_mb": sysctl_int("iogpu.wired_limit_mb"),
               "settings": {k: v for k, v in cfg.items() if k not in ("runtime", "model")},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "responsive_criteria": {**RESPONSIVE, "planned_s": PLANNED_S},
               "quiet_criteria": QUIET}
        subprocess.run(["sudo", "-n", "/usr/sbin/purge"], check=True)  # cold file cache: load_s is a cold load
        res["start_gate"] = start_gate()
        baseline = snapshot()
        with Monitor() as mon:
            try:
                mon.phase = "load"
                mon.pids_fn = rt.pids
                res["load_s"] = round(rt.start(), 2)
                if hasattr(rt, "load_duration"):
                    res["load_s_reported"] = round(rt.load_duration, 2)
                if hasattr(rt, "ready"):
                    res["worker_ready"] = rt.ready
                res["cmdline"] = rt.cmdline
                prompt = rt.prompt(corpus())
                if cfg["runtime"] == "llamacpp":
                    (OUT / f"prompt-{Path(cfg['model']['local']).stem}.txt").write_text(prompt)
                mon.phase = "gen@8k"
                res["req_8k_cold"] = rt.generate(prompt, GEN_TOKENS, cache=False)
                res["req_8k_cached"] = rt.generate(prompt, GEN_TOKENS, cache=True)
                mon.phase = "gen@short"
                res["req_short"] = rt.generate(prompt[:300], 128, cache=False)
                if hasattr(rt, "derive_decode"):
                    rt.derive_decode(prompt, res)
            except Exception as e:  # record the failure; it is a result too
                res["error"] = clean(f"{type(e).__name__}: {e}")
            finally:
                mon.phase = "unload"
                try:
                    rt.stop()
                except Exception as e:
                    res["stop_error"] = clean(str(e))
                time.sleep(3)
        res["memory"] = mon.summary(baseline)
        res["contaminated"] = mon.contaminated
        res["aborted"] = mon.tripped
        res["responsive"] = responsive(res["memory"], mon.tripped)
        gen8k = [r["gen_tps"] for k in ("req_8k_cold", "req_8k_cached") if (r := res.get(k)) and r.get("gen_tps")]
        res["gen_tps_at_8k"] = round(statistics.mean(gen8k), 2) if gen8k else None
        res["usable"] = bool(res["gen_tps_at_8k"] and res["gen_tps_at_8k"] >= 5 and res["responsive"])
    res["log_excerpt"] = interesting(log_path.read_text(errors="ignore"))
    log_path.write_text(clean(log_path.read_text(errors="ignore")))
    res["machine"] = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                      "mem_gib": round(psutil.virtual_memory().total / GiB), "macos": platform.mac_ver()[0]}
    out = OUT / "contaminated" if mon.contaminated else OUT
    out.mkdir(parents=True, exist_ok=True)
    stamp = f".{int(time.time())}" if mon.contaminated else ""
    if mon.contaminated:
        log_path.rename(out / f"{label}{stamp}.log")
    (out / f"{label}{stamp}.json").write_text(json.dumps(res, indent=2))
    (out / f"{label}{stamp}.samples.json").write_text(json.dumps({"samples": mon.samples}, separators=(",", ":")))
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["list"]:
        print("\n".join(CONFIGS))
        sys.exit()
    wired = 0
    if args[:1] == ["--wired"]:
        wired, args = int(args[1]), args[2:]
        if wired > WIRED_MAX_MB:
            sys.exit(f"--wired {wired} refused: {WIRED_MAX_MB} MB is the cap (14336 froze WindowServer)")
    for n in args:
        for attempt in range(QUIET["retries"] + 1):
            r = run(n, wired)
            if not r.get("contaminated"):
                break
            print(json.dumps({"config": n, "contaminated": r["contaminated"], "retry": attempt < QUIET["retries"]}), flush=True)
        else:
            continue
        brief = {k: r.get(k) for k in ("config", "load_s", "gen_tps_at_8k", "responsive", "usable", "aborted", "error")}
        brief |= {k: r["memory"][k] for k in ("peak_proc_rss_gib", "peak_wired_gib", "swapouts_gib", "pressure", "window_ms", "timer_late_ms", "spawn_ms")}
        brief |= {k: (r.get(k) or {}).get(f) for k, f in (("req_8k_cold", "prompt_tps"), ("req_8k_cold", "gen_tps"), ("req_8k_cached", "gen_tps"), ("req_short", "gen_tps"))}
        print(json.dumps(brief), flush=True)
