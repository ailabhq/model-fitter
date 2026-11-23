#!/usr/bin/env python3
"""Baseline benchmark: off-the-shelf quantized builds on one Apple Silicon Mac.

Usage: bench.py CONFIG [CONFIG ...]     (see CONFIGS below; `bench.py list` prints them)

Environment:
  MF_ROOT     scratch dir holding models/, llama.cpp/, omlx-venv/ (never inside the repo)
  MF_OUT      where result JSON goes (default: ./results)

For each config: start the runtime, time the load, send an ~8000-token prompt and
generate 256 tokens (twice, the second time from the prompt cache), then a short
prompt for a shallow-context reference. A sampler records system memory and two
responsiveness probes the whole time. Results carry no paths, usernames or hostnames.
"""
import ctypes
import json
import os
import platform
import re
import signal
import socket
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
TEXT_DIR = ROOT / "llama.cpp" / "docs"
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


class Monitor:
    """Samples memory every 0.5 s and runs two responsiveness probes."""

    def __init__(self):
        self.samples, self.timer_late, self.spawn = [], [], []
        self.foreign = {}  # process name -> peak RSS of processes > 1 GiB outside the runtime under test
        self.pids_fn = lambda: []
        self.phase = "idle"
        self.tripped = None  # reason the kill-switch fired, if it did
        self._stop = threading.Event()
        self.t0 = time.time()

    def _trip(self, reason: str):
        """Kill-switch: stop the runtime before the machine becomes unusable."""
        if self.tripped:
            return
        self.tripped = f"{reason} at t={time.time() - self.t0:.0f}s in phase {self.phase}"
        for pid in self.pids_fn():
            try:
                p = psutil.Process(pid)
                for q in p.children(recursive=True) + [p]:
                    q.kill()
            except psutil.Error:
                pass

    def _foreign(self):
        ours = set()
        for pid in self.pids_fn() + [os.getpid()]:
            try:
                p = psutil.Process(pid)
                ours |= {pid} | {c.pid for c in p.children(recursive=True)}
            except psutil.NoSuchProcess:
                pass
        for p in psutil.process_iter(["name", "memory_info"]):
            mi = p.info["memory_info"]
            if p.pid not in ours and mi and mi.rss > GiB:
                self.foreign[p.info["name"]] = max(self.foreign.get(p.info["name"], 0), mi.rss)

    def _sample(self):
        n, critical_since = 0, None
        while not self._stop.is_set():
            s = snapshot()
            self.samples.append({"t": round(time.time() - self.t0, 2), "phase": self.phase,
                                 **s, "proc_rss": tree_rss(self.pids_fn())})
            if s["pressure"] >= 4:
                critical_since = critical_since or time.time()
                if time.time() - critical_since > KILL["critical_s"]:
                    self._trip(f"memory pressure critical for > {KILL['critical_s']} s")
            else:
                critical_since = None
            if n % 10 == 0:
                self._foreign()
            n += 1
            self._stop.wait(0.5)

    def _timer(self):  # how late a 20 ms sleep wakes up
        while not self._stop.is_set():
            t = time.perf_counter()
            time.sleep(0.02)
            late = time.perf_counter() - t - 0.02
            self.timer_late.append((time.time() - self.t0, late))
            if late > KILL["timer_late_s"]:
                self._trip(f"20 ms timer woke {late:.1f} s late")

    def _spawner(self):  # how long launching a trivial process takes
        while not self._stop.is_set():
            t = time.perf_counter()
            subprocess.run(["/usr/bin/true"])
            took = time.perf_counter() - t
            self.spawn.append((time.time() - self.t0, took))
            if took > KILL["spawn_s"]:
                self._trip(f"process spawn took {took:.1f} s")
            self._stop.wait(1.0)

    def __enter__(self):
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._sample, self._timer, self._spawner)]
        for th in self.threads:
            th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        for th in self.threads:
            th.join()

    def summary(self, baseline: dict) -> dict:
        s = [x for x in self.samples if x["phase"] != "idle"] or self.samples
        first, last = self.samples[0], self.samples[-1]
        late = sorted(v for _, v in self.timer_late)
        spawn = sorted(v for _, v in self.spawn)
        pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
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
            "timer_late_ms": {"p50": round(pct(late, .5) * 1e3, 1), "p99": round(pct(late, .99) * 1e3, 1), "max": round(late[-1] * 1e3, 1)},
            "spawn_ms": {"p50": round(pct(spawn, .5) * 1e3, 1), "p99": round(pct(spawn, .99) * 1e3, 1), "max": round(spawn[-1] * 1e3, 1)},
            "foreign_procs_over_1gib": {clean(k): round(v / GiB, 2) for k, v in self.foreign.items()},
        }


# Provisional operational reading of "the machine stays responsive" for this
# baseline only; the Usable definition itself is decided elsewhere on the map.
RESPONSIVE = {"max_pressure_level_below": 4, "swapouts_gib_max": 0.25, "timer_p99_ms_max": 50, "spawn_max_ms_max": 1000}
# Kill-switch: abort the run (kill the runtime) on any of these. A run at iogpu.wired_limit_mb=14336
# froze WindowServer for 41 s, so this bounds the harm a config can do.
KILL = {"critical_s": 15, "timer_late_s": 1.0, "spawn_s": 1.5}
WIRED_MAX_MB = 13312


def responsive(m: dict) -> bool:
    return (m["max_pressure_level"] < RESPONSIVE["max_pressure_level_below"]
            and m["swapouts_gib"] <= RESPONSIVE["swapouts_gib_max"]
            and m["timer_late_ms"]["p99"] <= RESPONSIVE["timer_p99_ms_max"]
            and m["spawn_ms"]["max"] <= RESPONSIVE["spawn_max_ms_max"])


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

    def version(self):
        out = subprocess.run([OMLX_BIN / "python", "-c", "import importlib.metadata as m;print('omlx', m.version('omlx'), 'mlx', m.version('mlx'))"],
                             capture_output=True, text=True)
        return out.stdout.strip()

    def start(self) -> float:
        guard = ["--memory-guard", self.cfg["guard"]] if "guard" in self.cfg else []
        cmd = [str(OMLX_BIN / "omlx"), "serve", "--model-dir", str(self.path.parent), "--port", str(PORT), *guard]
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
        usage, text = {}, ""
        for line in self._complete(prompt, n).iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("error"):
                raise RuntimeError(f"stream error: {d['error']}")
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                text += d["choices"][0]["text"]
        rnd = lambda v: round(v, 2) if isinstance(v, (int, float)) else None
        return {"wall_s": round(time.perf_counter() - t, 2), "prompt_n": usage.get("prompt_tokens"),
                "ttft_s": rnd(usage.get("time_to_first_token")), "prompt_tps": rnd(usage.get("prompt_tokens_per_second")),
                "prompt_s": rnd(usage.get("prompt_eval_duration")), "gen_n": usage.get("completion_tokens"),
                "gen_tps": rnd(usage.get("generation_tokens_per_second")), "sample": text[:300]}

    def stop(self):
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


RUNTIMES = {"llamacpp": LlamaCpp, "ollama": Ollama, "omlx": OMLX}

# ---------------------------------------------------------------- one config


def interesting(log_text: str) -> list:
    pat = re.compile(r"offload|fit|Mapped model buffer|model buffer size|KV buffer|compute buffer|working set|recommendedMax|"
                     r"warn|error|exceed|repack|n_ctx|wired", re.I)
    return [clean(l) for l in log_text.splitlines() if pat.search(l)][:80]


QUIET = {"pressure_level": 1, "compressor_gib_max": 2.0, "cpu_percent_max": 40}


def quiet_gate(timeout=1800) -> float:
    """Block until the machine is quiet: normal pressure, small compressor, idle CPU, no downloads."""
    t = time.perf_counter()
    while True:
        s = snapshot()
        downloading = any("download" in " ".join(p.info["cmdline"] or []) and "hf" in " ".join(p.info["cmdline"] or [])
                          for p in psutil.process_iter(["cmdline"]) if p.info["cmdline"])
        cpu = psutil.cpu_percent(interval=2)
        if (s["pressure"] == QUIET["pressure_level"] and s["compressor"] / GiB <= QUIET["compressor_gib_max"]
                and cpu <= QUIET["cpu_percent_max"] and not downloading):
            return {"waited_s": round(time.perf_counter() - t, 1), "cpu_percent": cpu, "compressor_gib": round(s["compressor"] / GiB, 2)}
        if time.perf_counter() - t > timeout:
            raise TimeoutError(f"machine not quiet: pressure={s['pressure']} compressor={s['compressor'] / GiB:.1f}GiB cpu={cpu}% downloading={downloading}")
        time.sleep(10)


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
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "responsive_criteria": RESPONSIVE}
        subprocess.run(["sudo", "-n", "/usr/sbin/purge"], check=True)  # cold file cache: load_s is a cold load
        res["quiet_gate"] = quiet_gate()
        baseline = snapshot()
        with Monitor() as mon:
            try:
                mon.phase = "load"
                mon.pids_fn = rt.pids
                res["load_s"] = round(rt.start(), 2)
                if hasattr(rt, "load_duration"):
                    res["load_s_reported"] = round(rt.load_duration, 2)
                res["cmdline"] = rt.cmdline
                prompt = rt.prompt(corpus())
                if cfg["runtime"] == "llamacpp":
                    (OUT / f"prompt-{Path(cfg['model']['local']).stem}.txt").write_text(prompt)
                mon.phase = "gen@8k"
                res["req_8k_cold"] = rt.generate(prompt, GEN_TOKENS, cache=False)
                res["req_8k_cached"] = rt.generate(prompt, GEN_TOKENS, cache=True)
                mon.phase = "gen@short"
                res["req_short"] = rt.generate(prompt[:300], 128, cache=False)
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
        res["aborted"] = mon.tripped
        res["responsive"] = responsive(res["memory"]) and not mon.tripped
        gen8k = [r["gen_tps"] for k in ("req_8k_cold", "req_8k_cached") if (r := res.get(k)) and r.get("gen_tps")]
        res["gen_tps_at_8k"] = round(statistics.mean(gen8k), 2) if gen8k else None
        res["usable"] = bool(res["gen_tps_at_8k"] and res["gen_tps_at_8k"] >= 5 and res["responsive"])
    res["log_excerpt"] = interesting(log_path.read_text(errors="ignore"))
    log_path.write_text(clean(log_path.read_text(errors="ignore")))
    res["machine"] = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                      "mem_gib": round(psutil.virtual_memory().total / GiB), "macos": platform.mac_ver()[0]}
    (OUT / f"{label}.json").write_text(json.dumps(res, indent=2))
    (OUT / f"{label}.samples.json").write_text(json.dumps({"samples": mon.samples}, separators=(",", ":")))
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
        r = run(n, wired)
        brief = {k: r.get(k) for k in ("config", "load_s", "gen_tps_at_8k", "responsive", "usable", "error")}
        brief |= {k: r["memory"][k] for k in ("peak_proc_rss_gib", "peak_wired_gib", "swapouts_gib", "max_pressure_level", "timer_late_ms", "spawn_ms")}
        brief |= {k: (r.get(k) or {}).get(f) for k, f in (("req_8k_cold", "prompt_tps"), ("req_8k_cold", "gen_tps"), ("req_8k_cached", "gen_tps"), ("req_short", "gen_tps"))}
        print(json.dumps(brief))
