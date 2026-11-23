# Baseline: off-the-shelf quantized builds of Inkling 2 Mini and Qwen3.8-27B on a Mac mini M4 16 GB

Ticket: [#7](https://github.com/ailabhq/model-fitter/issues/7). Measured 2026-11-23 on the reference Device profile. Harness, raw JSON (per-run summary and 0.5 s samples) and sanitized server logs are in [`baseline/`](baseline/).

## Answer in short

- **No off-the-shelf build is cleanly Usable on 16 GB.** One comes close: **Qwen3.8-27B UD-IQ3_XXS (10.9 GB) on llama.cpp generates 6.7–7.1 tok/s at 8K**. The cost is memory: macOS swaps 3–7 GiB of other apps' memory out (14 GiB under Ollama), and pressure sits at "warn" for 65–90% of the run with a few seconds at "critical". The latency probes stayed fine (20 ms timer p99 ≤ 11 ms, process spawn ≤ 270 ms). Whether that counts as "the machine stays responsive" is the open definition question this baseline hands on.
- **Prefill is slow everywhere.** An 8K prompt takes ~140 s on Qwen (58 tok/s) and ~400 s on Inkling via llama.cpp (20 tok/s).
- **The working set has a hard edge at about 12.5 GiB.** Qwen UD-Q3_K_XL (13.15 GB) fails every way:
  - default `--fit`: 8 layers left CPU-mapped, thrashing at **0.36 tok/s** with 124 GiB swapped out;
  - all layers on GPU: Metal **out-of-memory** on the first batch;
  - `iogpu.wired_limit_mb=14336`: **WindowServer stopped responding for 41 s** and the watchdog fired. That is the "machine responsive" ceiling, found the hard way.
- **Inkling 2 Mini runs at only 3.1 tok/s on llama.cpp, though its weights fit.** unsloth's mainline GGUFs keep the expert down-projections in native **NVFP4**, and llama.cpp's Metal backend has **no NVFP4 matmul** (`ggml-metal-device.m` L1530–1535). Those experts (3.6 GiB in UD-IQ3_XXS) stay CPU-mapped, are computed on the CPU, and are re-read from SSD under memory pressure (199 GiB of page-ins in one run). The larger UD quants are mostly NVFP4: 8.0 GiB in UD-Q3_K_XL, 10.9 GiB in UD-IQ4_XS.
- **Inkling on MLX (oMLX oQ2.5e, 11.6 GB) decodes at ~20–24 tok/s at short context, but can't hold 8K.**
  - oMLX's default memory guard refuses to load it (ceiling ~8 GB; ~10.4 GB with `aggressive`).
  - With the guard off it loads, but prefill hits Metal out-of-memory: from 512 tokens up at the default limit, and at 8K even with the limit raised to 13312 MB.
- **Default GPU working set is 11.84 GiB (74%), not ~2/3.** llama.cpp (Metal) may allocate past it with a warning; MLX may not.
- **Ollama adds nothing for these models and is worse for memory.** On the same GGUF it matches llama.cpp's speed, but it disables mmap for llama-server by default. The weights then sit in anonymous memory and can't be dropped as clean file pages: 14 GiB of swap-outs against 3–7 GiB for llama.cpp.

## Results

Context 8,704; 8,000-token prompt; 256 tokens generated at 8K depth (two samples), 128 at short context. `wired MB` is `iogpu.wired_limit_mb` (default = 0 → 11.84 GiB working set). Load is cold (`purge` first).

| Config | Build size | Wired limit | Load s | Prefill tok/s @8K | Gen tok/s @8K | Gen tok/s short | Peak wired GiB | Swapped out GiB | Pressure: s critical / s warn (of run s) | Timer p99 / spawn max ms | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen UD-IQ3_XXS, llama.cpp `--fit` | 10.9 GB | default | 14.8 | 58.0 | **6.75** | 6.95 | 12.3 | 2.85 | 1.5 / 224 (252) | 10 / 31 | speed ✓, swaps |
| Qwen UD-IQ3_XXS, llama.cpp all-GPU | 10.9 GB | default | 9.4 | 57.6 | **7.08** | 7.47 | 12.9 | 7.35 | 0.5 / 167 (244) | 10 / 149 | speed ✓, swaps |
| Qwen UD-IQ3_XXS, llama.cpp `--fit` | 10.9 GB | 13312 | 20.0 | 55.9 | **6.71** | 7.12 | 12.6 | 4.78 | 2 / 233 (262) | 10 / 82 | speed ✓, swaps |
| Qwen UD-IQ3_XXS, Ollama 0.40.16 | 10.9 GB | default | 9.4 | 57.3 | **6.75** | 7.04 | 12.8 | 14.02 | 4.5 / 224 (252) | 11 / 270 | speed ✓, swaps most |
| Qwen UD-Q3_K_XL, llama.cpp `--fit` | 13.1 GB | default | 27.2 | 19.5 | **0.36** | – | 13.2 | 124 | 63 / 612 (854) | 12 / 2,440 | thrash, stopped by hand |
| Qwen UD-Q3_K_XL, llama.cpp all-GPU | 13.1 GB | default | 9.8 | – | – | – | – | – | – | – | Metal OOM |
| Qwen UD-Q3_K_XL, llama.cpp `--fit` | 13.1 GB | 14336 | – | 51.6 | – | – | – | – | – | – | **WindowServer hung 41 s** in generation |
| Inkling UD-IQ3_XXS, llama.cpp `--fit` | 11.7 GB | default | 11.7 | 20.0 | **3.07** | 3.78 | 10.5 | 0.62 | 0 / 592 (629) | 10 / 82 | too slow |
| Inkling oQ2.5e, oMLX 0.8.3 | 11.6 GB | default | – | – | – | – | – | – | – | – | refused (guard: 8 GB ceiling) |
| Inkling oQ2.5e, oMLX guard `aggressive` | 11.6 GB | default | – | – | – | – | – | – | – | – | refused (guard: 10.4 GB ceiling) |
| Inkling oQ2.5e, oMLX guard off | 11.6 GB | default | 21–26 | OOM | OOM | **~20–24** | 13.6 | 4.97 | 5 / 18 (44) | 10 / 119 | prompts of 64–256 tokens work; 512 and up: Metal OOM |
| Inkling oQ2.5e, oMLX guard off | 11.6 GB | 13312 | 20.2 | OOM | OOM | – | 14.7 | 7.55 | 8 / 19 (50) | 22 / 816 | 8K prefill: Metal OOM |

Short-context oMLX speed is wall clock from a follow-up probe (guard off, default limit): 128 tokens plus a 64–256-token prompt in 5.3–6.9 s; the first guard-off run's short request gave the same (128 + 69 tokens in 6.97 s). oMLX's own `generation_tokens_per_second` was unusable: it reported more than 30,000 tok/s because the stream arrived in one burst.

All generated text was checked to be coherent continuations of the prompt, so Inkling's loop implementation in llama.cpp works (fix #32704 included).

## Where llama.cpp put things

Placement comes from a separate, unmeasured load at debug verbosity (`baseline/results/placement-*.txt`).

- **Qwen UD-IQ3_XXS:** `--fit` projected 10,539 MiB against 12,123 MiB free and offloaded **66/66 layers** with no changes. "fit" and "all-GPU" are therefore the same placement; their speed gap is noise.
- **Qwen UD-Q3_K_XL:** projected 12,525 MiB against 12,123 MiB. `--fit` filled **57 layers (11,080 MiB) on the GPU and left 8 CPU-mapped**. Together with ~4 GB of desktop load, those CPU pages are evicted and re-read on every token.
- **Inkling UD-IQ3_XXS:** projected 8,759 MiB and offloaded **49/49 layers**. But 3,893 MiB stays `CPU_Mapped`: the 20 NVFP4 `ffn_down_exps` tensors (3,645 MiB), which Metal cannot multiply, plus `token_embd`. KV buffers are 272 MiB global + 240 MiB sliding, exactly as [the runtime research](https://github.com/ailabhq/model-fitter/blob/research/runtime-support/research/runtime-support.md) computed.

NVFP4 share of each unsloth mainline Inkling quant, from the GGUF headers:

| Build | Size | NVFP4 (CPU-only on Metal) |
|---|---|---|
| UD-Q2_K_XL | 9.1 GB | 2.0 GiB |
| UD-IQ3_XXS | 11.7 GB | 3.6 GiB |
| UD-Q3_K_XL | 13.6 GB | 8.0 GiB |
| UD-IQ4_XS | 15.7 GB | 10.9 GiB |

## What this means for the map

[INFERENCE: analysis of the measurements above]

- **Runtimes.** llama.cpp is the only runtime that produced a Usable-speed result. Ollama matched its speed but doubled-to-quadrupled the swap. oMLX decodes Inkling ~6× faster than llama.cpp, but its memory guard and prefill allocations keep it from running at all on 16 GB at 8K.
- **Fit plan.**
  - **Dense Qwen:** the off-the-shelf ceiling is ~11 GB (≈ 3.2 bpw) resident on the GPU. Anything that leaves layers CPU-mapped collapses below 1 tok/s, because the desktop's own memory forces those pages out. A Fit must budget for the desktop (~4–6 GB here), not just the GPU limit.
  - **Inkling:** the precision of the experts matters less than whether their quant type has a Metal kernel. Either requantize the NVFP4 experts to a Metal-supported type, or use MLX with a smaller prefill footprint.
- **Wired limit.** Raising `iogpu.wired_limit_mb` bought nothing for builds that already fit (Qwen IQ3_XXS: same speed, more swap). At 14336 MB it froze the GUI. It is not a safe lever on 16 GB above ~13 GB.
- **Usable.** "Machine stays responsive" needs a measurable definition. Latency probes alone pass the Qwen IQ3_XXS runs; memory-pressure and swap-out criteria fail them.
- **Prefill.** 8K prompts take 2–7 minutes. Time to first token at 8K is a usability fact the Usable bar doesn't yet mention.

## Not measured, and why

- **Inkling on Ollama:** Ollama ships only a 20 GB MLX tag, and its released binary can't load Inkling GGUFs (runtime research).
- **Qwen on MLX:** no sub-4-bit MLX build exists. `mlx-community/Qwen3.8-27B-oQ3.5e-mtp` is an empty repo; the smallest real build is 4-bit at 16.1 GB.
- **oMLX oQ3e (13.7 GB), and Inkling UD-Q3_K_XL / UD-IQ4_XS on llama.cpp:** skipped because measured results dominate them. oQ3e is 1.1 GiB larger than oQ2.5e, which already OOMs at 512 tokens. The two GGUFs carry 2–3× the CPU-only NVFP4 of UD-IQ3_XXS (3.07 tok/s) and exceed RAM, which is the pattern behind the Qwen UD-Q3_K_XL thrash and freeze.
- **Vision**, MTP / speculative decoding, KV-cache quantization: out of this baseline's scope (defaults only).

## Method

**Machine.** Mac mini M4 (10-core CPU, 10-core GPU), 16 GB, 120 GB/s, macOS 26.6.2. ~22.8 GB of free disk; one build on disk at a time, deleted after measuring.

**Quiet machine.** Heavy background jobs (an image-generation queue, a test suite, a stray headless Chrome) were stopped first. Before each run the harness ran `purge` (cold file cache) and waited for:
- normal memory pressure;
- compressor ≤ 2 GiB;
- CPU ≤ 40% (macOS `mediaanalysisd` kept 1–2 cores busy);
- no download in progress.

The usual desktop stayed up (~4–6 GB: terminal, agent runtimes): the realistic "machine in use" case. Every run also records any other process above 1 GiB; none appeared.

**Runtimes.**
- llama.cpp master `b6d5ce4` (2026-11-23), built locally, including #32694 and #32704.
- Ollama 0.40.16, with the GGUF imported via `ollama create`.
- oMLX 0.8.3 (mlx 0.32.4, mlx-vlm 0.7.11).

Defaults throughout, except the flags named in each config: `-c 8704 -np 1 -fa on`, all-GPU = `--fit off -ngl 999`, `--memory-guard`, `iogpu.wired_limit_mb`.

**Workload.**
- Prompt: 8,000 tokens of llama.cpp's own Markdown docs, cut with the model's tokenizer, sent as a raw completion.
- Generation: 256 tokens at temperature 0, EOS ignored, then the same request again from the prompt cache; then a 64-token prompt with 128 generated.
- Speeds are the runtime's own timings (llama.cpp, Ollama) or wall clock (oMLX, see above).

**Memory and responsiveness.** Sampled every 0.5 s:
- wired, compressor, swap;
- swap-outs, page-outs, page-ins;
- macOS memory-pressure level;
- RSS of the runtime's process tree (unreliable for Metal/mmap: wired is the meaningful figure).

Two probes run alongside: lateness of a 20 ms sleep, and time to launch `/usr/bin/true` (once a second). After the WindowServer freeze, a kill-switch was added: it kills the runtime if pressure stays critical > 15 s, the timer is > 1 s late, or a spawn takes > 1.5 s. It never had to fire.

**Provisional "responsive" flag in the JSON:** never critical, ≤ 0.25 GiB swapped out, timer p99 ≤ 50 ms, spawn max ≤ 1 s. It is strict and only a placeholder; the raw numbers are there to apply the real definition once it is decided.
