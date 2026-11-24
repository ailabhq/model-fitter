# Inkling 2 Mini on llama.cpp with Metal NVFP4, at 8K on the M4 16 GB

Map ticket: "How fast is Inkling 2 Mini on llama.cpp with Metal NVFP4 (≥ 51831d47d) at 8K context on the M4 16 GB?"

## Answer

**Usable.** With Metal NVFP4, unsloth's Inkling UD-IQ3_XXS on llama.cpp `--fit` generates **13.1 tok/s at 8K**, up from 3.07 on the Baseline build. Prefill is 47.6 tok/s, so an 8K prompt takes 168 s to the first token. The machine stays **Responsive**: 0.5 s of critical pressure, a 12 ms worst window check, and 2.7 GiB of other apps swapped out. The smaller UD-Q2_K_XL fits wholly on the GPU and does better on every measure: **15.3 tok/s at 8K**, 104 tok/s prefill (77 s to the first token), no critical pressure and 0.02 GiB swapped out. Its Quality loss is unmeasured and almost certainly larger.

| Build | Where the weights go | Gen tok/s @8K | 8K prefill | Peak wired | Swap-outs / page-ins | Responsive |
|---|---|---|---|---|---|---|
| UD-IQ3_XXS (11.7 GB), Baseline `b6d5ce4` | all 20 NVFP4 down-projections on the CPU (3.6 GiB) | 3.07 | 400 s (20 tok/s) | 10.5 GiB | – / 199 GiB | Baseline placeholder flag only |
| UD-IQ3_XXS, `--fit` | 49/49 layers on the GPU, 29 of them with part of their experts on the CPU (1.42 GiB) | **13.09** | 168 s (47.6 tok/s) | 12.85 GiB | 2.73 / 40.1 GiB | yes |
| UD-IQ3_XXS, all on the GPU (`--fit off -ngl 999`) | needs 12,425 MiB against 12,123 MiB free | – | Metal out-of-memory in prefill | 13.66 GiB | – | – |
| same, `-ub 256` | 11,862 MiB | – | Metal out-of-memory in prefill | 13.55 GiB | – | – |
| same, `-ub 128` | 11,600 MiB | – | killed: 1,355 ms window hang 42 s in | 13.42 GiB | – | **no** |
| UD-Q2_K_XL (9.1 GB), `--fit` | everything on the GPU but the token embedding (235 MiB) | **15.32** | 77 s (104 tok/s) | 12.11 GiB | 0.02 / 9.1 GiB | yes |

### Facts later tickets depend on

- **`--fit` places Inkling correctly on 16 GB; forcing everything onto the GPU does not work.** For UD-IQ3_XXS, `--fit` projects 12,425 MiB with every expert on the GPU, against 12,123 MiB free and its 1,024 MiB margin. It then moves part of the experts of 29 of the 49 layers to the CPU: 744 MiB memory-mapped plus 729 MiB repacked into RAM. All-GPU runs out of Metal memory in prefill at `-ub` 512 and 256, and at 128 it loads but hangs WindowServer for 1.4 s. The Metal budget is the ceiling, as it was for MLX ("Can MLX hold Inkling oQ2.5e at 8K…").
- **`-ub` is llama.cpp's prefill-chunk lever, but it doesn't buy residency here.** The Metal compute buffer at 8,704 context is 1,047 MiB at the default `-ub 512`, 523 MiB at 256 and 280 MiB at 128. With `--fit`, a smaller `-ub` only shrinks the experts on the CPU from 1.42 GiB to 0.84 GiB (256) or 0.60 GiB (128), because `--fit` keeps its 1 GiB margin (placement files `placement-ink-iq3xxs-nvfp4-fit-ub{256,128}.txt`). These `--fit -ub` variants weren't benchmarked.
- **The memory-mapped experts are re-read from SSD.** IQ3_XXS read 40 GiB of page-ins in a 254 s run, with pressure at "warning" for 229 s of it, and pushed 2.7 GiB of other apps' memory to swap. It still passed the Responsive rule by a wide margin. So for llama.cpp Inkling, deliberate expert streaming isn't needed at IQ3_XXS: `--fit`'s own spill is Usable. Q2_K_XL, with nothing on the CPU, decodes 17% faster. `[INFERENCE]` That gap is partly the spill and partly Q2_K_XL's fewer bits; the runs don't separate the two.
- **llama.cpp vs MLX for Inkling on this Mac:** MLX oQ2.5e generates 22 tok/s at 8K and prefills at 160–190 tok/s, against 13–15 and 48–104 here. llama.cpp needs no code patch (MLX needs oMLX's prefill step patched), but it does need a `master` build at or after `51831d47d`: still no release has it, and `--version` reports `1.0.0-dev (build 12719)`.
- **A benchmark run must own a quiet machine.** A first IQ3_XXS `--fit` run overlapped another session's 4-process CPU job starting 2 minutes in. It measured 7.57 tok/s, 20.5 GiB swapped out and a 723 ms window hang, and wasn't a valid measurement (kept in `results/pre-gate/`). The harness now waits for the machine to stay quiet for 60 s before each run, and stops and retries any run during which other processes use more than 1.5 cores for 15 s (`QUIET` in `bench.py`). Every new run in the tables passed that gate. The other processes averaged 0.2–0.4 cores during them.

## Setup

- Mac mini M4, 16 GB, macOS 26.6.2. Usual desktop and agent sessions open; `iogpu.wired_limit_mb` default (0 → 12,713 MB `recommendedMaxWorkingSetSize`, 12,123 MiB free for Metal at load).
- llama.cpp `master` at `22b99122e` (build 12719), which contains `51831d47d` ("metal : add NVFP4 support"). Built with CMake, Release, Metal, static.
- Models: [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) at `8078d149`, `Mainline_Support/` files: UD-IQ3_XXS (11.66 GB: 3,645 MiB NVFP4, 3,225 MiB IQ3_XXS, 1,868 MiB IQ2_S, 729 MiB IQ4_NL…) and UD-Q2_K_XL (9.14 GB: 2,283 MiB IQ2_S, 2,005 MiB NVFP4, 1,094 MiB Q2_0, 835 MiB IQ2_XXS, 759 MiB IQ1_S…). Downloaded one at a time and deleted after.
- Workload as in the Baseline: an 8,000-token prompt from llama.cpp's docs at `b6d5ce4`, 256 tokens generated twice (the second time from the prompt cache), then a 69-token prompt with 128 tokens generated. Greedy, EOS suppressed, `-c 8704 -np 1 -fa on`. Cold load (`purge` first).
- Harness: the Baseline's [`bench.py`](baseline/harness/bench.py) as changed for "Can MLX hold Inkling oQ2.5e at 8K…" (Responsive rule as judgement and SIGKILL kill-switch), plus the quiet gate above. Placement from [`placement.sh`](baseline/harness/placement.sh) at `-lv 4`.

## Method notes

- "Gen tok/s @8K" is the mean of the cold and cached 8K requests (12.97 / 13.20 for IQ3_XXS, 15.33 / 15.32 for Q2_K_XL), as llama.cpp's server reports them.
- Placement is read from `common_params_fit_impl` in an unmeasured load with the same arguments, because the benchmark runs at default verbosity. `--fit` reports "49 layers (29 overflowing), 10952 MiB used, 1170 MiB free" for IQ3_XXS.
- Output check: the first 300 characters at 8K are identical between IQ3_XXS and Q2_K_XL and continue the documentation coherently.
- Failed runs aren't judged for responsiveness: they are under a minute long.

## All runs

| Config | Load s | 8K prefill s | Prefill tok/s @8K | Gen tok/s @8K | Gen tok/s short | Peak wired GiB | Swap-outs GiB | Page-ins GiB | Critical s (share, longest) | Window max ms | Timer p99 / max ms | Launch max ms | Other procs, cores (mean / max) | Responsive | Usable | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ink-iq3xxs-llamacpp-nvfp4-fit | 24.26 | 168.02 | 47.61 | 13.09 | 12.98 | 12.85 | 2.73 | 40.07 | 0.5 (0.2%, 0.5) | 12.3 | 5.3 / 16.5 | 52.5 | 0.26 / 0.88 | yes | **yes** | ok |
| ink-iq3xxs-llamacpp-nvfp4-allgpu-ub128 | 15.98 | – | – | – | – | 13.42 | 0.25 | 20.22 | 0.0 (0.0%, 0.0) | 1354.7 | 10.1 / 17.5 | 14.7 | 0.25 / 0.69 | – | no | killed: window check stuck for >= 500 ms at t=42s in phase gen@8k |
| ink-iq3xxs-llamacpp-nvfp4-allgpu-ub256 | 18.58 | – | – | – | – | 13.55 | 0.69 | 21.91 | 0.5 (2.3%, 0.5) | 4.9 | 10.1 / 15.3 | 57.8 | 0.25 / 0.34 | – | no | Metal out-of-memory in prefill (server: "Compute error") |
| ink-iq3xxs-llamacpp-nvfp4-allgpu | 16.78 | – | – | – | – | 13.66 | 0.48 | 20.29 | 0.5 (2.6%, 0.5) | 4.9 | 10.1 / 15.9 | 104.2 | 0.43 / 0.51 | – | no | Metal out-of-memory in prefill (server: "Compute error") |
| ink-q2kxl-llamacpp-nvfp4-fit | 6.48 | 76.67 | 104.34 | 15.32 | 17.18 | 12.11 | 0.02 | 9.05 | 0.0 (0.0%, 0.0) | 9.1 | 10.2 / 25.2 | 12.7 | 0.2 / 0.62 | yes | **yes** | ok |

Not valid, kept for the record: `results/pre-gate/` (the run that overlapped another session) and `results/contaminated/` (two runs stopped when the other session's load began, before the quiet gate existed).
