# Does a candidate's memory cost add to the desktop's load on the M4 16 GB?

Map ticket: "Does a candidate's memory cost add to the desktop's load on the M4 16 GB, so one failed run can rule out the next candidates?"

## Answer

**Only below the edge, and there it buys nothing. At the edge it doesn't add, but failure is still monotone. So a failed run may rule out larger candidates only when it failed for memory.**

- **Two kinds of failure, cleanly separated.** All 20 kills in 48 runs fall into one of two groups:
  - **Memory failures (16):** pressure reached *critical*, and the run hung WindowServer, the sleep timer or process launch during the cold load (17–21 s in; once at 255 s).
  - **Stalls (4):** a single ≥ 500 ms pause in generation with pressure never above *warning*, and none recurred in the other replicate.
- **A memory failure rules out every larger candidate of the same placement, at that desktop state.** In every (replicate, load) where an all-GPU candidate failed for memory, every larger all-GPU candidate failed too: 5 of 5 such cells.
- **A stall is not evidence about the next candidate.** In one cell the 17408-context candidate stalled while the larger 26112 passed. Stalls hit 4 of 48 runs, on a Usable build, without memory pressure.
- **Counterfactual prediction of the next candidate's numbers (observed run + footprint difference) is not worth using:**
  - Generation speed of the all-GPU candidates doesn't move with load at all (7.07–7.22 tok/s in every completed run from 0 to 1.5 GiB of extra load), so there is nothing to predict.
  - Peak wired memory follows the planned GPU difference on a quiet machine. Under load the prediction error (0.07–0.22 GiB) is within replicate noise (mean 0.26 GiB), and ρ is 0.8–3.9: no gain.
  - Swap-outs add between the two smaller candidates (ρ 0.22–0.46) but not toward the largest (ρ 1.48–1.91; prediction error 1.2–1.6 GiB). Near the Metal ceiling, cost grows faster than footprint.
- **The edge comes in the cold load, so candidates that differ only in context hit it together.** All four candidates (planned totals 10.73–11.89 GiB) pass at up to 1.5 GiB of extra incompressible memory, apart from stalls and one 26112 memory failure. At 2 and 4 GiB, 15 of 16 runs fail for memory (the exception, `-ngl 58` at 2 GiB, survived but was too slow). If memory costs added, the 8704-context candidate would have 1.16 GiB more room than the 26112 one. It doesn't: both fail at 2 GiB in both replicates. `[INFERENCE]` The kills come 17–21 s into a ~15 s cold load, as the weights are read and wired, before the KV-cache difference between the candidates is fully in use. So when a run fails for memory, the next candidate has to load fewer weights; a shorter context doesn't help.
- **The desktop is never "the same load" from run to run.** At a fixed synthetic load, the desktop's own committed memory (wired + anonymous + compressor, just before the load) ranged 4.7–7.8 GiB between runs minutes apart (up to 12.4 at 2 GiB). Measured that way, the room left for the Fit doesn't separate pass from fail either: Usable at −2.65 GiB, memory-killed at −2.17 GiB.

### For "When a Source model is not Usable at 4-bit, what does the Fit plan try, and in what order?"

- After a **memory failure** (critical pressure reached, or Metal out-of-memory), skip every remaining candidate of the same placement whose planned footprint is at least as large, and back off on **weights**, not context. The skip holds only within the same plan run, minutes apart. Over hours the desktop drifts by GiBs (the quiet gate below).
- After a **stall** (≥ 500 ms pause, pressure never critical), rerun the same candidate once before treating it as failed. Skip nothing on its account.
- Never predict a pass. The 26112-context candidate passed at 0.5 and 1 GiB in one replicate and failed at 0 and 1.5 GiB in the other.

### Facts later tickets depend on

- **The Responsive rule fails about 1 run in 12 with no memory pressure.** 4 of 48 runs failed on a single ≥ 500 ms pause, pressure never above warning, and none of the four stalled in the other replicate: `-ngl 58` on the quiet machine and at 0.5 GiB, the 17408-context build at 1 GiB, and the 26112-context build on the quiet machine. Two were the all-GPU builds that are Usable in every other run at those loads. Under the rule as decided, one such pause fails the Fit.
- **psutil's "available" doesn't measure the desktop's load on macOS.** It counts idle anonymous pages, so a 4 GiB holder of touched data showed *more* available memory at the gate than a 2 GiB one in some runs. The quiet gate's fixed 8 GiB floor also blocked runs for over 2 hours once the desktop had grown by 1.5 GiB. Lowering it to 6 GiB for the last 28 runs let them run, and the analysis uses committed memory instead.
- **The extra load leaves RAM first.** Every holder process was evicted completely during its run (RSS 0.0 GiB at the minimum), even at 0.5 GiB.

## Setup

- Mac mini M4, 16 GB, macOS 26.6.2, usual desktop and agent sessions open, `iogpu.wired_limit_mb` default. llama.cpp `master` `22b99122e` (build 12719), as in "How fast is Inkling 2 Mini on llama.cpp with Metal NVFP4…".
- **Source and candidates:** `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS (10.9 GB) at `4af51b35`. Only one Source fits on the disk at once (6.9 GB free at the start), and Inkling oQ2.5e had been deleted. Four neighbouring candidates, all with the same 8,000-token prompt, so only memory differs:

  | Candidate | llama.cpp args | Planned total GiB | On the GPU GiB |
  |---|---|---|---|
  | `gpu-c8704` | `--fit off -ngl 999 -c 8704` | 10.73 | 10.29 |
  | `gpu-c17408` | `--fit off -ngl 999 -c 17408` | 11.31 | 10.86 |
  | `gpu-c26112` | `--fit off -ngl 999 -c 26112` | 11.89 | 11.43 |
  | `ngl58-c8704` | `--fit off -ngl 58` (8 layers memory-mapped on the CPU) | 10.82 | 9.20 |

  The planned footprints are the sum of llama.cpp's own buffer sizes (model, KV, recurrent state, compute, output) from an unmeasured load at `-lv 4` (`results/placement-*.txt`). Each context step adds 544 MiB of KV plus 38–42 MiB of compute buffer.
- **Load:** a holder process with 0, 0.5, 1, 1.5, 2 or 4 GiB of anonymous memory, every page touched with random bytes so the compressor can't shrink it. A fresh holder for every run, started before the cold-cache `purge` and the quiet gate. The gate asks the desktop for the same available memory as without the holder: 8 GiB minus the holder for the first 20 runs, 6 GiB minus the holder for the last 28 (the gate note above). The holder is a child of the runner, so the Monitor doesn't count it as another workload.
- **Runs:** each candidate at each load, in two replicates in reverse orders (replicate 1: loads ascending, candidates ascending; replicate 2: both descending), through `bench.py`'s quiet gate, Monitor, Responsive rule and kill-switch. 48 valid runs. 8 more were stopped as contaminated by another process's CPU use and retried (`results/*/*/contaminated/`).
- **ρ** = Var(obs_j − obs_i) / Var(obs_j) over the (replicate, load) cells where both candidates completed, as defined in the ticket. Killed runs are left out: their metrics stop at the kill. The prediction errors compare three predictors for j under load: i's run + planned footprint difference, i's run + the quiet-machine gap, and j's own quiet-machine run.

Harness: [`additivity/run.py`](additivity/run.py) (holder and grid), the four candidates in [`bench.py`](baseline/harness/bench.py), and [`additivity/summarize.py`](additivity/summarize.py). All tables, including every ρ, are in [`additivity/summary.md`](additivity/summary.md).

## Outcomes

| Rep | Load GiB | ngl58-c8704 | gpu-c8704 | gpu-c17408 | gpu-c26112 |
|---|---|---|---|---|---|
| 1 | 0 | too slow | Usable | Usable | Usable |
| 1 | 0.5 | stall | Usable | Usable | Usable |
| 1 | 1 | Usable | Usable | **stall** | Usable |
| 1 | 1.5 | Usable | Usable | Usable | Usable |
| 1 | 2 | memory | memory | memory | memory |
| 1 | 4 | memory | memory | memory | memory |
| 2 | 0 | stall | Usable | Usable | stall |
| 2 | 0.5 | too slow | Usable | Usable | Usable |
| 2 | 1 | Usable | Usable | Usable | Usable |
| 2 | 1.5 | too slow | Usable | Usable | memory |
| 2 | 2 | too slow | memory | memory | memory |
| 2 | 4 | memory | memory | memory | memory |

"memory": killed with pressure at critical (critical share 0.06–0.31 of the run, mostly during the cold load). "stall": killed on one ≥ 500 ms pause in generation, pressure never above warning. `ngl58-c8704` generates 3.3–5.2 tok/s with no trend in load, so it lands on both sides of the 5 tok/s line. It reads 19–80 GiB back from the SSD per run.

## Metrics of the completed runs

| Candidate | Gen tok/s @8K | Peak wired GiB, quiet | Swap-outs GiB, quiet | Swap-outs GiB, 0.5–1.5 GiB load |
|---|---|---|---|---|
| gpu-c8704 | 7.07–7.21 | 12.38 / 12.52 | 0.60 / 0.76 | 1.30–2.53 |
| gpu-c17408 | 7.12–7.21 | 12.93 / 12.98 | 1.51 / 1.77 | 2.21–3.14 |
| gpu-c26112 | 7.12–7.22 | 13.79 (the other replicate stalled) | 6.66 | 3.20–5.56 |
| ngl58-c8704 | 3.34–5.24 | 11.51 | 0.78 | 1.36–2.90 |

| Pair (i → j) | ρ peak wired | ρ swap-outs | ρ gen tok/s | Swap error: i + footprint Δ / j's quiet run (GiB) |
|---|---|---|---|---|
| gpu-c8704 → gpu-c17408 | 2.97 | 0.46 | 1.14 | 0.24 / 0.88 |
| gpu-c17408 → gpu-c8704 | 1.17 | 0.22 | 0.51 | 0.24 / 1.35 |
| gpu-c8704 → gpu-c26112 | 3.88 | 1.91 | 0.45 | 1.22 / 2.37 |
| gpu-c17408 → gpu-c26112 | 0.83 | 1.48 | 0.38 | 1.61 / 2.10 |
| gpu-c8704 → ngl58-c8704 | 1.26 | 0.32 | 0.97 | 0.37 / 1.11 |

Replicate noise (same candidate and load, |rep 1 − rep 2|): peak wired mean 0.26 / max 0.81 GiB, swap-outs mean 0.70 / max 4.28 GiB, gen tok/s mean 0.11 / max 0.59. The tok/s ρ values are ratios of variances at the noise floor (Var ≈ 0.001) and carry no signal for the all-GPU candidates. Critical-pressure share is ≈ 0 in every completed run, so its ρ is noise too.

## Method notes

- Gen tok/s @8K is the mean of the cold and cached 8K requests, as in the earlier notes.
- Desktop committed memory = wired + anonymous + compressor pages in the Monitor's snapshot just before the load. It includes the holder.
- The first two loads (2 and 4 GiB) were too coarse: every candidate failed. The 0.5 / 1 / 1.5 GiB levels were added after replicate 1's first 12 runs.
- The run waited for the other session's Reference-pack scoring ("Draft Reference packs…") to finish. That session deleted the Qwen file after scoring it, so it was downloaded again at the same revision (same size, 10,934,860,704 bytes) before the last 28 runs.
