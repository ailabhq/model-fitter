# Does a group-probed sensitivity-weighted mix lose no more than Unsloth's UD-IQ3_XXS at the same size?

Map ticket: "Does a group-probed sensitivity-weighted mix lose no more than Unsloth's UD-IQ3_XXS at the same size, for both reference models?" (map: "Map: sarto, Usable Fits of new models on small Apple Silicon Macs"). This branch is never merged. Status: **Qwen answered, Inkling not yet probed** (GPU budget; see the end).

## Answer so far (Qwen3.8-27B)

**No, as decided, but the measurement works except for one tensor.** The group probes resolve every Qwen group, and the weights they give track Unsloth's choices. The mix the unchanged solver builds from them is Rejected. One tensor causes that: `token_embd`, which the solver sends to Q1_0 by extending a single Q3_K probe linearly to 16× the error. With `token_embd` at Unsloth's Q2_K and the rest re-solved under the same ceiling, the mix ties Unsloth: 0.0608 vs 0.0597 mean KLD, so it misses the pass rule by 0.001.

Scores on the Source's response tokens against the Qwen draft pack, `llama-mf-score` at `-c 8704 -fa on -b 128 -ub 128`:

| | Unsloth UD-IQ3_XXS | Group mix, as decided | Group mix, `token_embd` Q2_K, over size | Group mix, `token_embd` Q2_K, same size |
|---|---|---|---|---|
| File bytes | 10,934,860,704 | 10,934,843,840 | 11,173,231,040 (+0.24 GB) | 10,934,854,080 |
| `token_embd` | Q2_K | **Q1_0** | Q2_K | Q2_K (held) |
| Mean KLD / top-1 | **0.0597 / 91.8%** | 0.449 / 78.8% | 0.0547 / 92.1% | **0.0608 / 91.7%** |
| p99 KLD | 0.53 | 4.32 | 0.52 | 0.58 |
| 8K slice (512 positions) | 0.019 / 97.7% | 0.186 / 90.6% | 0.015 / 98.0% | 0.017 / 98.0% |
| Band (provisional) | Lossy | **Rejected** | Lossy | Lossy |
| Chat / code / multilingual / tool calls KLD | 0.078 / 0.043 / 0.074 / 0.031 | 0.61 / 0.34 / 0.40 / 0.54 | – | 0.076 / 0.044 / 0.083 / 0.020 |
| Pass (≤ 0.060, band and 8K no worse) | – | no | not same size | **no, by 0.001** |

- **As decided, the mix fails.** It is 7.5× Unsloth's loss, uniformly over every domain, as damage to every token would be.
- **`token_embd` is the cause.** The same type map with only `token_embd` raised to Q2_K goes from 0.449 to 0.055. That is below Unsloth, though at 0.24 GB more. The probe measured `token_embd` at Q3_K only: sₜ = 6.4e-8, the smallest of all 82 groups. The solver then priced Q1_0 (table error 89,530, 16× Q3_K's 5,644) at 0.006 KLD, and took the 0.24 GB it saved.
- **Size-matched with `token_embd` held at Q2_K, it's a tie.** The rest of the mix pays for those 0.24 GB, mostly in early `ffn_down`/`ffn_up` and some `attn_qkv_gate`. Position-weighted it is 0.0011 worse than Unsloth. Item-weighted it is 0.0015 ± 0.0022 better (17 items, 7 worse). The 8K slice and tool calls are better; multilingual is worse.
- **Holding a tensor's type is a floor,** which "How does sarto's converter choose a type for each tensor…" ruled out. This run is a diagnosis, not a candidate rule. Under the ticket's rules, the remedy is the second probe type per group. It measures each group's slope, so it would have priced `token_embd` at Q1_0 from a second point near Q2_K instead of a straight line from Q3_K.
- **Speed is in line.** The as-decided mix runs at 6.92 tok/s at 8K under `--fit`, Responsive (longest window stall 41 ms, critical pressure 1.1 s of 251 s). Unsloth ran at 7.07 tok/s in the table-mix ticket. The other two files weren't benchmarked.

## The probes

82 Qwen groups (one role × one eighth of depth; `output` and `token_embd` alone). 145 calibration sequences, 131,598 tokens, 99,928 of them response positions.

- **Every group resolved.** The excess over the matched control is 7.7–36 paired standard errors (`results/qwen-groups/summary.md`). No group needed the upper bound.
- **The controls behave as the SE says.** The repeated-seed control (the `ffn_up` group of each eighth) differs from the first control by z = 0.18, 0.73, 2.00, −0.80, −1.41, −0.58, −0.70, −0.56: spread ≈ 1.1, so the paired SE is right. The control itself falls with depth, 0.00051 in the first eighth to 0.00020 in the last.
- **Excess grows with depth.** For example, `ffn_down` goes 0.0009 → 0.0037 and `attn_v` 0.00015 → 0.0010 between the first and last eighths.

sₜ relative to the `ffn_gate`/`ffn_up` groups' geometric mean, next to the weights Unsloth's types imply (`implied.py`: the lower bound at the unweighted solve's λ, as in "How does sarto's allocator weigh one tensor's quantization error…"). First eighth → middle → last:

| Role | Measured sₜ | Unsloth-implied |
|---|---|---|
| `ssm_beta_alpha` | 9.5 → 24 → 12 | 411 → 431 → 395 |
| `attn_v` | 4.1 → 20.5 → 18.3 | 14.4 → 13.7 → 13.8 |
| `attn_k` | 1.9 → 8.5 → 9.4 | 4.1 → 6.6 → 12.3 |
| `ffn_down` | 2.1 → 6.6 → 9.8 | 3.5 → 4.5 → 18.5 |
| `ssm_out` | 2.8 → 3.8 → 3.8 | 6.7 → 2.4 → 7.2 |
| `ffn_gate` | 0.29 → 1.4 → 1.7 | 0.49 → 1.1 → 3.7 |
| `attn_qkv_gate` | 0.44 → 0.99 → 0.69 | 0.62 → 0.79 → 0.98 |
| `output` | 4.9 | 3.9 |
| `token_embd` | 0.18 | 0.37 |

- Most groups land within 2× of Unsloth, with the same depth ramp for `ffn_down`, `ffn_gate`/`ffn_up` and `attn_k`.
- `ssm_beta_alpha` is the exception: 10–30 measured against ~400 implied. Unsloth's 8.5 bpw for it isn't explained by its effect on the output at IQ3_XXS. The group mix gives it Q5_K (5.4 bpw), and the size-matched mix with Q2_K `token_embd` still ties.
- Bytes by role in the as-decided mix are close to Unsloth's (`ffn_down` 3.18 vs 3.12 bpw, `output` 4.50 vs 4.50, `attn_v` 5.19 vs 4.94). The table mix of "Does sarto's table-allocated mix lose less quality…" had starved exactly these. The exception is `token_embd`, at 1.12 vs 2.62 bpw.

## Inkling 2 Mini: not probed yet

- **The pod run stopped at the mapping check, before any probe.** transformers 5.24.0 doesn't know the checkpoint's `modelopt` quantization label and skips it on load ("Unknown quantization type, got modelopt … we will skip the quantization"). The loaded `lm_head.weight` then differs from the checkpoint's `model.llm.unembed.weight`, and `check_units` stops the run.
- `[INFERENCE]` The draft packs loaded the same revision with the same transformers version through `build_pack.py`. Either their load took a different path (config or class), or the same skip went unnoticed there. Which one has to be checked before probing, and it can be checked on CPU.
- **Pending:**
  - the Inkling probes: about 82 groups (`make_groups` on the table's 226 tensors); `[INFERENCE]` about 45–75 min of H100;
  - the conversion with the NVFP4 copy (`run_gmix.sh inkling`, wired, untested end to end);
  - scoring against the Inkling draft pack.

## Cost

- **$7.30 of RunPod in total** (balance $10.89 → $3.59).
  - First an H100 NVL (secure, $3.19/h). It turned out to be capped at 310 W, about 1 GHz, 30–60 s per pass, so it was replaced after the first eighth (the 10 block-0 groups).
  - Then an H100 SXM (secure, $4.39/h; no community H100 available): the other 72 Qwen groups in about 50 min, plus setup and the failed Inkling start.
  - A whole-model pass of Qwen takes 18.8 s from layer 0 and 10.5 s from layer 32; one group with its control takes 17–45 s.
  - That is above the 1.5 H100-hour budget for both models' share. `[INFERENCE]` The NVL pod (about 50 min, ≈ $2.7) and the failed Inkling start account for about half of it.
- **Block 0 ran on the NVL, the rest on the SXM.** A group's swap and control always ran on the same GPU in the same pass setup, so each excess is paired within one machine.
- **NAS conversions:** 43–44 min each (`llama-quantize` from the F32 GGUF, every tensor named). Copying 11 GB to the Mac over SMB takes about 15 min.

## Method

- `sensitivity-mix/pod/group_probe.py`, the measurement as decided:
  - **Groups:** one role × one eighth of depth (`make_groups`). Qwen's 64 layers give 8 per block; a fused tensor is one member; MTP (`blk.64`) is outside.
  - **Swap:** every member through ggml's quantize-then-dequantize at its probe type (`probe.dequant`, unchanged from the per-tensor probes: IQ3_XXS, Q3_K for `output`/`token_embd`, Unsloth's imatrix), rounded to BF16.
  - **Control:** every member + N(0, (1e-3·RMS)²).
  - **Rerun point:** Qwen reruns from the stored input of the group's first layer.
  - **Scoring:** response positions only. Per-sequence KLD sums are kept for swap, control and, once per eighth, a second control.
- `sensitivity-mix/sens.py` computes:
  - excess = Σ(swap − control) / Σ positions, clipped at 0, with the ratio-estimator paired SE;
  - sₜ = excess / Σ eₜ, or (excess + 2 SE) / Σ eₜ when unresolved;
  - `kld_share` = sₜ·eₜ, which `solve_weighted.py` reads unchanged (`--sens-col kld_share`);
  - a per-group power law when a second type is given (`--second`; the remedy, not run).
- **Solver inputs:** those of the table-mix ticket (Unsloth's size as ceiling; header and other bytes from `layout.py`; MTP fixed at Unsloth's types). Run without weights, it still reproduces both table mixes tensor for tensor and byte for byte.
- **Conversion:** `convert_sens.sh` in the `sarto-ref` image. Both solved files landed on the solver's estimate to the byte. sha256: as decided `results/mixes/qwen-gmix.file.json`; size-matched `qwen-gmix-emb.file.json`; over size `qwen-gmix-embq2k.file.json`.
- **Scoring:** `score_fit.py` at `-b 128 -ub 128` behind the 8 GiB quiet gate, through `run_gmix.sh`.
  - A first attempt at a 6 GiB gate was killed in load: 4.3 s of critical pressure and a 2.9 s window stall next to the desktop (`results/aborted/`).
  - The scorer needs about 8 GiB available on this Mac.
- **Smoke test:** `pod/smoke.py` now also runs `group_probe.py` (both types) and `sens.py` on tiny random Qwen3.5 and Inkling 2 Loop models. Nulls are 0, every tensor is in exactly one group, and the repeat controls come out.
- transformers 5.24.0, torch 2.15.0+cu132, flash-linear-attention 0.6.0, TF32 off.

## Files

- `sensitivity-mix/pod/group_probe.py`, `pod/pod_run.sh` (stages `gprobe-*`, `gprobe2-*`), `pod/probe.py` (`dequant` shared), `pod/smoke.py`.
- `sensitivity-mix/sens.py`, `implied.py`, `run_gmix.sh`, `convert_sens.sh`, `bench_gmix.py`.
- `sensitivity-mix/results/qwen-groups/`:
  - `groups.jsonl`: raw per-sequence sums;
  - `sens.jsonl`: per GGUF tensor: sₜ, group, probe type, swap/control/excess KLD, paired SE, resolved;
  - `groups.json`, `summary.md`, `implied.md`.
- `sensitivity-mix/results/mixes/qwen-gmix*.{types,solve,file}.json`: the three type maps.
- `sensitivity-mix/results/qwen-gmix*-llamacpp-fit-b128.json`: scores. `qwen-gmix-llamacpp-fit.json`: the 8K benchmark. `results/aborted/`: the killed first scoring run.
