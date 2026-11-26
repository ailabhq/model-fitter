# Does sarto's table-allocated mix lose less quality than Unsloth's UD-IQ3_XXS at the same size?

Map ticket: "Does sarto's table-allocated mix lose less quality than Unsloth's UD-IQ3_XXS at the same size, for both reference models?" (map: "Map: sarto, Usable Fits of new models on small Apple Silicon Macs"). This branch is never merged.

## Answer

**No, for both reference models.** At the same file size, the table-allocated mix loses about twice (Qwen) to three times (Inkling) as much as Unsloth's UD-IQ3_XXS on the Source's response tokens. The tables rank it as the better mix. Under the provisional bands, Qwen's table mix is Rejected, where Unsloth's is Lossy.

| | Qwen3.8-27B: Unsloth | Qwen3.8-27B: table mix | Inkling 2 Mini: Unsloth | Inkling 2 Mini: table mix |
|---|---|---|---|---|
| File bytes | 10,934,860,704 | 10,934,850,240 | 11,663,310,784 | 11,663,310,368 |
| Response tokens: mean KLD / top-1 | **0.060 / 91.8%** | 0.108 / 88.3% | **0.020 / 95.4%** | 0.056 / 92.1% |
| p99 KLD | 0.27 | 0.92 | 0.20 | 0.53 |
| 8K slice (512 positions) | 0.019 / 97.7% | 0.032 / 95.5% | 0.004 / 99.6% | 0.011 / 98.0% |
| Band (response tokens, provisional) | Lossy | **Rejected** (KLD > 0.10) | Lossy | Lossy |
| Chat / code / multilingual / tool calls KLD | 0.078 / 0.043 / 0.074 / 0.031 | 0.129 / 0.081 / 0.151 / 0.042 | 0.026 / 0.016 / 0.026 / 0.011 | 0.070 / 0.050 / 0.063 / 0.039 |
| Total table error (column A) | 2.30e5 | 1.22e5 | 3.80e5 | 2.71e5 |
| `--fit` placement at 8K | – | – | `-ngl 49`, experts of 29 layers on CPU | identical arguments, 3 of 3 runs |
| 8K benchmark, tok/s (Responsive) | 7.07 (yes) | 6.51 (yes) | 11.36 (yes, third attempt) | 11.91 (yes) |

- **Every domain is worse**, for both models, by 1.3–3.7× in KLD.
- **The tables predict the opposite.** By the tables' own error, the table mix is the better mix (about half Unsloth's error for Qwen, 71% for Inkling). The measured loss shows that this per-tensor error doesn't rank mixes by Quality loss.
- **The lower loss doesn't come from spilling.** Each pair has the same size to within 10.5 KB. For Inkling, `llama-fit-params` gives both files the same `--fit` arguments in all three runs. Qwen loads wholly on the GPU either way.

## Which roles the tables starved

Same byte budget, redistributed. Full per-role and per-layer data are in `results/mixes/*-roles.json`.

**Qwen3.8-27B** (GiB, bpw):

| Role | Unsloth | Table mix |
|---|---|---|
| `ffn_down` | 2.14, 3.17 | **1.49, 2.21** (IQ2_XXS in 43 of 64 layers) |
| `output` | 0.67, 4.50 | **0.51, 3.44** |
| `ssm_out` | 0.63, 3.59 | **0.54, 3.06** |
| `attn_v` / `attn_k` | 5.15 / 4.41 bpw | **3.78 / 3.74 bpw** |
| `ssm_beta_alpha` | 8.50 bpw | **3.50 bpw** |
| `ffn_gate` / `ffn_up` | 1.99 / 2.08 | 2.35 / 2.35 (3.49 bpw) |
| `attn_qkv_gate` | 1.45 | 1.61 |
| `token_embd` | 0.39, 2.62 | 0.51, 3.44 |

**Inkling 2 Mini:**

| Role | Unsloth | Table mix |
|---|---|---|
| Attention `q`/`k`/`v`/`gate`/`output`/`rel` | 0.58 GiB, 4.5–8.5 bpw | **0.26 GiB, 3.4 bpw** |
| `output` / `token_embd` | 5.50 / 4.50 bpw | **3.44 / 3.44 bpw** |
| `ffn_gate_up_exps` (routed experts) | 5.32 GiB, 2.80 bpw | 5.81 GiB, 3.06 bpw |
| `ffn_down_exps` | NVFP4 layers 0–19, IQ4_NL 20–23 | the same types |

**The allocation is flat in depth.** The table mix gives a role nearly the same type at every layer:
- Qwen `ffn_gate`: IQ3_S in every layer.
- Inkling attention and experts: one type each in all 24 layers.

Unsloth's bits change with depth:
- Qwen `ffn_down`: 2.1 → 4.25 bpw from the first layers to the last.
- Inkling `attn_q`: 8.5 bpw in the middle layers, 4.5–5.5 in the last.

The per-row relative error is about the same at every depth, so the tables can't express how much each tensor's error moves the output.

**The starved roles are the ones upstream boosts.** Upstream's fixed boosts (`ffn_down` ×6, `output` ×3, `attn_v` ×1.25) and its inner-product adjustment for attention target exactly the roles the table mix starves. For Inkling, the tables also can't see that attention runs on every token while each routed expert serves a few. So they trade dense attention bits for expert bits.

## Other facts

- **NVFP4 doesn't stay lossless through an F32 conversion.** Requantizing Inkling's dequantized NVFP4 experts back to NVFP4 at the pin has table error comparable to IQ4_NL (layer 0: `ffn_down` 1,001 vs 1,451; `gate_up` 1,836 vs 1,516). The ggml NVFP4 block has no per-tensor second-level scale, and the pinned toolchain never writes the `.scale` tensors.
  - So "Inkling's NVFP4 experts can stay as they are … with no added loss" holds only if the converter copies the Source's NVFP4 bytes and their scale.
  - In this test, that loss falls on the table mix's 20 NVFP4 `ffn_down_exps` tensors. Unsloth's file keeps the exact bytes.
  - `[INFERENCE]` That is part of the Inkling gap, but it can't be most of it: Qwen has no NVFP4 and shows a gap as large.
- **The two error columns barely differ.**
  - Column A is target sizing's error as is. Column B is the plain imatrix-weighted error: `ffn_down` measured without the simulated-rotation clipping, no inner-product factor.
  - Allocated with B, the mixes differ from A in 26 of 417 tensors (Qwen) and 17 of 226 (Inkling). B wasn't converted.
- **Scoring batch size doesn't move the numbers.** Unsloth Inkling scores 0.0204 / 95.41% at `-b 128 -ub 128` against 0.0203 / 95.36% at `-b 512` (draft packs ticket). Both Inkling builds were scored at `-b 128` here, because a `-b 512` run of the table mix was a memory failure next to today's desktop (40.8 GiB swapped, critical pressure; kept in `results/aborted/`).
- **Speed is in line; Inkling runs were flaky today.**
  - Qwen: the table mix ran at 6.51 tok/s against Unsloth's 7.07 in the same session, both Responsive. The table mix swapped more (7.98 vs 3.66 GiB).
  - Inkling: the table mix ran at 11.91 tok/s and Unsloth at 11.36, both Responsive. Unsloth swapped more (15.9 vs 8.8 GiB).
  - Unsloth Inkling passed only at its third attempt. First came a 920 ms stall in load behind a 6 GiB gate, then an 818 ms stall at 8K with 21 GiB swapped behind the 8 GiB gate. Two days earlier it passed at 13.1 tok/s with the same binary and arguments.
  - Other sessions were running on the Mac today, and Inkling's placement leaves about 1 GiB free. Speed and Responsive therefore depend on the desktop at that moment; the scores don't. The failed runs are kept in `results/failed-*`.
- **Table cost.** The patched `llama-quantize --dry-run` on the F32 GGUFs, with 5 threads on the NAS's Pentium Gold 8505:
  - Qwen: 410 tensors × 26 types in 21 min.
  - Inkling: 226 tensors in 2.8 h, under disk contention. Its 3-D expert tensors sample 256 rows per expert.
  - The compressed tables are 0.2 MB (Qwen) and 0.1 MB (Inkling).

## Method

- **Tables** (`tables.patch`, `Dockerfile`):
  - llama.cpp `22b99122e`, `target_bpw_type()` patched for this test:
    - every candidate's error is appended to a TSV (`MF_TABLE_DUMP`);
    - the Metal types upstream's pool lacks are added: Q1_0, Q2_0, TQ2_0, MXFP4, NVFP4.
  - Everything else is upstream's error:
    - sampled rows (256–4096 per tensor or per expert);
    - imatrix values winsorized per row;
    - per-row error normalized by the row's weighted norm, trimmed mean × rows.
  - Inputs:
    - the F32 GGUFs of "Can a per-tensor streaming converter build…" (pinned converter, so fused `attn_qkv_gate`/`ssm_beta_alpha`);
    - Unsloth's imatrix, adapted to the fused names for Qwen.
  - Neither imatrix has entries for `output` or `token_embd`, so those tensors get no IQ types and an unweighted error, as in upstream.
- **Solver** (`solve.py`):
  - The rules as decided: lower convex hull per tensor, a Lagrangian solve, then a greedy fill under the whole-file ceiling; ties break by name; no boosts, caps or floors.
  - Only llama.cpp's never-quantize list keeps its type.
  - Source cap: Inkling's layer 0–19 experts (NVFP4) get no type larger than 4.5 bpw.
- **Ceiling.** Unsloth's file size. Header bytes and non-quantizable tensors are taken from the reference conversion with Unsloth's types, so both files carry the same fixed part.
- **MTP.** Qwen's `blk.64` keeps Unsloth's types (0.33 GiB) outside the allocation, so the sizes stay comparable. The decided rule drops MTP, which would have given the table mix 0.33 GiB more.
- **Conversion** (`convert_mix.sh`). `llama-quantize` from the F32 GGUF on the NAS, naming every tensor: the reference path that the streaming converter matched tensor for tensor. Each file landed on the solver's estimate to the byte.
- **Scoring.**
  - The draft packs ticket's `score_fit.py` and `llama-mf-score`, on the M4 16 GB against the draft Reference packs, at `-c 8704 -fa on -b 128 -ub 128`.
  - Quiet gate, Monitor and kill-switch as before.
  - Response tokens only, the band being the worse of the whole corpus and the 8K slice.
  - Unsloth Qwen's score is the draft packs ticket's run with the same binary and arguments. Unsloth Inkling was rescored here at `-b 128`.
- **Benchmark** (`bench_mix.py`): bench.py's 8K run under `--fit`, with no other arguments.
- **Unsloth files.**
  - Qwen `4af51b35` (sha256 `c0b7c303…`) and Inkling `8078d149` (sha256 `fd9cceda…`), as published.
  - The table mixes are sha256 `732e487b…` (Qwen) and `2bac0a1d…` (Inkling). Weights aren't committed.

## Files

- `table-mix/tables.patch`, `table-mix/Dockerfile`: the table dump.
- `table-mix/solve.py`: the solver.
- `table-mix/convert_mix.sh`: the conversion.
- `table-mix/roles.py`: the per-role comparison.
- `table-mix/bench_mix.py`: the benchmark configs.
- `table-mix/results/tables/*.tsv.gz`: the error tables. Columns: tensor, type, bytes, error A, error B, rows sampled, elements, has imatrix.
- `table-mix/results/mixes/`: both type maps and the per-role and per-layer comparisons.
- `table-mix/results/*.json`, `*.scores.bin`: scores and benchmarks. Failed and aborted runs sit in their own directories.
