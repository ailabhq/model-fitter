# Running Inkling's NVFP4 experts on Apple GPUs

Ticket: [#14](https://github.com/ailabhq/model-fitter/issues/14). Researched 2026-11-23.
Terms follow [GLOSSARY.md](../GLOSSARY.md). `[INFERENCE]` marks a conclusion of this note that no source states directly. Every other claim links to the source that owns it, or to a measurement made for this note (marked **measured here**).

## Answer

- **Metal NVFP4 has landed in llama.cpp, so requantizing is no longer required.** [ggml-org/llama.cpp#32825][pr32825] ("metal : add NVFP4 support", by ggerganov) was merged on 2026-11-23 as [`51831d47d`][merge]. It adds NVFP4 mat-vec, `MUL_MAT_ID`, mat-mat, `get_rows` and `cpy` kernels and removes NVFP4 from the Metal exclusion list. The benchmarked build `b6d5ce4` is an ancestor of the merge, so it predates the support. **No release contains it yet**: the newest tag, `b12651`, predates the merge ([releases][releases]). A Fit must build llama.cpp from `master` at or after `51831d47d` until a release ships it.
- **Measured here on the reference M4** (`51831d47d`, `test-backend-ops`): all NVFP4 Metal tests pass (MUL_MAT 52/52, MUL_MAT_ID 73/73, GET_ROWS). NVFP4 runs as fast as the other 4-bit Metal types. Decode-shaped mat-vec: NVFP4 1.12 ms, Q4_K 1.06–1.11, MXFP4 1.20, IQ4_NL 1.35. Prefill-shaped mat-mat: 1.40 TFLOPS, against MXFP4 1.41 and Q4_K 1.32. Keeping the native NVFP4 experts on a new build therefore costs **no extra size and no Quality loss**. End-to-end Inkling tok/s on the new build is not measured; downloading weights was out of scope.
- **MXFP4 has had Metal kernels since 2025-08-05** ([#15091 gpt-oss][pr15091]). It is no help here, because converting NVFP4 to MXFP4 loses accuracy (see below).
- **If a Fit still has to requantize** (an older llama.cpp, or a tighter size budget), the row length restricts the targets. `ffn_down_exps` rows are **1152 wide, which is not a multiple of 256**, so K-quants and 256-block i-quants cannot be used for them. The block-32 types remain: IQ4_NL, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, MXFP4. The 2304-wide `ffn_gate_up_exps` rows can also take Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3 types. All of these have Metal kernels.
- **Cost of requantizing an already-4-bit source:**
  - **Q8_0 is near-free in Quality loss, but costs +89 % size.** The added error is about 1/16 of NVFP4's own error (synthetic, measured here). The down-projections alone grow from 3.56 to 6.72 GiB; all 20 NVFP4 layers grow from 10.68 to 20.17 GiB.
  - **A 4-bit target (IQ4_NL, Q4_K, Q4_0) costs no size, but adds about as much error against the Source as quantizing a BF16 model to that type would.** The only published measurement for Inkling: ggml-org's IQ4_NL (NVFP4 experts → IQ4_NL with imatrix, everything else Q8_0) has **KLD 0.0095 and 95.5 % same top token**, against **0.0030 / 97.6 %** for its all-Q8_0 file. Both are measured against the BF16 upcast ([ggml-org card][ggml-ink]).
  - **MXFP4 is the worst 4-bit target.** Its power-of-two scales "severely degrade accuracy" ([Egiazarian et al.][mrgptq]). In the synthetic test it adds 1.6× IQ4_NL's error.
  - unsloth keeps native NVFP4 down-projections in every quant checked (UD-Q2_K_XL through UD-Q4_K_XL). In UD-Q4_K_XL, all NVFP4 experts stay native (headers read here and in the [Baseline][baseline]).
- **MLX runs NVFP4 natively on Metal.** `mode="nvfp4"` (group 16, E4M3 scale) has Metal kernels and takes an optional float32 `global_scale` ([mx.quantize][mlx-modes], [fp_quantized.metal][mlx-metal]). `[INFERENCE]` That is why oMLX could run Inkling's experts on the GPU without requantizing them.

## Upstream status (llama.cpp)

| Item | State | Source |
|---|---|---|
| Metal NVFP4 in the benchmarked build `b6d5ce4` | `ggml_metal_supports_mul_mat_op` returns false for `F8_E4M3`, `NVFP4`, `TQ1_0`; `GET_ROWS` also excludes NVFP4 | [ggml-metal-device.m L1528–1537][old-excl], [L1992][old-getrows] |
| [#20456][pr20456] "metal : add NVFP4 quantization support" (third-party) | **Still open, never merged.** Opened 2026-03-12 and approved by ggerganov the same day; last activity 2026-06-23. Claims ~11× pp / ~25× tg over CPU. Superseded in effect by #32825 `[INFERENCE]` | PR page |
| [#32825][pr32825] "metal : add NVFP4 support" (ggerganov) | **Merged 2026-11-23** as [`51831d47d`][merge]. "Follow the MXFP4 path": UE4M3 scale decode, NVFP4 dequantizers, a mat-vec kernel with its `MUL_MAT_ID`, small-batch and few-row MMA variants, `mul_mm`/`mul_mm_id` and tensor-op variants, `get_rows` and `cpy`. Tested by the author on an M5 Max; the second commit also cites an M2 Ultra. A reviewer reports "the same PPL/KLD as the CPU at every batch size" for a dense NVFP4 model on an M5 Pro | [PR][pr32825], [commit][merge] |
| Metal exclusion list after the merge | Only `F8_E4M3` and `TQ1_0` remain excluded | [ggml-metal-device.m L1528–1550][new-excl] |
| Release containing it | **None yet** as of 2026-11-23. The newest release is `b12651`; `git describe` places the merge at `b12653-19-g51831d47d` | [releases][releases] |
| MXFP4 on Metal | Present since 2025-08-05 (gpt-oss) | [#15091][pr15091] |

The Inkling graph applies the per-expert NVFP4 outer scale as a separate F32 `ggml_mul` after `ggml_mul_mat_id` ([llama-graph.cpp L1898–1915][graph-ws]), which Metal already supported. unsloth's GGUFs carry these as `blk.N.ffn_*_exps.scale` F32 [128] tensors (headers read here). `[INFERENCE]` No other op stands between the new kernels and full GPU execution of the experts.

**Measured here (reference M4, 10-core GPU, `MTLGPUFamilyApple9`, llama.cpp `51831d47d`):**

| `test-backend-ops` | Result |
|---|---|
| `test -b MTL0 -o MUL_MAT -p type_a=nvfp4` | 52/52 passed |
| `test -b MTL0 -o MUL_MAT_ID -p type_a=nvfp4` | 73/73 passed |
| `test -b MTL0 -o GET_ROWS -p type=nvfp4` | passed |
| `perf -o MUL_MAT`, m=4096, k=14336, **n=1** (decode) | NVFP4 1120 µs · Q4_0 1002 · Q4_K 1057/1114 · MXFP4 1203 · IQ4_NL 1350 · IQ4_XS 1367 · Q8_0 2105 |
| `perf -o MUL_MAT`, m=4096, k=14336, **n=512** (prefill) | NVFP4 1.40 TFLOPS · MXFP4 1.41 · Q4_K 1.32 · Q8_0 1.31 · IQ4_NL 1.21 |

`perf` repeats one matrix, so absolute numbers overstate cache effects. Only the ratios between types are meaningful.

`[INFERENCE]` **Residency caveat for the 16 GB Mac.** In the Baseline, UD-IQ3_XXS projected 8,759 MiB on the GPU and left the NVFP4 down-projections (3,645 MiB) CPU-mapped, against 12,123 MiB free ([baseline][baseline]). Moving all of them onto the GPU needs ~12,404 MiB, which is slightly over that figure and over `--fit`'s 1 GiB margin. `--fit` may therefore keep one or two layers' experts on the CPU even on the new build. This is far better than all 20, but it needs measuring.

## Inkling's NVFP4 tensors, as stored

From the GGUF headers (tensor-info section only, read over HTTP ranges, no weights):

| File | `ffn_down_exps` layers 0–19 | `ffn_gate_up_exps` layers 0–19 | Shapes (ggml order) |
|---|---|---|---|
| [unsloth UD-IQ3_XXS][unsloth-tree] | **NVFP4** 3.56 GiB | IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS | down `[1152, 2304, 128]`, gate_up `[2304, 2304, 128]`; per-expert `.scale` F32 `[128]` |
| [unsloth UD-Q4_K_XL][unsloth-tree] | **NVFP4** 3.56 GiB | **NVFP4** 7.12 GiB | same |
| [ggml-org IQ4_NL][ggml-ink] | IQ4_NL 3.56 GiB | IQ4_NL 7.12 GiB | same; no `.scale` tensors (the converter folds the outer scale in) |

- The source stores NVFP4 as E2M1 values with an E4M3 scale per 16 weights, plus a per-expert `amax / outer_scale_denominator` factor. llama.cpp's converter multiplies all three out to float ([conversion/inkling2.py L33–65][conv]).
- ggml's `block_nvfp4` holds 64 weights in 36 bytes (4.5 bpw): four UE4M3 sub-block scales and 32 bytes of packed E2M1. `block_mxfp4` holds 32 weights in 17 bytes (4.25 bpw) with one E8M0 scale ([ggml-common.h L214–227][blocks]).
- The NVFP4 experts total 20 layers × 3 × 128 × 2304 × 1152 = 20.4 G weights. Of these, 6.8 G are down-projections ([config.json][ink-config]).

## Which Metal types a Fit could requantize to

`llama-quantize` demotes a type whose block does not divide the row length, logging "falling back": IQ4_XS/IQ3_*/IQ2_* → IQ4_NL, Q2_K/Q3_K → Q4_0, Q4_K → Q5_0, Q5_K → Q5_1, Q6_K → Q8_0 ([llama-quant.cpp L445–494][fallback]). For 1152-wide down-projection rows, any K-quant request silently becomes a block-32 type.

Sizes are computed from block sizes and the weight counts above. Added error comes from a **synthetic** probe (measured here; [method](#synthetic-requantization-probe)). It is the relative RMS distance from the NVFP4 values (the Source) after requantizing, with no imatrix.

| Target | bpw | Metal (`b6d5ce4`) | Allowed for down (1152) | Down, 20 layers | All NVFP4 experts | Added error vs NVFP4 Source (synthetic) | Same type from Gaussian "BF16" (synthetic) |
|---|---|---|---|---|---|---|---|
| **NVFP4 (keep)** | 4.50 | ❌ (✅ from `51831d47d`) | ✅ | 3.56 GiB | 10.68 GiB | **0** | — |
| MXFP4 | 4.25 | ✅ | ✅ | 3.36 | 10.09 | 0.117 | 0.115 |
| IQ4_XS | 4.25 | ✅ | ❌ → IQ4_NL | — | 6.72 (gate_up only) | 0.075 | 0.077 |
| IQ4_NL | 4.50 | ✅ | ✅ | 3.56 | 10.68 | 0.074 | 0.076 |
| Q4_0 | 4.50 | ✅ | ✅ | 3.56 | 10.68 | 0.086 | 0.086 |
| Q4_K | 4.50 | ✅ | ❌ → Q5_0 | — | 7.12 (gate_up only) | 0.070 | 0.071 |
| Q5_0 | 5.50 | ✅ | ✅ | 4.35 | 13.05 | 0.041 | 0.043 |
| Q5_K | 5.50 | ✅ | ❌ → Q5_1 | — | 8.70 (gate_up only) | 0.035 | 0.036 |
| Q6_K | 6.56 | ✅ | ❌ → Q8_0 | — | 10.38 (gate_up only) | 0.018 | 0.018 |
| Q8_0 | 8.50 | ✅ | ✅ | 6.72 | 20.17 | **0.006** | 0.005 |
| IQ3_XXS / Q3_K | 3.06 / 3.44 | ✅ | ❌ → IQ4_NL / Q4_0 | — | gate_up only | 0.215 / 0.148 | 0.213 / 0.151 |

Reading the synthetic columns `[INFERENCE]`:

- NVFP4's own RTN error on the same data is 0.095. Measured against the Source (which is what Quality loss measures for Inkling), a 4-bit requantization of an NVFP4 tensor costs **about the same as quantizing a BF16 tensor to that type**: the two right-hand columns match to within ±0.004. There is no extra penalty for an "already-4-bit" source relative to that Source. The penalty is relative to the unpublished pre-NVFP4 weights: 0.12–0.15 instead of 0.095.
- Gaussian weights without imatrix are a toy. Real expert weights have outliers, and imatrix shifts the 4-bit numbers. Use the columns only to rank targets; use measured KLD (below) for magnitude.
- No 4-bit requantization preserves NVFP4 exactly. MXFP4 cannot represent the E4M3 scale mantissas or two different 16-weight scales inside one 32-block. IQ4_NL, Q4_0 and Q4_K use different value grids. Q8_0 is not exact either, but its step (block max/127) is far below NVFP4's (sub-block max/3 near the top of the E2M1 grid).

## Published Quality-loss evidence for requantized Inkling experts

| Build | What happens to the NVFP4 experts | Size | Mean KLD | Same top token | Reference / corpus |
|---|---|---|---|---|---|
| ggml-org Q8_0 | → Q8_0 (as is every other matrix) | 27.8 GB | 0.0030 | 97.6 % | BF16 upcast, wikitext-2 40×512 ([card][ggml-ink]) |
| ggml-org IQ4_NL | → **IQ4_NL with imatrix**; everything else Q8_0 | 17.6 GB | **0.0095** | **95.5 %** | same |
| unsloth UD-Q4_K_XL | **kept NVFP4**; MXFP8 experts → Q8_0 | 17.63 GB | 0.0107 | 98.0 % | UD-Q8_K_XL, seven domains ([card][unsloth-card]) |
| unsloth UD-IQ3_XXS | down kept NVFP4; gate_up → IQ2/IQ3 | 11.66 GB | 0.0269 | 94.7 % | same |

- `[INFERENCE]` The ggml-org pair isolates the experts best. Going from Q8_0 to IQ4_NL on the NVFP4 experts alone adds roughly 0.0065 KLD and costs 2.1 points of top-token agreement, while saving 10.2 GB. The two producers' rows cannot be compared with each other, because their references and corpora differ.
- unsloth's reference "keeps the original NVFP4 expert weights", and its guide asks for a build "with … per-expert NVFP4 scales" ([unsloth guide][unsloth-guide]). In unsloth's own recipe, native NVFP4 beats any requantization at the same size.
- llama.cpp's own warning on requantizing: `--allow-requantize` "can severely reduce quality compared to quantizing from 16bit or 32bit" ([quantize README L54][rq-readme]); without the flag, a quantized input is refused ([llama-quant.cpp L3627][rq-code]).
- **No primary source** measures NVFP4 → MXFP4, NVFP4 → Q4_K or NVFP4 → Q8_0 on Inkling specifically. The closest published format comparison finds that "MXFP4's power-of-two scale quantization severely degrades accuracy due to high induced error" ([Egiazarian et al., ICLR 2026][mrgptq]).

## MLX

- `mx.quantize` modes: `affine` (1–8 bits, group 32/64/128), `mxfp4` (group 32, E8M0), `mxfp8` (group 32, E8M0), `nvfp4` (group 16, E4M3). `nvfp4` takes an optional float32 `global_scale` ([ops.cpp L4856–4890][mlx-modes]). For `gather_qmm`, the MoE path, `global_scale` is "Only supported on the GPU" ([ops.cpp L5030–5031][mlx-gqmm]).
- The NVFP4 Metal kernels (`qmm`, `qmv_fast`, with and without a global scale) are instantiated in [fp_quantized.metal L74–75, L104–107, L255–261][mlx-metal].
- mlx-vlm, mlx-lm and Ollama's MLX runner keep Inkling's pre-quantized NVFP4/MXFP8 experts as they are ([runtime research][runtime]). In MLX the experts were already on the GPU, which matches the oMLX ~6× decode gap in the Baseline.

## What this means for a Fit plan `[INFERENCE]`

1. **Default:** keep NVFP4 experts native and require llama.cpp ≥ `51831d47d`. Pin a release tag once one exists. This costs nothing in size or Quality loss, and removes the CPU and SSD path behind the 3.1 tok/s.
2. **Requantize only to buy size, never for kernel support.** For down-projections, IQ4_NL (same size, imatrix) is the published option; Q5_0 (+0.8 GiB) or Q8_0 (+3.2 GiB) are the low-loss options. For gate_up, Q4_K/IQ4_XS go down to 4.25–4.5 bpw. Avoid MXFP4 as a target.
3. Requantizing an existing GGUF in place needs `--allow-requantize`. It reads the whole input and writes a full output, so it needs both on disk: ~2 × 11.7 GB for UD-IQ3_XXS. That is above the ~21 GB of free disk ([low-bit research][lowbit]).
4. Quality loss of any requantized Fit has to be measured against the NVFP4 Source, or the BF16 upcast, as the Quality-loss research recommends.

## Open questions

- Inkling tok/s and residency at 8K on the reference Mac with llama.cpp ≥ `51831d47d` (UD-IQ3_XXS; also UD-Q4_K_XL, now that NVFP4 is GPU-capable).
- Whether `--fit` places all 20 NVFP4 down-projections on the GPU within the 11.84 GiB default working set.
- Measured KLD for NVFP4 → Q5_0/Q8_0 down-projections on Inkling. No one has published it.

## Synthetic requantization probe

Measured here with ggml's own quantizers (libggml at `51831d47d`, CPU). Method: 512 rows of N(0, 1) weights, at the row lengths 1152 and 2304. Quantize to NVFP4 and dequantize: this is the stand-in Source. Then requantize that tensor to each target and dequantize again. Report the relative RMS error against the NVFP4 tensor ("added error"). As a control, also quantize the Gaussian weights directly to each target. No imatrix. Row length 1152 gives the same numbers for every type that accepts it. The probe source is not committed, because it is throwaway.

[pr32825]: https://github.com/ggml-org/llama.cpp/pull/32825
[merge]: https://github.com/ggml-org/llama.cpp/commit/51831d47d0a47b8a7d07c9ff262ebe4b0a211790
[pr20456]: https://github.com/ggml-org/llama.cpp/pull/20456
[pr15091]: https://github.com/ggml-org/llama.cpp/pull/15091
[releases]: https://github.com/ggml-org/llama.cpp/releases
[old-excl]: https://github.com/ggml-org/llama.cpp/blob/b6d5ce4bc9a3691d26b3207cc6c9c63b732d67e8/ggml/src/ggml-metal/ggml-metal-device.m#L1528-L1537
[old-getrows]: https://github.com/ggml-org/llama.cpp/blob/b6d5ce4bc9a3691d26b3207cc6c9c63b732d67e8/ggml/src/ggml-metal/ggml-metal-device.m#L1992
[new-excl]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/ggml/src/ggml-metal/ggml-metal-device.m#L1528-L1550
[graph-ws]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/src/llama-graph.cpp#L1898-L1915
[conv]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/conversion/inkling2.py#L33-L65
[blocks]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/ggml/src/ggml-common.h#L214-L227
[fallback]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/src/llama-quant.cpp#L445-L494
[rq-readme]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/tools/quantize/README.md?plain=1#L54
[rq-code]: https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/src/llama-quant.cpp#L3627
[ggml-ink]: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/blob/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9/README.md
[unsloth-card]: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/8078d149286eec7193795413b9e0643da6d11fa1/README.md
[unsloth-tree]: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1
[unsloth-guide]: https://unsloth.ai/docs/models/inkling-2-mini
[ink-config]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/config.json
[mrgptq]: https://arxiv.org/abs/2509.23202
[mlx-modes]: https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/python/src/ops.cpp#L4856-L4890
[mlx-gqmm]: https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/python/src/ops.cpp#L5030-L5031
[mlx-metal]: https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/mlx/backend/metal/kernels/fp_quantized.metal
[baseline]: https://github.com/ailabhq/model-fitter/blob/research/baseline-benchmarks/research/baseline-benchmarks.md
[runtime]: https://github.com/ailabhq/model-fitter/blob/research/runtime-support/research/runtime-support.md
[lowbit]: https://github.com/ailabhq/model-fitter/blob/research/low-bit-quantization/research/low-bit-quantization.md
