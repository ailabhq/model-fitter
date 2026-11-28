# Is ggml-org's BF16 upcast of Inkling 2 Mini value-exact to its NVFP4/MXFP8 Source?

Map ticket: "Is ggml-org's BF16 upcast of Inkling 2 Mini value-exact to its NVFP4/MXFP8 Source, and if not, can an exact one be written?" (map: "Map: Source-precision Fits, the Source's own weights run from disk for overnight batches"). This branch is never merged.

The prior reading of the converter, in "Which runtimes can run Qwen3.8-27B at BF16 and Inkling 2 Mini at native NVFP4/MXFP8...", said the NVFP4 part of the upcast is not bit-exact because the converter folds the FP32 per-expert scale in before the BF16 rounding. That was marked `[INFERENCE]`. This note measures it.

## Answer

**The MXFP8 experts are value-exact in the BF16 file. The NVFP4 experts are not.** Every stored BF16 bit of a sampled NVFP4 weight is the round-to-nearest-even image of the f32 value `E2M1 x E4M3 x fp32(amax / denominator)`. The scale is in the product before the rounding. An f32 file of that same product keeps every expert value, because the BF16 file is the rounding of it. A native NVFP4 repack is the smaller file. On the four whole NVFP4 experts the repacked bytes equal ggml-org's published NVFP4 tensor, and all 40 scales match `fp32(amax / denominator)`. [Issue #49](https://github.com/ailabhq/sarto/issues/49) measured that ggml's decoder of that layout, times the scale, matches this reference except the sign of zero, which does not change a product. The MXFP8 experts, widened to BF16, were exact in this run.

Headline counts, from [`compare.out`](bf16-upcast/compare.out). `numeric_diff` counts values whose f32 differs, with +0 and -0 treated as the same value. `rne_match` counts stored BF16 bits that equal round-to-nearest-even of the reference f32. `max_ulp` is the largest absolute error divided by the BF16 unit in the last place of that value's binade. Half an ulp is 0.5.

| Sample | Values | Numeric diffs | RNE match | max abs | mean abs | max rel | mean rel | max ulp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4, 4 whole experts | 15,925,248 | 14,832,692 (93.139%) | 15,925,248 / 15,925,248 | 0.000809133 | 2.33899e-05 | 0.00343821 | 0.00126015 | 0.472458 |
| NVFP4, every layer, 3 experts, 2-3 rows | 552,960 | 515,012 (93.137%) | 552,960 / 552,960 | 0.000470281 | 2.52711e-05 | 0.00376565 | 0.00138947 | 0.499939 |
| MXFP8, 4 whole experts | 15,925,248 | 0 | 15,925,248 / 15,925,248 | 0 | 0 | 0 | 0 | 0 |
| MXFP8, every layer, same row grid | 110,592 | 0 | 110,592 / 110,592 | 0 | 0 | 0 | 0 | 0 |

The two percents are 14,832,692 / 15,925,248 and 515,012 / 552,960, printed to three decimal places. The error columns are the script's printed values. No sampled value differed only in the sign of zero. No sample produced an infinity. The relative-error bucket above 2^-8 is empty for every NVFP4 row in the output.

### What the differences are

The reference is the pinned converter's own dequantization, `conversion/inkling2.py` `_dequant_expert_chunk` at `22b99122e`. For NVFP4 it multiplies `fp32(amax) / fp32(denominator)` into the block product and returns f32. For MXFP8 it multiplies E4M3 by `2^(UE8M0 - 127)` and returns f32. There is no per-expert scale on MXFP8.

Round-to-nearest-even BF16 is `gguf-py` `BF16.quantize_blocks` at the same commit, which is `ggml_compute_fp32_to_bf16`. On both samples, the BF16 file matches that rounding on every value.

The fold is visible in a second comparison. Rounding the NVFP4 product before the per-expert scale matches the stored bits only on the zeros: 1,092,556 / 15,925,248 in the whole-expert sample, and 37,865 / 552,960 in the grid. Those two counts are the zero counts. Every nonzero stored BF16 differs from the unscaled product. In the four whole experts, the 1,092,556 exact values are the zeros. The grid's exact bucket is 37,948 against 37,865 zeros, so 83 nonzero values land on a BF16 number after the scale multiply. All 83 are in the layer 3 `w2` grid rows (exact 518, zeros 435). Every other NVFP4 grid slice has an exact count equal to its zero count. 83 / 515,095 nonzero grid values is 0.0161%.

The per-expert scales themselves, over all 40 NVFP4 tensors and all 128 experts, run from 3.77341e-05 to 0.000316986.

BF16 has 7 explicit mantissa bits. Just above a power of two, one ulp is 2^-7 of the value, so round-to-nearest-even stops at 2^-8 relative, which is 0.00390625. The largest relative error in the grid is 0.00376565, which is 0.00376565 / 0.00390625 = 0.9640064 of that bound, on the pair (-0.0078420304, -0.0078125). The largest absolute error in the whole-expert sample is 0.000809133, on (-0.46794087, -0.46875). That magnitude sits in [2^-2, 2^-1), where one BF16 ulp is 2^-9 = 0.001953125 and half an ulp is 0.0009765625. The grid's largest absolute error is 0.000470281, on (-0.12843597, -0.12890625), in [2^-3, 2^-2), where half an ulp is 0.00048828125. Both sit under half an ulp. The largest error in ulps is 0.499939, on layer 17 `w13`, a different value from either max-abs pair. The earlier inference said the error was at most 2^-9 relative. 2^-9 is the half-ulp bound at the top of a binade. The bound at the bottom of a binade is twice that, and the sample uses it. Nothing in the sample exceeds half an ulp, so the file is that rounding.

MXFP8 block-scale bytes in the four whole experts sit between 113 and 117, none of them 255. The unbiased exponents are -14 to -10. An E4M3 significand times a power of two in that range stays a normal BF16 number. The smallest subnormal E4M3 magnitude is 2^-9, times 2^-14 is 2^-23, and BF16's smallest normal is 2^-126. The four whole experts (15,925,248 values) and the grid (110,592 values) are both exact. The grid rereads rows from some of those same experts, so the two counts are not a count of distinct values.

### An exact file, and what it weighs

Sizes are decimal GB (10^9 bytes). The payload arithmetic is in [`compare.out`](bf16-upcast/compare.out). Both GGUF files pack tensors with no gap and no tail. Every one of the 704 BF16-file tensors has a logical size divisible by 32 (`0/704` not divisible), so doubling a BF16 payload stays 32-aligned and the F32 sizes add no padding.

The `path_i2` line in that print keeps the BF16 file's 11,977,408-byte prefix and swaps payloads. A writer also adds 40 tensor records. Each record is 32 bytes plus the name: a `uint64` length, the name, a `uint32` dimension count, one `uint64` dimension, a `uint32` type, and a `uint64` offset. Those 40 names total 2,360 bytes. The header grows from 11,977,383 to 11,979,743, and alignment to 32 moves the data start from 11,977,408 to 11,979,744. The file is 22,924,073,664 - 11,977,408 + 11,979,744 = 22,924,076,000 bytes (22.924076 GB, 21.349709 GiB). The 20,480 bytes of scale payload are already inside the 22,924,073,664.

| File | Bytes | GB | What it keeps |
|---|---:|---:|---|
| ggml-org `Inkling-2-Mini-BF16.gguf` | 52,226,509,504 | 52.226510 | Measured file. MXFP8 experts exact, NVFP4 experts rounded. |
| `--outtype f32` on the pinned converter | 104,293,518,016 | 104.293518 | Every BF16 payload stored as f32. Exact for the experts, because the BF16 file is the rounding of that f32. |
| F32 only on the 40 NVFP4 expert tensors | 92,995,144,384 | 92.995144 | Same BF16 file, with layers 0-19 `ffn_down_exps` and `ffn_gate_up_exps` widened. MXFP8 stays BF16. Not a stock flag. |
| Path I2, native NVFP4 plus BF16 MXFP8 | 22,924,076,000 | 22.924076 | Layers 0-19 copied as NVFP4 with a per-expert f32 scale. Layers 20-23 and the other tensors stay as in the BF16 file. Includes the 2,360-byte header growth. |
| ggml-org `Inkling-2-Mini-NVFP4.gguf` | 17,627,963,488 | 17.627963 | Layers 0-19 are the native copy. Layers 20-23 and the other matrices are Q8_0. Not exact. |

The Source safetensors file is 19,526,019,168 bytes (19.526019 GB). The pinned converter skips the vision tower and the MTP tensors. Path I2 is text tensors only, and it is still larger because the MXFP8 experts and the other matrices are stored as BF16. Of the 22,924,076,000 bytes, the MXFP8 experts are 8,153,726,976 and the other BF16 tensors are 3,144,646,656.

### What the repack takes on the pinned llama.cpp

The pin is `22b99122e` (2026-11-24). At that commit `Inkling2Model.dequant_model` always dequantizes both expert formats and writes no `.scale`. There is no preserve flag. `--outtype f32` is the stock exact path, at 104.294 GB. `--outtype bf16` is the file measured here.

The runtime at the pin already runs a native file:

- The loader reads optional `blk.{i}.ffn_{gate,down,gate_up}_exps.scale` with shape `{n_expert}` ([llama-model.cpp L1998-L2013][loader], mapped at [L2136-L2158][wmap]).
- `build_lora_mm_id` multiplies that scale after `mul_mat_id` ([llama-graph.cpp L1921-L1948][mmid]). For NVFP4 it sets the matmul accumulation to BF16 before the scale multiply.
- Inkling's graph passes `ffn_gate_up_exps` and `ffn_down_exps` through `build_moe_ffn` ([inkling2.cpp L393-L406][ink]).

The pin needs a writer. The loader, the graph, and the NVFP4 matmul already apply `.scale`. The writer copies each NVFP4 expert with the nibble shuffle in `to_ggml_nvfp4` (4 UE4M3 scales and 32 bytes per 64 values, gate rows then up rows) and writes `fp32(amax / denominator)` as `.scale`. On the four whole NVFP4 experts, that byte string equals ggml-org's published NVFP4 tensor, and all 40 scales match `fp32(amax / denominator)` bit for bit. MXFP8 has no exact ggml type at the pin. `F8_E4M3` is one byte per element and has no E8M0 block scale. Those layers are stored as the BF16 widening measured above, which is exact on the sample. The other tensors stay as the BF16 conversion already stores them. The first 8 rows of `token_embd.weight`, 36,864 bytes, match `model.llm.embed.weight` byte for byte. This note does not compare the rest of the non-expert tensors.

Two runtime facts bound whether path I2 counts as Source precision. They are properties of the pin, not new measurements of logits:

- On the CPU, NVFP4's `vec_dot_type` is Q8_0 ([ggml-cpu.c L296-L299][cpu]) and BF16's is BF16 (L404-L407). An NVFP4 expert that `--fit` leaves on the CPU is multiplied by quantized activations. Path I2 stays at Source precision only while every NVFP4 expert runs on Metal.
- ggml's NVFP4 decoder maps the E2M1 code `0b1000` (-0) to +0. [Can sarto's converter keep a Source's NVFP4 tensors bit-exact...](https://github.com/ailabhq/sarto/issues/49) measured that on this Source. It does not change a product or a sum. The BF16 file does not have this quirk: its zeros matched the reference in sign as well as in magnitude (`neg0_only = 0`, and the RNE match covers the zeros).

llama.cpp master gained the writer on 2026-11-27 in [#33062][pr33062] (`4c7344447cb1`). That commit is not in the pin. `Inkling2Model.__init__` stores `_preserve_nvfp4` when the requested file type is still `GUESSED`, before `ModelBase.__init__` rewrites `GUESSED`. The rewrite sets `MOSTLY_BF16` on the first 2D BF16 tensor it sees, and this Source's `model.llm.embed.weight` is BF16. `dequant_model` then repacks NVFP4 and writes `.scale`. MXFP8 still dequantizes. It is marked for Q8_0 only when `--fp8-as-q8` is set. Without that flag, the rewritten file type stores it as BF16. `--outtype auto` without `--fp8-as-q8` is path I2. The PR's own command for an nvfp4-plus-q8_0 file is `--outtype auto --fp8-as-q8`, then `llama-quantize` to Q8_0 while `blk.([0-9]|1[0-9]).ffn_(gate_up|down)_exps.weight` stays NVFP4. This note did not re-run that command. The measured ggml-org file has the tensor types that command produces: `general.file_type` 7 (`MOSTLY_Q8_0`), layers 0-19 NVFP4, layers 20-23 Q8_0. The four whole MXFP8 experts in that file differ from the reference on 15,925,108 of 15,925,248 values (99.999%). Mean absolute error is 0.00010992, mean relative error is 0.0248586, and the largest relative error is 1, where the reference value 1.7166138e-05 is stored as 0. Max absolute error is 0.00141907. 71 of the zeros that the numeric-diff count treats as equal differ only in sign.

## Method

- [`compare.py`](bf16-upcast/compare.py) reads by HTTP range and writes no weights. The committed [`compare.out`](bf16-upcast/compare.out) is the run that also prints the header-adjusted path I2 size. Two earlier runs of the measurement loop printed the same FULL and GRID aggregate counts. Each run fetched the embed rows twice, and both copies matched the GGUF. The committed run read 133,458,320 bytes.
- The decision this run was built to settle: the upcast is exact only when the numeric-diff count is 0. The fold hypothesis is confirmed only when every stored BF16 bit equals round-to-nearest-even of the folded f32, and the unscaled product does not. There is no timing noise. A wrong layout or a truncated range would not land on that rounding. The embed control is the check that the range reader is aimed at the right bytes.
- Source `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, file `model-00001-of-00001.safetensors` (19,526,019,168 bytes, safetensors header ends at byte 148,416). NVFP4 experts are layers 0-19, tensors `model.llm.layers.{L}.mlp.experts.{w2,w13}_weight` (U8), `_scale` (F8_E4M3), `_amax` (F32 `[128]`), `_outer_scale_denominator` (F32 `[1]`). MXFP8 experts are layers 20-23, weight F8_E4M3 and scale U8. Shapes: `w2` is `[128, 2304, 576]` packed or `[128, 2304, 1152]` for MXFP8. `w13` is `[128, 2304, 1152]` packed or `[128, 2304, 2304]` for MXFP8. The stored MXFP8 block is `(1, 32)` elements. `hf_quant_config.json` says `[32, 32]`, and the converter warns and uses the stored shape.
- ggml-org `Inkling-2-Mini-GGUF` at `0bdc580634ceac4c4d921f452777bf44228de9af`.
  - `Inkling-2-Mini-BF16.gguf`, 52,226,509,504 bytes, 704 tensors, `general.file_type` 32 (`MOSTLY_BF16`), `general.architecture` `inkling2_loop`. Expert tensors `blk.{L}.ffn_{down,gate_up}_exps.weight` are BF16 on all 24 layers. There is no `.scale` tensor.
  - `Inkling-2-Mini-NVFP4.gguf`, 17,627,963,488 bytes, 744 tensors, `general.file_type` 7 (`MOSTLY_Q8_0`). Layers 0-19 of those expert tensors are NVFP4 plus an f32 `.scale` of 128 elements. Layers 20-23 are Q8_0.
- Whole experts, every value: NVFP4 `(layer, tensor, expert)` = `(0, w2, 0)`, `(0, w13, 77)`, `(19, w2, 127)`, `(19, w13, 3)`. MXFP8 `(20, w2, 0)`, `(20, w13, 127)`, `(23, w2, 50)`, `(23, w13, 1)`. `w13` is deinterleaved to gate rows then up rows, which is the converter's chunk order. That layout is the one that matched.
- Grid: layers 0-23, both `w2` and `w13`, experts 0, 63 and 127, logical rows 0 and 1000, and row 2000 on `w13` so the up half is included.
- `mean abs` is the arithmetic mean of `|stored - ref|`. `mean rel` is the arithmetic mean of `|stored - ref| / |ref|` over nonzero reference values. It is not an RMS. `max ulp` divides the absolute error by the BF16 ulp of the reference binade, `2^(floor(log2(|ref|)) - 7)`, and uses `2^-133` when `|ref|` is below `2^-126`. A BF16 subnormal would require a magnitude below `2^-126`. The MXFP8 bound above stays at or above `2^-23`. The smallest nonzero NVFP4 block product is `0.5 * 2^-9 = 2^-10`, and the smallest measured scale is 3.77341e-05, so that product is `2^-10 * 3.77341e-05 = 3.68497e-08`.
- The positive control ran before the expert reads. `model.llm.embed.weight` against `token_embd.weight`, first 8 rows. The source range was fetched twice and both copies matched the GGUF.

## Files

- `bf16-upcast/compare.py`, `bf16-upcast/compare.out`: the range-read comparison and the run.
- `bf16-upcast/gguf_header.py`: the header reader from the NVFP4-copy note, with retries on the range read.
- `bf16-upcast/decisions.tsv`: the decision trail for this measurement.

[loader]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-model.cpp#L1998-L2013
[wmap]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-model.cpp#L2136-L2158
[mmid]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-graph.cpp#L1921-L1948
[ink]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/models/inkling2.cpp#L393-L406
[cpu]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-cpu/ggml-cpu.c#L296-L299
[pr33062]: https://github.com/ggml-org/llama.cpp/pull/33062
