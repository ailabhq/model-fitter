# Can sarto's converter keep a Source's NVFP4 tensors bit-exact, with their per-tensor scale, on the pinned llama.cpp?

Map ticket: "Can sarto's converter keep a Source's NVFP4 tensors bit-exact, with their per-tensor scale, on the pinned llama.cpp?" (map: "Map: sarto, Usable Fits of new models on small Apple Silicon Macs"). This branch is never merged.

## Answer

**Yes. The pinned runtime already reads and applies the `.scale` tensors. Only the pinned Inkling converter drops them, and a per-tensor copy closes that gap.** The copy is a lossless byte shuffle of the Source's packed E2M1 values and E4M3 block scales into ggml's `block_nvfp4`, plus one F32 `.scale` per expert holding `amax / outer_scale_denominator`.

- **Checked on 6 experts by HTTP range reads** (layers 0 and 19, `w2` and `w13`; [`copy_check.out`](nvfp4-copy/copy_check.out)):
  - the mapping gives the same bytes as Unsloth's published file and as the pin's own ModelOpt repacker;
  - Unsloth's `.scale` equals `fp32(amax / denominator)` for all 40 NVFP4 tensors × 128 experts;
  - ggml's dequantization at the pin (C and gguf-py), times `.scale`, equals the pinned converter's reference dequantization value for value.
- **The one bitwise difference is the sign of zero.** E2M1 code `0b1000` is −0 in the reference and +0 in ggml. That is about 3.4% of values per expert, and it has no effect on any product or sum.
- **A kept NVFP4 tensor costs nothing beyond the Source.** It takes 4.5 bpw, the same bytes as the Source's packing, plus 512 bytes of `.scale` per tensor. Its added error against the Source is 0.
- **The error tables are wrong for these tensors.** They charge NVFP4 a requantization error (`ffn_down` 20,125, `gate_up` 36,559, against IQ4_NL's 29,878 and 30,345), where a copy's is 0.
- **No llama.cpp patch and no newer pin are needed.** What's needed is a converter step: sarto's own, or the opt-in path in Unsloth's open llama.cpp PR, which isn't in the pin or in master.

### 1. The pinned loader, graph and Metal kernels read `.scale`

- **Loader.** After each architecture's own tensors, a generic pass loads optional `<weight>.scale` tensors. For MoE experts the shape is `{n_expert}`: `ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`, and `ffn_gate_up_exps` as `ffn_up_exps_s` ([llama-model.cpp L1943–2013][loader]). Every `X.weight` with an `X.scale` of `ne[0] == weight->ne[2]` is then mapped to it in `w_scales` ([L2136–2161][wmap]).
- **Graph.** `build_lora_mm_id` looks the weight up in `w_scales` and multiplies the `mul_mat_id` result by the selected experts' scales: `ggml_get_rows` on the scale vector, then `ggml_mul` ([llama-graph.cpp L1924–1952][mmid]). `build_moe_ffn` calls it for `gate_up_exps` and `down_exps` ([L2567][gu], [L2701][dn]), and Inkling's graph passes `ffn_gate_up_exps` and `ffn_down_exps` to `build_moe_ffn` ([inkling2.cpp L393–406][ink-graph]).
- **History.**
  - Per-expert scales for fused `gate_up` came from Unsloth ([#32657][pr32657], `a1f422260`, 2026-11-20).
  - Applying every `.scale` automatically came in [#32669][pr32669] (`19bc41dbb`, 2026-11-21).
  - Both are ancestors of `22b99122e`, and so is Metal NVFP4 ([#32825][pr32825], `51831d47d`).
- **Metal.** NVFP4 is off the Metal exclusion list at the pin, which leaves only `F8_E4M3` and `TQ1_0` there ([ggml-metal-device.m L1528–1537, L1983, L1998][metal]). The scale step uses only F32 `get_rows`/`repeat`/`mul`.
- **The pinned Metal build already runs a file with these scales.** Unsloth UD-IQ3_XXS, which carries the 40 `.scale` tensors, scored KLD 0.020 / 95.4% top-1 on the M4 at the pin ([#47][i47]). Unsloth reports perplexity "about 130 instead of 7.65" when the scales are lost ([Mainline_Support README][ureadme]).
- **Runtime arithmetic, not stored values.** The scale is applied after the dot product (`(Σ qᵢxᵢ)·s`, not `Σ (qᵢs)xᵢ`). NVFP4 `mul_mat_id` also allows BF16 accumulation ([llama-graph.cpp L1938–1940][mmid]), and the CPU path quantizes activations to Q8_0 ([ggml-cpu.c L296–301][cpu]). These are runtime choices shared by every Fit. `[INFERENCE]` They are Quality loss the Reference pack measures, not lost bits in the file.

### 2. How Unsloth's file was produced, and what's in the pin

- **The pinned Inkling converter dequantizes.** `Inkling2Model.dequant_model` turns each packed expert into F32 chunks with `_dequant_expert_chunk`: E2M1 × E4M3 × `amax/denominator` ([conversion/inkling2.py L18–65, L116–180][ink-conv]). Then it drops the companions. It writes no `.scale`.
- **The pin has a ModelOpt NVFP4 passthrough, but it never fires for Inkling.**
  - `ModelBase._nvfp4_pack` repacks ModelOpt bytes into `block_nvfp4` ("Preserves original E4M3 scale bits as UE4M3"). `_flush_nvfp4_experts` writes a `.scale` tensor per expert stack ([conversion/base.py L1309–1332, L1426–1444][base-pack]).
  - Inkling's `config.json` does say `quant_method: modelopt, quant_algo: NVFP4`, so `_generate_nvfp4_tensors` runs ([L1521–1544][base-prep]).
  - It only matches `X.weight` with `X.weight_scale` / `X.weight_scale_2` ([L1356–1361][base-match]). Inkling's tensors are `experts.w2_weight`, `w2_weight_scale`, `w2_weight_amax`, `w2_weight_outer_scale_denominator` ([Source index][src-index]), so nothing matches. The experts fall through to the dequantizing path.
- **Unsloth's converter is an opt-in path in its own open PR.** [#32695][pr32695] by danielhanchen (open, head `6196d980`) adds `INKLING2_NVFP4_NATIVE=1`. It "keeps NVFP4 experts as GGML NVFP4 with a per-expert scale": `_nvfp4_expert_rows` is the same nibble/scale shuffle, gate/up rows are de-interleaved, and `a / d` is written as `.scale` ([fork inkling2.py L69–89, L207–216, L342–347][fork]).
  - The earlier open PR [#32187][pr32187] (head `a45f6626`) has the same in `nvfp4_to_ggml`, "bit-exact".
  - Neither is in `22b99122e` or in master at `a866b4ad`, whose `conversion/inkling2.py` history has no such commit.
  - `[INFERENCE]` Unsloth built its root files with this path, then `llama-quantize`. Its public fork's `master` has no `conversion/inkling2.py`, so the exact build can't be named. The bytes match this path exactly (below).
- **Unsloth's mainline files are the root files with a different layout.** The `Mainline_Support` files "hold the same tensors as the root … GGUFs, byte for byte", with only metadata and names changed ([README][ureadme]). The UD-IQ3_XXS header has 20 NVFP4 `ffn_down_exps` (layers 0–19) and 40 F32 `[128]` `.scale` tensors. One of these is on every layer 0–19 `ffn_gate_up_exps`, even where that tensor is IQ2/IQ3 ([`unsloth_header.out`](nvfp4-copy/unsloth_header.out)). Its metadata names no converter; only `quantize.imatrix.*` keys are set.
- **The pinned `llama-quantize` keeps a copied NVFP4 tensor as it is.** A tensor whose target type equals its current type is copied in row slabs without requantizing ([llama-quant.cpp L3563–3592][q-copy]). `.scale` is 1-D and isn't a `weight`, so it is never quantized ([L291–297][q-which]). A pipeline "copy NVFP4 + `.scale` into the input GGUF, then quantize the rest" therefore works at the pin, as long as these tensors are named NVFP4. Any other type needs `--allow-requantize` ([L3626][q-req]).

### 3. The layout mapping is exact

| | Source (ModelOpt-style, `nvfp4_nearest`, `amax_denominator`) | ggml `block_nvfp4` at the pin |
|---|---|---|
| Values | `uint8 [E, rows, cols/2]`: element 2i in the low nibble, 2i+1 in the high | 32 bytes per 64 values; in each 16-value sub-block, byte j holds element j (low) and j+8 (high) ([ggml-quants.c L592–612][deq]) |
| Block scale | `float8_e4m3fn [E, rows, cols/16]` | 4 `uint8` UE4M3 per block, first in the block ([ggml-common.h L221–227][blk]) |
| Global scale | `amax F32 [E]` / `outer_scale_denominator F32 [1]`: one per expert | none in the block; F32 `.scale [n_expert]` beside the tensor |
| `w13` | gate and up rows interleaved (row 2i gate, 2i+1 up) | gate rows, then up rows (`ffn_gate_up_exps`; the graph splits at `n_ff`) |

- **Values and block scales** map by a nibble re-pairing inside each 16-group, plus a regrouping of 4 scales and 32 bytes into 36-byte blocks. Nothing is rounded. Both rows (2304 and 1152) are multiples of 64.
- **Decoding agrees.**
  - ggml decodes UE4M3 with the same exponent/mantissa formula as the reference, subnormals (`man·2⁻⁹`) included, and halves it. It doubles E2M1 (`kvalues_fp4`) ([ggml-impl.h L519–535][ue4m3], [ggml-common.h L1124–1129][kv]).
  - Each product is exact in F32, and multiplying by the F32 `.scale` is then the same single rounding the reference does (`values *= float32(a) / float32(d)`).
  - ggml decodes `0x7F` as 0 and ignores the sign bit. In the 6 experts checked, no block scale is zero, negative, subnormal or NaN, and the largest is `0x7e` (448). The copy step should refuse a sign bit or `0x7F`, as #32695 does.
- **Fused `gate_up` is not a problem for Inkling.** `w13_weight_amax` is `F32 [128]`, so one global scale per expert covers both gate and up rows. A single `.scale [n_expert]` holds it exactly.
  - `[INFERENCE]` A Source with different gate and up global scales could still be exact if written as separate `ffn_gate_exps` / `ffn_up_exps` with their own `.scale` (the loader takes both, [L1999–2010][loader]). That only works for an architecture whose graph takes separate gate and up tensors, and Inkling's passes only `gate_up` ([inkling2.cpp L393–406][ink-graph]).
  - Folding two global scales into one fused tensor would need requantizing the block scales.
- **Folding the global scale into the block scales would not be exact.** Inkling's per-expert scales are 3.8e-5 to 3.2e-4. Multiplied into E4M3 block scales of at most 448, they land in or below UE4M3's subnormal range.
  - With ggml's own rounding, only 3.4–3.5% of values survive exactly, at relative RMS error 0.057–0.083 ([`copy_check.out`](nvfp4-copy/copy_check.out)).
  - #32268's description says the same: the scale "is too small to fold into the FP8 block scales" ([#32268][pr32268]). That's why `.scale` has to stay a separate tensor.
- `[INFERENCE]` **Metal's in-kernel dequantization stays exact.** An unscaled E2M1 × E4M3 product has at most 5 significant bits and lies between 2⁻¹⁰ and 2688, so it is exact even in F16. This holds only because the global scale isn't folded in.
- **Compressed-tensors NVFP4 Sources** go through the pin's passthrough with `weight_scale_2 = 1 / weight_global_scale` computed in F32 ([base.py L1522–1543][base-prep]). `[INFERENCE]` If such a Source's reference divides by the global scale, that reciprocal can differ by one ulp. Inkling's format doesn't have this.

### 4. What a kept NVFP4 tensor costs

A copy costs 0 added error against the Source. Here is what the table-mix error tables charge the same 40 tensors (layers 0–19) today, when NVFP4 is requantized from the dequantized F32 ([`table_cost.out`](nvfp4-copy/table_cost.out), tables from [#47][tablemix]). Error A and B are equal for these tensors.

| Role (20 tensors each) | Type | GiB | Table error |
|---|---|---|---|
| `ffn_down_exps` | **NVFP4, copied** | 3.560 (+10 KB `.scale`) | **0** |
| | NVFP4, requantized (table) | 3.560 | 20,125 |
| | IQ4_NL | 3.560 | 29,878 |
| | Q5_0 | 4.351 | 8,680 |
| | Q8_0 | 6.724 | 206 |
| `ffn_gate_up_exps` | **NVFP4, copied** | 7.119 (+10 KB `.scale`) | **0** |
| | NVFP4, requantized (table) | 7.119 | 36,559 |
| | IQ4_NL | 7.119 | 30,345 |
| | IQ4_XS | 6.724 | 31,015 |
| | Q5_0 | 8.701 | 8,491 |
| | Q8_0 | 13.447 | 197 |

- At 4.5 bpw, a copied NVFP4 tensor is the cheapest-and-exact choice for these tensors. Nothing smaller is exact, and Q8_0's 89% more bytes still leave about 200 of error.
- **The table totals in #47 change.** Unsloth UD-IQ3_XXS's table total (3.80e5) charges its copied `ffn_down` NVFP4 20,125 that it doesn't lose. `[INFERENCE]` About 3.60e5 is the real figure. The table mix's 20,125 on the same tensors was real loss, because it requantized them.
- **Minimal change, at the pin:**
  1. In sarto's streaming converter, for each Source NVFP4 expert tensor, write the shuffled bytes as GGML type 40 and `fp32(amax/denominator)` as `<name>.scale` (F32 `[n_expert]`). Refuse sign-bit or NaN block scales. This is `to_ggml` in [`copy_check.py`](nvfp4-copy/copy_check.py), about 10 lines, and it reads one expert at a time, as the streaming converter already does.
  2. In the error tables, give "NVFP4, copied" error 0 for any tensor the Source stores as NVFP4.
  3. Other types for these tensors may quantize the unscaled values and keep `.scale`, as Unsloth's IQ2/IQ3 `gate_up` do. `[INFERENCE]` That changes nothing for scale-invariant quantizers.
- **llama.cpp alternative:** #32695's `INKLING2_NVFP4_NATIVE=1` path, if it merges. A pin bump would then let the pinned converter do step 1. The runtime side needs nothing.

## Method

- llama.cpp source read at `22b99122e` (the pin), by `git show`. PR metadata from GitHub.
- [`copy_check.py`](nvfp4-copy/copy_check.py), about 60 MB of HTTP range reads and no weights on disk:
  - Source `thinkingmachines/Inkling-2-Mini` at `c9d02830`. The safetensors blob is sha256 `65e60f51…`, the same at `7a67ae23`.
  - Unsloth `Mainline_Support` UD-Q4_K_XL (all 40 tensors NVFP4) and UD-IQ3_XXS at `8078d149`.
  - For 6 experts it repacks the Source bytes and compares them with Unsloth's bytes and with the pin's `ModelBase._nvfp4_pack`, lifted from `conversion/base.py` by its AST (importing the module needs `transformers`).
  - It then dequantizes with `dequantize_row_nvfp4` from `libggml-base` built at the pin and with gguf-py's `NVFP4`, multiplies by `.scale`, and compares bitwise with `_dequant_expert_chunk`'s formula.
  - It also compares all 40 `.scale` tensors in both Unsloth files with `fp32(amax/denominator)`.
  - Run: `LLAMA_CPP=<llama.cpp at 22b99122e> LIBGGML_BASE=<its libggml-base> python3 copy_check.py`.
- [`unsloth_header.py`](nvfp4-copy/unsloth_header.py): UD-IQ3_XXS's header by range, uses [`gguf_header.py`](nvfp4-copy/gguf_header.py) from the streaming-converter branch.
- [`table_cost.py`](nvfp4-copy/table_cost.py): sums the table-mix error tables over the 40 tensors.
- Nothing heavy ran on the Mac. The Mac's only part is the CPU dequantization of 6 experts.

## Files

- `nvfp4-copy/copy_check.py`, `copy_check.out`: the copy and its checks.
- `nvfp4-copy/unsloth_header.py`, `unsloth_header.out`, `gguf_header.py`: Unsloth's NVFP4 and `.scale` tensors and metadata (the chat template is left out).
- `nvfp4-copy/table_cost.py`, `table_cost.out`: table error and bytes per candidate type.

[loader]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-model.cpp#L1943-L2013
[wmap]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-model.cpp#L2136-L2161
[mmid]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-graph.cpp#L1924-L1952
[gu]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-graph.cpp#L2567
[dn]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-graph.cpp#L2701
[ink-graph]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/models/inkling2.cpp#L393-L406
[metal]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-metal/ggml-metal-device.m#L1528-L1537
[cpu]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-cpu/ggml-cpu.c#L296-L301
[ink-conv]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/conversion/inkling2.py#L18-L180
[base-pack]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/conversion/base.py#L1309-L1444
[base-match]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/conversion/base.py#L1356-L1361
[base-prep]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/conversion/base.py#L1521-L1544
[q-copy]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-quant.cpp#L3563-L3592
[q-which]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-quant.cpp#L291-L297
[q-req]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/src/llama-quant.cpp#L3626
[deq]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-quants.c#L592-L612
[blk]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-common.h#L221-L227
[kv]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-common.h#L1124-L1129
[ue4m3]: https://github.com/ggml-org/llama.cpp/blob/22b99122e1dfe1d8a5175bfa2eaa8389699c0b92/ggml/src/ggml-impl.h#L519-L535
[pr32657]: https://github.com/ggml-org/llama.cpp/pull/32657
[pr32669]: https://github.com/ggml-org/llama.cpp/pull/32669
[pr32825]: https://github.com/ggml-org/llama.cpp/pull/32825
[pr32695]: https://github.com/ggml-org/llama.cpp/pull/32695
[pr32187]: https://github.com/ggml-org/llama.cpp/pull/32187
[pr32268]: https://github.com/ggml-org/llama.cpp/pull/32268
[fork]: https://github.com/danielhanchen/llama.cpp/blob/6196d9802e0757a5218f16f91d9a6830793ba800/conversion/inkling2.py#L69-L89
[ureadme]: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/8078d149286eec7193795413b9e0643da6d11fa1/Mainline_Support/README.md
[src-index]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/c9d028302cd0f035fdf8e7e91f089a4ce94d2195/model.safetensors.index.json
[i47]: https://github.com/ailabhq/sarto/issues/47
[tablemix]: https://github.com/ailabhq/sarto/blob/research/table-mix/research/table-mix.md
