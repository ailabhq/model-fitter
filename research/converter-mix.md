# How sarto's converter chooses a type for each tensor: supporting facts

Assets for [How does sarto's converter choose a type for each tensor to hit a target file size?](https://github.com/ailabhq/sarto/issues/35). This branch is never merged. llama.cpp citations are at the pin `22b99122e`.

## 1. Upstream `llama-quantize --target-size` / `--target-bpw`

- **Origin.** Added by [llama.cpp#15550](https://github.com/ggml-org/llama.cpp/pull/15550), merged 2026-11-23. It lives in a single private function, `target_bpw_type()` (`src/llama-quant.cpp:844-3071`), with no public API.
- **How it chooses.**
  1. For every quantizable tensor, it quantizes and dequantizes 256–4096 sampled rows into each of 21 candidate types (`:887-909`).
  2. It scores each type with an imatrix-weighted output-error MSE (`compute_quant_error`, `:1260-1460`). Attention Q/K use a weighted cosine error instead.
  3. It prunes each tensor's options to the Pareto front, then to the lower convex hull (`:2376-2407`).
  4. It allocates with a Lagrangian solve over μ, then a greedy fill (`:2676-2871`).
  5. On top of that sit fixed boosts: dense `ffn_down` 6×, `output` 3×, `attn_v` 1.25×, tied `token_embd` 2.5× (`:916-920`). Two role caps follow, each re-running the allocation (`:2876-3067`).
- **What it needs.**
  - An imatrix; it throws without one (`:3499`).
  - A local GGUF, read through `llama_model_loader`.
  - Two passes over the weights: sampling, then the full quantize. The header is written last.
  - Its `--dry-run` still samples, and the chosen mix is only logged.
- **Gaps.**
  - No NVFP4 or MXFP4 among its candidates.
  - The routed-expert cap matches only separate `ffn_gate_exps`/`ffn_up_exps`, so it does nothing for fused `ffn_gate_up_exps` (Inkling).
  - Padding and output-header bytes aren't budgeted; `target_size` is tested to ±8 KiB.
  - The only test fixture is a 4-layer dense model.
- **Verdict.** It can't run inside a single-pass streaming converter as is. Its allocation step needs only per-tensor (type, bytes, error) tables plus shapes.

## 2. Unsloth UD-IQ3_XXS mixes by role

Bytes are computed from shapes × the ggml block size; their totals match the published files.

### Qwen3.8-27B (`unsloth/Qwen3.8-27B-GGUF`, 10,934,860,704 B)

| Role | Types (count) | GiB | Share |
|---|---|---|---|
| `attn_qkv` (linear layers) | IQ3_XXS 21, IQ3_S 17, IQ2_XXS 4, IQ2_S 3, Q2_K 2, IQ4_XS 1 | 0.904 | 8.9% |
| `attn_gate` (linear layers) | IQ3_XXS 24, IQ3_S 13, IQ2_S 6, Q2_K 2, IQ2_XS 1, IQ2_XXS 1, IQ4_XS 1 | 0.540 | 5.3% |
| `ssm_out` | IQ3_S 23, IQ4_XS 11, IQ3_XXS 8, Q3_K 3, Q4_K 2, IQ2_S 1 | 0.631 | 6.2% |
| `ssm_alpha`/`ssm_beta` | Q8_0 | 0.023 | 0.2% |
| `attn_q` (full layers) | IQ3_XXS 6, Q2_K 4, IQ3_S 4, IQ2_XXS 1, IQ4_XS 1 | 0.358 | 3.5% |
| `attn_k` / `attn_v` | Q4_K, IQ3_S, IQ4_XS / Q4_K, Q5_K | 0.089 | 0.9% |
| `attn_output` | IQ4_XS 8, IQ3_S 3, Q4_K 2, IQ3_XXS 2, IQ2_XXS 1 | 0.225 | 2.2% |
| `ffn_gate` / `ffn_up` / `ffn_down` | IQ1_S … IQ4_XS, Q4_K | 6.012 | 59.1% |
| `token_embd` / `output` | Q2_K / Q4_K | 1.055 | 10.3% |
| MTP (blk.64) | Q6_K, Q8_0 | 0.327 | 3.2% |
| 1-D and small tensors | F32 | 0.009 | 0.1% |

Bits rise with depth:
- Dense FFN: layers 0–19 sit at 1–2.5 bpw (`ffn_gate.0` and `ffn_up.0` are IQ1_S), the middle at 3 bpw, and layers 54–63 at IQ4_XS.
- `attn_v` never goes below Q4_K.
- The order isn't strictly monotone, which points to per-tensor sensitivity scoring rather than a fixed schedule.

### Inkling 2 Mini (`unsloth/Inkling-2-Mini-GGUF`, 11,663,310,784 B)

| Role | Types (count) | GiB | Share |
|---|---|---|---|
| `ffn_gate_up_exps` | IQ3_XXS 13, IQ2_S 9, IQ2_XS 1, IQ2_XXS 1 | 5.320 | 49.0% |
| `ffn_down_exps` (1152 wide) | NVFP4 20 (layers 0–19), IQ4_NL 4 (layers 20–23, MXFP8 in the Source) | 4.271 | 39.4% |
| Attention q/gate/output/k/v/rel | mostly Q8_0 in layers 0–17, Q4_K–Q6_K in layers 18–23 | 0.580 | 5.3% |
| `token_embd` / `output` | Q4_K / Q5_K | 0.540 | 5.0% |
| Router, loop controllers, norms, small projections | F32 / BF16 | ≈0.14 | 1.3% |

- **Experts dominate:** routed experts are 88% of the file.
- **NVFP4 kept only for `down`:** `gate_up` is requantized to 2–3 bits even in the NVFP4 layers.
- **`ffn_down_exps` has few type options:** at width 1152, only block-32/64/128 types fit: NVFP4, MXFP4, IQ4_NL, Q4_0/1, Q5_0/1, Q8_0, Q2_0, Q1_0.
- **MTP and vision** aren't in the text GGUF.
- **No imatrix entries:** neither file has entries for `token_embd` or `output`, and Qwen also has none for MTP.

### Fused tensors at the pin

`adapt_mix.py` (research/streaming-converter) gives a fused tensor the type of its largest part. That moves Qwen's 48 `attn_gate` tensors to their `attn_qkv` type, for +2,949,120 B over the published mix. A fused tensor needs a type chosen for the whole tensor.

## 3. `llama-fit-params` on a header-only GGUF

`header_only.py` runs the research converter's `stream_convert.main()`, except that it writes only the tensor infos and then stops. It produces:
- **`OUT`:** the header alone.
- **`OUT.sparse.gguf`:** the same file truncated *up* to `data_offset + Σ pad(nbytes)`, leaving the data region as a sparse hole.

| | Qwen3.8-27B (adapted UD-IQ3_XXS mix) | Inkling 2 Mini (UD-IQ3_XXS mix) |
|---|---|---|
| Header only, no padding | exit 1: `tensor 'blk.0.attn_norm.weight' data is not within the file bounds` | exit 1 (same check) |
| Sparse, padded | exit 0, identical output in 3 runs | exit 0, identical output in 3 runs |
| Padded size vs the real conversion's size | 10,937,802,944 B, equal | 11,663,287,840 B, equal |
| Disk used by the sparse file | ≈ 10.5 MiB | ≈ 11.4 MiB |
| Placement at `-c 8192` | `-ngl -1`: all on GPU, 10,882 MiB projected, 1,241 MiB free | `-ngl 49 -ot …`: experts of 28 layers on CPU, 1,056 MiB free |
| Wall time | 0.19–0.22 s | 2.21–2.22 s |

- **Why the unpadded file fails.** The loader rejects any tensor with `offs + nbytes > file size` (`src/llama-model-loader.h:45-46`). This check runs before the fit's no-alloc path.
- **Why only the header matters.** The fit pass loads with `no_alloc` and no mmap (`common/fit.cpp:71-72`, `:179-180`), and returns before it reads any data (`src/llama-model.cpp:2299-2300`). Its answer therefore depends only on the header and the file size.
- **Not compared against a real file.** No real GGUF of either mix remained on disk, so the placement wasn't compared with one directly.

Files: `header_only.py`, and each run's stdout/stderr (`*.out`, `*.err`). `ink.run1.out` holds the full Inkling `-ot` list.

Commands, with paths as placeholders:

```
cmake --build <llama.cpp>/build --target llama-fit-params
python header_only.py Qwen/Qwen3.8-27B <out>/qwen-hdr.gguf --types <mix>/qwen-adapted.types.json \
  --imatrix <mix>/qwen-adapted.imatrix.gguf --ftype MOSTLY_IQ3_XXS
llama-fit-params -m <out>/qwen-hdr.sparse.gguf -c 8192 [-fitp on]
```
