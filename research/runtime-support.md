# How llama.cpp, MLX and Ollama run Inkling 2 Mini and Qwen3.8-27B on a 16 GB Apple Silicon Mac

Ticket: [#2](https://github.com/ailabhq/model-fitter/issues/2). Researched 2026-11-23 from source code at these commits (links below are permalinks):

| Project | Commit | Date |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830) | `a0677c3` | 2026-11-23 |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/tree/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0) | `1bf8da0` | 2026-11-21 |
| [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm/tree/36565bc13a1b0314acff6cbcbc8328cf675d85e4) | `36565bc` | 2026-11-23 |
| [ollama/ollama](https://github.com/ollama/ollama/tree/318ee284aa3b5558de57efefff730ea33c4a9bef) | `318ee28` (latest release v0.40.17, 2026-11-10) | 2026-11-22 |

Nothing was downloaded or benchmarked. Sizes come from Hugging Face / ollama.com listings. KV numbers are computed from each model's `config.json` and checked against the runtime cache code.

## Answer in short

- **Qwen3.8-27B** (dense, hybrid Gated-DeltaNet + full attention, arch `qwen35`) works in all three runtimes, vision included (in MLX, vision needs mlx-vlm). It is dense, so every token reads every weight. On 16 GB the only quants that fit are ≤ ~3.5 bpw GGUF (unsloth `UD-IQ3_XXS` 10.9 GB … `UD-Q3_K_XL` 13.2 GB). No MLX or Ollama build that fits 16 GB is published: the smallest official ones are 4-bit, ~16–18 GB. KV is small: **64 KiB/token** (16 of 64 layers keep KV) plus a fixed ~150 MiB recurrent state, so 8K costs ~0.65 GiB and 32K ~2.15 GiB.
- **Inkling 2 Mini** (looped MoE, arch `inkling2_loop`) is supported on llama.cpp `master` (text + vision, merged 2026-11-20). In MLX it is supported by **mlx-vlm** (text + vision); for **mlx-lm** it is only an open PR. In Ollama it runs **only through the MLX runner, as one 20 GB NVFP4 tag**: the GGUF path fails on the released Ollama binary. Looping does **not** add weights, but it **does** multiply cache: every one of the 48 layer visits keeps its own KV/conv cache. Even so, KV stays small because 40 of the 48 visits are 1024-token sliding-window layers: **32 KiB/token** plus ~240 MiB fixed (8K ≈ 0.5 GiB, 32K ≈ 1.2 GiB). The weights are the problem: the native checkpoint is ~20 GB and Thinking Machines recommends ≥ 32 GB Macs. On 16 GB the options are GGUF ≤ ~3-bit (unsloth `UD-Q2_K_XL` 9.1 GB … `UD-Q3_K_XL` 13.6 GB), or MLX 2.5–3-bit community quants (oMLX `oQ2.5e`/`oQ3e`, 11.5/13.7 GB).
- **Memory beyond the GPU limit:** only **llama.cpp** gives real control. Weights are mmapped by default; `--cpu-moe` / `--n-cpu-moe N` / `-ot` keep MoE expert tensors in CPU memory (they match Inkling's tensor names); `--fit` (on by default) shrinks context and moves layers or experts off the GPU to fit `recommendedMaxWorkingSetSize`. **mlx-vlm** has a generic SSD expert-offload (`moe_offload`), but it is not tested on Inkling. mlx-lm and Ollama-MLX need the whole model resident: they wire up to the recommended working set and otherwise fall back to pageable memory, with a warning. Raising `iogpu.wired_limit_mb` (sudo sysctl) is the documented MLX lever.

## Comparison table

✅ = supported on main/release; 🟡 = partial / PR / community; ❌ = not supported.

| | llama.cpp | MLX: mlx-lm | MLX: mlx-vlm | Ollama (v0.40.17) |
|---|---|---|---|---|
| **Qwen3.8-27B text** | ✅ arch `qwen35` ([convert](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/conversion/qwen.py#L734-L736); [GGUF metadata](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)) | ✅ `qwen3_5` ([model](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/models/qwen3_5/model.py)) | ✅ `qwen3_5` | ✅ GGUF + MLX tags ([library](https://ollama.com/library/qwen3.8/tags)) |
| **Qwen3.8-27B vision** | ✅ `mmproj` via mtmd, projector `qwen3vl_merger` ([convert map](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/conversion/__init__.py#L445)) | ❌ vision and MTP weights dropped ([checkpoint.py](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/models/qwen3_5/checkpoint.py#L56-L58)) | ✅ ([models/qwen3_5](https://github.com/Blaizzy/mlx-vlm/tree/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/qwen3_5)) | ✅ "Text, Image" on all tags |
| **Inkling 2 Mini text incl. loop** | ✅ arch `inkling2_loop` ([PR #32694](https://github.com/ggml-org/llama.cpp/pull/32694), merged 2026-11-20; [inkling2-loop.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/models/inkling2-loop.cpp)) | 🟡 open [PR #2086](https://github.com/ml-explore/mlx-lm/pull/2086) (stacked on #2061) | ✅ native checkpoint, no conversion ([README](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/inkling2/README.md)) | ✅ MLX runner only ([mlxrunner/model/inkling2](https://github.com/ollama/ollama/tree/318ee284aa3b5558de57efefff730ea33c4a9bef/mlxrunner/model/inkling2)); GGUF fails: released binary pins llama.cpp `b12257` ([LLAMA_CPP_VERSION](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/LLAMA_CPP_VERSION), [issue #19817](https://github.com/ollama/ollama/issues/19817), fix in open [PR #19861](https://github.com/ollama/ollama/pull/19861)) |
| **Inkling 2 Mini vision** | ✅ projector `inkling2v` ([mtmd/models/inkling2v.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/mtmd/models/inkling2v.cpp)) | 🟡 in PR #2086 | ✅ | ✅ ([vision.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/mlxrunner/model/inkling2/vision.go)) |
| **Quant formats** | GGUF: all `ggml` types (Q2_K…Q8_0, IQ1–IQ4, K-quants, NVFP4/MXFP4 types exist in [ggml.h](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/include/ggml.h#L258-L259)); Inkling converter **dequantizes** the native NVFP4/MXFP8 experts first ([conversion/inkling2.py](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/conversion/inkling2.py#L116-L175)) | MLX affine 2–8 bit, mxfp4, nvfp4, mxfp8 | same modes ([convert.py](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/convert.py#L541-L545)), plus AWQ and mixed recipes; Inkling loads native NVFP4/MXFP8 experts as-is | GGUF import is not requantized ([import.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/import.mdx#L35)); safetensors `create -q` only `int4, int8, nvfp4, mxfp4, mxfp8` ([quant.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/mlx/quant/quant.go#L19-L23)). **Nothing below 4 bit** |
| **Weights > GPU limit** | mmap by default; `--cpu-moe`, `--n-cpu-moe N`, `-ot regex=CPU`; `--fit` auto-placement (see below) | no offload; wires up to recommended working set | generic SSD expert offload `moe_offload` (not verified on Inkling) | GGUF: llama-server `--fit` defaults, no expert flags exposed; MLX runner: no offload, warns and uses pageable memory |
| **KV-cache quantization** | `-ctk/-ctv` (f16, q8_0, q4_0, …) ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2421-L2440)) | `--kv-bits`, `--max-kv-size` ([generate.py](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py#L186-L224)) | similar flags | `OLLAMA_KV_CACHE_TYPE`, needs flash attention ([faq.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/faq.mdx#L375-L385)) |
| **Default context** | from model, or reduced by `--fit` (min 4096) | model max | model max | **4K below 24 GiB VRAM**: must set ≥ 8K for Usable ([context-length.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/context-length.mdx)) |

### Published weight sizes vs a 16 GB machine

The 16 GB M4 Mac mini has **120 GB/s** memory bandwidth ([Apple tech specs](https://support.apple.com/en-ca/121555)). Sizes are file sizes from the Hugging Face tree API / ollama.com tag pages, in decimal GB.

| Model | Runtime | Build | Size | Fits 16 GB with room for OS + 8K KV? |
|---|---|---|---|---|
| Inkling 2 Mini | native (all MLX) | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) NVFP4 + MXFP8 | ~20 | ❌ (card: "Apple silicon devices with 32GB or more memory") |
| Inkling 2 Mini | Ollama | `inkling-2-mini:26b-a5b-nvfp4` (= `latest`), MLX ([tags](https://ollama.com/library/inkling-2-mini/tags)) | 20 | ❌ only tag offered |
| Inkling 2 Mini | mlx-lm PR / mlx-vlm | [mlx-community/Inkling-2-Mini-4bit](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit): experts kept at NVFP4/MXFP8, the rest 4-bit | 17.3 | ❌ |
| Inkling 2 Mini | oMLX ([jundot/omlx](https://github.com/jundot/omlx)) | [mlx-community/Inkling-2-Mini-oQ3e-mtp](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ3e-mtp): experts affine 3-bit, 3.95 bpw avg | 13.0 + 0.64 draft | 🟡 tight |
| Inkling 2 Mini | oMLX | [mlx-community/Inkling-2-Mini-oQ2.5e-mtp](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ2.5e-mtp) | 10.9 + 0.64 draft | ✅ |
| Inkling 2 Mini | llama.cpp | [unsloth GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) `UD-IQ1_S` / `UD-Q2_K_XL` / `UD-IQ3_XXS` / `UD-Q3_K_XL` / `UD-IQ4_XS` / `UD-Q4_K_M` | 7.8 / 9.1 / 11.7 / 13.6 / 15.7 / 16.6 | ✅ ≤ IQ3_XXS; 🟡 Q3_K_XL; only with expert offload above that |
| Inkling 2 Mini | llama.cpp | [ggml-org GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) `IQ4_NL` / `Q8_0`; mmproj Q8_0 0.35 | 17.6 / 27.8 | ❌ resident; mmap/expert offload only |
| Qwen3.8-27B | llama.cpp | [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-IQ2_XXS` / `UD-Q2_K_XL` / `UD-IQ3_XXS` / `UD-IQ3_S` / `UD-Q3_K_XL` / `UD-IQ4_XS` | 7.2 / 10.0 / 10.9 / 12.0 / 13.2 / 14.4 | ✅ ≤ IQ3_S; 🟡 Q3_K_XL; ❌ ≥ IQ4_XS |
| Qwen3.8-27B | llama.cpp | [ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) `Q4_K_M`; mmproj Q8_0 0.63 | 19.4 | ❌ |
| Qwen3.8-27B | mlx-lm / mlx-vlm | [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (smallest mlx-community build) | 16.1 | ❌ |
| Qwen3.8-27B | Ollama | `qwen3.8:27b-q4_K_M` (GGUF) / `27b-mlx` / `27b-nvfp4` | 18 / 18 / ~19 | ❌ |

## Per runtime

### llama.cpp

**Support.**
- Qwen3.8-27B converts as `qwen35`: a hybrid of linear-attention (Gated DeltaNet) layers and full-attention layers. It keeps a recurrent state plus a KV cache, and it supports recurrent-state rollback for speculative decoding ([llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-arch.cpp#L1468-L1510)).
- Inkling 2 Mini is `inkling2_loop`. The loader turns the 24 physical layers into `n_phys + (n_passes-1)*(core_end-core_start)` = 24 + 2×12 = **48 cache slots**. The repeated slots share the physical weights (`layers[il] = layers[layer_phys(il)]`), and "each cache slot keeps its own KV and conv state" ([inkling2-loop.cpp L3-L40, L65-L70](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/models/inkling2-loop.cpp#L3-L70)).
- Vision uses mtmd with an `inkling2v` / `qwen3vl_merger` mmproj.
- Ollama's open PR describes a follow-up "loop-slot correctness fix" upstream (#32704) that is already in current master ([ollama PR #19861](https://github.com/ollama/ollama/pull/19861)). Use a llama.cpp build from 2026-11-21 or later.

**Quantization.**
- The Inkling converter reads the native NVFP4 (layers 0–19) and MXFP8 (layers 20–23) expert banks ([hf_quant_config.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/hf_quant_config.json)) and dequantizes them to float before writing GGUF ([conversion/inkling2.py](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/conversion/inkling2.py#L116-L175)). Every Inkling GGUF quant is therefore a *re*-quantization of an already-4-bit source. Quality loss needs measuring against the native checkpoint, not BF16.
- `llama-quantize` with an imatrix ([ggml-org ships `imatrix.gguf`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)) produces any ggml type.

**Memory beyond the GPU limit** (all flags from [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp)):
- **Load mode.** `-lm/--load-mode auto|none|mmap|mlock|mmap+mlock|dio`; the default `auto` means "mmap, unless a device does not support it" ([L2665-L2683](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2665-L2683)). The Metal backend can wrap mmapped host memory directly (`buffer_from_host_ptr = true`, [ggml-metal.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal.cpp#L778)). Weights not wired can therefore be paged back in from SSD by the OS.
- **MoE expert placement.** `-ot/--override-tensor <regex>=<buffer type>`, `-cmoe/--cpu-moe` ("keep all MoE weights in the CPU") and `-ncmoe/--n-cpu-moe N` ("…of the first N layers") ([L2729-L2750](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2729-L2750)). Both use the regex `\.ffn_(up|down|gate|gate_up)_(ch|)exps` ([common.h L1292](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.h#L1292)). Inkling's experts are `blk.N.ffn_gate_up_exps` / `blk.N.ffn_down_exps` ([inkling2.cpp L155-L156](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/models/inkling2.cpp#L155-L156)), so they match. `N` counts the 24 physical layers, because loop slots share tensors.
- **`--fit` (default on).** It "adjust[s] unset arguments to fit in device memory". The default target margin is 1 GiB per device and the minimum context is 4096 ([common.h L504-L509](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.h#L504-L509)). Order of steps: (2) reduce context, (3) fill layers back to front, "for a MoE model … with all MoE tensors in system memory", (4) move MoE layers back onto the GPU while they fit ([fit.cpp L460-L839](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/fit.cpp#L460-L839)). It "assumes system memory is unlimited" ([fit.h L25](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/fit.h#L25)). On a unified-memory Mac, the CPU side and the GPU side come out of the same 16 GB. So `--fit` only respects the Metal limit, not total RAM; the CPU share then relies on mmap paging.
- **Metal limit.** "Device memory" on Metal is `recommendedMaxWorkingSetSize`, with free = that minus `currentAllocatedSize`. The code notes that "it's possible to allocate more than `recommendedMaxWorkingSetSize`" and logs a warning when it does ([ggml-metal-device.m L1501-L1510, L2080-L2091](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal-device.m#L1501-L1510)). Raising `iogpu.wired_limit_mb` raises that value [INFERENCE: Apple does not document the sysctl; the link comes from the MLX docs quoted below]. On macOS ≥ 15, llama.cpp keeps GPU buffers in residency sets and "keep[s] the memory wired for 3 minutes" after the last use; `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` overrides this and `GGML_METAL_NO_RESIDENCY` disables it ([L933-L985, L1266-L1268](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal-device.m#L933-L985)).
- **Sliding-window cache.** By default the SWA cache is `pad256(min(n_ctx, n_swa + n_ubatch))` cells; `--swa-full` makes it full-size ([llama-kv-cache-iswa.cpp L71-L81](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-kv-cache-iswa.cpp#L71-L81)). For Inkling with the default `n_ubatch` 512, that is 1536 cells per local slot.

### MLX (mlx-lm and mlx-vlm)

**Support.**
- mlx-lm `main` has `qwen3_5` for **text only**: "mlx-lm does not run the vision encoder or the multi-token prediction head yet" ([checkpoint.py L56-L58](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/models/qwen3_5/checkpoint.py#L56-L58)).
- mlx-lm has **no Inkling** on main. [PR #2086](https://github.com/ml-explore/mlx-lm/pull/2086) (open) adds text + image, with "each repeat has its own cache". It reports peak memory of 24.3 GB at 4K and 24.6 GB at 16K context, with 32 tok/s on an M3 Ultra, using the native weights.
- mlx-vlm `main` supports Inkling 2 Mini directly from the native checkpoint, text + image. "The decoder visits 24 distinct layers, running the middle 12 three times … Each visit has its own attention and convolution cache" ([README](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/inkling2/README.md)). Its notes say "Inference uses floating-point activations, not the reference runtime's activation quantization. Strict numerical parity with the reference is not established."
- In mlx-vlm, sliding visits use a window-bounded `AttentionSinkKVCache(1024)` and global visits an unbounded one ([language.py L945-L955](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/inkling2/language.py#L945-L955); [cache.py L642-L701](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/cache.py#L642-L701)).
- mlx-vlm also has Qwen3.8 vision ([models/qwen3_5](https://github.com/Blaizzy/mlx-vlm/tree/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/qwen3_5)).

**Quantization.**
- Modes are `affine`, `mxfp4`, `nvfp4`, `mxfp8`, with RTN or AWQ, plus mixed-bit recipes ([convert.py L541-L558](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/convert.py#L541-L558)).
- For Inkling, the quantizer only touches modules that have `to_quantized`; Inkling defines no `quant_predicate` that would opt quantized modules in ([quant_utils.py L119-L122](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/quant_utils.py#L119-L122)). The already-quantized NVFP4/MXFP8 expert banks are kept, which is why `mlx-community/Inkling-2-Mini-4bit` is still 17.3 GB.
- Going below ~4 bit on the experts needs a dequantize → requantize pass [INFERENCE from the code; `convert --dequantize` exists]. The oMLX `oQ2.5e`/`oQ3e` checkpoints did exactly that ("requantizes the routed experts of all layers to affine 3-bit"), but they need oMLX ≥ 0.8.1, a third-party MLX server ([model card](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ3e-mtp)).

**Memory beyond the GPU limit.**
- mlx-lm wires memory up to `max_recommended_working_set_size` ([utils.py L84-L93](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/utils.py#L84-L93)). It warns when a model is too large, and the README says "If the model fits in RAM then it can often be sped up by increasing the system wired memory limit … `sudo sysctl iogpu.wired_limit_mb=N` … larger than the size of the model in megabytes but smaller than the memory size of the machine" ([README L272-L290](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/README.md#L272-L290)).
- MLX's `set_wired_limit` doc: "Setting a wired limit larger than system wired limit is an error. You can increase the system wired limit with `sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>`", and "The wired limit should remain strictly less than the total memory size" ([mlx python/src/memory.cpp](https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/python/src/memory.cpp#L118-L149)).
- mlx-lm and stock MLX have no CPU or disk offload: weights must be resident, and beyond the wired limit they live in pageable memory that macOS may compress or swap.
- **mlx-vlm `moe_offload`** pages routed experts from an SSD store. It "mmaps it and keeps a byte-budgeted LRU of resident experts, evicting the coldest ones once touched experts approach the GPU's recommended working set". Usage: repack once with `python -m mlx_vlm moe_offload --build … --out …`, then serve with `--expert-cache-gb` ([moe_offload.py L1-L13](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/moe_offload.py#L1-L13); [docs/usage.md L361-L373](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/docs/usage.md#L361-L373)). Inkling's sanitizer maps `experts.w13_weight` to `switch_mlp.{gate,up}_proj`, which is the layout the offloader's regex expects ([inkling2.py L186-L203](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/inkling2/inkling2.py#L186-L203)). However, no Inkling-specific test or doc exists, so Inkling + offload is **unverified**.
- KV can be quantized with `--kv-bits` / `--quantized-kv-start` and capped with `--max-kv-size` (rotating) ([generate.py L186-L224](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py#L186-L224)).

### Ollama

**Support.**
- Two engines: GGUF through a bundled `llama-server`, and safetensors through the MLX runner.
- **Qwen3.8**: library tags `27b-q4_K_M` (18 GB, GGUF), `27b-mlx` (18 GB), `27b-nvfp4`, `27b-mxfp8` (30 GB), `27b-bf16`, plus MTP / DFlash2 variants, all "Text, Image", 256K context ([tags](https://ollama.com/library/qwen3.8/tags)).
- **Inkling 2 Mini**: one artifact, `26b-a5b-nvfp4` = `latest`, MLX, 20 GB, Text + Image, 1M context ([tags](https://ollama.com/library/inkling-2-mini/tags)). Implemented in the MLX runner, with a separate cache per layer visit: window-bounded for sliding layers, plus conv state ([model.go L138-L150](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/mlxrunner/model/inkling2/model.go#L138-L150)).
- Inkling GGUFs **do not load** on release v0.40.17: `unknown model architecture: 'inkling2_loop'` / `unknown projector type: inkling2v` ([issue #19817](https://github.com/ollama/ollama/issues/19817)). The bundled llama.cpp is pinned to `b12257` ([LLAMA_CPP_VERSION](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/LLAMA_CPP_VERSION)); the bump is in open [PR #19861](https://github.com/ollama/ollama/pull/19861).

**Quantization.** `ollama create -q` supports only `int4, int8, nvfp4, mxfp4, mxfp8` for safetensors ([quant.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/mlx/quant/quant.go#L19-L23); [create/classify.go L126](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/create/classify.go#L126)). Pre-quantized Inkling expert banks are detected and kept as NVFP4/MXFP8 ([create/inkling2.go L38-L56](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/create/inkling2.go#L38-L56)). GGUFs are imported as-is ([import.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/import.mdx#L35)). **Ollama cannot produce a sub-4-bit Fit itself**; it can only serve a GGUF made elsewhere, and only for architectures its pinned llama.cpp knows.

**Memory beyond the GPU limit.**
- GGUF path: Ollama passes a "minimal set" of flags and lets llama-server auto-detect `-ngl` (so `--fit` applies). It exposes only `num_gpu`, `num_ctx`, the KV type and similar options; there is no `-ot`/`--n-cpu-moe` passthrough ([llama_server.go L422-L471](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/llm/llama_server.go#L422-L471)).
- MLX runner: it wires `min(active model memory, max recommended working set)`. Above that it logs "MLX model exceeds the recommended working set; performance may be degraded" and runs from pageable memory ([runner.go L329-L361](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/mlxrunner/runner.go#L329-L361)).
- The default context is 4K on machines with < 24 GiB VRAM ([context-length.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/context-length.mdx)). Set `OLLAMA_CONTEXT_LENGTH` / `num_ctx` ≥ 8192 for Usable.

## KV-cache and state memory per context length

### Inputs from `config.json`

**Inkling 2 Mini** ([config.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/config.json)):
- 24 layers, `hidden_size` 2304.
- `local_layer_ids` = all layers except **5, 11, 17, 23** (global), with `sliding_window_size` 1024.
- Global and local attention both use `num_key_value_heads` 8 and `head_dim` 128.
- `loop_config` `core_start` 6, `core_end` 18, `num_passes` 3.
- Layer visits: 6 (layers 0–5) + 12×3 (layers 6–17) + 6 (layers 18–23) = **48**. Global visits: layer 5 once, layers 11 and 17 three times each, layer 23 once = **8**. Local visits: **40**.
- KV per token per visit at 16-bit: 2 (K,V) × 8 × 128 × 2 B = **4 KiB**.
- The cached K/V really are just `n_kv × head_dim` (the relative-position `r_proj` and the sink are computed from the query side): [mlx-vlm language.py L262-L300, L439-L500](https://github.com/Blaizzy/mlx-vlm/blob/36565bc13a1b0314acff6cbcbc8328cf675d85e4/mlx_vlm/models/inkling2/language.py#L262-L300).
- Short-conv state: llama.cpp uses `2·n_embd_k_gqa + 2·n_embd` = 6656 channels × (kernel 4 − 1) × f32 ≈ 78 KiB per visit, so ×48 ≈ **3.7 MiB** ([inkling2.cpp L62-L79](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/models/inkling2.cpp#L62-L79)).
- Formula: **8 × 4 KiB = 32 KiB per token** (global) + fixed local part. The fixed local part is 40 × 1536 × 4 KiB = **240 MiB** in llama.cpp, or 40 × 1024 × 4 KiB ≈ 160 MiB minimum in MLX. Without the loop it would be 4 global visits (16 KiB/token) and 20 local, so **looping doubles the cache**.

**Qwen3.8-27B** ([config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)):
- 64 layers, `full_attention_interval` 4, so **16** full-attention layers and 48 `linear_attention` layers.
- Full attention: `num_key_value_heads` 4, `head_dim` 256 → 2 × 4 × 256 × 2 B = 4 KiB per layer per token → **64 KiB per token**.
- Linear attention (Gated DeltaNet), fixed size, `mamba_ssm_dtype` float32 (llama.cpp also allocates recurrent state as F32):
  - SSM state: 48 value heads × 128 × 128 × 4 B = 3 MiB per layer.
  - Conv state: 3 × (48·128 + 2·16·128) × 4 B = 120 KiB per layer.
  - Total for 48 layers: **≈ 150 MiB** ([hparams n_embd_r/n_embd_s](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-hparams.cpp#L208-L272)).

### Totals

Totals for one sequence, f16/bf16 KV (q8_0 KV ≈ 0.53× of the KV part). The llama.cpp defaults (`n_ubatch` 512) are used for the Inkling sliding part. Compute buffers, the vision tower and draft models come on top.

| Context | Inkling 2 Mini (f16) | Inkling 2 Mini (q8_0 KV) | Qwen3.8-27B (f16) | Qwen3.8-27B (q8_0 KV) |
|---|---|---|---|---|
| 4K | 0.36 GiB | 0.19 GiB | 0.40 GiB | 0.28 GiB |
| **8K** | **0.49 GiB** | 0.26 GiB | **0.65 GiB** | 0.41 GiB |
| 16K | 0.74 GiB | 0.39 GiB | 1.15 GiB | 0.68 GiB |
| 32K | 1.24 GiB | 0.66 GiB | 2.15 GiB | 1.21 GiB |
| 64K | 2.24 GiB | 1.19 GiB | 4.15 GiB | 2.27 GiB |
| 128K | 4.24 GiB | 2.25 GiB | 8.15 GiB | 4.40 GiB |
| 256K (Qwen max) | 8.24 GiB | 4.38 GiB | 16.15 GiB | 8.65 GiB |
| 1M (Inkling max) | 32.24 GiB | 17.13 GiB | n/a | n/a |

Cross-check: mlx-lm PR #2086 measured peak memory 24.3 GB at 4K vs 24.6 GB at 16K with its kernels on. That small growth is consistent with ~32 KiB/token of global KV (12K extra tokens ≈ 0.4 GB).

## Implications for model-fitter

[INFERENCE: analysis, not measured]

- **KV is not the constraint at 8K for either model** (≤ 0.65 GiB). Weights are.
- **Qwen3.8-27B (dense):** every token reads every weight. At 120 GB/s, an ~11 GB IQ3_XXS gives a theoretical ceiling of ~11 tok/s with everything resident. Any SSD paging of a dense model touches every page, every token. A Usable Fit of Qwen3.8-27B on 16 GB is therefore realistically limited to ≤ ~3.5 bpw GGUF on llama.cpp, fully resident with the wired limit raised. In MLX it needs a custom sub-4-bit conversion, because none is published.
- **Inkling 2 Mini (MoE + loop):** per-token reads ≈ 5.4B effective active params, about 2.7 GB at ~4 bpw, so bandwidth allows far more than 5 tok/s if resident. That makes expert placement the key Fit plan lever:
  - llama.cpp `--n-cpu-moe` + mmap for ~4-bit GGUFs (15–18 GB);
  - fully-resident ≤ 3-bit GGUF / oQ2.5e–oQ3e MLX;
  - mlx-vlm `moe_offload` once validated on Inkling.

  The looped core layers 6–17 run the same physical expert banks in all three passes (routing may differ per pass), which should favor any expert LRU cache.
- **Ollama is not a viable runtime for Inkling 2 Mini on 16 GB today.** It has only a 20 GB tag, cannot quantize below 4 bit, and its released binary cannot load Inkling GGUFs. For Qwen3.8, Ollama works only with an imported ≤ 3.5-bit GGUF and an explicit `num_ctx` ≥ 8192.
- **Raising `iogpu.wired_limit_mb`** is needed in every runtime to let the GPU keep ~12–13 GB resident on a 16 GB machine. It needs sudo and resets on reboot [INFERENCE: sysctl semantics]. It must stay below total RAM (MLX doc), and the Usable "machine stays responsive" bar caps how far it can go. That cap should be measured by the benchmark ticket.
