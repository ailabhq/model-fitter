# What existing Mac image-generation tools do for new text-to-image models on 16 GB Apple Silicon, and which of sarto's gaps remains

Map ticket: "What do existing Mac image-generation tools already do for new text-to-image models on 16 GB Apple Silicon, and which of sarto's gaps remains?" for [Map: image Fits, new text-to-image models on small Apple Silicon Macs](https://github.com/ailabhq/sarto/issues/36). Researched 2026-11-25 from repos at pinned commits, release notes, docs and Hugging Face (HF) model cards and API dates. Nothing was downloaded or run. `[INFERENCE]` marks a conclusion no source states directly.

| Project | Commit / version read | Date |
|---|---|---|
| [leejet/stable-diffusion.cpp][sd] | `52491bd` (`master-1133`) | 2026-11-25 |
| [comfyanonymous/ComfyUI][cu] | `a2103ae` (v0.46.0 is 2026-11-18) | 2026-11-25 |
| [city96/ComfyUI-GGUF][cg] | `6ea2651` (no tags) | 2026-01-12 |
| [drawthingsai/draw-things-community][dt] | `cb2d4c6` (v26.1023.0 is 2026-10-23) | 2026-11-24 |
| [filipstrand/mflux][mf] | `3d1d17a` (v.0.24.0 is 2026-10-27) | 2026-11-25 |
| [invoke-ai/InvokeAI][inv] | `1bd3954` (v6.14.4 stable, v7.0.0-beta.2) | 2026-11-25 |
| [huggingface/diffusers][df] | `3d33683` (v0.43.0 is 2026-11-12) | 2026-11-25 |
| [unslothai/unsloth][us] (Unsloth Studio / Desktop) | `2039322` (v0.1.925-beta) | 2026-11-25 |
| [AtomicBot-ai/Atomic-Chat][ac] + [atomic-chat-core][acc] | `c45dc5d` / `03097b5` (v2.4.7) | 2026-11-25 |
| [ollama/ollama][ol] | `eec8e0b` (v0.32.5, last with images); `924cc28` (v0.40.18) | 2026-07 / 2026-11-24 |
| [godly-devotion/MochiDiffusion][mo] | `e9e0d75` (v7.0) | 2026-11-15 |
| [apple/ml-stable-diffusion][ap] | `ea2805d` (last code change 2025-07-03) | 2026-09-11 |
| [divamgupta/diffusionbee-stable-diffusion-ui][db] | `daf82f8` (release 2.5.3 is 2024-08-14) | 2023-12-28 |
| [argmaxinc/DiffusionKit][dk], [ml-explore/mlx-examples][mx] | `498e5db` (archived), `c52cc74` | 2025-04-11, 2025-03-25 |

## Answer

- **Release lag is no longer a gap.** For every reference model that can fit 16 GB, at least one Mac tool had a 4-bit build that fits within 0–3 days of the public release. stable-diffusion.cpp gave **day-0** support to Qwen-Image-2.1, FLUX.3 Image Klein and MiniMax-H3-Image, and supported the other reference models in its docs within 1–7 days. Its GGUFs (leejet, unsloth, Abiray, vantagewithai, realrebelai) appear within 0–3 days. ComfyUI core lands 0–1 days after release, or before it, and runs the same GGUFs through ComfyUI-GGUF. Unsloth Studio and Atomic Chat package those GGUFs on Mac within 0–6 days. mflux runs MLX builds within 1–6 days for Z-Image, FLUX.2/FLUX.3 Klein and Qwen-Image-2.1, but takes 45–230 days for ERNIE, Krea 2 and Qwen-Image-2512. The caveat is that no one shows a build is *Usable on a 16 GB Mac*: every "fits 16 GB" claim is a file-size estimate or a GPU measurement.
- **Measured Quality loss is partly covered, by a few quantizers, not by any tool.** Since September 2026, unsloth and Atomic Chat publish LPIPS, SSIM and PSNR of each GGUF against the BF16 denoiser on fixed prompts and seeds. Atomic Chat adds confidence intervals, a separately measured text encoder and public renders. mflux publishes PSNR against bf16 for a handful of models, and Apple's Core ML converter does so for Stable Diffusion only. Every number is for one component at a time and was rendered on NVIDIA GPUs (or one M4 Pro 64 GB for mflux). It covers about five recent models, and no tool shows the number for the combination it actually runs. Nobody measures the whole pipeline as run on the Mac (encoder, denoiser and VAE precisions together) against the Source.
- **Per-component precision under a memory limit is still an open gap.** Every serious tool lets the user set the text encoder, denoiser and VAE separately, but only **by hand**. The automatic memory logic that exists picks **placement**, not precision: sd.cpp `--auto-fit`, ComfyUI's unloading and VAE tiling, Atomic Chat's offload policy, Draw Things' streaming tiers. Or it marks the **one** denoiser file that fits: Unsloth Studio's recommended/TIGHT/OOM tags, Atomic Chat's ok/maybe/no verdict. No tool chooses the encoder, denoiser and VAE precisions jointly against a memory budget, let alone against a measured Quality loss. Only Atomic Chat warns about swapping before a render, and only the removed Ollama image runner refused a load that would not fit.

So the go condition ("existing tools leave a gap on 16 GB Macs in at least one of the three") is **met by the third gap, and by the whole-pipeline part of the second**.

## Reference models and dates

The lag is counted from the **public** release. HF `createdAt` is when the repo was created, often private and days earlier. Where no first-party public date was found, `createdAt` is used, so that lag is an upper bound.

| Model | Public release (source) | HF repo created | Smallest Mac build that fits 16 GB [INFERENCE from file sizes] |
|---|---|---|---|
| FLUX.2 [dev] (32B) | 2025-11-25 ([BFL][bfl2]) | 2025-11-22 | **none**: Q2_K DiT is 12.85 GB before a ≥ 12 GB Mistral encoder ([city96 GGUF][c96]) |
| Z-Image-Turbo (6B) | 2025-11-27 (GitHub repo initialized 2025-11-26, [Tongyi-MAI/Z-Image][zgh]) | 2025-11-25 | yes: Q8_0 DiT 6.57 GB + Qwen3-4B Q4_K_M 2.49 GB ([sd.cpp z_image.md][sd-z]) |
| Qwen-Image-Edit-2511 (20B) | ≤ 2025-12-20 (unsloth GGUF existed then); `createdAt` used | 2025-12-17 | borderline: Q4_K_M 12.84 GB + 4.68 GB encoder ([sd.cpp][sd-qe]) |
| Qwen-Image-2512 (20B) | 2025-12-31 ([QwenLM commit "qwen image 2512 release"][q2512]) | 2025-12-30 | borderline, as 2511 |
| FLUX.2 [klein] 4B/9B | 2026-01-15 ([BFL][bflk]) | 2026-01-14 | yes: 4B Q4_0 2.46 GB, 9B Q4_0 5.61 GB ([leejet][sd-k]) |
| ERNIE-Image (8B) | 2026-04-15 (baidu/ERNIE-Image GitHub repo created 2026-04-14) | 2026-04-07 | yes: Q4_K_M 5.01 GB + Ministral 3B ([sd.cpp][sd-e]) |
| Krea 2 (12B) | 2026-06-23 ([Krea on X][krea]) | 2026-06-18 | yes: Q4_K_M 7.21 GB + Qwen3-VL-4B 2.49 GB ([sd.cpp krea2.md][sd-kr]) |
| Qwen-Image-2.1 (7B) | 2026-09-20 (sd.cpp README calls its 2026-09-20 support "Day-0"; first-party date not found) | 2026-09-14 | yes: Q4_0 4.19 GB + Qwen3-VL-8B Q4_K_M 5.02 GB ([leejet][sd-q21]) |
| FLUX.3 Image Klein (5.6B) | 2026-10-15 ([BFL blog][bfl3]) | 2026-10-13 | yes: Q4/Q5 3.5–4.5 GB + Qwen3-VL-4B Q8_0 4.28 GB ([Atomic Chat card][acf3]) |
| MiniMax-H3-Image (33B) | 2026-11-10 ([MiniMax release notes][mmrn]) | 2026-11-03 | **none with the Source encoder**: the 32B encoder is 18 GB at Q4_K_M ([sd.cpp minimax_h3_image.md][sd-mm]). It only fits by swapping in the P-H3 4B encoder, which is not the Source pipeline. |

## Per-tool table

"Lag" is the number of days from the public release to the first **released** build of the tool that runs the model ("−" means before the release; "n" means not supported as of 2026-11-25). Rows below the line are tools that do not track new models.

| Tool | Source / license | Lag, models that fit 16 GB (Z-Image-T, klein, ERNIE, Krea 2, Qwen-Img-2.1, FLUX.3 Klein) | Per-component precision | Memory limits | Swap / hang | Quality measurement |
|---|---|---|---|---|---|---|
| **stable-diffusion.cpp** | open, MIT | 4, 2, 1, 1, **0, 0** (also FLUX.2 dev 5, Edit-2511 7, H3-Image 0) | Per component, by hand: a separate file per `--diffusion-model`/`--llm`/`--vae`; `--type` global; `--tensor-type-rules` per tensor pattern; f32/f16/q8_0…q2_K, INT8 ConvRot; imatrix | `--auto-fit` (default on) places each component on GPU, then RAM, then disk; `--params-backend`, `--offload-to-cpu`, `--mmap`, `--max-vram`, layer streaming; VAE decode retries with tiles | none documented | none in docs ("can lead to a loss of quality") |
| **ComfyUI** (+ ComfyUI-GGUF) | open, GPL-3.0 (+ Apache-2.0) | −1, 0, −2, 0, 1, 0 (FLUX.2 dev 0, 2512 1, H3-Image 0); GGUFs 0–7 | Per component, by hand: separate loader nodes; `--*-unet`, `--*-text-enc`, `--*-vae` dtype flags; GGUF Q2_K…Q8_0, IQ4; on MPS **fp8 refused** and int8 has no compute | Mac forced to SHARED; RAM read from the system; unloads after use; VAE tiles on OOM or when the estimate exceeds free memory | none found | none (sample images) |
| **Draw Things** | open engine GPL-3.0 (draw-things-community); App Store GUI not checked | 11, 5, 5, 27, 8, 8 (FLUX.2 dev 27, 2512 5, Edit-2511 18; H3-Image on `main` 2026-11-20, unreleased) | Per component in each model entry, chosen by Draw Things; the user picks a variant: fp16, bf16, q8p/q6p/q5p, 8/6/4-bit "S" (SpinQuant) | Reads RAM, sets tiers and a weight-streaming budget (16 GB → 8 GiB tier); memory saver, tiled decode by hand | none found | none (qualitative release notes) |
| **mflux** (MLX) | open, MIT | 6, 3, 53, 45, 1, 1 (2512 231, Edit-2511 295; FLUX.2 dev and H3-Image n) | One `-q 3/4/5/6/8`; *which* components it touches is hard-coded per model (Qwen and Krea encoders stay bf16); `--small-text-encoder`, `--small-decoder` | `--low-ram` (1 GB cache, VAE tiling, frees encoder before denoising), strip VAE decode by default; `--ssd-stream` on `main` only; nothing automatic | none: docs show `vm_stat` by hand | **PSNR vs bf16 for a few models** (FLUX.3 Klein q8/q6/q4: 44.5/28.2/25.2 dB, 1 seed) |
| **Unsloth Studio / Desktop** | open, Apache-2.0 core, AGPL-3.0 `studio/` | n/a, n/a, n/a, n/a, 2, 4 (H3-Image GGUF in app 6); Mac route is GGUF via its sd.cpp fork | GGUF denoiser picked by the user; "Precision: Auto" picks bf16/FP8/INT8 on NVIDIA; encoder INT8 fixed | GGUF sizes tagged **recommended / TIGHT / OOM** for the device; Low-VRAM, CPU offload | "TIGHT may use system memory" | **unsloth GGUF cards: LPIPS, SSIM and PSNR vs the bf16 DiT** since Qwen-Image-2.1 |
| **Atomic Chat** | open, Apache-2.0 | n/a, n/a, n/a, n/a, 1, — (Qwen-Image-2.1-Turbo 0, H3-Image 3); runs sd.cpp | User picks the denoiser quant from a curated ladder; one encoder file per family; encoder **always on the CPU under Metal** | Estimates resident bytes (DiT + VAE + encoder + 1.5 GiB/MP), sets the offload policy automatically, retries under a fallback policy on OOM, tiles VAE above 1 MP | **ok/maybe/no verdict, "expect heavy swapping"** warning before rendering | **Atomic Chat GGUF cards: LPIPS with 95% CI, SSIM, PSNR; encoder measured separately; renders public** |
| **InvokeAI** | open, Apache-2.0 | 29, 12, 107, 38, 29 (alpha only), 18 (alpha only); H3-Image n | Separate installed model per component, by hand; GGUF, SDNQ, int8, fp8/nvfp4 files; on Mac fp8 is unpacked to bf16, NF4 does not load | On Mac models "load whole into system memory"; RAM cache fixed at 50% of RAM − 2 GB (6 GB on 16 GB); auto tiled decode | none for macOS | none for quantization (PSNR only for decode tricks) |
| **diffusers on MPS** | open, Apache-2.0 | merge −2, 0, −4, −1, −2, 0; PyPI 11, 49, 16, 10, 16, 8; ships the Source pipeline in bf16 | `PipelineQuantizationConfig` with a backend per component, by hand; Quanto and SDNQ run on MPS | model, sequential and group offload; VAE slicing and tiling; attention slicing advised < 64 GB; on unified memory, CPU offload does not lower peak RAM [INFERENCE] | MPS doc warns the system "automatically swaps"; nothing detects it | none (side-by-side image) |
| **Ollama** (image gen) | open, MIT | 44, 1; others n. Image generation **removed** 2026-07-28 (v0.32.6) | One quant for encoder + DiT at import (int8/int4); VAE unquantized | Loads all components; **refused a load larger than free memory** | refuses rather than swaps | none |
| **Mochi Diffusion** | open, GPL-3.0 | 340, 44; others n | Original weights via iris.c; Core ML for SD | Compute-unit choice by hand | none | none |
| — **DiffusionBee** | AGPL-3.0, unmaintained (last release 2024-08-14) | n for all | not documented | not documented | none | none |
| — **Apple ml-stable-diffusion** | open, MIT, dormant | n for all (SD 1.x/2.x/XL/3 only) | 2–8-bit palettization of UNet + encoder, mixed-bit per layer, by hand at conversion | `reduceMemory` loads just in time | none | **PSNR of latents vs fp16** (SDXL only) |
| — **DiffusionKit**, **mlx-examples** | MIT, archived / example code | n for all (SD3, FLUX.1 only) | fp16 flags / one 4-bit switch | `low_memory_mode` | none | none |
| — **LM Studio** | closed | no local image generation (a plugin calls HF cloud inference) | — | — | — | — |

## 1. Release lag, per tool

### stable-diffusion.cpp

It ships a rolling `master-N` release on every merge, so the merge date is the release date ([releases][sd-rel]).

| Model | Public | Support commit → release | First 16 GB build (maker, HF created) | Lag |
|---|---|---|---|---|
| FLUX.2 [dev] | 2025-11-25 | 2025-11-30 [`52b67c5`][sd-c1] | none fits | 5 |
| Z-Image-Turbo | 2025-11-27 | 2025-12-01 [`34a6fd4`][sd-c2] | leejet Q4_0/Q8_0, 2025-11-30 | 4 |
| Qwen-Image-Edit-2511 | ≤ 2025-12-20 | 2025-12-24 [`a0adcfb`][sd-c3] | unsloth Q4_K_M, 2025-12-20 (borderline) | ≤ 7 |
| FLUX.2 [klein] | 2026-01-15 | 2026-01-17 [`9565c7f`][sd-c4] | leejet 4B/9B, 2026-01-17 | 2 |
| ERNIE-Image | 2026-04-15 | 2026-04-16 [`5c243db`][sd-c5] | unsloth, 2026-04-14 | 1 |
| Krea 2 | 2026-06-23 | 2026-06-24 [`8caa3f9`][sd-c6] | realrebelai, 2026-06-23 | 1 |
| Qwen-Image-2.1 | 2026-09-20 | 2026-09-20 [`137f740`][sd-c7], "Day-0" | leejet, 2026-09-20 | 0 |
| FLUX.3 Image Klein | 2026-10-15 | 2026-10-15 [`8cd731e`][sd-c8] | Abiray 2026-10-15, unsloth 10-16, leejet 10-18 | 0 |
| MiniMax-H3-Image | 2026-11-10 | 2026-11-10 [`32d516c`][sd-c9] | none with the Source encoder | 0 |

### ComfyUI and GGUF builds

ComfyUI support is dated by the model-support commit and the first tag after it. The macOS desktop app bundles a core version that was not checked [INFERENCE: its lag can be longer].

| Model | ComfyUI commit → first tag | Lag | GGUF for ComfyUI-GGUF (maker, created) | GGUF lag |
|---|---|---|---|---|
| FLUX.2 [dev] | 2025-11-25 [`6b573ae`][cu-c1] → v0.3.72 2025-11-25 | 0 | city96, 2025-11-25 | 0 |
| Z-Image-Turbo | 2025-11-25 [`e9aae31`][cu-c2] → v0.3.73 2025-11-26 | −1 | jayn7, 2025-11-27 | 0 |
| FLUX.2 [klein] | 2026-01-15 [`3b83223`][cu-c3] → v0.9.2 2026-01-15 | 0 | unsloth, 2026-01-15 | 0 |
| Qwen-Image-2512 | same architecture; Comfy-Org weights 2025-12-31 | 0 | unsloth, 2025-12-30 | −1 |
| ERNIE-Image | 2026-04-12 [`31283d2`][cu-c4] → v0.19.0 2026-04-13 | −2 | unsloth / vantagewithai, 2026-04-14 | −1 |
| Krea 2 | 2026-06-22 [`2a61015`][cu-c5] → v0.26.0 2026-06-23 | 0 | vantagewithai / realrebelai, 2026-06-23 | 0 |
| Qwen-Image-2.1 | 2026-09-19 [`6bfaacc`][cu-c6] → v0.37.0 2026-09-21 | 1 | leejet / Abiray, 2026-09-20 | 0 |
| FLUX.3 Image Klein | 2026-10-15 [`20b4606`][cu-c7] → v0.41.0 2026-10-15 | 0 | Abiray 2026-10-15; BFL's own Comfy int8 2026-10-14 | 0 |
| MiniMax-H3-Image | 2026-11-10 [`ede4e5f`][cu-c8] → v0.45.0 2026-11-10 | 0 | unsloth, 2026-11-12 | 2 |

On a Mac, the Comfy-Org fp8/nvfp4 repacks do not help: `supports_cast` is False for fp8 on MPS ([model_management.py L1305-L1316][cu-fp8]), and `supports_int8_compute` is False because `torch._int_mm` is missing on MPS ([L2110-L2117][cu-int8]). The 16 GB route is a GGUF through ComfyUI-GGUF, which dequantizes Q2_K…Q8_0 and IQ4 in torch ([dequant.py L287-L301][cg-dq]). city96 has published only one GGUF since 2025-08 (FLUX.2 dev); third parties supply the rest ([city96 on HF][c96u]).

### Draw Things

Draw Things quantizes the weights itself (`.q8p/.q6p/.q5p` and "S" variants, [Quantizer.swift L45-L150][dt-q]) and hosts them outside HF. Dates are GitHub releases of draw-things-community; the App Store build may differ by a few days [INFERENCE].

| Model | Release | Lag |
|---|---|---|
| Z-Image-Turbo | [v1.20251207.0][dt-r1] 2025-12-08: q8p, 8-bit S, q6p denoiser; q8p Qwen3-VL-4B encoder; f16 VAE ([ModelZoo.swift L1209-L1235][dt-mz1]) | 11 |
| FLUX.2 [dev] | [v1.20251219.0][dt-r2] 2025-12-22 | 27 |
| Qwen-Image-2512 / Edit-2511 | [v1.20260101.0][dt-r3] 2026-01-04; precision fix 2026-01-06 ("more plastic than it is", [v1.20260105.0][dt-r3b]) | 4 / ≤ 18 |
| FLUX.2 [klein] | [v1.20260120.0][dt-r4] 2026-01-20 | 5 |
| ERNIE-Image | [v1.20260418.1][dt-r5] 2026-04-20 | 5 |
| Krea 2 | [v1.20260716.0][dt-r6] 2026-07-20 | 27 |
| Qwen-Image-2.1 | [v26.0928.0][dt-r7] 2026-09-28, with a SpinQuant "4-bit S … for lower RAM usage" | 8 |
| FLUX.3 Image Klein | [v26.1023.0][dt-r8] 2026-10-23 | 8 |
| MiniMax-H3-Image | commit [`f064901`][dt-h3] "Add MiniMax H3 Image" on `main` 2026-11-20; no release yet | ≥ 15 |

### mflux

| Model | First release | Lag | Notes |
|---|---|---|---|
| Z-Image-Turbo | [v.0.13.0][mf-r1] 2025-12-03 | 6 | `-q 8` peak 7.5 GB ([z_image README L205-L213][mf-z]) |
| FLUX.2 [klein] | [v.0.15.0][mf-r2] 2026-01-18 | 3 | 4B `-q 8` peak 11.6 GB ([flux2 README L155-L160][mf-k]) |
| Qwen-Image-2512 | [v.0.19.0][mf-r3] 2026-08-18 | 230 | |
| Qwen-Image-Edit-2511 | name in v.0.19.0, correct weights only in [v.0.22.0][mf-r5] 2026-10-08 | ≤ 295 | "Until now those names loaded 2509" |
| ERNIE-Image | [v.0.18.0][mf-r4] 2026-06-07 | 53 | |
| Krea 2 | [v.0.18.1][mf-r6] 2026-08-07 | 45 | encoder stays bf16 |
| Qwen-Image-2.1 | [v.0.20.0][mf-r7] 2026-09-21 | 1 | 8-bit at 512² ≈ 13 GB, ≈ 8 GB with `--low-ram` ([v.0.22.0][mf-r5]) |
| FLUX.3 Image Klein | [v.0.23.0][mf-r8] 2026-10-16 | 1 | q4 peaked at **18.5 GB** on an M4 Pro, measured before the v.0.24.0 strip decode ([flux3 README L131-L141][mf-f3]) |
| FLUX.2 [dev], MiniMax-H3-Image | not supported | n | |

Its pre-quantized builds trail: `mflux-community` q3–q8 repos came in batches in 2026-07…10, 200+ days after Z-Image and klein ([HF list][mfc]). `mlx-community/Qwen-Image-2512-*bit` appeared 2026-01-01.

### InvokeAI, diffusers, Unsloth Studio, Atomic Chat, Ollama, Mochi

- **InvokeAI** (first tag including the PR, rc/alpha counted): Z-Image-Turbo 29 ([v6.10.0rc1][inv-r1]), klein 12 ([v6.11.0][inv-r2]), Qwen-Image-2512 ≈ 120 ([v6.13.0.rc1][inv-r3]), ERNIE 107 ([v6.14.0-rc1][inv-r4]; GGUF only 179), Krea 2 38, FLUX.2 dev 264, Qwen-Image-2.1 29 and FLUX.3 Klein 18 in v7 alphas only ([v7.0.0-alpha.3][inv-r5], [alpha.4][inv-r6]); MiniMax-H3-Image is an open PR ([#10240][inv-pr]).
- **diffusers**: the pipeline usually merges on launch day or before (Z-Image [#12703][df-p1] −2, FLUX.2 dev [#12711][df-p2] 0, klein [#12982][df-p3] 0, ERNIE [#13432][df-p4] −4, Krea 2 [#14045][df-p5] −1, Qwen-Image-2.1 [#14804][df-p6] −2, FLUX.3 [#15046][df-p7] 0, H3-Image [#15230][df-p8] 0). The PyPI release follows 2–78 days later. The default load is the bf16 Source, which does not fit 16 GB for any reference model except klein 4B [INFERENCE].
- **Unsloth Studio**: Qwen-Image-2.1 in [v0.1.812-beta][us-r1] 2026-09-22 (2), Qwen-Image-2.1-Turbo in [v0.1.906-beta][us-r2] 2026-10-12, FLUX.3 Klein in [v0.1.909-beta][us-r3] 2026-10-19 (4), MiniMax-H3-Image GGUF "for CPU, Mac and AMD" in [v0.1.918-beta][us-r4] 2026-11-16 (6). Its own GGUFs: Qwen-Image-2.1 2026-09-21, FLUX.3 Klein 2026-10-16, MiniMax-H3-Image 2026-11-12.
- **Atomic Chat** added sd.cpp image generation in [v2.0.44][ac-r1] 2026-09-21 ("Qwen 2.1, Krea 2 Turbo, FLUX and other models"), Qwen-Image-2.1-Turbo the day it appeared ([v2.2.3][ac-r2] 2026-10-09), and MiniMax-H3-Image in [v2.4.5][ac-r3] 2026-11-13 (3). Its curated catalog lives in a remote config and can add a model without an app release ([decision 2026-09-10][ac-cat]). Its FLUX.3 Klein GGUFs are dated 2026-10-20 (5).
- **Ollama** shipped experimental MLX image generation in [v0.14.0][ol-r1] 2026-01-10 (Z-Image-Turbo, 44) and klein in [v0.14.3][ol-r2] 2026-01-16 (1). It **removed** it in commit [`4713800`][ol-rm] (2026-07-28), and v0.40.18 rejects image models ([routes.go L494][ol-rt]).
- **Mochi Diffusion** runs klein (44, [v6.0][mo-r1]) and Z-Image-Turbo (340, [v6.3][mo-r2]) through iris.c at the original precision. The README says Z-Image "may fail to run on Macs with less than 24GB RAM" ([README L122][mo-rm]).

## 2. Per-component precision

Every tool that runs a split pipeline lets the user give the text encoder, the denoiser and the VAE different precisions, and none decides that split from the machine's memory:

- **stable-diffusion.cpp:** each component is its own file (`--diffusion-model`, `--llm`, `--vae`) at any GGUF or safetensors type. `--type` converts everything at load; `--tensor-type-rules "^vae\.=f16,model\.=q8_0"` sets a type per tensor pattern ([common.cpp L502-L505][sd-ttr], [quantization_and_gguf.md][sd-qg]). `--auto-fit` decides placement only, never type ([backend.md L142-L203][sd-af]).
- **ComfyUI:** mutually exclusive dtype flags per component, `--fp16/bf16/fp8_*-unet`, `--*-text-enc`, `--*-vae` ([cli_args.py L79-L103][cu-cli]). In practice the user picks a file per loader node. On MPS the default compute is fp16, or bf16 from macOS 14 ([model_management.py L1922-L1996][cu-dt]).
- **Draw Things:** each ModelZoo entry fixes the denoiser, encoder and VAE files separately (Z-Image: q8p/q6p denoiser, q8p encoder, f16 VAE). The user picks among full, 8-bit S and 6-bit variants; no code maps RAM to a variant ([DeviceCapability.swift][dt-dc]).
- **mflux:** one `-q` value. The per-model code decides whether the encoder follows it ([quantization_strategies.md L6-L35][mf-qs]): Qwen-Image, Qwen-Image-2.1 and Krea 2 keep the encoder bf16, while Z-Image, klein and ERNIE quantize it with the DiT. No RAM query in `src` ([w8a8.py L33][mf-w8] is the only `device_info` use).
- **InvokeAI:** components are separate installed models; `precision: auto` only picks the compute dtype (fp16 on MPS) ([devices.py L218-L245][inv-dev]). The Mac guide says to "use a smaller build of the model" by hand ([low-vram-mode.mdx L67-L70][inv-lv]).
- **diffusers:** `PipelineQuantizationConfig(quant_mapping=…)` sets a backend per component ([quantization/overview.md L20-L92][df-q]). Of the backends, the docs say Quanto and SDNQ run on MPS ([quanto.md L25][df-qu], [sdnq.md L14][df-sd]); BFL fp8/nvfp4 dequantize per forward off CUDA ([bfl.md L61-L63][df-bfl]).
- **Unsloth Studio:** "Precision: Auto … picks what fits" on NVIDIA, with the encoder "always … INT8-ConvRot". On a Mac the route is the GGUF denoiser the user picks ([FLUX.3 guide][us-f3]).
- **Atomic Chat:** the user picks a denoiser quant from a curated Q3_K_M…Q8_0 ladder; the encoder is one file per family ([decision 2026-09-10][ac-cat]).
- **Ollama (removed):** one int8/int4 type for encoder and DiT linears, VAE skipped ([create.go L174-L220][ol-cr]).
- **Apple ml-stable-diffusion:** `--quantize-nbits 2/4/6/8` for UNet and text encoder, plus per-layer mixed-bit recipes chosen at conversion ([README L161-L245][ap-q]).

## 3. Memory limits, swapping and hangs

- **stable-diffusion.cpp:** `--auto-fit` (default) puts diffusion, then encoder, then VAE on the first device with room: GPU (free − 512 MiB), then CPU RAM (reserving max(2 GiB, 10%)), then another GPU, then **disk** ([backend.md L142-L201][sd-af]). It also has `--params-backend diffusion=disk,te=cpu,vae=cpu`, `--offload-to-cpu`, `--mmap`, `--max-vram` and segmented execution ([performance.md L115-L203][sd-pf]). VAE decode retries with smaller tiles ([L97-L102][sd-vt]). No swap detection is documented.
- **ComfyUI:** on MPS the VRAM state is forced to SHARED, overriding `--lowvram/--novram` ([L567-L583][cu-vs]). Free memory comes from system virtual memory ([L297-L298][cu-mem]). Models are unloaded after use unless `--highvram`, `torch.mps.empty_cache()` is called ([L2164-L2165][cu-ec]), and the VAE retries tiled on OOM or tiles up front when its estimate exceeds free memory ([sd.py L1377-L1425][cu-vae]). There is no swap check. On unified memory, macOS swapping can replace the OOM those fallbacks wait for [INFERENCE].
- **Draw Things:** reads `physicalMemory`, sets tiers and a weight-streaming budget (16 GB → `_8GiB`), and measures available memory from Mach VM stats ([DeviceCapability.swift L65-L128, L334-L389][dt-dc]). Tiled decode and memory saver are manual; CPU offload is CUDA-only. No swap warning was found.
- **mflux:** `--low-ram` sets a 1 GB MLX cache, tiles the VAE and deletes the encoder before denoising and the transformer before decode ([memory_saver.py L12-L80][mf-ms]). Swap is checked by hand with `vm_stat` per the README ([L780-L781][mf-sw]).
- **InvokeAI:** on Apple silicon models "load whole into system memory"; partial loading and VRAM settings do nothing ([low-vram-mode.mdx L17-L20][inv-lv]). The RAM cache is 50% of RAM − 2 GB, and dynamic sizing from memory pressure was rejected ([model_cache.py L57-L59, L2573-L2580][inv-mc]).
- **diffusers:** offload, group offload to disk, VAE slicing and tiling ([memory.md][df-mem]). The MPS page warns that performance "is very sensitive to memory pressure … the system automatically swaps" and advises attention slicing below 64 GB ([mps.md L64-L78][df-mps]). Nothing is automatic.
- **Atomic Chat:** estimates the resident set as DiT + VAE + encoder + 1.5 GiB per megapixel, checks it against Metal's ceiling **and** against RAM minus about 2.1 GiB with the CPU-side encoder counted. It shows "expect heavy swapping and a very slow image" when the result is "no" ([fit.ts][ac-fit]). It chooses the offload policy before load and retries under a fallback on OOM ([decision 2026-09-30][acc-oom]). It tiles the VAE above 1 MP ([args.ts L213-L313][acc-args]). The fit header cites an 8 GB Mac that "swapped until the engine stopped" on Z-Image Q3_K_M (QA 2026-10-23).
- **Unsloth Studio:** tags GGUF sizes **recommended / TIGHT / OOM** for the device; "A TIGHT model may use system memory and run more slowly" ([diffusion-image docs][us-doc]). Its "Mac with 16 GB" row for FLUX.3 Klein is labeled an estimate; measurements are from an A100 and an L4 ([FLUX.3 guide][us-f3]).
- **Ollama (v0.32.5):** refused to load when the tensor total exceeded free memory ([server.go L67-L90][ol-ld]). This is the only hard refusal found.

## 4. Quality measurement against the Source

| Who | What is measured | Metric and set | Scope limits |
|---|---|---|---|
| unsloth GGUF cards (Qwen-Image-2.1, -Turbo, FLUX.3 Klein, MiniMax-H3-Image, MOSS-Image; none before 2026-09) | each GGUF denoiser vs the bf16 DiT, same prompt, seed, noise and encoder | LPIPS (AlexNet) mean/max, SSIM, PSNR. FLUX.3 Klein: 19 renders, `UD-Q4_K_XL` LPIPS 0.038 ([card][us-f3c]). H3-Image: 8 prompts × 2 seeds with 95% interval, on a CUDA sd.cpp build ([card][us-h3c]) | denoiser only; encoder swaps measured once (Qwen-Image-2.1 card); NVIDIA only; no Mac run |
| Atomic Chat GGUF cards (FLUX.3 Klein, Qwen-Image-2.1-Turbo, MiniMax-H3-Image) | each GGUF vs BF16, **and every other published Klein GGUF**; encoder quants measured separately | LPIPS with 95% CI, "vs. new seed" scale, SSIM, PSNR, text-rendering subset; 48 captions × 2 seeds; renders and logs public in [a metrics dataset][acm] ([card][acf3]) | denoiser and encoder measured one at a time, not combined; GPU; ComfyUI cross-check on 8 captions |
| mflux model READMEs | q8/q6/q4 vs bf16 | PSNR: FLUX.3 Klein 44.5/28.2/25.2 dB, one seed ([L131-L139][mf-f3]); Qwen-Image-2.1 edit q8 29.5 dB; MOSS 8-bit 36.6 dB | a few models; no numbers for Z-Image, klein, ERNIE, Krea 2, Qwen-2512 |
| Apple ml-stable-diffusion | palettized UNet vs fp16 | PSNR of denoised latents (SDXL: 8-bit ≈ 65 dB, 2.81-bit MBP ≈ 67 dB) ([README L188-L199][ap-p]) | Stable Diffusion only, dormant |
| stable-diffusion.cpp, ComfyUI, ComfyUI-GGUF, city96, Comfy-Org, Draw Things, InvokeAI, diffusers, Ollama, Mochi, DiffusionBee | — | none; sample images or prose ("near-lossless", "slightly different image") | — |

None of these is a Quality loss in the map's sense. None is the whole pipeline as a Fit runs it: they vary one component and keep the others at the reference. None is rendered by the runtime and precisions a 16 GB Mac would use. None is shown by the tool for the build it is about to run. The same cards show why the whole pipeline matters: the MiniMax-H3-Image card notes that a Q4_K_M encoder alone moves the image by LPIPS 0.17 ([card L120][us-h3c]), and Atomic Chat finds a Q4_K_M encoder moves Klein's picture more (0.088) than its 4-bit denoiser does (0.081) ([card][acf3]).

## Verdict on the map's three gaps

| Gap | Closed? | Evidence |
|---|---|---|
| A Usable build of a new model within days of release | **Closed** for builds that fit; **open** for proof that they are Usable on 16 GB | stable-diffusion.cpp 0–4 days and ComfyUI −2–1 days on the six models that fit, with GGUFs within 0–3 days; Unsloth Studio and Atomic Chat package them within 0–6 days; mflux 1–6 days for Z-Image, the Klein models and Qwen-Image-2.1. No source measures seconds per image or Responsiveness on a 16 GB Mac: "fits" is a file size or a GPU number. Unsloth's 16 GB Mac row is an explicit estimate ([FLUX.3 guide][us-f3]), and mflux measured FLUX.3 Klein q4 at an 18.5 GB peak. |
| A measured Quality loss | **Partly closed**: per component, for a few new models | unsloth and Atomic Chat publish LPIPS, SSIM and PSNR against BF16 on fixed prompts and seeds, and Atomic Chat publishes the renders; mflux publishes PSNR for a few models. Still open: the whole pipeline's loss for the precisions actually combined, rendered on the runtime that runs the Fit, for every new model, reported by the tool. |
| Precision chosen per component under a memory limit | **Open** | Per-component precision is everywhere, but only by hand. Memory-driven automation picks placement (sd.cpp `--auto-fit`, ComfyUI, Atomic Chat, Draw Things) or flags one denoiser size (Unsloth recommended/TIGHT/OOM, Atomic Chat ok/maybe/no). No tool trades encoder, denoiser and VAE precision jointly against a budget or a measured loss. Swap: Atomic Chat warns, removed Ollama refused, the rest are silent. |

## Facts later tickets depend on

- **Runtime candidates by lag:** stable-diffusion.cpp (ggml + Metal, GGUF) and ComfyUI (PyTorch MPS) both reach day 0 or day 1 on the newest reference models; sd.cpp does it without PyTorch, and its GGUF k-quants run in ggml's Metal backend rather than being dequantized in torch as in ComfyUI-GGUF [INFERENCE]. MLX's only tracking tool is mflux: 1–6 days for Z-Image, the Klein models and Qwen-Image-2.1, 45–230 days for ERNIE, Krea 2 and Qwen-Image-2512. diffusers merges early, but its MPS quantization is Quanto or SDNQ only.
- **Community builds exist per component:** GGUF denoisers (leejet, unsloth, Atomic Chat, Abiray, vantagewithai, realrebelai) and GGUF encoders (unsloth Qwen3/Qwen3-VL GGUFs, Atomic Chat encoder GGUFs) are separate files, so a Fit can mix makers per component. Atomic Chat states Klein's encoder shards are byte-identical to `Qwen/Qwen3-VL-4B-Instruct` ([card][acf3]).
- **Reference packs in the wild:** Atomic Chat's metrics dataset (renders, per-tensor layouts, logs; created 2026-10-20) is the nearest published precedent for an image Reference pack ([dataset][acm]).
- **Encoder placement on Metal:** Atomic Chat always runs the text encoder on the CPU under Metal, and counts it against RAM because offloading on unified memory "saves no RAM" ([fit.ts][ac-fit]). On unified memory, CPU offload does not lower peak RAM; only disk-backed placement (sd.cpp `diffusion=disk`, diffusers group offload to disk, mflux `--ssd-stream` on `main`) does [INFERENCE].
- **MPS limits:** fp8 storage refused (ComfyUI) or unpacked to bf16 (InvokeAI); no int8 matmul in PyTorch MPS; NF4/bitsandbytes not usable on macOS in InvokeAI.
- **Models that cannot be a 16 GB Fit with the Source pipeline:** FLUX.2 [dev] and MiniMax-H3-Image, short of disk streaming.

[sd]: https://github.com/leejet/stable-diffusion.cpp/tree/52491bd9f1fc15281791723bdff13baab7fd6819
[sd-rel]: https://github.com/leejet/stable-diffusion.cpp/releases
[sd-z]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/models/z_image.md
[sd-qe]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/models/qwen_image_edit.md#L9-L11
[sd-k]: https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF
[sd-e]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/models/ernie_image.md
[sd-kr]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/models/krea2.md
[sd-q21]: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF
[sd-mm]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/models/minimax_h3_image.md
[sd-c1]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-378-52b67c5
[sd-c2]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-385-34a6fd4
[sd-c3]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-444-a0adcfb
[sd-c4]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-473-9565c7f
[sd-c5]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-570-5c243db
[sd-c6]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-721-8caa3f9
[sd-c7]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-883-137f740
[sd-c8]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-968-8cd731e
[sd-c9]: https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1088-32d516c
[sd-ttr]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L502-L505
[sd-qg]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/quantization_and_gguf.md
[sd-af]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L142-L203
[sd-pf]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/performance.md#L115-L203
[sd-vt]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/performance.md#L97-L102
[cu]: https://github.com/comfyanonymous/ComfyUI/tree/a2103ae69d98d8ef94c150f5a472b89775eed929
[cu-c1]: https://github.com/comfyanonymous/ComfyUI/commit/6b573ae0cb
[cu-c2]: https://github.com/comfyanonymous/ComfyUI/commit/e9aae31fa2
[cu-c3]: https://github.com/comfyanonymous/ComfyUI/commit/3b832231bb
[cu-c4]: https://github.com/comfyanonymous/ComfyUI/commit/31283d2892
[cu-c5]: https://github.com/comfyanonymous/ComfyUI/commit/2a61015582
[cu-c6]: https://github.com/comfyanonymous/ComfyUI/commit/6bfaacc67c
[cu-c7]: https://github.com/comfyanonymous/ComfyUI/commit/20b46065ef
[cu-c8]: https://github.com/comfyanonymous/ComfyUI/commit/ede4e5fe8d
[cu-fp8]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L1305-L1316
[cu-int8]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L2110-L2117
[cu-cli]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/cli_args.py#L79-L103
[cu-dt]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L1922-L1996
[cu-vs]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L567-L583
[cu-mem]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L297-L298
[cu-ec]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L2164-L2165
[cu-vae]: https://github.com/comfyanonymous/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/sd.py#L1377-L1425
[cg]: https://github.com/city96/ComfyUI-GGUF/tree/6ea2651e7df66d7585f6ffee804b20e92fb38b8a
[cg-dq]: https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/dequant.py#L287-L301
[c96]: https://huggingface.co/city96/FLUX.2-dev-gguf
[c96u]: https://huggingface.co/city96
[dt]: https://github.com/drawthingsai/draw-things-community/tree/cb2d4c61d1b58816ce1843481e74293a240e5bf4
[dt-q]: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Apps/ModelQuantizer/Quantizer.swift#L45-L150
[dt-mz1]: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/ModelZoo/Sources/ModelZoo.swift#L1209-L1235
[dt-dc]: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/DataModels/Sources/DeviceCapability.swift#L65-L389
[dt-r1]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20251207.0
[dt-r2]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20251219.0
[dt-r3]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20260101.0
[dt-r3b]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20260105.0
[dt-r4]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20260120.0
[dt-r5]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20260418.1
[dt-r6]: https://github.com/drawthingsai/draw-things-community/releases/tag/v1.20260716.0
[dt-r7]: https://github.com/drawthingsai/draw-things-community/releases/tag/v26.0928.0
[dt-r8]: https://github.com/drawthingsai/draw-things-community/releases/tag/v26.1023.0
[dt-h3]: https://github.com/drawthingsai/draw-things-community/commit/f064901cd
[mf]: https://github.com/filipstrand/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7
[mf-r1]: https://github.com/filipstrand/mflux/releases/tag/v.0.13.0
[mf-r2]: https://github.com/filipstrand/mflux/releases/tag/v.0.15.0
[mf-r3]: https://github.com/filipstrand/mflux/releases/tag/v.0.19.0
[mf-r4]: https://github.com/filipstrand/mflux/releases/tag/v.0.18.0
[mf-r5]: https://github.com/filipstrand/mflux/releases/tag/v.0.22.0
[mf-r6]: https://github.com/filipstrand/mflux/releases/tag/v.0.18.1
[mf-r7]: https://github.com/filipstrand/mflux/releases/tag/v.0.20.0
[mf-r8]: https://github.com/filipstrand/mflux/releases/tag/v.0.23.0
[mf-z]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/z_image/README.md#L205-L213
[mf-k]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/flux2/README.md#L155-L160
[mf-f3]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/flux3/README.md#L131-L141
[mf-qs]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/docs/quantization_strategies.md#L6-L35
[mf-w8]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/common/quantization/w8a8.py#L33
[mf-ms]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/instances/memory_saver.py#L12-L80
[mf-sw]: https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/common/README.md#L780-L781
[mfc]: https://huggingface.co/api/models?author=mflux-community
[inv]: https://github.com/invoke-ai/InvokeAI/tree/1bd3954850c1eb1efac041854f8f0ba098f1b47a
[inv-r1]: https://github.com/invoke-ai/InvokeAI/releases/tag/v6.10.0rc1
[inv-r2]: https://github.com/invoke-ai/InvokeAI/releases/tag/v6.11.0
[inv-r3]: https://github.com/invoke-ai/InvokeAI/releases/tag/v6.13.0.rc1
[inv-r4]: https://github.com/invoke-ai/InvokeAI/releases/tag/v6.14.0-rc1
[inv-r5]: https://github.com/invoke-ai/InvokeAI/releases/tag/v7.0.0-alpha.3
[inv-r6]: https://github.com/invoke-ai/InvokeAI/releases/tag/v7.0.0-alpha.4
[inv-pr]: https://github.com/invoke-ai/InvokeAI/pull/10240
[inv-dev]: https://github.com/invoke-ai/InvokeAI/blob/1bd3954850c1eb1efac041854f8f0ba098f1b47a/invokeai/backend/util/devices.py#L218-L245
[inv-lv]: https://github.com/invoke-ai/InvokeAI/blob/1bd3954850c1eb1efac041854f8f0ba098f1b47a/docs/src/content/docs/configuration/Optimization/low-vram-mode.mdx#L17-L70
[inv-mc]: https://github.com/invoke-ai/InvokeAI/blob/1bd3954850c1eb1efac041854f8f0ba098f1b47a/invokeai/backend/model_manager/load/model_cache/model_cache.py#L57-L59
[df]: https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8
[df-p1]: https://github.com/huggingface/diffusers/pull/12703
[df-p2]: https://github.com/huggingface/diffusers/pull/12711
[df-p3]: https://github.com/huggingface/diffusers/pull/12982
[df-p4]: https://github.com/huggingface/diffusers/pull/13432
[df-p5]: https://github.com/huggingface/diffusers/pull/14045
[df-p6]: https://github.com/huggingface/diffusers/pull/14804
[df-p7]: https://github.com/huggingface/diffusers/pull/15046
[df-p8]: https://github.com/huggingface/diffusers/pull/15230
[df-q]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/quantization/overview.md#L20-L92
[df-qu]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/quantization/quanto.md#L25
[df-sd]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/quantization/sdnq.md#L14
[df-bfl]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/quantization/bfl.md#L61-L63
[df-mem]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/optimization/memory.md#L170-L407
[df-mps]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/optimization/mps.md#L64-L78
[us]: https://github.com/unslothai/unsloth/tree/2039322f9bdb5c1b8bd58d834e86cbda705d7c0d
[us-doc]: https://unsloth.ai/docs/basics/diffusion-image
[us-f3]: https://unsloth.ai/docs/models/flux-3
[us-r1]: https://github.com/unslothai/unsloth/releases/tag/v0.1.812-beta
[us-r2]: https://github.com/unslothai/unsloth/releases/tag/v0.1.906-beta
[us-r3]: https://github.com/unslothai/unsloth/releases/tag/v0.1.909-beta
[us-r4]: https://github.com/unslothai/unsloth/releases/tag/v0.1.918-beta
[us-f3c]: https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/12ef13ce920d53f45fe805093910c899668b9e76/README.md
[us-h3c]: https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF/blob/0d25c7349e007529691bbc0f4426b579fce7de01/README.md
[ac]: https://github.com/AtomicBot-ai/Atomic-Chat/tree/c45dc5d6354a460e43e97fa6c9619a1dd5fc46b3
[acc]: https://github.com/AtomicBot-ai/atomic-chat-core/tree/03097b5927afc7d585292dbe3c946638c32d788e
[ac-r1]: https://github.com/AtomicBot-ai/Atomic-Chat/releases/tag/v2.0.44
[ac-r2]: https://github.com/AtomicBot-ai/Atomic-Chat/releases/tag/v2.2.3
[ac-r3]: https://github.com/AtomicBot-ai/Atomic-Chat/releases/tag/v2.4.5
[ac-cat]: https://github.com/AtomicBot-ai/Atomic-Chat/blob/c45dc5d6354a460e43e97fa6c9619a1dd5fc46b3/docs/decisions/2026-09-10-curate-the-diffusion-model-catalog-in-atomic-chat-conf.md
[ac-fit]: https://github.com/AtomicBot-ai/Atomic-Chat/blob/c45dc5d6354a460e43e97fa6c9619a1dd5fc46b3/web-app/src/lib/diffusion/fit.ts
[acc-oom]: https://github.com/AtomicBot-ai/atomic-chat-core/blob/03097b5927afc7d585292dbe3c946638c32d788e/docs/decisions/2026-09-30-fall-back-to-offload-when-the-gpu-runs-out-of-memory.md
[acc-args]: https://github.com/AtomicBot-ai/atomic-chat-core/blob/03097b5927afc7d585292dbe3c946638c32d788e/src/diffusion/args.ts#L213-L313
[acf3]: https://huggingface.co/AtomicChat/FLUX-3-Image-Klein-GGUF/blob/b88bc03edf40e24846eaf7270b2ba9c2a9a1637f/README.md
[acm]: https://huggingface.co/datasets/AtomicChat/FLUX-3-Image-Klein-GGUF-metrics/tree/c04c238a1c02c6000b461828c668d6077f648459
[ol]: https://github.com/ollama/ollama/tree/eec8e0b9458b8a01be0c216a9cc53eefde24ef50
[ol-r1]: https://github.com/ollama/ollama/releases/tag/v0.14.0
[ol-r2]: https://github.com/ollama/ollama/releases/tag/v0.14.3
[ol-rm]: https://github.com/ollama/ollama/commit/4713800b08b2ddf5e14acf8398953cf7b12f169b
[ol-rt]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/server/routes.go#L494
[ol-cr]: https://github.com/ollama/ollama/blob/eec8e0b9458b8a01be0c216a9cc53eefde24ef50/x/create/create.go#L174-L220
[ol-ld]: https://github.com/ollama/ollama/blob/eec8e0b9458b8a01be0c216a9cc53eefde24ef50/x/imagegen/server.go#L67-L90
[mo]: https://github.com/godly-devotion/MochiDiffusion/tree/e9e0d75503c61ba274bdd9bb53b5ad88d0cdb866
[mo-r1]: https://github.com/godly-devotion/MochiDiffusion/releases/tag/v6.0
[mo-r2]: https://github.com/godly-devotion/MochiDiffusion/releases/tag/v6.3
[mo-rm]: https://github.com/godly-devotion/MochiDiffusion/blob/e9e0d75503c61ba274bdd9bb53b5ad88d0cdb866/README.md#L122
[ap]: https://github.com/apple/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2
[ap-q]: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L161-L245
[ap-p]: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L188-L199
[db]: https://github.com/divamgupta/diffusionbee-stable-diffusion-ui/releases/tag/2.5.3
[dk]: https://github.com/argmaxinc/DiffusionKit/tree/498e5dba5fb48b0f01cb6b5c2292a6dbea67a317
[mx]: https://github.com/ml-explore/mlx-examples/tree/c52cc748f876b898720720dd0e4c4632c9790ebd/flux
[bfl2]: https://bfl.ai/blog/flux-2
[bflk]: https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
[bfl3]: https://bfl.ai/blog/flux-3-image-klein
[zgh]: https://github.com/Tongyi-MAI/Z-Image
[q2512]: https://github.com/QwenLM/Qwen-Image/commits/main
[krea]: https://x.com/krea_ai/status/2069435590995812396
[mmrn]: https://platform.minimax.io/docs/release-notes/models
