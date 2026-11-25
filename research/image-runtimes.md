# Which runtimes run current text-to-image pipelines on a 16 GB Apple Silicon Mac, with which per-component precisions, offload and seed-determinism controls?

Map ticket: "Which runtimes run current text-to-image pipelines on a 16 GB Apple Silicon Mac, with which per-component precisions, offload and seed-determinism controls?" for "Map: image Fits, new text-to-image models on small Apple Silicon Macs". Researched 2026-11-25 by reading source at the commits below. Nothing was built, run or downloaded; sizes come from the Hugging Face tree API.

`[INFERENCE]` marks a conclusion no source states directly. The reference Device profile is the Mac mini M4 16 GB (default Metal working set 11.84 GiB, ~21 GB free disk).

## Pinned commits

| Runtime | Repo | Commit | Commit date |
|---|---|---|---|
| stable-diffusion.cpp | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/52491bd9f1fc15281791723bdff13baab7fd6819) | `52491bd` (tag `master-1133-52491bd`) | 2026-11-25 |
| its ggml (submodule, leejet's fork) | [leejet/ggml](https://github.com/leejet/ggml/tree/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539) | `eeda9ec` | 2026-11-14 |
| mflux | [filipstrand/mflux](https://github.com/filipstrand/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7) (now mflux-community/mflux) | `3d1d17a` (latest release v0.24.0, 2026-10-27) | 2026-11-25 |
| MLX | [ml-explore/mlx](https://github.com/ml-explore/mlx/tree/5996d291ee88db9535b33eec458ca241428c0d2b) | `5996d29` | 2026-11-25 |
| diffusers | [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8) | `3d33683` (latest release v0.43.0, 2026-11-12) | 2026-11-25 |
| PyTorch (MPS) | [pytorch/pytorch](https://github.com/pytorch/pytorch/tree/be9af63a628c1a99030b8e7ab45d2339f81db461) | `be9af63` | 2026-11-25 |
| ComfyUI | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI/tree/a2103ae69d98d8ef94c150f5a472b89775eed929) | `a2103ae` (latest release v0.46.0, 2026-11-18) | 2026-11-25 |
| ComfyUI-GGUF | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/tree/6ea2651e7df66d7585f6ffee804b20e92fb38b8a) | `6ea2651` | 2026-01-12 |
| Draw Things (open part) | [drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community/tree/cb2d4c61d1b58816ce1843481e74293a240e5bf4) | `cb2d4c6` (release v26.1023.0, 2026-10-23) | 2026-11-24 |
| Core ML | [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2) | `ea2805d` (last code change 2025-07-03) | 2026-09-11 |

Other repos read (DiffusionKit, mlx-examples, swift-coreml-diffusers, coremltools, InvokeAI, SD.Next, candle, Ollama, LocalAI, koboldcpp, Nunchaku, DiffusionBee) are pinned in their sections below.

## Answer

- **Four runtimes run current DiT/flow-matching pipelines on a 16 GB Mac and are worth measuring: stable-diffusion.cpp, mflux, ComfyUI (+ComfyUI-GGUF) on MPS and diffusers on MPS.** Draw Things is a fifth, but it keeps its own weight format and exposes no latents. Core ML (ml-stable-diffusion, DiffusionKit) stops at SD3-medium and is dormant. DiffusionBee is dead, Ollama removed its image generation on 2026-07-28, Nunchaku is CUDA-only, and candle is example code.
- **Native low-bit Metal kernels:** only stable-diffusion.cpp (GGUF k- and i-quants, dequantized per tile inside the Metal matmul) and mflux (MLX affine 3–8-bit, weight-only) have them. ComfyUI-GGUF and diffusers' GGUF/fp8 paths store weights low-bit but dequantize to fp16/bf16 in PyTorch at every forward, and fp8 on MPS is storage only.
- **Per-component precision:** stable-diffusion.cpp (one file per component plus `--tensor-type-rules` per tensor regex), diffusers (`PipelineQuantizationConfig` per component) and ComfyUI (separate UNET/CLIP/VAE loaders) set it freely. mflux has one `-q` per model, with some text encoders pinned to BF16. Draw Things picks its codecs per tensor at conversion.
- **Weights larger than memory:** only stable-diffusion.cpp (`--mmap`, `--params-backend disk`, segmented streaming under `--auto-fit`) and Draw Things (`externalOnDemand` file-backed weights) stream from disk. MLX, diffusers and ComfyUI need the working weights resident; beyond RAM they swap.
- **Text encoder out of the way before denoising:** every viable runtime has it. mflux always frees it; ComfyUI runs it on the CPU by default on a Mac and unloads models after use; stable-diffusion.cpp releases its staged copies and can put it on the CPU or disk; diffusers needs `encode_prompt` then `del`. All except mflux and Draw Things can run a chosen component on the CPU. VAE tiling exists everywhere except Core ML. stable-diffusion.cpp and ComfyUI retry a failed VAE decode with tiles on their own.
- **Determinism:** no runtime documents bit-identical repeat runs on Metal, so the baseline must test it. The initial noise is backend-independent in stable-diffusion.cpp (Philox, matches torch-CUDA noise), ComfyUI (CPU `torch.manual_seed`, fp32) and diffusers with a CPU `Generator`. mflux uses MLX's own RNG, so its seeds match no other runtime. ComfyUI's SDE/ancestral samplers draw per-step noise on the MPS device.
- **Fixed initial latents in and per-step denoiser output out:** diffusers has both as plain API (`latents=`, `callback_on_step_end`). ComfyUI has both (`LoadLatent` + `SamplerCustomAdvanced` + `DisableNoise`; an in-process callback gets `x0` and `x` per step), but over HTTP it only streams JPEG previews. stable-diffusion.cpp reads per step through its C preview callback (denoised and noisy tensors) and takes fixed noise only through its DGL graph scripting. mflux reads per step (`latents`, `denoised`) but takes no initial latents without patching its noise function. Draw Things and Core ML can feed in neither.
- **Handover:** HTTP servers in stable-diffusion.cpp (`sd-server`: OpenAI-style and A1111-style), ComfyUI (`/prompt` + websocket) and Draw Things (gRPC + A1111). CLIs in stable-diffusion.cpp, mflux and Draw Things. diffusers is a library only.
- **Support lag:** in 2026, new models land in ComfyUI, diffusers and stable-diffusion.cpp on day 0–1 (MiniMax-H3-Image: all three on 2026-11-10/11). Draw Things takes 2–21 days. mflux takes days for small models and skips the big ones (no FLUX.2 [dev], no MiniMax-H3-Image).
- **Licences and pinning:** all can be pinned to a commit. stable-diffusion.cpp and mflux are MIT, diffusers Apache-2.0, ComfyUI and Draw Things GPL-3.0. stable-diffusion.cpp vendors **leejet's own ggml fork**, so it cannot share sarto's llama.cpp pin.

## Runtime × capability table

"Per-step out" means reading the denoiser's prediction or the latents after each step from code, not a preview image.

| | stable-diffusion.cpp | mflux (MLX) | diffusers on MPS | ComfyUI (+ComfyUI-GGUF) on MPS | Draw Things | Core ML (ml-stable-diffusion) |
|---|---|---|---|---|---|---|
| **Current architectures** | FLUX.1, FLUX.2 dev + klein, FLUX.3 Image Klein, Qwen-Image 1 / 2.1 (+Edit), Z-Image, SD3.5, Chroma, MiniMax-H3-Image, MOSS-Image (+Edit), SenseNova U1.5, HiDream-O1. No HunyuanImage | Z-Image(-Turbo), FLUX.2 klein (+edit), FLUX.3 Image Klein, Qwen-Image (+Edit-2511), Qwen-Image 2.1, MOSS-Image, SenseNova U1.5. No FLUX.2 dev, HunyuanImage, MiniMax-H3-Image, SD3.5 | FLUX.1/.2 dev + klein, Qwen-Image 1 / 2.1 (+Edit), Z-Image, HunyuanImage 2.1, MiniMax-H3-Image (Modular only), Chroma, HiDream, SD3.5. No MOSS, SenseNova | FLUX.1/.2 (+klein), Qwen-Image 1 / 2.1 (+Edit), Z-Image, HunyuanImage 2.1, MiniMax-H3-Image, MOSS-Image, SenseNova U1.5, HiDream, Chroma, SD3.5 | SD3.5, FLUX.1, FLUX.2 dev + klein, Qwen-Image 2.1 (+Edit-2511), Z-Image, HiDream, ERNIE, MOSS-Image, MiniMax-H3-Image | SD1/2, SDXL, SD3-medium only |
| **Per-component precision** | Yes: a file per component, `--tensor-type-rules` regex, `--type` | Partly: one `-q` per model; some TEs forced BF16; VAE ~BF16 | Yes: per-component dtype and `PipelineQuantizationConfig` | Yes: UNET/CLIP/VAE loaders, `weight_dtype` | Yes: codec per tensor/file at conversion | Yes, at conversion (one `.mlpackage` each) |
| **Quant types on Metal** | GGUF q4_0/1, q5_0/1, q8_0, q2_K–q6_K, iq1_s/m, iq2, iq3, iq4_nl/xs; INT8 convrot | MLX affine 3/4/5/6/8-bit (group 64); no GGUF | GGUF Q/K/IQ2–IQ4, quanto int8/int4, fp8 storage; all dequantized in torch | fp8 e4m3fn/e5m2 (storage only); GGUF Q2_K–Q8_0, IQ4_NL/XS via ComfyUI-GGUF; dequantized in torch | q5p/q6p/q8p palettized, fp16 (own `.ckpt`) | palettized 2/4/6/8-bit |
| **CPU fallback / activation quant** | FP8, NVFP4, TQ1_0 matmuls rejected by Metal → CPU [INFERENCE]; INT8 convrot quantizes activations on the GPU | None; weight-only | Op errors unless `PYTORCH_ENABLE_MPS_FALLBACK=1`; no activation quant | int8 compute off on MPS; no activation quant | [INFERENCE] in-kernel dequant, fp16 activations | SD3 MMDiT must run fp32 on CPU+GPU |
| **Unload TE before denoising** | Yes (auto-fit releases staged copies; `te=disk`) | Yes, always | Yes (`encode_prompt`, `del`) | Yes (unload after use; TE on CPU by default) | Yes (`offloadModels`) | Yes (`reduceMemory`) |
| **Component on CPU** | Yes (`--backend te=cpu,vae=cpu`) | No | Yes (`.to("cpu")`, offload hooks) | Yes (TE default on Mac, `--cpu-vae`) | Not exposed | Whole pipeline only [INFERENCE] |
| **Weights > RAM** | Yes (`--mmap`, disk params, segmented streaming) | No (lazy load, then resident) | No [INFERENCE] | mmap, then swap [INFERENCE] | Yes (`externalOnDemand`) | No |
| **VAE tiling** | Yes, plus automatic retry with smaller tiles | Yes (`--vae-tiling`, `--low-ram`) | Yes (+ slicing) | Yes, plus automatic tiled retry on OOM | Yes (`tiledDecoding`) | No |
| **OOM behaviour** | VAE retried; other failures error out; macOS form undocumented | swap, then Python exception from Metal malloc | swap, then `RuntimeError: MPS backend out of memory` at the watermark | node error, server stays up [INFERENCE] | undocumented | app crash (README) |
| **Seed RNG** | Philox `cuda` (default, = torch-CUDA noise) or `cpu` (= ComfyUI), same on every backend | MLX RNG (matches nothing else) | CPU `Generator` recommended (= any machine) | CPU `torch.manual_seed` fp32; SDE samplers draw on MPS | emulates torch CPU / CUDA RNG | numpy, torch-CPU or CUDA emulation |
| **Bit-exact repeat on same Mac** | undocumented: test | undocumented: test | undocumented: test | `--deterministic` "might not" guarantee it | undocumented: test | per compute unit [INFERENCE] |
| **Feed fixed initial latents** | Only via DGL scripting (`tensor::load` + `add_noise`) | No (patch `create_noise`) [INFERENCE] | Yes (`latents=`) | Yes (`LoadLatent` + `SamplerCustomAdvanced` + `DisableNoise`) | No (init image only) | No |
| **Per-step out** | Yes: preview callback with denoised and noisy tensors; graph-eval hook | Yes: callbacks get `latents` and `denoised` | Yes: `callback_on_step_end` reads/replaces `latents` | In process: callback `(i, x0, x)`; over HTTP only JPEG previews | Preview images only | Yes: `progressHandler` latents |
| **Handover** | C library, `sd-cli`, `sd-server` (OpenAI + A1111 API) | CLI + Python library; no server | Python library only | HTTP `/prompt` + websocket; Desktop app | App, gRPC server, A1111 HTTP, CLI | Swift library + CLI |
| **License** | MIT | MIT | Apache-2.0 | GPL-3.0 (GGUF pack Apache-2.0) | GPL-3.0 + CLA | MIT |
| **Build / pin** | CMake `-DSD_METAL=ON`; pin sha, ggml fork follows via submodule | pip from git sha + `mlx==` | pip from git sha + torch | git sha + locked torch, `comfy-aimdo` | `swift build` or Bazel; sha or dated tag | SwiftPM revision |

Wrappers add nothing to measure: InvokeAI and SD.Next wrap diffusers (both already run MiniMax-H3-Image), LocalAI pins stable-diffusion.cpp `3b12781`, and koboldcpp bundles its own stable-diffusion.cpp fork.

## Support lag

Release dates are the authors' own news entries: [Qwen-Image README L39](https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/README.md#L39), [FLUX.2 README L17-L18](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/README.md#L17-L18), [Z-Image README L42](https://github.com/Tongyi-MAI/Z-Image/blob/26f23eda626ffadda020b04ff79488e1d72004cd/README.md#L42), [HunyuanImage-2.1 README L40](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/blob/307df8801d176740dafb67b2872c831cb9362cf9/README.md#L40). MiniMax-H3-Image has no dated news entry; its Hugging Face repo was created 2026-11-03 and secondary reports give 2026-11-10. Support dates are merge-commit dates or the runtime's own news. "—" means not supported at the pin; blank means not checked.

| Model (release) | stable-diffusion.cpp | mflux | diffusers | ComfyUI | Draw Things |
|---|---|---|---|---|---|
| Qwen-Image (2025-08-04) | 2025-10-12 (~10 wk) | | 2025-08-04 (`8e53cd9`, day 0) | 2025-08-05 (`c012400`, 1 d) | |
| HunyuanImage-2.1 (2025-09-08) | — | — | 2025-10-23 (`a138d71`, ~6 wk) | supported | |
| FLUX.2 [dev] (2025-11-25) | 2025-11-30 (5 d) | — | 2025-11-25 (`5ffb73d`, day 0) | ≤ 2025-11-26 (day 0–1) | 2025-12-15 (`2693a0d`, ~3 wk) |
| Z-Image-Turbo (2025-11-26) | 2025-12-01 (5 d) | 2025-12-03 (PR #284, 7 d) | 2025-11-25 (`4088e8a`, day 0) | 2025-11-25 (`e9aae31`, day 0) | 2025-12-03 (`88cb386`, 7 d) |
| FLUX.2 [klein] (2026-01-15) | 2026-01-18 (3 d) | 2026-01-18 (PR #323, 3 d) | 2026-01-15 (`61f1756`, day 0) | 2026-01-15 (`3b83223`, day 0) | 2026-01-16 (`da7527c`, 1 d) |
| MiniMax-H3-Image (~2026-11-10) | 2026-11-11 ("Day-0") | — | 2026-11-10 (`875373a`, day 0) | 2026-11-10 (`ede4e5f`, day 0) | 2026-11-20 (`f064901`, ~10 d) |

stable-diffusion.cpp's [news list](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/README.md#L18-L36) also claims day-0 support for Qwen-Image-2.1 (2026-09-20), MOSS-Image and FLUX.3 Image Klein (2026-10-16). SenseNova U1.6, released 2026-11-24, is supported nowhere yet; the latest any runtime has is U1.5.

## What has to shrink to fit 16 GB

Component sizes as published (BF16 unless noted), from the Hugging Face tree API at the repos' current revisions. With the text encoder freed before denoising, the largest single component must fit in the 11.84 GiB working set [INFERENCE].

| Source model | Denoiser | Text encoder | VAE | Needs [INFERENCE] |
|---|---|---|---|---|
| FLUX.2 [klein] 4B | 7.75 GB | 8.04 GB | 0.17 GB | Fits at BF16, one component at a time |
| Z-Image-Turbo (6B) | 24.62 GB (FP32 files; ~12.3 GB at BF16) | 8.04 GB | 0.17 GB | 8-bit denoiser |
| FLUX.2 [klein] 9B | 18.16 GB | 16.38 GB | 0.17 GB | 8-bit denoiser and text encoder |
| Qwen-Image (20B) | 40.86 GB | 16.58 GB | 0.25 GB | ~4-bit denoiser, 8-bit text encoder |
| FLUX.2 [dev] (32B) | 64.45 GB | 48.02 GB | 0.34 GB | ~2–3-bit, or streaming from disk |
| MiniMax-H3-Image (33.1B) | 66.28 GB ([model card L126-L130](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/README.md)) | 66.7 GB | 2.34 GB (FP32) | Streaming from disk; even Q4 exceeds the working set |

Only stable-diffusion.cpp and Draw Things can run a component whose quantized weights still exceed the working set. mflux, diffusers and ComfyUI need it resident.

## Shortlist for the M4 baseline task

1. **stable-diffusion.cpp**: the lead candidate. It has native GGUF k/i-quants on Metal with precision per component, disk streaming for models larger than memory, CPU or disk placement per component, VAE tiling with retry, backend-independent Philox seeds, per-step tensors through the C API, an HTTP server, MIT licence and day-0 support in 2026. Measure: its own ggml fork, the Metal speed warning in its build docs, and same-machine determinism.
2. **mflux**: the MLX candidate the map makes eligible. Native MLX 3–8-bit kernels, automatic text-encoder freeing, `--low-ram` and per-step callbacks. Limits: one bit width per model, no CPU placement, no streaming, no fixed-latent input, an RNG of its own, and no large models (FLUX.2 [dev], MiniMax-H3-Image).
3. **ComfyUI + ComfyUI-GGUF on MPS**: the broadest day-0 coverage, real per-component loaders, text encoder on the CPU by default, automatic tiled VAE, CPU-seeded noise, latents in and per step in process, and an HTTP API. Limits: all low-bit weights are dequantized in PyTorch (no low-bit Metal kernels), fp8 is storage only, and it is GPL-3.0.
4. **diffusers on MPS**: the control. It runs the Source pipeline as its authors publish it, the same code a rented GPU uses for a Reference pack, and it is the only one with fixed latents in and per-step latents out as plain arguments. Expect it to be the slowest and to need one component resident at a time.

Optional: **Draw Things (`gRPCServerCLI`)**, for speed comparison only. It is fast to add models and streams from disk, but it uses its own `.ckpt` format, exposes no latents and is GPL-3.0. Not candidates: Core ML (no current DiT pipeline, dormant), DiffusionKit and mlx-examples (FLUX.1/SD3 era), candle (examples), DiffusionBee (dead), Ollama (image generation removed), Nunchaku (CUDA only), and the wrappers InvokeAI, SD.Next, LocalAI and koboldcpp.

# Per-runtime details

The sections below give each runtime's answers to the ticket's six bullets with permalinks at the pinned commits.

## stable-diffusion.cpp

Research only; nothing built or run. S = `https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819`, G = `https://github.com/leejet/ggml/blob/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539`.

### Pinned commits

| repo | sha | date (UTC) | note |
|---|---|---|---|
| leejet/stable-diffusion.cpp | 52491bd9f1fc15281791723bdff13baab7fd6819 | 2026-11-25T18:38:21Z | HEAD of master, released as tag `master-1133-52491bd` (2026-11-25T19:07:44Z) |
| leejet/ggml (submodule `ggml`) | eeda9ec48fc84a3df0bb9dcc19ae9053075a2539 | 2026-11-14T17:11:12+08:00 | leejet's fork, commit "Merge branch 'master' into sd.cpp" |

**ggml is not shared with llama.cpp.** sd.cpp uses ggml as a git submodule that points to **leejet's fork** `https://github.com/leejet/ggml.git` ([S/.gitmodules#L1-L3](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/.gitmodules#L1-L3)), pinned by the gitlink sha above. The fork adds types/ops that upstream ggml does not have, such as F8_E4M3/E5M2 tensors, INT8 convrot and NVFP4. A `SD_USE_UPSTREAM_GGML` build switch exists. It turns FP8 into F16 at load ([S/src/model_io/safetensors_io.cpp#L89-L99](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/src/model_io/safetensors_io.cpp#L89-L99)) and rejects FP8 GGUF files ([S/src/model_io/gguf_io.cpp#L65-L69](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/src/model_io/gguf_io.cpp#L65-L69)). Its README credits [ggml-org/ggml](https://github.com/ggml-org/ggml) "working in the same way as llama.cpp" ([S/README.md#L40](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/README.md#L40)). Other submodules: web UI frontend, libwebp, libwebm, cutlass.

### Answers to questions 1–6

#### 1. Architectures and how long support takes
README model list ([S/README.md#L43-L93](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/README.md#L43-L93)):
- **Image models:** SD1.x/2.x/XL, SD3/SD3.5, FLUX.1 dev/schnell, FLUX.2 dev/klein, FLUX.3 Image Klein/Klein Base, MOSS-Image Base/SFT, Chroma, Chroma1-Radiance, Qwen-Image, Qwen-Image 2.1, Z-Image (with DGL ControlNet), Z-Image L2P, MiniMax-H3-Image, SenseNova U1.5, Ovis-Image, HiDream-O1-Image.
- **Edit models:** FLUX.1-Kontext-dev, Qwen-Image-Edit series, MOSS-Image Edit.
- **Video models:** Wan2.1/2.2, MiniMax-H3, HunyuanVideo 1.5.
- **Docs only:** `docs/models/` also has pages for Ernie-Image, LongCat-Image, Lumina-type families (`lens`, `iris`, …), PixArt, Ideogram4, Krea2 and others, not checked one by one.
- **Not seen:** SenseNova **U1.6** (only U1.5 is listed). No HunyuanImage (2.x/3) image page appears, only HunyuanVideo 1.5 [INFERENCE from the file list].

How long support took (README news, [S/README.md#L16-L36](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/README.md#L16-L36), against Hugging Face repo `createdAt`, which is the repo creation date and can be earlier than the public release):

| model | HF repo created | sd.cpp support | lag |
|---|---|---|---|
| Qwen-Image | 2025-08-02 | 2025-10-12 | about 10 weeks |
| FLUX.2-dev | 2025-11-22 | 2025-11-30 | about 1 week |
| Z-Image(-Turbo) | 2025-11-25 | 2025-12-01 | about 6 days |
| FLUX.2-klein-4B | 2026-01-14 | 2026-01-18 | 4 days |
| MiniMax-H3-Image | released 2026-11-10 | 2026-11-11 ("Day-0") | about 1 day |
| Qwen-Image-2.1 | — | 2026-09-20 ("Day-0") | — |
| MOSS-Image, FLUX.3 Klein | — | 2026-10-16 ("Day-0") | — |

In 2026 new models now get support on day 0 or day 1. In 2025 the lag ran from weeks to months.

#### 2. Precision per component
- **Separate file per component:** the CLI takes `--diffusion-model`, `--vae`, `--llm`, `--t5xxl`, `--clip_l`, and so on ([S/examples/common/common.cpp#L388-L457](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L388-L457)). Each file can use its own dtype and quant.
- **Global override:** `--type` sets one weight type for everything and converts at load time ([S/examples/common/common.cpp#L747-L750](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L747-L750); [S/docs/guides/quantization_and_gguf.md#L1-L9](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/quantization_and_gguf.md#L1-L9)).
- **Per-tensor override:** `--tensor-type-rules "^vae\.=f16,model\.=q8_0"` sets the type per tensor-name regex ([S/examples/common/common.cpp#L502-L505](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L502-L505)).
- **Conversion:** `-M convert` writes GGUF or safetensors ahead of time.
- **Metal matmul kernels in the pinned fork:**
  - Supported: q4_0/1, q5_0/1, q8_0 and the k-quants q2_K…q6_K.
  - Supported i-quants: iq1_s, iq1_m, iq2_xxs/xs/s, iq3_xxs/s, iq4_nl/xs ([G/src/ggml-metal/kernels/mul_mm.metal#L909-L922](https://github.com/leejet/ggml/blob/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539/src/ggml-metal/kernels/mul_mm.metal#L909-L922)).
  - How they run: each tile is dequantized to half inside the kernel (`dequantize_*` → `simdgroup_half8x8`). Weights stay quantized in memory and are expanded on the fly. Activations are not quantized.
- **Rejected by Metal mul_mat:** `F8_E4M3`, `NVFP4` and `TQ1_0` ([G/src/ggml-metal/ggml-metal-device.m#L1525-L1535](https://github.com/leejet/ggml/blob/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539/src/ggml-metal/ggml-metal-device.m#L1525-L1535)). FP8 can only be copied/converted to F16 or BF16 on Metal ([G/…/ggml-metal-device.m#L1969-L1971](https://github.com/leejet/ggml/blob/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539/src/ggml-metal/ggml-metal-device.m#L1969-L1971)). On the fork build, FP8 safetensors stay F8_E4M3 ([S/src/model_io/safetensors_io.cpp#L94-L97](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/src/model_io/safetensors_io.cpp#L94-L97)). So **FP8 matmuls fall back to CPU through the ggml scheduler**, unless you re-type them with `--type`/rules [INFERENCE: standard ggml_backend_sched behaviour for ops a backend does not support].
- **INT8 (ComfyUI `int8_tensorwise`/convrot):** runs natively on Metal on Apple GPU family 7 or newer. Weights stay INT8 and **activations are rotated and quantized on the GPU**. On macOS 26 it uses TensorOps `matmul2d`; otherwise it uses SIMD-group kernels ([S/docs/guides/int8_convrot.md#L100](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/int8_convrot.md#L100)).
- **BF16:** BF16 × F16 mul_mat is unsupported on Metal ([G/…/ggml-metal-device.m#L1541-L1543](https://github.com/leejet/ggml/blob/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539/src/ggml-metal/ggml-metal-device.m#L1541-L1543)). In that case the op falls back to CPU or needs an F32 src1 [INFERENCE].

#### 3. Memory
- **Placement per component:** `--backend te=cpu,vae=metal,diffusion=…`. The old `--clip-on-cpu`/`--vae-on-cpu` flags are now deprecated aliases ([S/examples/common/common.cpp#L519-L598](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L519-L598); [S/docs/guides/backend.md#L455-L470](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L455-L470)).
- **Where weights live:** `--params-backend` can be `disk`, `cpu` or a GPU, per component. `--offload-to-cpu` is shorthand for `*=cpu` ([S/docs/guides/performance.md#L111-L128](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/performance.md#L111-L128)).
- **Default `--auto-fit on`:**
  - It places diffusion, then TE, then VAE weights on the GPU, then RAM, then another GPU, then disk ([S/docs/guides/backend.md#L186-L195](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L186-L195)).
  - "Ending a component's execution releases its staged GPU copies…" ([S/docs/guides/backend.md#L201-L203](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L201-L203)). In practice the TE's GPU copies are released before denoising whenever its parameters are offloaded. Parameters assigned permanently to a backend stay resident.
  - Automatic graph segmentation streams weights per segment and prefetches the next one, so a DiT larger than the GPU budget can still run ([S/docs/guides/backend.md#L228-L233](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L228-L233)).
  - `--max-vram` caps the budget ([S/examples/common/common.cpp#L541-L545](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L541-L545)).
- **mmap:** `--mmap` memory-maps the model ([S/examples/common/common.cpp#L585-L587](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L585-L587)), and the C API has the matching `enable_mmap` key. Mapped files count toward budgets at full size ([S/docs/guides/backend.md#L182-L184](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L182-L184)). Disk-backed params reload on demand, so a model larger than RAM can run, slowly.
- **VAE tiling:** `--vae-tiling`, `--vae-tile-size`, `--vae-relative-tile-size` ([S/examples/common/common.cpp#L1254](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L1254), [#L1602-L1607](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L1602-L1607)).
- **OOM:**
  - A VAE decode that fails to allocate is **retried automatically with smaller tiles**, even without `--vae-tiling`.
  - Other execution failures are not retried, and encode has no OOM retry ([S/docs/guides/performance.md#L97-L102](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/performance.md#L97-L102)).
  - Failed budget checks log free/total and tracked memory ([S/docs/guides/backend.md#L180-L181](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/backend.md#L180-L181)).
  - No doc says how a Metal command-buffer OOM shows up on macOS [INFERENCE: upstream ggml-metal reports a failed command buffer and aborts; unified memory may swap before that].
  - `docs/build.md` still warns that Metal is "highly inefficient" for very large matrices ([S/docs/build.md#L243-L251](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/build.md#L243-L251)).

#### 4. Determinism and access
- **RNG:** `--rng` / `--sampler-rng` accept `std_default`, `cuda` and `cpu` ([S/examples/common/common.cpp#L752-L757](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L752-L757)).
  - `cuda` is the default and matches stable-diffusion-webui's GPU RNG. `cpu` matches ComfyUI's RNG. The README advertises this as "cross-platform reproducibility" ([S/README.md#L128-L130](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/README.md#L128-L130)).
  - `cuda` is Philox and is computed the same way on every backend ([S/docs/dgl.md#L843-L851](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl.md#L843-L851); `src/core/rng_philox.hpp`). So the initial noise matches torch-CUDA noise for the same seed, independent of the backend.
- **Seed:** the CLI's default seed is 42 ([S/examples/cli/README.md#L20-L21](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/cli/README.md#L20-L21)).
- **Bit-identical runs:** no doc states that Metal produces the same bits across runs, and no doc lists nondeterministic Metal ops [INFERENCE: probably deterministic on one machine, because ggml-metal kernels use fixed reduction orders; not verified].
- **Feeding fixed initial latents:**
  - Not possible through the plain C API key table, which has `init_image`/`mask_image`/`ref_images` but no noise/latent key ([S/docs/guides/c_api.md#L395-L497](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/guides/c_api.md#L395-L497)).
  - Possible through the built-in DGL graph scripting: `tensor::load` reads `.npy` ([S/docs/dgl.md#L494-L504](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl.md#L494-L504)), then `add_noise`/`sample` take a caller-supplied noise tensor ([S/docs/dgl.md#L853-L887](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl.md#L853-L887)).
- **Reading per-step output:**
  - `sd_set_progress_callback` gives step, steps and time.
  - `sd_set_preview_callback(cb, mode none|proj|tae|vae, interval, denoised, noisy)` delivers per-step frames as `sd_tensor_t`, in denoised and/or noisy form.
  - `sd_set_backend_eval_callback` is a raw ggml graph-eval hook (sees any tensor) ([S/include/stable-diffusion.h#L287-L339](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/include/stable-diffusion.h#L287-L339)).
  - DGL `tensor::save` can dump any tensor to `.npy`.

#### 5. Handover
- C library (`stable-diffusion.h`, opaque `sd_opt_*` key/value API).
- `sd-cli`.
- `sd-server` with a built-in Web UI, an OpenAI-compatible `/v1/images/generations` and `/v1/images/edits`, an A1111-style `/sdapi/v1/txt2img|img2img|progress`, a native job API and API-key auth ([S/examples/server/api.md#L28-L66](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/server/api.md#L28-L66)).
- Prebuilt release binaries.
- Python binding `william-murray1204/stable-diffusion-cpp-python`, listed in [S/docs/ecosystem.md#L12](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/ecosystem.md#L12). Not checked; it likely lags HEAD [INFERENCE].

#### 6. Build and pinning
- **License:** MIT ([S/LICENSE](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/LICENSE)).
- **Build on macOS:** `cmake .. -DSD_METAL=ON` ([S/docs/build.md#L243-L251](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/build.md#L243-L251)).
- **Pinning:** pin any sd.cpp commit, check out with `--recursive`, and the ggml fork sha comes with it through the gitlink. Every master commit also gets a tag of the form `master-<n>-<sha7>`.
- **Caution:** HEAD moves several times a day (3 tagged builds on 2026-11-25), and the `--clip-on-cpu`-era flags are being deprecated. Pin, and re-check the flags whenever you bump the pin.

### Verdict
**Viable, and probably the strongest baseline candidate.**
- **Models:** It is a single C/C++ binary that tracks current models within 0–1 days (MiniMax-H3-Image on day 0).
- **Quantization on Metal:** It runs llama.cpp-style GGUF k-quants and i-quants natively on Metal, with the precision set separately for each component.
- **16 GB fit:** It has the memory controls an 11.84 GiB working set needs: auto-fit and offload, disk/mmap params with segmented streaming, CPU placement per component, and VAE tiling with automatic retry.
- **Determinism:** It gives torch-CUDA-compatible Philox seeds, and fixed noise plus per-step tensors are reachable through DGL and C callbacks.
- **Caveats:**
  - It uses leejet's ggml fork, not llama.cpp's, so it cannot share a llama.cpp pin.
  - FP8 checkpoints fall back to CPU on Metal; convert them to GGUF.
  - The docs still flag Metal performance on large matmuls.
  - Same-machine bit-determinism on Metal is undocumented and must be tested.
  - The API and flags churn fast, so pin the commit.

## MLX runtimes: mflux, DiffusionKit, mlx-examples

### Pinned commits
| Repo | SHA | Commit date |
|---|---|---|
| filipstrand/mflux (now redirects to mflux-community/mflux), v0.24.0 is latest release (2026-10-27) | 3d1d17a93ee45707f0e987eed2c494a3665296b7 | 2026-11-25 |
| ml-explore/mlx | 5996d291ee88db9535b33eec458ca241428c0d2b | 2026-11-25 |
| argmaxinc/DiffusionKit | 498e5dba5fb48b0f01cb6b5c2292a6dbea67a317 | 2025-04-11 (stale) |
| ml-explore/mlx-examples | 796f5b53cab69a3d48a44233ce21aae889e94a08 | 2026-04-06 |

M = `https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7`, X = `https://github.com/ml-explore/mlx/blob/5996d291ee88db9535b33eec458ca241428c0d2b`.

### 1. mflux (primary)
**1. Architectures.** The README model table ([M/README.md#L118-L144](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/README.md#L118-L144)) lists Z-Image (+Turbo), Krea 2, FLUX.2 **klein** 4B/9B (+base, kv, edit), "FLUX 3 Image Klein" (Oct 2026), Iris-3B, Ideogram 4, ERNIE-Image, Lens, Ming-Image, MOSS-Image, SenseNova-U1.5, DreamLite, Anima, Boogu, FIBO/FIBO 2, Qwen-Image (20B) + Qwen-Image-Edit/-2511, Qwen-Image 2.1 (+edit/reference), SeedVR2 upscaler and FLUX.1 (+Kontext, legacy). The source tree has no FLUX.2 [dev], HunyuanImage, MiniMax-H3-Image, SD3.5, Chroma, HiDream or Lumina (grep of `src/` and README found none of those names). Edit: FLUX.2 klein edit, Qwen-Image-Edit, Qwen 2.1 reference edit, FIBO 2 edit, FLUX.1 Kontext, plus img2img through `Img2Img`.
Support lag (model repo created on HF → mflux PR merged):
- Z-Image-Turbo: 2025-11-25 → PR #284 merged 2025-12-03 (~8 days).
- FLUX.2-klein-4B: 2026-01-14 → PR #323 merged 2026-01-18 (~4 days).
- SenseNova-U1.5-8B-MoT: 2026-08-19 → PR #989 merged 2026-11-12 (~3 months).
- Qwen-Image 2.1 support: PR #736 merged 2026-09-21. MOSS-Image PR #895 shows no merge (`merged_at` null), but the code is in `src/mflux/models/moss_image`, so it probably landed through another path [INFERENCE].
- FLUX.2-dev (created on HF 2025-11-22): still unsupported a year later.

**2. Precision.** `-q/--quantize` accepts {3,4,5,6,8} ([M/src/mflux/cli/defaults/defaults.py#L81](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/cli/defaults/defaults.py#L81), [parsers.py#L168](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/cli/parser/parsers.py#L168)). It applies MLX `nn.quantize` (affine, group size from the weight definition, default 64) ([weight_applier.py#L174-L203](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/common/weights/loading/weight_applier.py#L174-L203)). There is one bit width per model, not one per component. The exceptions:
- Some models mark the text encoder `skip_quantization` and keep it in BF16: Qwen-Image, Qwen-Image-Edit, Qwen-Image 2.1 t2i and Krea 2.
- Ming-Image has a separate `--text-encoder-quantize`.
- Lens ships a pre-quantized GPT-OSS encoder.
See [docs/quantization_strategies.md#L6-L50](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/docs/quantization_strategies.md#L6-L50) for the per-model TE-Q/DiT-Q/group-size table.

The VAE uses the default predicate `hasattr(module,"to_quantized")` ([z_image_weight_definition.py#L65-L66](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/z_image/weights/z_image_weight_definition.py#L65-L66)). Conv layers are not quantizable in MLX, so the VAE stays mostly BF16 [INFERENCE].

`mflux-save` writes quantized models, and loading reads the stored bits and group size ([weight_applier.py#L58-L127](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/common/weights/loading/weight_applier.py#L58-L127)). Compute defaults to bf16. `--compute-precision float16` is offered for M1 ([README#L159](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/README.md#L159)).

mflux does not read GGUF. It loads FP8 only for Ideogram 4 (a dedicated `fp8_linear.py`). On the MLX side, quantization modes are affine/mxfp4/nvfp4/mxfp8 ([X/python/mlx/nn/layers/quantized.py#L11-L23](https://github.com/ml-explore/mlx/blob/5996d291ee88db9535b33eec458ca241428c0d2b/python/mlx/nn/layers/quantized.py#L11-L23)). The default is weight-only: activations stay bf16/fp16, and the quantized matmul kernels dequantize weight groups in-kernel [INFERENCE from MLX kernel design]. Activation quantization exists only with `quantize_input=True` for nvfp4/mxfp8 ([L42-L44](https://github.com/ml-explore/mlx/blob/5996d291ee88db9535b33eec458ca241428c0d2b/python/mlx/nn/layers/quantized.py#L42-L44)), and mflux's affine path does not use it. There is no CPU fallback; everything runs on the Metal stream.

**3. Memory.**
- Text encoders are always deleted after encoding, even without `--low-ram`, for single-seed runs or when the prompt embeddings are cached ([callback_manager.py#L107-L131](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/callback_manager.py#L107-L131), [memory_saver.py#L45-L51,L80-L100](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/instances/memory_saver.py#L45-L100)).
- `--low-ram` does four things: it enables VAE tiling (when the model allows it), sets `mx.set_cache_limit` to 1 GB (or `--mlx-cache-limit-gb`), deletes the transformer after the loop and before VAE decode (single seed/image), and runs gc plus `mx.clear_cache` ([callback_manager.py#L92-L106](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/callback_manager.py#L92-L106), [memory_saver.py#L28-L33,L102-L113](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/instances/memory_saver.py#L28-L113)).
- `--vae-tiling` and `--vae-tile-size` (default 512, minimum 128) also work on their own ([parsers.py#L131-L134](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/cli/parser/parsers.py#L131-L134)). The FLUX.2 VAE never auto-tiles under low-RAM ([flux2 vae.py#L19](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/flux2/model/flux2_vae/vae.py#L19)).
- There is no CPU-offload or component-on-CPU option.
- Weights load lazily: `mx.load` safetensors builds a lazy `Load` primitive that reads file bytes on eval ([X/mlx/io/safetensors.cpp#L204-L211](https://github.com/ml-explore/mlx/blob/5996d291ee88db9535b33eec458ca241428c0d2b/mlx/io/safetensors.cpp#L204-L211)). There is no mmap, so a model larger than RAM cannot run; once evaluated, everything is in unified memory [INFERENCE].
- OOM: the MLX Metal allocator throws a C++ `runtime_error` that surfaces as a Python exception, with one of three messages: `[metal::malloc] Attempting to allocate … greater than the maximum allowed buffer size`, `Resource limit exceeded` or `[malloc] Unable to allocate` ([X/mlx/backend/metal/allocator.cpp#L116-L164](https://github.com/ml-explore/mlx/blob/5996d291ee88db9535b33eec458ca241428c0d2b/mlx/backend/metal/allocator.cpp#L116-L164)). Before that, it usually swaps: mflux's own comment reports swap use when the peak exceeds RAM ([callback_manager.py#L111-L117](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/callback_manager.py#L111-L117)).

**4. Determinism and access.**
- Noise comes from `mx.random.normal(key=mx.random.key(seed))` ([z_image_latent_creator.py#L8-L17](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/z_image/latent_creator/z_image_latent_creator.py#L8-L17)), an MLX threefry RNG that is independent of the torch/CUDA RNG. The same seed does not reproduce Diffusers images.
- The `--seed` help says "Random seed(s) for reproducibility" ([parsers.py#L148](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/cli/parser/parsers.py#L148)), and the docs show `--config-from-conf` replay ([qwen21/reference/README.md#L149-L174](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/reference/README.md#L149-L174)). No nondeterminism is documented. Bit-exact repeats on the same machine are likely, but this was not tested [INFERENCE].
- There is no CLI or API argument for initial latents. A library caller would have to replace `LatentCreator.create_noise` (monkeypatch) [INFERENCE]; img2img mixes an encoded image with noise ([latent_creator.py#L43-L68](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/common/latent_creator/latent_creator.py#L43-L68)).
- Per-step access works through the callback protocols `call_before_loop`/`call_in_loop`/`call_after_loop`, which receive `latents`. An in-loop callback can also declare `denoised` to get the model's per-step prediction ([callback.py#L23-L34](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/callback.py#L23-L34), [generation_context.py#L49-L60](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/callbacks/generation_context.py#L49-L60)).
- `--stepwise-image-output-dir` ([parsers.py#L316](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/cli/parser/parsers.py#L316)) decodes the image at every step. Boogu has no stepwise support.

**5. Handover.** Per-model CLIs (`mflux-generate-*`, `mflux-save`) and a Python library. Core has no server: HTTP servers are third-party (e.g. mfluxible, MFLUX-WEBUI), and `mflux.web` is a namespace reserved for external UIs ([README#L248-L300](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/README.md#L248-L300)).

**6. Build and pinning.** MIT ([LICENSE](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/LICENSE)). Pure Python; requires Python ≥3.11, `mlx>=0.32.0,<0.33.0` on darwin, and torch ≥2.13 as a dependency ([pyproject.toml#L24-L45](https://github.com/filipstrand/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/pyproject.toml#L24-L45)). Pin with `uv pip install "mflux @ git+https://github.com/mflux-community/mflux@<sha>"` and pin `mlx==` as well. MLX itself is MIT and builds from source with CMake/pip.

### 2. DiffusionKit (argmaxinc)
- Supports SD3-medium/SD3.5 (MMDiT) and FLUX.1 schnell/dev only, in MLX (plus a Core ML export path).
- The last commit is 2025-04-11, so there is nothing from FLUX.2, Qwen-Image or Z-Image.
- Precision: `w16`/`a16` (fp16 weights and activations) flags and pre-converted `argmaxinc/mlx-FLUX.1-*` checkpoints ([README#L105-L125](https://github.com/argmaxinc/DiffusionKit/blob/498e5dba5fb48b0f01cb6b5c2292a6dbea67a317/README.md#L105-L125)).
- `low_memory_mode` deletes T5/CLIP after conditioning and the MMDiT before decode ([mlx/__init__.py#L378-L384,L452-L456](https://github.com/argmaxinc/DiffusionKit/blob/498e5dba5fb48b0f01cb6b5c2292a6dbea67a317/python/src/diffusionkit/mlx/__init__.py#L378-L456)).
- `--seed` is available. It offers a CLI and a Python library. MIT.
- Verdict: unmaintained, not viable.

### 3. mlx-examples flux / stable_diffusion
- Reference example code, FLUX.1 schnell/dev only ([flux/txt2image.py#L46-L65](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/flux/txt2image.py#L46-L65)).
- `-q` is a single on/off switch that applies 4-bit affine quantization to the flow model, T5 and CLIP, limited to layers whose input dimension is a multiple of 512; the VAE is not quantized ([L29,L75-L78](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/flux/txt2image.py#L29-L78)).
- Also offers `--seed` and `--preload-models` (without it, models load lazily). MIT.
- Verdict: a demo, not a runtime.

### Verdict
mflux is the only viable MLX candidate for a 16 GB M4 baseline:
- It is actively maintained (commits on 2026-11-25, monthly releases) and supports most current small and medium models within days to weeks (Z-Image: 8 days, FLUX.2 klein: 4 days).
- 4–8-bit weight-only quantization, automatic TE eviction, `--low-ram`/VAE tiling, seed replay and per-step callbacks are all available.
- Limitations: it has no GGUF, no per-component bit widths (only skip or uniform), no CPU offload or mmap, and no FLUX.2-dev or MiniMax-H3 (33B). Its RNG differs from torch, so images cannot match other runtimes seed-for-seed.

DiffusionKit and mlx-examples are FLUX.1/SD3-era and not viable.

## diffusers on MPS, InvokeAI, SD.Next

Base permalinks: `D` = https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8 , `P` = https://github.com/pytorch/pytorch/blob/be9af63a628c1a99030b8e7ab45d2339f81db461

### Pinned commits
| repo | sha | date |
|---|---|---|
| huggingface/diffusers | 3d336834a7855493cb665b20db99bdcf0be8c7e8 | 2026-11-25 (latest release v0.43.0, 2026-11-12) |
| pytorch/pytorch | be9af63a628c1a99030b8e7ab45d2339f81db461 | 2026-11-25 |
| invoke-ai/InvokeAI | 1bd3954850c1eb1efac041854f8f0ba098f1b47a | 2026-11-25 (latest release v6.14.4, 2026-11-03) |
| vladmandic/sdnext | 6657d06112ed8e924e93eed099af44ea8d892417 | 2026-10-29 |

### diffusers (MPS)
1. **Architectures.** At HEAD, `src/diffusers/pipelines/` includes `flux` and `flux2` (`pipeline_flux2.py`, `pipeline_flux2_klein.py`, Klein inpaint and KV), `qwenimage` (T2I, img2img, inpaint, edit, edit_plus, edit_inpaint, layered, controlnet), `qwenimage21`, `z_image` (T2I, img2img, inpaint, omni, controlnet), `hunyuan_image` (+refiner), `chroma`, `hidream_image`, `lumina2`, `stable_diffusion_3`, `sana`, `cogview4`, `ernie_image`, `krea2`, `ideogram4`, `longcat_image`, `ovis_image`, `glm_image`, `bria_fibo2`, `joyimage`, `nucleusmoe_image` and others (directory listing at `D/src/diffusers/pipelines`). **MiniMax-H3-Image is integrated as Modular Diffusers blocks only** (`MiniMaxH3ImageModularPipeline`, [docs/source/en/api/pipelines/minimax_h3_image.md#L16](https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/api/pipelines/minimax_h3_image.md#L16), code in `src/diffusers/modular_pipelines/minimax_h3_image/`), added by `875373a` "MiniMax H3 Image (#15230)" on 2026-11-10. There is no MOSS-Image or SenseNova pipeline at this sha. **Lag** (date each pipeline dir was added, from git `--diff-filter=A`; model release dates from the authors' news entries cited under Support lag): FLUX.2 `flux2` added 2025-11-25 (5ffb73d4a), the same day as FLUX.2 [dev]'s release (≈0 days, day-0 coordinated); Qwen-Image `qwenimage` added 2025-08-04 (8e53cd959), the same day as the model; HunyuanImage `hunyuan_image` added 2025-10-23 (a138d71ec), about 6 weeks after HunyuanImage-2.1 (2025-09-08); `qwenimage21` added 2026-09-18 (6256aa766). Big-lab models often get day-0 support. Others wait weeks (HunyuanImage) or arrive as Modular blocks only (MiniMax-H3-Image, day 0). `Z-Image` landed 2025-11-25 (`4088e8a`) and FLUX.2 klein 2026-01-15 (`61f1756`), both day 0.
2. **Precision.** Each component is loaded separately (`from_pretrained(..., torch_dtype=...)` per model), and `PipelineQuantizationConfig` takes `components_to_quantize` / `quant_mapping` per component (`D/src/diffusers/quantizers/pipe_quant_config.py#L42-L52`). Backends present: bitsandbytes, gguf, torchao, quanto, modelopt, nunchaku, sdnq, autoround, bfl (`D/src/diffusers/quantizers`).
   - **GGUF:** weights stay uint8 and are "dynamically dequantized and cast to the configured `compute_dtype` during each module's forward pass" (`D/docs/source/en/quantization/gguf.md#L28`). Loading works only for a single model through `from_single_file`, not through pipelines (`#L16`). Documented types are BF16, Q4_0/1, Q5_0/1, Q8_0 and Q2_K–Q6_K (`#L68-L80`). The code also dequantizes IQ2_S, IQ3_XXS/S and IQ4_XS/NL (`D/src/diffusers/quantizers/gguf/utils.py#L75-L84`, `#L503-L515`). Dequantization is plain torch math, so it is device-agnostic and runs on MPS [INFERENCE]. Optional fused kernels are CUDA-only (`utils.py#L34-L38`) and the MMQ path is disabled (`#L89-L95`). Activations are not quantized.
   - **quanto:** the docs say it is "Device agnostic (e.g CUDA,XPU,MPS,CPU)" (`D/docs/source/en/quantization/quanto.md#L25`).
   - **torchao** and **sdnq** docs show `# or "mps"` (`torchao.md#L38`, `sdnq.md#L33`). Which torchao kernels actually run on MPS is not stated [INFERENCE: many int4/fp8 paths are CUDA-only].
   - **bitsandbytes:** the quantizer accepts MPS (`D/src/diffusers/quantizers/bitsandbytes/bnb_quantizer.py#L62`, `#L235-L236`). Kernel maturity depends on the bitsandbytes multi-backend [INFERENCE].
   - **fp8 storage:** `enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)` stores weights as fp8 and upcasts them for compute (`D/docs/source/en/optimization/memory.md#L412-L431`).
   - **CPU fallback:** an unsupported MPS op raises an error unless `PYTORCH_ENABLE_MPS_FALLBACK=1` (`P/docs/source/mps_environment_variables.md#L17`).
3. **Memory.**
   - **Offload:** `enable_model_cpu_offload` / `enable_sequential_cpu_offload` (`D/src/diffusers/pipelines/pipeline_utils.py#L1152`, `#L1273`) and `enable_group_offload` (`#L1343`) auto-detect the accelerator (`#L1162-L1165`). On unified memory, "offload to CPU" saves nothing physical; it only moves tensors between allocators [INFERENCE].
   - **Freeing the text encoder:** you can call `encode_prompt` and then `del` the encoder, passing `prompt_embeds=` to the pipeline (Flux2: `D/src/diffusers/pipelines/flux2/pipeline_flux2.py#L571-L576`, `#L760`).
   - **VAE:** tiling and slicing are available (`D/src/diffusers/models/autoencoders/vae.py#L899-L916`; QwenImage `autoencoder_kl_qwenimage.py#L744`).
   - **mmap:** safetensors are mmapped at load, but MPS tensors are copied into Metal buffers, so there is no mmap-beyond-RAM execution [INFERENCE].
   - **OOM:** the MPS allocator raises a Python `RuntimeError: MPS backend out of memory` (`P/aten/src/ATen/mps/MPSAllocator.mm#L537-L550`). The hard limit is `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (default 1.7 × recommended working set); `0.0` disables the limit and "may cause system failure" (`P/docs/source/mps_environment_variables.md#L12-L22`). Below the limit, macOS swaps, which the diffusers MPS doc warns degrades performance (`D/docs/source/en/optimization/mps.md#L64-L66`). The doc also warns that MPS cannot handle NDArrays larger than 2**32 (`#L42`).
4. **Determinism.**
   - **Generator:** the docs recommend a CPU `Generator` for reproducibility because CPU RNG is "more stable across machines" (`D/docs/source/en/using-diffusers/reusing_seeds.md#L24-L31`). `randn_tensor` samples on CPU when given a CPU generator and silently moves the result to MPS (`D/src/diffusers/utils/torch_utils.py#L86-L93`).
   - **RNG portability:** CPU torch RNG is identical across platforms, so a seed matches diffusers on any machine using a CPU generator. It does not match CUDA-generator noise [INFERENCE].
   - **Deterministic algorithms:** `enable_full_determinism()` calls `torch.use_deterministic_algorithms(True)` (`reusing_seeds.md#L85-L101`). MPS kernel run-to-run bit-exactness is not documented [INFERENCE: generally stable on the same machine and torch version].
   - **Fixed latents and per-step reads:** pipelines take `latents=` (`pipeline_flux2.py#L759`) and `callback_on_step_end` with `callback_on_step_end_tensor_inputs`, which exposes and can replace `latents` at every step (`#L821-L829`, `#L999-L1003`).
5. **Handover.** Python library only. There is no official server.
6. **Build and pinning.** Apache-2.0. Install a pinned commit with `pip install git+https://github.com/huggingface/diffusers@3d336834a7855493cb665b20db99bdcf0be8c7e8`. It is pure Python with no macOS build step, and it needs a PyTorch macOS arm64 wheel.

### InvokeAI
Apache-2.0. It is a diffusers/torch-based app with a web UI and a REST API server (FastAPI), and it supports macOS MPS. At `1bd3954` its model loaders cover FLUX.1, FLUX.2 dev, FLUX.2 Klein, Qwen-Image, Z-Image, SD3, CogView4, ERNIE-Image, Krea2, Ideogram4, Anima, Wan and **MiniMax-H3** (`https://github.com/invoke-ai/InvokeAI/tree/1bd3954850c1eb1efac041854f8f0ba098f1b47a/invokeai/app/invocations`, `*_model_loader.py`). Latest release: v6.14.4 (2026-11-03). Its node graph exposes noise, latents and denoise as separate nodes [INFERENCE].

### SD.Next
Apache-2.0. It is a diffusers-based web UI with an HTTP API and a macOS/MPS install path. At `6657d06` it has pipeline modules for flux, flux2, flux2_klein, flux3, qwen, qwen21, z_image, hyimage, hidream, chroma, lumina, sd3, **minimax**, **moss**, ovis, longcat, ernie, krea2, ideogram4 and more (`https://github.com/vladmandic/sdnext/tree/6657d06112ed8e924e93eed099af44ea8d892417/pipelines`, `model_*.py`). It ships SDNQ quantization, which is upstreamed into diffusers as a quantizer.

### Verdict
diffusers is a viable but heavy baseline for a 16 GB M4. It supports more current models than any other runtime here, with per-component dtype and quantization, CPU-generator seed reproducibility, `latents=` input and per-step callbacks, all pinnable by commit. Its weaknesses are that GGUF is dequantized in torch with no Metal kernels (slow), and offload gives no real saving on unified memory. Fitting the 11.84 GiB working set requires sequential component loading (encode → free TE → denoise → VAE tiling). InvokeAI and SD.Next are diffusers-based servers that wrap the same stack, and both also run MiniMax-H3-Image.

## ComfyUI and ComfyUI-GGUF

Canonical repo: `comfyanonymous/ComfyUI` now redirects to **Comfy-Org/ComfyUI** (the GitHub API returns `full_name: Comfy-Org/ComfyUI`). C = `https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929`, G = `https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a`.

### Pinned commits
| repo | sha | date | license |
|---|---|---|---|
| Comfy-Org/ComfyUI | a2103ae69d98d8ef94c150f5a472b89775eed929 | 2026-11-25 | GPL-3.0 |
| city96/ComfyUI-GGUF | 6ea2651e7df66d7585f6ffee804b20e92fb38b8a | 2026-01-12 | Apache-2.0 |
Latest ComfyUI releases: v0.46.0 (2026-11-18), v0.45.0 (2026-11-10), v0.44.0 (2026-11-04).

### 1. Architectures and support lag
- `comfy/ldm/` at the pinned commit includes flux, flux3, qwen_image, qwen_image21, lumina (Z-Image is a subclass of Lumina2), hidream, hidream_o1, chroma, chroma_radiance, minimax, moss_image, sensenova, hunyuan_video (HunyuanImage 2.1), ernie, kandinsky5, krea2, ideogram4, joyimage, omnigen and others. Model classes: `Flux2` [C/comfy/supported_models.py#L802](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L802), `MiniMaxH3` / `MiniMaxH3Image` [#L987-L1021](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L987-L1021), `ZImage(Lumina2)` [#L1240](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L1240), `SenseNovaU15` [#L1792](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L1792), `QwenImage` / `QwenImage21` [#L2175-L2204](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L2175-L2204), `HunyuanImage21` (+Refiner) [#L2265-L2290](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L2265-L2290). Editing works through reference-latent conditioning and the edit nodes. The edit node classes were not enumerated.
- **MiniMax:** the v0.46.0 MiniMax item is a **partner (paid API) node**: "[Partner Nodes] feat(MiniMax): add MiniMax H3 Image node" (PR #17661, v0.46.0 release notes). Local weights are supported separately: MiniMax-H3 (the video/audio model) since `57500fc5bc` "feat: Support MiniMax-H3 (CORE-375) (#15224)" on 2026-08-03, and **MiniMax-H3-Image** since `ede4e5fe8d` "feat: Support MiniMax-H3-image (CORE-708) (#17655)" on 2026-11-10, day 0. ComfyUI has no SenseNova U1.6 class yet. The latest one is `SenseNovaU15`, added by `3216c62e99` on 2026-09-01.
- Lag examples. ComfyUI commit dates come from the GitHub API. Release dates are approximated by Hugging Face `createdAt`, which is when the repo was created and may be earlier than the public release.
  - Qwen-Image: HF repo created 2025-08-02; release ≈2025-08-04 [INFERENCE]. ComfyUI commit `c012400240` "Initial support for qwen image model (#9179)" landed 2025-08-05, a lag of about 1 day.
  - Z-Image-Turbo: HF repo created 2025-11-25. ComfyUI commit `e9aae31fa2` "Z Image model (#10892)" landed 2025-11-25, so support arrived on day 0.
  - FLUX.2 [dev]: public release 2025-11-25 [INFERENCE]. A Flux2 fix commit, `58b8574661`, is dated 2025-11-26, so support was day 0 or day 1.
  - Qwen-Image 2.1: `6bfaacc67c` landed 2026-09-19. MOSS-Image: `e087d8f6cb` landed 2026-10-16.
  - Typical lag: 0–2 days for big labs, because Comfy-Org gets pre-release access [INFERENCE].

### 2. Per-component precision
- Components load separately: `UNETLoader` has `weight_dtype` ∈ {default, fp8_e4m3fn, fp8_e4m3fn_fast, fp8_e5m2} [C/nodes.py#L992-L1012](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/nodes.py#L992-L1012). `CLIPLoader` has `device` ∈ {default, cpu} [#L1017-L1042](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/nodes.py#L1017-L1042). `VAELoader` is at [#L776](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/nodes.py#L776).
- Global CLI flags: `--force-fp16`, `--bf16-unet`, `--fp8_e4m3fn-unet`, `--fp8_e4m3fn-text-enc`, `--cpu-vae` [C/comfy/cli_args.py#L87-L106](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/cli_args.py#L87-L106).
- **fp8 on MPS is storage only and is upcast for compute.** On a non-CPU device, `supports_fp8_compute` returns True only if `SUPPORT_FP8_OPS` is set or the device is NVIDIA SM≥8.9; MPS therefore gets False [model_management.py#L2055-L2075](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L2055-L2075). Weights are then manual-cast per layer to fp16/bf16 (`unet_manual_cast`, [#L1185-L1207](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L1185-L1207)).
- **int8 compute is disabled on MPS** because `torch._int_mm` has no MPS implementation [#L2131-L2137](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L2131-L2137). fp16 is used on MPS [#L1952](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L1952). bf16 is used only on macOS ≥14 [#L2013-L2016](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L2013-L2016). fp64 is not supported on MPS [#L2112](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L2112). Activations are never quantized.
- **ComfyUI-GGUF:** this pack loads GGUF files for the UNet and for CLIP/text encoders in separate loaders. It dequantizes weights on the fly in PyTorch, layer by layer, in the forward cast [G/ops.py#L176-L177](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/ops.py#L176-L177), so the dequant runs on whatever device holds the layer (MPS). Supported types: BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_NL and IQ4_XS; F32 and F16 pass through [G/dequant.py#L287-L300](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/dequant.py#L287-L300). Other i-quants are absent [INFERENCE: not in table → unsupported]. The `dequant_dtype` and `patch_dtype` options are at [G/nodes.py#L150-L156](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/nodes.py#L150-L156). The README warns about an MPS "buffer is not large enough" error with some torch versions on macOS Sequoia [G/README.md#L30](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/README.md#L30). There is no MLX quant type in core.
- **MLX node packs:** none is maintained to a usable standard. A search finds only 0–1-star repos (e.g. susky900213/ComfyUI-MLX-GEN, ramedeiros/comfyui-mlx, updated 2026-11). The older thoddnn/ComfyUI-MLX now returns 404.

### 3. Memory
- **On MPS, `vram_state = SHARED`** [#L590-L591](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L590-L591).
- **DynamicVRAM (comfy-aimdo) is off on MPS:** it is enabled only for NVIDIA and ROCm ≥7.14 [C/main.py#L285-L291](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/main.py#L285-L291), so Macs use the legacy ModelPatcher.
- **The text encoder runs on the CPU by default.** Under SHARED, `text_encoder_device()` returns CPU [#L1215-L1226](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L1215-L1226) [INFERENCE: SHARED ∉ {HIGH,NORMAL}]. `CLIPLoader device=cpu` forces this explicitly.
- **Unloading between stages:** models are unloaded to CPU RAM after use unless `--highvram` is set. Flags: `--highvram`, `--lowvram`, `--novram`, `--cpu`, `--reserve-vram`, `--disable-smart-memory` [cli_args.py#L171-L194](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/cli_args.py#L171-L194). `--cache-none` reduces RAM [#L142](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/cli_args.py#L142). On unified memory, "offload to CPU" frees no physical RAM [INFERENCE].
- **VAE on CPU:** `--cpu-vae` [model_management.py#L1275-L1278](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L1275-L1278).
- **VAE tiling:** the `VAEDecodeTiled` node [nodes.py#L348](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/nodes.py#L348). On OOM, ComfyUI automatically retries the decode tiled: "Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding" [comfy/sd.py#L1408-L1415](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/sd.py#L1408-L1415).
- **mmap:** safetensors are loaded with `safe_open`, which uses mmap by default; `--disable-mmap` copies tensors instead [comfy/utils.py#L203-L208](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/utils.py#L203-L208). Weights larger than RAM go through page cache and swap; there is no streaming guarantee [INFERENCE].
- **OOM on macOS:** `OOM_EXCEPTION` falls back to plain `Exception` when there is no CUDA, and `is_oom` checks for "out of memory" in the message [#L355-L371](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L355-L371). An MPS allocation failure therefore becomes a node error in the prompt, and the server stays up [INFERENCE]. macOS swaps before MPS refuses an allocation [INFERENCE].

### 4. Determinism and access
- **Initial noise is generated on the CPU in fp32** with `torch.manual_seed(seed)` [comfy/sample.py#L9-L38](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/sample.py#L9-L38). The noise is therefore device-independent and matches CPU torch RNG on any platform.
- **SDE/ancestral samplers draw per-step noise on the model device** (`torch.Generator(device=x.device)`) [comfy/k_diffusion/sampling.py#L78-L88](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/k_diffusion/sampling.py#L78-L88). On MPS this uses the MPS RNG, so it will not match CUDA.
- **`--deterministic`** says it "might not make images deterministic in all cases" [cli_args.py#L195](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/cli_args.py#L195), [model_management.py#L106-L108](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/model_management.py#L106-L108). The same seed probably gives the same image on the same Mac with Euler/flow samplers [INFERENCE, untested].
- **Fixed latents in:** `LoadLatent` / `SaveLatent` [nodes.py#L510-L565](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/nodes.py#L510-L565). For the noise itself, use `SamplerCustomAdvanced` with a NOISE input and `DisableNoise` (`Noise_EmptyNoise`) [comfy_extras/nodes_custom_sampler.py#L716-L732](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy_extras/nodes_custom_sampler.py#L716-L732), [#L985-L1055](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy_extras/nodes_custom_sampler.py#L985-L1055). With DisableNoise, a loaded latent can serve as the exact initial noise when denoise is 1 [INFERENCE: flow models scale x by sigma; check noise scaling].
- **Per-step output:** the sampler callback is called as `callback(i, denoised, x, total_steps)` [comfy/samplers.py#L997-L1005](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/samplers.py#L997-L1005). Previews are decoded from x0 and sent with progress [latent_preview.py#L120-L137](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/latent_preview.py#L120-L137). They reach clients as WebSocket binary `PREVIEW_IMAGE` events (JPEG), with `--preview-method` [server.py#L1446](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/server.py#L1446). Raw per-step latents need a custom node, model wrapper or library use [INFERENCE].

### 5. Handover
- Server mode: HTTP `POST /prompt` (API-format workflow JSON) [server.py#L1164](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/server.py#L1164), plus `/ws` for progress and previews. `--listen` defaults to 127.0.0.1.
- Also available: the Comfy Desktop app for macOS [README.md#L53](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/README.md#L53), and `comfy-cli` [#L71](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/README.md#L71).
- There is no standalone generate CLI: a headless run means starting the server and POSTing a workflow.

### 6. Build and pinning
- GPL-3.0 (core); GGUF pack Apache-2.0.
- From source: `pip install -r requirements.txt` [README.md#L82](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/README.md#L82). Apple Silicon instructions recommend a PyTorch nightly [docs/installation.md#L170-L172](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/docs/installation.md#L170-L172).
- Pure Python, so pinning is `git checkout <sha>` for both repos. Torch and `comfy-aimdo` are pip dependencies, so pin them via a lockfile.
- Releases are weekly.

### Verdict
**Viable, and the strongest breadth candidate.**
- **Why:** day-0/1 support for new models; truly separate per-component loaders and precisions; GGUF k-quants on Metal via ComfyUI-GGUF; TE placed on CPU by default on Mac; automatic tiled-VAE fallback; CPU-seeded noise that matches CPU torch RNG; and an HTTP API.
- **Caveats for a 16 GB M4:**
  - Every quant is dequantized to fp16/bf16 in PyTorch, so there are no native low-bit Metal kernels; it is slower than MLX or sd.cpp [INFERENCE].
  - DynamicVRAM is NVIDIA/AMD-only, so Macs get the legacy memory estimator.
  - fp8 saves storage only.
  - The GGUF pack was last committed 2026-01-12 and has known torch-version MPS bugs.
  - Determinism with SDE samplers on MPS is not guaranteed.

## Core ML, Draw Things and others

Scope: (a) Core ML stack (apple/ml-stable-diffusion, swift-coreml-diffusers, coremltools, Argmax DiffusionKit); (b) other runtimes: Draw Things (community repo, gRPCServerCLI, draw-things-cli), DiffusionBee, candle, Ollama, LocalAI, koboldcpp, Nunchaku. MLX runtimes (mflux, DiffusionKit-MLX), sd.cpp, ComfyUI and diffusers/MPS have their own sections above; here they only appear where another runtime wraps them.

### Pinned commits

| Repo | SHA | Commit date (UTC) | License (GitHub SPDX) |
|---|---|---|---|
| apple/ml-stable-diffusion | ea2805dc1945be20561c77e5f6d1d9a5a637cda2 | 2026-09-11 (root-file update only; last code change e12202c 2025-07-03; last release 1.1.1 on 2024-05-04) | MIT |
| huggingface/swift-coreml-diffusers | a561fae8b47707efa68f70e2fa1fea5ab4462ab9 | 2024-06-25 | Apache-2.0 |
| apple/coremltools | a207b1a3452eafe02f2d870f486287834af2c82d | 2026-09-30 (latest release 9.0 on 2025-11-10) | BSD-3-Clause |
| argmaxinc/DiffusionKit | 498e5dba5fb48b0f01cb6b5c2292a6dbea67a317 | 2025-04-11 | MIT |
| drawthingsai/draw-things-community | cb2d4c61d1b58816ce1843481e74293a240e5bf4 | 2026-11-24 (release v26.1023.0 on 2026-10-23) | GPL-3.0 (+CLA) |
| liuliu/swift-diffusion | 835fd60de370247ba6199db6fa45deb3031999d4 | 2026-11-17 | BSD-3-Clause |
| divamgupta/diffusionbee-stable-diffusion-ui | daf82f80e37b5a8c3e7b6caccedae09a3bf16dc5 | 2023-12-28 (last release 2.5.3 on 2024-08-14) | AGPL-3.0 |
| huggingface/candle | e91e46595c4ce467432564e47ea774873c7c3724 | 2026-11-25 | Apache-2.0 / MIT (GitHub SPDX: Apache-2.0) |
| ollama/ollama | 784dd544291825e8f5f064672dc2bd7f4d9345ed | 2026-11-25 | MIT |
| mudler/LocalAI | 3ecc82d605c036fd8ba333e590cf86430a65887b | 2026-11-25 | MIT |
| LostRuins/koboldcpp | 69dedbeb1de423945b7f5f9a8330008bcf330aac | 2026-11-22 | AGPL-3.0 |
| mit-han-lab/nunchaku | 302e0e97024ebd68688fe890e5df83731edf7b54 | 2026-09-06 | Apache-2.0 |

Abbreviations used in links below: `MLSD` = https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2, `DT` = https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4. Every link is written out in full.

---

### 1. Core ML: apple/ml-stable-diffusion (Python converter + Swift package + `StableDiffusionSample` CLI)

1. **Architectures.** SD 1.x/2.x, SDXL (+refiner), ControlNet (not for XL), and **SD3-medium** (MMDiT, `DiscreteFlowScheduler`) only. There is no FLUX, Qwen-Image, Z-Image, FLUX.2, HunyuanImage or MiniMax-H3 pipeline: the pipeline directory has only `StableDiffusionPipeline`, `StableDiffusionXLPipeline` and `StableDiffusion3Pipeline`, and the converter has no "flux"/"qwen" code path ([INFERENCE] from a grep of `torch2coreml.py` at the pin). SD3 conversion: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L346-L352. Img2img through `startingImage`/`strength`: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusionPipeline.Configuration.swift#L22-L26. There are no edit models. **Lag:** effectively unbounded. SD3-medium (June 2024) landed through Argmax's upstreaming. Nothing newer was added, and the repo has had no code commit since 2025-07-03.
2. **Precision.** Weights are fp16 by default. `--quantize-nbits {2,4,6,8}` k-means palettizes the **unet and text_encoder** (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L550). Mixed-bit palettization recipes go down to about 2.8–4 bits: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L186-L195. Each component is a separate `.mlpackage`, so in practice per-component precision is chosen at conversion time. The SDXL VAE is forced to fp32 unless you use the fp16-fix VAE (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L383). **The SD3 MMDiT needs fp32 and runs on CPU_AND_GPU only** (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L352). Activations are fp16 on the GPU and ANE (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L86). Palettized weights are decompressed just-in-time at runtime on macOS 14+ (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L165). Optional W8A8 activation quantization for the UNet targets the ANE on M4/A17 Pro (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L249-L259). No GGUF/fp8.
3. **Memory.** `reduceMemory` loads and unloads each model just-in-time, so the text encoder is freed before the denoiser loads (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusion3Pipeline.swift#L35-L69, https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L87). Compute units (`cpuOnly`/`cpuAndGPU`/`cpuAndNeuralEngine`/`all`) apply per pipeline through `MLModelConfiguration`, not per component ([INFERENCE]). There is no VAE tiling and no explicit mmap control (Core ML manages weights itself). OOM: the README only says "if your app crashes … add Increased Memory Limit" (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L728-L730), i.e. a process crash/jetsam ([INFERENCE] for macOS: swap first).
4. **Determinism and access.** `seed: UInt32` plus `rngType` ∈ {numpy (default), torch (torch CPU RNG emulation), nvidia (CUDA RNG emulation)}: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusionCLI/main.swift#L80-L81, https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusionPipeline.Configuration.swift#L36-L52. The README FAQ says images are not identical across compute units or vs PyTorch: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#L752-L759. No same-machine nondeterminism is documented ([INFERENCE] fixed compute unit ⇒ repeatable). Initial latents come from the internal `generateLatentSamples`, and the configuration has no field to inject them (https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusion3Pipeline.swift#L268-L284). Injecting them needs a fork. Per-step: `progressHandler(Progress)` exposes `currentLatentSamples` (x_t, or the denoised x0 with `useDenoisedIntermediates`) and decoded `currentImages`, and returning false cancels: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusion3Pipeline.swift#L114-L119, https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift#L414-L425.
5. **Handover.** Swift library (SwiftPM), the `StableDiffusionSample` CLI (`--sd3`, `--xl`, `--rng`, `--reduce-memory`, `--compute-units`), and a Python pipeline. No server.
6. **Build and pin.** MIT. `swift run` on macOS with Xcode. Pinnable by SHA (SwiftPM `revision:`).

**Verdict:** not viable as a baseline for current models. There are no FLUX/Qwen/Z-Image pipelines, the only DiT (SD3) is fp32 GPU-only, and the project is dormant. It is useful only as a reference for RNG emulation (numpy/torch/nvidia).

### 2. huggingface/swift-coreml-diffusers (app)
A SwiftUI app on top of ml-stable-diffusion. It runs models quantized with coremltools ≥7 and picks compute units automatically with an override (https://github.com/huggingface/swift-coreml-diffusers/blob/a561fae8b47707efa68f70e2fa1fea5ab4462ab9/README.md#L12-L25). It inherits all the limits above and adds no API (app only). Last commit 2024-06-25. Apache-2.0. **Verdict: not viable** (stale; SD1.x/2.x/SDXL only).

### 3. coremltools (converter/compression toolkit, not a runtime)
- Palettization `nbits ∈ {1,2,3,4,6,8}` (kmeans/uniform/unique/custom; per-tensor or per-grouped-channel LUTs): https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/optimize/coreml/_config.py#L579-L582, https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/optimize/coreml/_config.py#L723.
- Linear quantization int8/uint8/int4/uint4, granularity per_tensor/per_channel/per_block: https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/optimize/coreml/_config.py#L131-L165.
- Configs can be set per op/per op-type/per model, and each pipeline component is converted separately, so per-component precision is free.
- [INFERENCE] Converting a 12–33B DiT (FLUX/Qwen-Image/MiniMax-H3) needs a full PyTorch trace on a machine far larger than 16 GB, and Core ML models have static shapes. Nobody publishes such conversions in the primary sources checked.

### 4. Argmax DiffusionKit (Core ML + MLX)
- The Core ML part only converts the **SD3-medium** MMDiT and VAE (https://github.com/argmaxinc/DiffusionKit/blob/498e5dba5fb48b0f01cb6b5c2292a6dbea67a317/README.md#L52-L74). Swift Core ML inference was "upstreamed" to ml-stable-diffusion. The MLX Swift part is a "🚧" placeholder, and the Swift targets are 8-line stubs (https://github.com/argmaxinc/DiffusionKit/blob/498e5dba5fb48b0f01cb6b5c2292a6dbea67a317/README.md#L158-L163).
- FLUX.1 schnell/dev run **only through Python MLX** (`FluxPipeline(low_memory_mode=True, w16=True, …)`, `seed`): https://github.com/argmaxinc/DiffusionKit/blob/498e5dba5fb48b0f01cb6b5c2292a6dbea67a317/README.md#L115-L143.
- MIT. Last commit 2025-04-11. **No current DiT/flow model has a working Core ML pipeline in any repo examined.** **Verdict: not viable** (stale; MLX side superseded by mflux).

---

### 5. Draw Things (app + open-source draw-things-community: gRPCServerCLI, draw-things-cli, HTTP API; engine s4nnc/ccv + liuliu/swift-diffusion)

1. **Architectures (broadest on Apple Silicon).** `ModelVersion` at the pin covers SD1/2/XL, SD3/3.5-Large, PixArt, AuraFlow, FLUX.1, HiDream-I1/O1, Qwen-Image and Qwen-Image 2.1, Z-Image, ERNIE-Image, FLUX.2 dev/9B/4B (klein), flux3_5b, Cosmos 2.5, Ideogram 4, Krea 2, **MOSS-Image, MiniMax-H3-Image**, plus Wan/LTX/HunyuanVideo video models: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/SwiftDiffusion/Sources/Samplers/Sampler.swift#L4-L45. Edit variants are sampler modifiers `kontext`, `qwenimageEditPlus`, `qwenimageEdit2511`, `qwenimageLayered`, `editing`: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/SwiftDiffusion/Sources/Samplers/Sampler.swift#L63-L77. HunyuanImage and SenseNova are not in the enum. **Lag** (HF repo `createdAt` vs first appearance in the public mirror, found with `git log -S` on Sampler.swift; the mirror syncs from a private repo, so app support may come earlier):
   - Z-Image(-Turbo): HF 2025-11-25 → 88cb386, 2025-12-03 (~8 d).
   - FLUX.2 dev: HF 2025-11-22 (public ~11-25) → 2693a0d, 2025-12-15 (~3 wk).
   - Qwen-Image-Edit-2511: HF 2025-12-17 → b2c1960, 2025-12-20 (3 d).
   - FLUX.2 klein 4B: HF 2026-01-14 → da7527c, 2026-01-16 (2 d).
   - ERNIE-Image: HF 2026-04-07 → 2c999ec, 2026-04-16 (9 d).
   - MOSS-Image: HF 2026-10-11 → a32aa0b, 2026-10-20 (9 d).
   - MiniMax-H3-Image: HF repo 2026-11-03 (release 2026-11-10) → f064901, 2026-11-20 (~10 d).
2. **Precision.** Draw Things uses its own `.ckpt` (SQLite-backed s4nnc) format, not GGUF/safetensors at runtime. Weight codecs are palettized `q5p`/`q6p`/`q8p` (plus `q4p`, [INFERENCE] from the q-family naming; not seen at the pin) and fp16 `ezm7`. The ModelQuantizer picks codecs **per tensor/per component** (e.g. q8p for some, q6p/q5p for others): https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Apps/ModelQuantizer/Quantizer.swift#L45-L129. Text encoders and denoisers ship as separate files (e.g. `flux_2_klein_4b_q6p.ckpt`), so precision is chosen per component file. [INFERENCE] Palettized weights are dequantized to fp16 inside Metal kernels on the fly. Activations are fp16/bf16, with a high-precision VAE fallback by device class (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/DataModels/Sources/DeviceCapability.swift#L796-L807). No GGUF image models (grep finds GGUF only in SwiftLLM).
3. **Memory.** `externalOnDemand` streams weights file-backed (mmap/fread from disk) per model family when the device class is below a threshold or when forced, so weights larger than RAM can be streamed: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/DataModels/Sources/DeviceCapability.swift#L809-L863. `ModelPreloader.offloadModels()` unloads models (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/LocalImageGenerator/Sources/ModelPreloader.swift#L944). `tiledDecoding` (VAE tiling) and tiled diffusion are generation-config fields (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/DataModels/PreGeneratedSPM/config_data_model_generated.swift#L747). Running a component on the CPU is not a user option ([INFERENCE]). OOM behaviour is not documented ([INFERENCE]: the Metal allocation fails and the process aborts, or the system swaps).
4. **Determinism and access.** `SeedMode` = legacy / torchCpuCompatible / scaleAlike / nvidiaGpuCompatible, i.e. torch CPU and CUDA RNG emulation: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/DataModels/PreGeneratedSPM/config_data_model_generated.swift#L33-L41. The CLI takes `--seed UInt32` (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Apps/DrawThingsCLI/DrawThingsCLI.swift#L550-L551). Same-machine run-to-run determinism is not documented ([INFERENCE] likely, since seed and kernels are fixed). Per step: the gRPC `GenerateImage` stream returns `currentSignpost` (step) and `previewImage` bytes: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/GRPC/Models/Sources/imageService/imageService.proto#L4-L9, https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/GRPC/Models/Sources/imageService/imageService.proto#L135-L139. The CLI has live preview with `--disable-preview` (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Apps/DrawThingsCLI/DrawThingsCLI.swift#L623-L624). Previews are decoded images, not raw latents. There is no documented way to feed initial latents; only an init image plus strength (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Apps/DrawThingsCLI/DrawThingsCLI.swift#L547-L548). Raw latents would need the Swift library ([INFERENCE]).
5. **Handover.** Four options:
   - gRPC server (`gRPCServerCLI`, prebuilt macOS/Linux binaries): https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/README.md#L78-L96.
   - A1111-compatible HTTP API `/sdapi/v1/txt2img|img2img|options`: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/HTTPAPIServer/Sources/HTTPAPIServer.swift#L22-L38.
   - `draw-things-cli generate` (also on Homebrew `--HEAD`), which takes `--config-json` for the full config (seedMode, tiling…): https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/README.md#L37-L76.
   - The app.
6. **Build and pin.** GPL-3.0 with a CLA (https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/LICENSE.md#L1-L3). Builds with `swift build --target gRPCServerCLI` or Bazel on macOS. Reproducible ("byte-by-byte matching") builds are claimed: https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/README.md#L17-L35, https://github.com/drawthingsai/draw-things-community/blob/cb2d4c61d1b58816ce1843481e74293a240e5bf4/README.md#L200-L204. Pinnable by SHA or by dated release tags (v26.1023.0, 2026-10-23). The model implementations live in liuliu/swift-diffusion (BSD-3, HEAD 835fd60, 2026-11-17).

**Verdict: viable, and the strongest non-PyTorch candidate.** It is the fastest to support new models (days), runs on Metal with q5p–q8p per-component weights, streams weights from disk, tiles the VAE, emulates torch/CUDA RNG, and exposes a CLI, HTTP and gRPC with step previews. Caveats: a proprietary `.ckpt` format (weights must be converted or downloaded from its catalog, not GGUF), GPL-3.0, raw latents are not exposed, and OOM behaviour is undocumented.

### 6. DiffusionBee
Electron app. Last release 2.5.3 (2024-08-14) added FLUX.1 on arm64/macOS 13+. The last commit is 2023-12-28 (`gh api …/releases`). AGPL-3.0. App only, no API. **Verdict: not viable** (dead; no models after FLUX.1).

### 7. huggingface/candle (Rust, Metal backend)
- Examples include `flux` (FLUX.1 schnell/dev; `--quantized` loads GGUF `flux1-schnell.gguf` through `quantized_var_builder`; `--seed`, `--cpu`): https://github.com/huggingface/candle/blob/e91e46595c4ce467432564e47ea774873c7c3724/candle-examples/examples/flux/main.rs#L24-L99, https://github.com/huggingface/candle/blob/e91e46595c4ce467432564e47ea774873c7c3724/candle-examples/examples/flux/main.rs#L165-L191. Also `stable-diffusion`, `stable-diffusion-3` (MMDiT), `wuerstchen`, and `z_image` (added 3a0d1cb, 2026-01-02, ~5 weeks after Z-Image's 2025-11-25 HF date).
- Z-Image runs bf16 (f32 on CPU), mmaps safetensors, takes `--seed` through `device.set_seed`, and builds noise through `get_noise` in the example, so initial latents and per-step access are editable code: https://github.com/huggingface/candle/blob/e91e46595c4ce467432564e47ea774873c7c3724/candle-examples/examples/z_image/main.rs#L149-L154, https://github.com/huggingface/candle/blob/e91e46595c4ce467432564e47ea774873c7c3724/candle-examples/examples/z_image/main.rs#L228, https://github.com/huggingface/candle/blob/e91e46595c4ce467432564e47ea774873c7c3724/candle-examples/examples/z_image/main.rs#L401-L411.
- No FLUX.2, Qwen-Image or edit models. Precision is one dtype per example except FLUX's GGUF transformer. Library plus example CLIs, no server. Apache-2.0/MIT. Pinnable cargo git rev.
- **Verdict: not viable** as a baseline (few architectures, example-grade); fine as a library if hand-written pipelines are acceptable.

### 8. Ollama
Image generation was added experimentally on its MLX engine (33ee716, 2026-01-08, "experimental MLX backend and engine with imagegen support"; 2584940, 2026-01-09, "Add z-image image generation prototype"). It was **removed** in 4713800 (2026-07-28, "imagegen: remove MLX image generation code (#16615)"). At the pin, the server and CLI reject image models: https://github.com/ollama/ollama/blob/784dd544291825e8f5f064672dc2bd7f4d9345ed/server/routes.go#L493-L496, https://github.com/ollama/ollama/blob/784dd544291825e8f5f064672dc2bd7f4d9345ed/cmd/cmd.go#L1024-L1026. **Verdict: not viable.**

### 9. LocalAI
A Go server (OpenAI-compatible `/v1/images/generations`, [INFERENCE] from the project's API, not re-read). Image backends are `backend/python/diffusers` (on Mac this is diffusers on MPS, see the diffusers section) and `backend/go/stablediffusion-ggml`, which **pins sd.cpp commit 3b12781fac1a22eda27ad0f08d0884ef816f24d7**: https://github.com/mudler/LocalAI/blob/3ecc82d605c036fd8ba333e590cf86430a65887b/backend/go/stablediffusion-ggml/Makefile#L10-L11. Capabilities equal the pinned sd.cpp or diffusers, wrapped behind HTTP. MIT. **Verdict:** a viable wrapper but adds nothing for a baseline. Use sd.cpp directly.

### 10. koboldcpp
Bundles a fork of sd.cpp (`otherarch/sdcpp`). It is a single binary with an HTTP API (A1111-style; [INFERENCE] not re-read) and a large set of image flags:
- `--sdmodel` (safetensors or GGUF), `--sdllm` (text encoder/LLM GGUF), `--sdvae`.
- Per-component devices: `--sdclipdevice`/`--sdvaedevice` (GPU or CPU).
- `--sdoffloadcpu`, `--sdquant 0/1/2` (= off/q8/q4 load-time requantization).
- `--sdtiledvae`, `--sdflashattention`, `--sdvramlimit`.

Source: https://github.com/LostRuins/koboldcpp/blob/69dedbeb1de423945b7f5f9a8330008bcf330aac/koboldcpp.py#L15991-L16015. Architectures follow its sd.cpp sync (see the stable-diffusion.cpp section). AGPL-3.0. Pinnable by SHA or release tag. **Verdict:** viable as an sd.cpp-plus-server packaging, but its sd.cpp is a fork with its own lag. Prefer upstream sd.cpp for the baseline.

### 11. Nunchaku (SVDQuant W4A4)
A CUDA engine (INT4/FP4 for NVIDIA 20xx–50xx; the README's release notes are all about GPU generations: https://github.com/mit-han-lab/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/README.md#L20, https://github.com/mit-han-lab/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/README.md#L55). The README never mentions Mac, MPS or Metal ([INFERENCE]: no Apple Silicon support). **Verdict: not viable on a Mac.**

---

### Verdict
- **Core ML (ml-stable-diffusion, swift-coreml-diffusers, DiffusionKit): not viable.** No FLUX/FLUX.2/Qwen-Image/Z-Image Core ML pipeline exists. SD3's MMDiT runs fp32 GPU-only. All three repos are dormant (last code commits 2025-07, 2024-06, 2025-04). Core ML is useful only for its RNG-emulation design (numpy/torch/nvidia sources) and the `progressHandler` latent access pattern.
- **Draw Things: viable**, and the only non-PyTorch, non-GGUF runtime that tracks Nov-2026 models (MiniMax-H3-Image ~10 d after release). It has per-component q5p–q8p weights, disk-streamed weights, VAE tiling, torch/CUDA RNG emulation, and a CLI, HTTP (A1111) and gRPC with step previews, all pinnable by SHA (GPL-3.0). Its limits for a reference baseline are the proprietary `.ckpt` format, no raw latent in/out, and undocumented OOM and determinism.
- **Others:** Ollama dropped image generation, DiffusionBee is dead, Nunchaku is CUDA-only, and candle is example-grade. LocalAI and koboldcpp are just wrappers around sd.cpp, so use upstream sd.cpp instead.
