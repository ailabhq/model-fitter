# Image Quality loss: final images at fixed seeds, or the denoiser's error step by step

Map: "Map: image Fits, new text-to-image models on small Apple Silicon Macs". Ticket: "How should an image Fit's Quality loss against the Source's own outputs be measured: final images at fixed seeds, or the denoiser's error step by step from the Source's own latents?". Researched 2026-11-25.

Terms follow [GLOSSARY.md](../GLOSSARY.md). `[INFERENCE]` marks a conclusion of this note that no source states directly. Every other claim links to the source that owns it. No benchmark was run and no weights were downloaded.

## Answer

**Measure the denoiser step by step on the Source's own latents ("teacher-forced"), and put no final-image metric in the band.** It is the image counterpart of scoring the Source's own response tokens. The Fit never runs its own trajectory, so it cannot drift into a different composition. Its noise floor between full-precision implementations is small and published. Its Mac cost is set by how many (prompt, step) pairs are scored, not by full generations.

1. **Headline: per-step denoiser error.** For each scored (prompt, step) pair, the Fit's denoiser gets the Source's stored latent `x_t`, the stored timestep/sigma and the Fit's own text conditioning. Its output is compared with the Source's stored output at that step. The proposed score is the relative L2 error of the implied clean-latent prediction, `‖σ_t (v_fit − v_src)‖ / ‖x̂0_src‖`, which weights each step by how much it moves the latent `[INFERENCE]`. Report the mean, p90 and an early/late split, because early steps set the composition ([MixDQ][mixdq], [TCEC][tcec]).
2. **Text encoder line:** relative L2 and per-token cosine of the Fit's prompt embeddings against the pack's embeddings, over the positions the denoiser consumes. The headline already includes this loss, because the Fit's own embeddings condition its denoiser.
3. **VAE line:** the Source's final latent decoded by the Fit's VAE, compared in pixels (PSNR) with the Source's decoded image. This is deterministic and has no trajectory.
4. **Pipeline check (no forward passes):** the Fit's runtime must reproduce the pack's sigma schedule, latent layout and step count exactly. Teacher forcing takes `t` from the pack, so it would otherwise miss a scheduler mismatch `[INFERENCE]`. This is the image counterpart of the text tokenizer check.

**Why not final images.** Two full-precision runs of the same model already differ in the final image by as much as a quantized Fit:
- FLUX 3 Klein in BF16 on two NVIDIA GPUs: **LPIPS 0.139, SSIM 0.699** ([Conde et al. 2026][nondet]).
- FLUX.1-dev with MXFP8, against its own BF16: **LPIPS 0.11** ([PyTorch][pt-bw]).
- The same mflux code on two Apple Max chips: framing and details differ, with a mean pixel difference of **27/255** ([mflux VALIDATION][mflux-val-golden]).
- Set metrics (FID, ImageReward) need thousands of images and can *improve* under quantization ([SVDQuant][svdq-t1]).
- On the M4 16 GB, a 12B-class DiT at 1024² costs an estimated 25–35 s per step, so 16 prompts × 28 steps is about 3–4 hours `[INFERENCE]`.

**What a pack holds, per prompt** (FLUX.1-dev-class at 1024², 28 steps):
- the initial noise tensor itself, never just a seed (runtimes use different RNGs);
- the Source's latent `x_t` and denoiser output `v_t` at every step: 28 × 2 × 0.5 MiB in FP16;
- the text embeddings, about 4 MiB for T5's 512 × 4096;
- the final latent, the Source's PNG, and the sampler settings.

That is **about 34 MB per prompt, about 1.1 GB for 32 prompts**, or about 0.5 GB if only 8 steps per prompt are kept. Text packs are 7 MB.

**Cost:**
- **GPU:** about 7 s per prompt for the BF16 trajectory on an H100 ([flux-fast][flux-fast-chart]). Recomputing the reference outputs in FP32 adds about 2 min per prompt if all 28 steps are done `[INFERENCE]`. About 1–1.5 H100-hours for 32 prompts, roughly **$5–7** at the H100 rate paid for the text packs.
- **Mac:** (prompt, step) pairs × ~30 s. A 20-minute budget buys about 40 pairs on the M4 16 GB for a 12B-class DiT at 1024² `[INFERENCE]`.

**Not established by any source:** that teacher-forced per-step error ranks Fits the same way as free-running final-image differences on these models. Theory says it does to first order ([TCEC][tcec]). Error accumulation, which it does not see, is real ([AccuQuant][accuquant]). The first draft image pack must measure this correlation off-device before thresholds are set (see [Open items](#open-items)).

## 1. Final-image comparison at fixed seeds

### What the quantization papers and tools measure

| Source | Metrics against the full-precision model's own images | Paired or set | Prompts |
|---|---|---|---|
| [SVDQuant][svdq-metrics] (ICLR 2025, FLUX.1, PixArt-Σ, SDXL) | LPIPS, PSNR ("similarity"); FID against ground truth, ImageReward ("quality"); appendix: SSIM, CLIP-IQA, CLIP score | LPIPS/PSNR/SSIM paired; FID set | 5K MJHQ + 5K sDCI |
| [PyTorch × diffusers, Blackwell MXFP8/NVFP4][pt-bw] (2026-04) | mean LPIPS (AlexNet) against BF16 images, same generator | paired | DrawBench |
| [MixDQ][mixdq] (ECCV 2024) | SSIM for content change, SQNR for quality change (sensitivity); FID, CLIP score, ImageReward (final) | paired + set | 32 prompts for sensitivity |
| [AccuQuant][accuquant] (NeurIPS 2025) | FID2FP32 (FID with the FP model's images as reference), LPIPS, PSNR, SSIM, CLIP score | set + paired | 5K for text-to-image |
| [Apple ml-stable-diffusion, MBP][apple-mbp] | PSNR between **denoised latents** of compressed and float16 models, same seed and prompts | paired (latent) | a few test prompts |
| [Conde et al. 2026][nondet] (reproducibility study) | bit-identity, MAE, SSIM, LPIPS | paired | up to 100 per set |

None of the quantization papers reviewed here reports DINO similarity. CLIP enters only as CLIP score (text–image alignment) or CLIP-IQA, never as image-to-image similarity against the full-precision output.

What the published numbers look like for a 12B DiT, paired against its own 16-bit images:

| FLUX.1-dev variant | LPIPS ↓ | PSNR ↑ | Source |
|---|---|---|---|
| INT8 W8A8 (SVDQuant), 50 steps, MJHQ | 0.089 | 27.0 | [SVDQuant Table 1][svdq-t1] |
| MXFP8 (selective), 28 steps, DrawBench | 0.11 | – | [PyTorch][pt-bw] |
| NVFP4 W4A4 (SVDQuant) | 0.208 | 21.4 | [SVDQuant Table 1][svdq-t1] |
| INT4 W4A4 (SVDQuant) | 0.223 | 21.0 | [SVDQuant Table 1][svdq-t1] |
| NF4 W4A16 (weight-only, closest to what Mac runtimes do) | 0.272 | 19.5 | [SVDQuant Table 1][svdq-t1] |
| NVFP4 (torchao, selective) | 0.44 | – | [PyTorch][pt-bw] |

Qwen-Image is more sensitive: MXFP8 0.34, NVFP4 0.41 LPIPS ([PyTorch][pt-bw]). PyTorch's blog uses "LPIPS of ~0.1 usually meaning that the images are nearly indistinguishable" as its rule of thumb ([PyTorch][pt-bw]).

### How each metric reacts to small drift that changes composition

- **Pixel metrics (PSNR, SQNR, MAE) mostly measure composition.** MixDQ measured that an image with *changed content but high quality* loses 3.51 dB of SQNR, while one with *similar content but unacceptable quality* loses only 0.26 dB ([MixDQ §3.2][mixdq]). Apple's own MBP write-up says: "The same seed produces an equivalent composition, but will not preserve the same details… PSNR only compares how much pixels differ overall, but does not care about the subjects" ([HF SDXL Core ML][hf-mbp]).
- **SSIM** is the metric MixDQ picks specifically *because* it reacts to content change ([MixDQ §3.2][mixdq]). So it is the most composition-sensitive of the paired metrics.
- **LPIPS** is perceptual, but it still compares two specific images. The reproducibility study's worst full-precision case is a Qwen-Image BF16 batch effect that turns stripes into a chevron crosshatch: **LPIPS 0.388, SSIM 0.228** ([Conde et al. 2026][nondet]). That is no quantization at all, and it scores like an NVFP4 Fit.
- **Set metrics (FID, FID2FP32)** don't care about composition, but they need thousands of images: SVDQuant and AccuQuant use 5K prompts ([SVDQuant][svdq-metrics], [AccuQuant][accuquant]). FID against ground truth also does not track fidelity to the Source. SVDQuant's 4-bit FLUX.1-dev reaches **lower** FID than BF16 (INT W4A4 19.9 vs 20.3), and FLUX.1-schnell's NF4 and INT4 do too (18.9 and 18.3 vs 19.2) ([SVDQuant Table 1][svdq-t1]).
- **Reward models (ImageReward) and CLIP score** don't care about composition either, but they reward or ignore deviation from the Source. SVDQuant's NVFP4 FLUX.1-schnell scores ImageReward 0.968 against 0.938 for BF16 ([SVDQuant Table 1][svdq-t1]).
- **The seed does not fix the starting noise across runtimes:**
  - diffusers samples the noise with `torch.randn`, and its docs warn that a GPU `Generator` "can diverge from CPU results" ([diffusers reusing_seeds][diffusers-seeds]).
  - mflux draws FLUX noise with `mx.random.normal(key=mx.random.key(seed))` ([mflux][mflux-noise]), a different generator.
  - stable-diffusion.cpp ships RNGs that imitate torch CPU and torch CUDA `randn` (`--rng std_default|cuda|cpu`) ([sd.cpp rng_mt19937][sdcpp-rng-cpu], [rng_philox][sdcpp-rng-cuda], [options][sdcpp-rng-opt]).
  - Apple's Core ML pipeline uses NumPy's RNG unless `--rng torch/cuda` is set ([ml-stable-diffusion FAQ][apple-faq]).

  So a "fixed seed" comparison across runtimes compares different noise unless the pack stores the noise tensor itself. mflux's own diffusers comparison does exactly that: "the same initial noise… [is] injected into both paths… rather than pretending two different RNGs provide the same input" ([mflux VALIDATION][mflux-val-ref]).

## 2. Step-by-step denoiser error from the Source's own latents

### Has it been used?

Yes, as a calibration objective, a sensitivity measure and a port-parity test. None of the sources reviewed here uses it as a published quality score for a finished quantized model.

- **PTQD** defines the quantized network's per-step signal-to-noise ratio as `SNR^Q(t) = ‖ε_θ(x_t,t)‖ / ‖Δε_θ(x_t,t)‖`, on the same input `x_t` for both models. It shows `SNR^Q` falls as `t → 0` (it reaches about 1 for W4A4) and rises with bit width, and uses it to pick a bit width per step ([PTQD §4.3][ptqd]).
- **Apple's Mixed-Bit Palettization pre-analysis** runs the pipeline for **one denoising step** (`num_inference_steps=1`, `output_type="latent"`, a shared RNG state) and computes PSNR between the compressed and float16 latents for every layer ([`run_pipe`][apple-runpipe], [README][apple-mbp]). Float16 SDXL scores ~82 dB, naive 8-bit ~65 dB and the 2.81-bit MBP recipe ~67 dB ([README][apple-mbp]).
- **mflux** tests its MLX ports against diffusers this way. A script exports diffusers transformer outputs on fixed inputs to `.npz` ([export_parity_reference.py][mflux-export]). The MLX test then feeds the same inputs and asserts relative L2 `< 0.02` in BF16 and `< 1e-5` in FP32 ([test_flux3_transformer_parity.py][mflux-paritytest]). This is the pack/scorer pattern proposed here, at unit-test scale `[INFERENCE]`.
- **Q-Diffusion-family PTQ** calibrates on activations collected along the full-precision trajectory. AccuQuant names this family: "Previous methods minimize the quantization error at each denoising step individually" ([AccuQuant Fig. 1][accuquant]).

### Does it predict final-image differences?

**Partly, by theory; no source measures the correlation directly.**

- **For:** TCEC derives a closed form in which the cumulative error is a linear propagation of per-step errors. It drops the Jacobian term because a well-trained denoiser is Lipschitz with `L < 0.3`, and reports the simplification costs "< 1%" accuracy. It also finds the cumulative error at a step depends mainly on the two preceding steps' per-step errors ([TCEC §3.1–3.2, App. A][tcec]). Under that model, per-step errors on the Source's path determine the final latent error to first order `[INFERENCE]`. PTQD's per-step `SNR^Q` drives a bit allocation that improves FID ([PTQD][ptqd]). Apple's per-layer one-step PSNR drives recipes whose images keep the composition ([HF SDXL Core ML][hf-mbp]).
- **Against:** AccuQuant shows the per-step error stays "relatively constant" while the accumulated error "increases drastically" ([AccuQuant Fig. 2a][accuquant]). Calibrating on single steps (group size 1) gives worse FID and FID2FP32 than calibrating on 5-step groups ([AccuQuant §4.3, App. C][accuquant]). A per-step metric therefore cannot *see* accumulation, only its inputs. TCEC's FLUX visualization shows FP16-vs-W4A8 error growing step by step and concentrating at edges and textures ([TCEC App. E][tcec]).
- **Few-step models are the exception:** with 1–2 steps there is no trajectory, and per-step error *is* the final error. MixDQ notes a 1-step model lacks the "recovery phase" that later steps give multi-step models ([MixDQ App. B.1][mixdq]).

So the per-step score is a sound *ranking* signal that leaves out how sampling amplifies or heals error `[INFERENCE]`. It is the same bargain the text method made: teacher-forced KLD on the Source's tokens instead of free generation, with no compounding.

### What the pack stores and how big it is

Worked for a FLUX.1-dev-class pipeline (12B DiT, 1024², 28 steps, guidance-distilled):
- diffusers packs the 16-channel latent at 1/8 resolution into 2×2 patches: `in_channels=64`, `num_channels_latents = in_channels // 4`, `vae_scale_factor = 8` ([transformer_flux.py][dif-flux-cfg], pipeline_flux.py [L208][dif-flux-vsf], [L810][dif-flux-nch], [L555–574][dif-flux-latents]). At 1024² that is 4096 tokens × 64 = **262,144 values, 0.5 MiB in FP16/BF16**.
- T5 conditioning is `max_sequence_length=512` × `joint_attention_dim=4096` = **4 MiB in FP16**, plus a 768-wide pooled CLIP vector ([pipeline_flux.py][dif-flux-msl], [transformer_flux.py][dif-flux-cfg]).
- diffusers' Euler flow-match step upcasts to FP32, adds `dt · v` and casts back to the model dtype ([scheduler step][dif-fm-step]). So each step's `x_t` is the exact BF16 tensor the Source saw, and storing it is lossless.

| Item, per prompt | Why | Size (FLUX.1-class, 1024²) |
|---|---|---|
| Initial noise `x_T` | RNGs differ across runtimes (§1) | 0.5 MiB |
| `x_t`, timestep and sigma for every step | denoiser input; the schedule is pinned by the pack, not by the Fit's scheduler | 28 × 0.5 MiB = 14 MiB |
| Reference output `v_t` for every step, computed in FP32 and stored as FP16 | the target; FP16 storage rounding (~2⁻¹¹ relative) is far below the BF16 floor in §4 `[INFERENCE]` | 14 MiB |
| Prompt embeddings (all positions the denoiser reads), plus pooled vector; negative-prompt embeddings for CFG models | text-encoder line | ~4 MiB |
| Final latent, Source PNG | VAE line; human inspection | 0.5 MiB + ~1–2 MB `[INFERENCE]` |
| Sampler settings: steps, guidance, shift, resolution, seed, Source revision | pipeline check, provenance | KB |

**About 34 MB per prompt: about 0.55 GB for 16 prompts or 1.1 GB for 32.** Keeping only 8 steps per prompt (stratified over the schedule) cuts it to about 15 MB per prompt, about 0.5 GB for 32 `[INFERENCE]`. Latent elements scale as `C × (H/f) × (W/f)`, so other pipelines are computed the same way. For CFG models, store the guided output: the Fit computes both branches anyway `[INFERENCE]`. Editing variants also need the VAE-encoded input image. That belongs to the Prompt set ticket.

The reference outputs come from two runs on the rented GPU `[INFERENCE]`:
- The trajectory comes from the Source in its published dtype, with diffusers and the authors' sampler settings. That is "the Source's own outputs".
- The stored `v_t` is recomputed in FP32 from the same weights at those exact latents, so the reference is "computed at full precision" as the Glossary already requires for Reference packs.
- The BF16 outputs from the trajectory run are free and serve as the pack cross-check (§4), as transformers vs vLLM did for text.

## 3. Text encoder

- **Direct embedding comparison is cheap and has a published floor.** For Qwen-Image-2.1's Qwen3-VL encoder, mflux measured these against diffusers ([VALIDATION][mflux-val-num], [actual inputs][mflux-val-actual]):

  | Comparison | Relative L2 |
  |---|---|
  | MLX FP32 vs Torch FP32, text-only | 6.5e-6 |
  | MLX FP32 vs Torch FP32, text positions of a real prompt | 8.1e-6 to 1.16e-5 |
  | **BF16 vs FP32, same text positions** | **2.9–3.6%** |
  | Torch BF16 vs Torch FP32 | 5.4% |
  | MLX BF16 vs Torch FP32 | 4.7% |

  The same note warns that encoder differences at image positions are "not a useful standalone quality indicator because the transformer replaces those positions" ([mflux VALIDATION][mflux-val-actual]). Compare only the positions the denoiser consumes.
- **Text-encoder loss shows up as content change.** MixDQ traces many of SDXL-Turbo's most sensitive layers to outliers in the text embeddings (the BOS token). Naive W8A8 "changes the content significantly"; adding BOS-aware handling of the text embeddings recovers it, moving FID from 103.95 to 31.65 and CLIP score from 0.148 to 0.265 ([MixDQ §3.1, §5.2][mixdq]). MixDQ also finds that the layers reading the text (cross-attention, FFN) drive content change, while self-attention and convolutions drive quality ([MixDQ §3.2][mixdq]).
- **How each method captures it:**
  - Final images capture it, mixed with every other source of drift.
  - The step-by-step method captures it exactly when the Fit's own embeddings condition its denoiser on the Source's latents. The early steps, where content is set, then carry the text-encoder error `[INFERENCE]`.
  - For attribution, a diagnostic run of a few steps with the *pack's* embeddings separates denoiser loss from text-encoder loss, at one extra forward per scored step `[INFERENCE]`.
  - The embedding line itself costs one text-encoder pass per prompt.

## 4. Noise floor: two full-precision implementations

### Final images

| Comparison (same prompt, seed and initial latent) | SSIM | LPIPS | Other | Source |
|---|---|---|---|---|
| SDXL fp16, RTX A6000 vs RTX 6000 Ada | 0.987 | 0.008 | MAE 1.46/255 | [Conde et al. 2026][nondet] |
| Qwen-Image-2.1 bf16, A6000 vs Ada, 768², 40 steps, fixed latents and conditioning | 0.970 | 0.019 | MAE 2.31/255 | [Conde et al. 2026][nondet] |
| HiDream-O1 bf16, A6000 vs Ada, 512², 50 steps | 0.913 | 0.050 | MAE 5.66/255 | [Conde et al. 2026][nondet] |
| **FLUX 3 Klein bf16, A6000 vs Ada, 512², 50 steps** | **0.699** | **0.139** | – | [Conde et al. 2026][nondet] |
| Qwen-Image-2.1 bf16, H100, batch 8 vs batch 1 (same GPU) | 0.875 | 0.088 (worst 0.388) | – | [Conde et al. 2026][nondet] |
| Qwen-Image-2.1 edit, **mflux (MLX) vs diffusers (PyTorch MPS) on one Mac**, 1024², 40 steps, BF16 transformer, same injected noise and conditioning | – | – | **PSNR 49.9 dB**, mean 0.39/255 | [mflux VALIDATION][mflux-val-ref] |
| Qwen-Image-2.1, mflux, same code, M4 Max vs the golden image made on M5 Max (attribution not established; MLX TF32 behaviour on M5 is a hypothesis) | – | – | **54.5% of channel values outside tolerance, mean 27.3/255**; "framing and facial/fur details differ" | [mflux VALIDATION][mflux-val-golden] |
| Same mflux code with and without its text cache and fused prologue | – | – | mean 1.56/255 | [mflux VALIDATION][mflux-val-golden] |

The reproducibility study is a preprint submitted 2026-11-23, not yet peer reviewed. Its replays are bit-exact, so its differences come from the execution setting, not run-to-run noise ([Conde et al. 2026][nondet]). No source compares diffusers on CUDA with a Mac runtime on final images.

**Consequence.** Floors range from LPIPS 0.008 to 0.139 by model, and from PSNR 49.9 dB to a recomposed image on Mac. Any final-image threshold would be model-specific and could sit on top of a Fit's real loss: FLUX.1-dev MXFP8 is 0.11 ([PyTorch][pt-bw]). This repeats the prompt-token finding from the text tickets, where reference noise passed the Near-lossless line.

### Per step, text encoder and VAE

| Comparison (identical inputs, one forward) | Relative L2 / error | Source |
|---|---|---|
| Qwen-Image-2.1 transformer, MLX FP32 vs diffusers FP32, real weights | 1.8e-5 to 4.5e-5 | [mflux VALIDATION][mflux-val-num] |
| Same, actual 1024² prefill | 7.2e-6 | [mflux VALIDATION][mflux-val-actual] |
| **Same, MLX BF16 vs FP32** | **0.0146** | [mflux VALIDATION][mflux-val-actual] |
| mflux FLUX 3 transformer parity test bound, BF16 / FP32 | < 0.02 / < 1e-5 | [test][mflux-paritytest] |
| Text encoder BF16 vs FP32, text positions | 0.029–0.036 | [mflux VALIDATION][mflux-val-actual] |
| VAE FP32 decode, MLX vs diffusers | max abs 1.76e-5, mean 5.3e-7 | [mflux VALIDATION][mflux-val-num] |
| Apple Core ML vs PyTorch, per component, minimum allowed at conversion | PSNR ≥ 35 dB on random inputs | [torch2coreml.py][apple-psnr] |

**Consequence:**
- Against an FP32-computed reference, a BF16 Mac implementation of the same weights should sit near 1.5–2% per-step relative error, the analogue of the text pack's 0.0005–0.0025 KLD `[INFERENCE]`.
- The pack cross-check (BF16 diffusers outputs against the FP32 reference, on the same GPU) measures that floor per pack.
- No source publishes a 4-bit Fit's per-step error on a 12B DiT. Where Fits land relative to the ~1.5% floor is the open measurement for the first draft image pack. PTQD only shows `SNR^Q` falling toward 1 at late steps for W4A4 LDMs ([PTQD][ptqd]).

## 5. Cost for a 12B-class DiT at 1024²

**GPU, per prompt (pack build):**
- FLUX.1-dev BF16, 1024², 28 steps on an H100: **7.30 s** eager, 4.62 s with `torch.compile` ([flux-fast chart][flux-fast-chart]; [settings][flux-fast-readme]). On a B200 it is 2.10 s at batch 1 ([PyTorch][pt-bw]).
- Recomputing the stored outputs in FP32: an H100 runs FP32 without tensor cores, roughly 15× slower than BF16, so about 4 s per step. That is ~2 min per prompt for 28 steps, or ~30 s for 8 kept steps `[INFERENCE]`.
- The trajectory run already yields the BF16 cross-check for free.
- **32 prompts ≈ 5 min of BF16 trajectories + 15–60 min of FP32 recompute, plus weight download.** That is about 0.5–1.5 H100-hours. At the rate paid for the text packs (one H100 session of about 32 min cost about $2.40, per the comment on "Draft Reference packs: what Quality loss do Qwen UD-IQ3_XXS and Inkling oQ2.5e have on the M4 16 GB?"), that comes to **roughly $3–7 per pack** `[INFERENCE]`.

**Mac, per Fit (scoring):**
- The only published FLUX.1 timings on Apple chips with a GPU-core count close to the M4's are for an **M4 Pro (20 GPU cores)**, FLUX at 20 steps and 1024² ([Draw Things MFA 2.0][dt-mfa]):

  | Runtime | Quantization | Total | Per step |
  |---|---|---|---|
  | Draw Things | 5-bit | ~235 s | ~12 s |
  | DiffusionKit | 4-bit | ~247 s | ~12 s |
  | ComfyUI + GGUF | 8-bit | ~303 s | ~15 s |
  | mflux | 8-bit | ~338 s | ~17 s |

- The M4 in the reference Mac has 10 GPU cores, so assume **~25–35 s per step** for a 12B-class DiT at 1024² `[INFERENCE]`. On an M4 Max (40 cores), mflux denoised Qwen-Image-2.1 BF16 at 1024² in ~591 s for 40 steps, ~15 s per step, with other work running ([mflux VALIDATION][mflux-val-gen]).

| Method | Forward passes | M4 16 GB estimate `[INFERENCE]` |
|---|---|---|
| Final images, 16 prompts × 28 steps | 448 denoiser + 16 TE + 16 VAE | **~3–4.5 h** |
| Final images, 4 prompts | 112 + 4 + 4 | ~50–65 min, for 4 data points against a floor of up to 0.139 LPIPS |
| **Step-wise, 40 (prompt, step) pairs**, e.g. 20 prompts × 2 stratified steps | 40 denoiser + 20 TE + 20 VAE decode | **~20–25 min** |
| Step-wise + text-encoder attribution on 10 of the pairs | +10 denoiser | +~5 min |

Step-wise scoring decouples the number of prompts from the number of steps. The pack keeps every step, and the scorer picks pairs to fit the time budget. The text method's 20-minute cap is therefore reachable on the M4. A full-trajectory method cannot get there for a 12B DiT at 1024² `[INFERENCE]`.

## Open items

- **Validate the proxy before setting thresholds.** On the rented GPU, quantize the reference model to a few levels (e.g. 8-bit, 4-bit weight-only, 3-bit) and compute both the per-step score and free-running final-image LPIPS/SSIM from the pack's noise. Check that the two rank the levels the same way `[INFERENCE]`. No source has done this for a flow-matching DiT.
- **The scorer needs a hook into the pinned runtime:** feed `x_t`, `t`, conditioning, read `v`. In mflux this is the same call its parity tests make ([test][mflux-paritytest]). In a C++ runtime it is new code, as `llama-mf-score` was for text `[INFERENCE]`. This belongs to the pinned-runtime ticket.
- **Thresholds, band mapping and how many (prompt, step) pairs to score** belong to the Quality loss ticket once a draft pack exists.

[svdq-metrics]: https://arxiv.org/html/2411.05007v4#S5.SS1
[svdq-t1]: https://arxiv.org/html/2411.05007v4#S5.T1
[pt-bw]: https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
[mixdq]: https://arxiv.org/html/2405.17873v2
[accuquant]: https://arxiv.org/html/2510.20348v1
[tcec]: https://arxiv.org/html/2508.12094v3
[ptqd]: https://arxiv.org/html/2305.10657v4
[nondet]: https://arxiv.org/abs/2611.30545
[hf-mbp]: https://huggingface.co/blog/stable-diffusion-xl-coreml
[apple-mbp]: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md?plain=1#L186-L199
[apple-faq]: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md?plain=1#L750-L760
[apple-runpipe]: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/mixed_bit_compression_pre_analysis.py#L244-L277
[apple-psnr]: https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/torch2coreml.py#L59-L97
[diffusers-seeds]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/docs/source/en/using-diffusers/reusing_seeds.md?plain=1#L21-L31
[dif-flux-latents]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/flux/pipeline_flux.py#L555-L574
[dif-flux-msl]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/flux/pipeline_flux.py#L606-L630
[dif-flux-cfg]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/models/transformers/transformer_flux.py#L616-L623
[dif-fm-step]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L492-L526
[mflux-noise]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/flux/latent_creator/flux_latent_creator.py#L5-L11
[mflux-val-num]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/reference/VALIDATION.md?plain=1#L23-L36
[mflux-val-gen]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/reference/VALIDATION.md?plain=1#L38-L44
[mflux-val-actual]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/reference/VALIDATION.md?plain=1#L56-L62
[mflux-val-ref]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/reference/VALIDATION.md?plain=1#L66-L68
[mflux-val-golden]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/reference/VALIDATION.md?plain=1#L97-L117
[mflux-export]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/scripts/flux3/export_parity_reference.py
[mflux-paritytest]: https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/tests/image_generation/test_flux3_transformer_parity.py#L70-L76
[sdcpp-rng-cpu]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/src/core/rng_mt19937.hpp#L11
[sdcpp-rng-cuda]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/src/core/rng_philox.hpp#L10-L11
[dif-flux-vsf]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/flux/pipeline_flux.py#L208
[dif-flux-nch]: https://github.com/huggingface/diffusers/blob/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/flux/pipeline_flux.py#L810
[sdcpp-rng-opt]: https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/examples/common/common.cpp#L751-L758
[flux-fast-chart]: https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/
[flux-fast-readme]: https://github.com/huggingface/flux-fast/blob/0a1dcc91658f0df14cd7fce862a5c8842784c6da/README.md?plain=1#L195-L198
[dt-mfa]: https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
