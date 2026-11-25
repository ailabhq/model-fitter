# Reference Source models for image Fits: one that must shrink to fit 16 GB, one that fits at 8-bit

Ticket: "Which two current open-weights text-to-image models should be the reference Source models: one that must shrink to fit 16 GB, and one that fits at 8-bit?" ([#40](https://github.com/ailabhq/sarto/issues/40)), part of "Map: image Fits, new text-to-image models on small Apple Silicon Macs".

Researched 2026-11-25. Nothing was downloaded or benchmarked. Sources:

- Hugging Face model cards, configs and licenses, read at the commits listed below.
- Parameter counts and dtypes, read from the safetensors headers of every weight file in the ungated repos (HTTP range requests, header only).
- For the gated repos (Krea 2, FLUX.2 [dev], FLUX.2 [klein] 9B, FLUX 3 Image Klein), the transformer count comes from the Hub's `safetensors` API. The other components are derived from file sizes and marked *derived*.

The reference Mac's default GPU working set is 11.84 GiB = **12.71 GB**. Sizes are decimal GB. "8-bit" means 1 byte per parameter, ignoring quantization scales. GGUF `Q8_0` is 8.5 bits per weight, about 6% more.

## Answer

- **Large: Qwen-Image-2.1, Turbo checkpoint** ([`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b) @ `fc13ba1`).
  - Components: a Qwen3-VL-8B text encoder (8.77 B), a 7.12 B single-stream DiT and a 0.34 B VAE. That is 16.22 B parameters: **32.4 GB at BF16 and ~16.2 GB at 8-bit**, so the whole pipeline is over 12.71 GB even at 8-bit.
  - Neither big component can stay at 8-bit next to the other. With `Q8_0` builds, the DiT (7.64 GB) and the text encoder (8.71 GB) together come to 16.4 GB. A Fit must shrink the text encoder, the DiT, or both, or unload the text encoder before denoising.
  - Recommended use: 8 steps, distilled, CFG 1. Default presets go up to 2048×2048. Editing (up to 10 references) is in the same checkpoint.
  - Every runtime checked supports it.
  - It has the largest set of Community builds of any new image model: unsloth's Turbo GGUF had 2.1 M downloads in the last 30 days.
  - License: Qwen Research License, non-commercial "research or evaluation" use. Publishing a Reference pack is evaluation [INFERENCE]. Commercial use of a Fit needs a separate license from Qwen.
  - The base checkpoint [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3) @ `d26bb61` has the same text encoder and DiT shapes (40 steps, not distilled). The same Fit plan applies to it when a non-distilled case is wanted.
  - **Runner-up: Krea 2 Turbo** ([`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo/tree/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7) @ `98e0fe1`).
    - Components: a 12.82 B DiT, a Qwen3-VL-4B text encoder (4.44 B, *derived*) and the Qwen-Image VAE. That is 17.4 B: 34.8 GB at BF16 and 17.4 GB at 8-bit. The DiT alone at `Q8_0` is 13.71 GB, over the limit.
    - 8 steps, distilled; no editing model.
    - License: Krea 2 Community License. Commercial use only under US$1 M revenue. Gated, so downloads need a Hugging Face token.
- **Small: Z-Image-Turbo** ([`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/f332072aa78be7aecdf3ee76d5c247082da564a6) @ `f332072`).
  - Components: a 6.16 B DiT, a Qwen3-4B text encoder (4.02 B) and the FLUX.1 VAE (0.08 B). That is 10.26 B: **~10.3 GB at 8-bit (9.6 GiB)**, so it fits under 11.84 GiB. At BF16 it is 20.5 GB, which does not fit, so the plan still has a real choice to make.
  - 1024×1024, 9 scheduler steps = 8 DiT forwards, guidance 0, distilled.
  - Apache-2.0, ungated: a Reference pack may be published freely.
  - The most-liked authors' text-to-image repo of the last 12 months on the Hub (5,659 likes; 819 k downloads in the last 30 days, plus 1.12 M for unsloth's GGUF).
  - Every runtime checked supports it.
  - **No editing variant is released**: Z-Image-Edit and Z-Image-Omni-Base are still "To be released".
  - Its DiT is published in **FP32** (24.6 GB on disk), while the card loads it in BF16.
  - **Runner-up: FLUX.2 [klein] 4B** ([`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) @ `e7b7dc2`).
    - Components: a 3.88 B DiT, Qwen3-4B and the FLUX.2 VAE. That is 7.98 B: 16.0 GB at BF16, 8.0 GB at 8-bit.
    - 4 steps, distilled. Apache-2.0, ungated. Editing is in the same checkpoint.
    - Pick it instead if the small reference must cover editing.

Rejected candidates and why:

- **MiniMax-H3-Image** (67 B, 134 GB at BF16): its license excludes the EU, UK, Korea and USA. It also cannot fit 16 GB even at 4-bit.
- **FLUX.2 [dev]** (56 B): non-commercial license; the 32 B DiT needs ~2-bit to fit.
- **Qwen-Image-2512** (28.9 B, Apache-2.0): the 20.4 B DiT alone is 11.85 GB at `Q4_0`. It is a fallback large pick only if a permissive license is required.
- **FLUX 3 Image Klein** (10.1 B): fits at 8-bit, but the card says to "always run" a separate 9 B prompt upsampler (18.8 GB). The license and the gate add friction.
- **MOSS-Image** (13.5 B): 13.5 GB at 8-bit, just over the limit; far less used.

## Candidate table

Release = Hub repo creation date. Usage = Hub downloads over the last 30 days and likes, read 2026-11-25, for the authors' repo. Per-component cells give parameters, then BF16 size (params × 2 bytes).

| Source model @ commit (release) | Text encoder(s) | Denoiser | VAE | Total BF16 / 8-bit | Fits 12.71 GB at 8-bit? | Default res · steps · distilled | Editing variant | License · pack publishable? | Usage |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b) `fc13ba1` (2026-10-09) | Qwen3-VL-8B, 8.767 B · 17.53 GB | 7.115 B · 14.23 GB | 0.338 B · 0.68 GB | 32.4 / 16.2 GB | **No** (15.1 GiB) | presets up to 2048² (example 1696×2528) · 8 · yes, CFG 1 | same checkpoint | Qwen Research (non-commercial) · yes, as evaluation [INFERENCE] | 29 k dl, 1,163 likes; Turbo GGUF 2.12 M dl |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3) `d26bb61` (2026-09-14) | same as Turbo | same | 0.338 B · 0.68 GB (published FP32, 1.35 GB) | 32.4 / 16.2 GB | **No** | 2048² · 40 · no | same checkpoint | Qwen Research · as above | 284 k dl, 3,728 likes |
| [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo/tree/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7) `98e0fe1` (2026-06-18) | Qwen3-VL-4B, ~4.44 B · 8.88 GB (*derived*) | 12.82 B · 25.64 GB (published 26.28 GB, 0.32 B FP32) | Qwen-Image VAE, ~0.127 B · 0.25 GB (*derived*, published FP32 0.51 GB) | 34.8 / 17.4 GB | **No** | 2048² · 8 · yes, CFG 0 | none published | Krea 2 Community, commercial < US$1 M revenue · yes, non-commercial [INFERENCE]; gated | 86 k dl, 2,086 likes |
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512/tree/25468b98e3276ca6700de15c6628e51b7de54a26) `25468b9` (2025-12-30) | Qwen2.5-VL-7B, 8.292 B · 16.58 GB | 20.430 B · 40.86 GB | 0.127 B · 0.25 GB | 57.7 / 28.8 GB | **No** | 1328² · 50 · no | separate Qwen-Image-Edit-2511 (Apache-2.0) | Apache-2.0 · yes | 37 k dl, 998 likes; GGUF 533 k dl |
| [FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/e8af80c9dd820ab07ee4171d024f03e5c4b3024b) `e8af80c` (2025-11-22) | Mistral-Small-3.2-24B, ~24.0 B · 48.02 GB (*derived*) | 32.223 B · 64.45 GB | 0.084 B · 0.17 GB (published FP32) | 112.6 / 56.3 GB | **No** | — · 50 (28 trade-off), guidance 4 · guidance-distilled | same checkpoint | FLUX Non-Commercial; card: outputs usable commercially · yes | 263 k dl, 2,624 likes |
| [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/8ced60068623def5b74b4d10558117cb61d6482d) `8ced600` (2026-11-03) | H3 Qwen3-VL text encoder, 33.357 B · 66.71 GB | 33.123 B · 66.25 GB | 0.584 B · 1.17 GB (published FP32) | 134.1 / 67.1 GB | **No** | min. area 2048² · 6 · yes, no CFG | same checkpoint | MiniMax community license, **excludes EU, UK, Korea, USA** · no | 1.04 M dl, 2,547 likes |
| [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9) `f972bd4` (2026-10-11) | Qwen3.5-9B class, 8.393 B · 16.79 GB | 5.082 B · 10.16 GB | 0.062 B · 0.12 GB (published FP32) | 27.1 / 13.5 GB | **No** (12.6 GiB) | — · 50 with CFG · no | no separate checkpoint checked; sd.cpp ships `moss-image-edit` recipes | Apache-2.0 · yes | 15 k dl, 1,027 likes |
| [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/f332072aa78be7aecdf3ee76d5c247082da564a6) `f332072` (2025-11-25) | Qwen3-4B, 4.022 B · 8.04 GB | 6.155 B · 12.31 GB (**published FP32, 24.62 GB**) | FLUX.1 AE, 0.084 B · 0.17 GB | 20.5 / 10.3 GB | **Yes** (9.6 GiB; `Q8_0` ≈ 10.2 GiB) | 1024² · 9 steps = 8 NFEs · yes, guidance 0 | **none released** | Apache-2.0 · yes | 819 k dl, 5,659 likes; unsloth GGUF 1.12 M dl |
| [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) `e7b7dc2` (2026-01-14) | Qwen3-4B, 4.022 B · 8.04 GB | 3.876 B · 7.75 GB | FLUX.2 AE, 0.084 B · 0.17 GB | 16.0 / 8.0 GB | **Yes** (7.4 GiB) | 1024² · 4 · yes, guidance 1 | same checkpoint | Apache-2.0 · yes | 449 k dl, 1,155 likes; Comfy-Org repack 1.63 M dl |
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein/tree/3b767ff9b5261d71645e2ec49d87d298d955238d) `3b767ff` (2026-10-13) | Qwen3-VL-4B-Instruct, ~4.44 B · 8.88 GB (*derived*), **plus required 9 B prompt upsampler (18.8 GB)** | 5.549 B · 11.10 GB | FLUX.2 AE, 0.084 B · 0.17 GB (published FP32) | 20.1 / 10.1 GB without upsampler; ~38.9 / ~19.1 GB with it | Yes without the upsampler; **no** with it | 1024² (card's memory figure) · 16 · yes, no guidance | same checkpoint | FLUX Kommunity: non-commercial, commercial outputs < US$5 M revenue · yes, non-commercial [INFERENCE]; gated | 128 k dl, 1,127 likes; Comfy repack 860 k dl |
| [FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/tree/1498bcb8aaebb230bae82c34b02ecc4087beebac) `1498bcb` (2026-01-14) | Qwen3-8B, ~8.19 B · 16.38 GB (*derived*) | 9.079 B · 18.16 GB | 0.084 B · 0.17 GB | 34.7 / 17.4 GB | **No** | 1024² · 4 · yes | same checkpoint | FLUX Non-Commercial · yes | 96 k dl, 1,891 likes |

### Where each number comes from

- **Parameters and dtypes, ungated repos.** Summed from the safetensors headers of every shard at the listed commit. Examples:
  - Qwen-Image-2.1-Turbo: `text_encoder` BF16 8.767 B, `transformer` BF16 7.115 B, `vae` BF16 0.338 B. The base repo stores the same VAE in F32.
  - Z-Image-Turbo: `transformer` **F32** 6.155 B (three shards, 24.62 GB: [tree](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/f332072aa78be7aecdf3ee76d5c247082da564a6/transformer)), `text_encoder` BF16 4.022 B, `vae` BF16 0.084 B. unsloth's BF16 GGUF of the DiT is 12.31 GB ([files](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/tree/b40f712c569ecfff42f2307dcc43399ae1d71318)).
- **Gated repos.** The Hub `safetensors` API (`/api/models/<id>?expand[]=safetensors`) reports the transformer only:
  - Krea 2: 12,498,501,632 BF16 + 321,571,404 F32.
  - FLUX.2 [dev]: 32,223,281,152 BF16.
  - FLUX 3 Image Klein: 5,548,681,984 BF16.
  - FLUX.2 [klein] 9B: 9,078,581,248 BF16.
  - The text encoder and VAE counts in these rows are file size ÷ 2 (BF16) or ÷ 4 (FP32), cross-checked against an identical ungated component: Qwen3-VL-4B is 4.438 B in `PrunaAI/P-Image-OSS`, and the FLUX.2 AE is 0.084 B in FLUX.2 [klein] 4B.
- **Component identities** come from the `text_encoder/config.json` at the same commits:
  - Qwen-Image-2.1: `Qwen3VLForConditionalGeneration`, 36 layers, hidden 4096.
  - Z-Image-Turbo and FLUX.2 [klein] 4B: `Qwen3ForCausalLM`, 36 layers, hidden 2560.
  - Qwen-Image-2512: `Qwen2_5_VLForConditionalGeneration`.
  - Krea 2: "Qwen3-VL-4B text encoder, Qwen-Image VAE" ([mflux krea2 README L6-L8](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/krea2/README.md#L6-L8)).
  - FLUX.2 [dev] text encoder: Mistral-Small-3.2-24B-Instruct-2506 ([sd.cpp recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/flux2/flux2-dev-edit.dgl)).
- **Steps, resolution, distillation and editing:**
  - Qwen-Image-2.1: 7B DiT with 32 layers; generation and editing in one model; 2048×2048 at 40 steps; presets up to 2752×1536 ([card L28](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/README.md#L28), [L64-L69](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/README.md#L64-L69), [L112-L124](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/README.md#L112-L124)). mflux states the recommended defaults as "40 steps and no guidance" ([qwen21 README L16](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/README.md#L16)).
  - Qwen-Image-2.1-Turbo: "8 denoising steps", CFG=1 by default; the schedule is stored as `sample_sigmas` in `model_index.json` ([card L29-L31](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/README.md#L29-L31), [L74](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/README.md#L74), [model_index.json](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/model_index.json)). mflux: "Use 1024 px or larger" for Turbo ([qwen21 README L92](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/README.md#L92)).
  - Z-Image-Turbo:
    - 6B family; Turbo distilled to 8 NFEs ([card L34-L36](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/f332072aa78be7aecdf3ee76d5c247082da564a6/README.md#L34-L36)).
    - Model zoo: Omni-Base and Edit are "To be released" ([L48-L51](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/f332072aa78be7aecdf3ee76d5c247082da564a6/README.md#L48-L51)), still so in the GitHub README at [`26f23ed`](https://github.com/Tongyi-MAI/Z-Image/blob/26f23eda626ffadda020b04ff79488e1d72004cd/README.md) (2026-02-09). No Tongyi-MAI edit repo exists on the Hub.
    - 1024×1024, `num_inference_steps=9` ("actually results in 8 DiT forwards"), `guidance_scale=0.0` ([L131-L134](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/f332072aa78be7aecdf3ee76d5c247082da564a6/README.md#L131-L134)).
  - Krea 2 Turbo: reference CLI `--steps 8 --cfg 0.0 --width 2048 --height 2048` ([card L263](https://huggingface.co/krea/Krea-2-Turbo/blob/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7/README.md#L263)); "timestep-distilled 8-step checkpoint" ([mflux krea2 README L8](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/krea2/README.md#L8)). The krea org on the Hub publishes only Raw, Turbo and LoRAs; there is no edit model.
  - FLUX.2 [klein] 4B: "4 billion parameter", T2I and multi-reference editing "in a single unified model", 1024², 4 steps, guidance 1.0 ([card L19](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/README.md#L19), [L29](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/README.md#L29), [L67-L70](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/README.md#L67-L70)).
  - FLUX 3 Image Klein:
    - The 5.6B label is the transformer only ([card L35](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/3b767ff9b5261d71645e2ec49d87d298d955238d/README.md#L35)).
    - "Always run a prompt upsampler" ([L41](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/3b767ff9b5261d71645e2ec49d87d298d955238d/README.md#L41)): a separate 9B model that Diffusers runs on every request ([L107](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/3b767ff9b5261d71645e2ec49d87d298d955238d/README.md#L107)), 18.8 GB ([L91](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/3b767ff9b5261d71645e2ec49d87d298d955238d/README.md#L91)).
    - Distilled, 16 steps, no guidance ([L49](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/3b767ff9b5261d71645e2ec49d87d298d955238d/README.md#L49)).
    - Peak 31 GB at BF16 at 1024² without the upsampler, on a B200 ([L189](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/3b767ff9b5261d71645e2ec49d87d298d955238d/README.md#L189)).
  - Qwen-Image-2512: Apache-2.0, 1328×1328, 50 steps ([card L2](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/README.md#L2), [L71](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/README.md#L71), [L87](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/README.md#L87)). Edit model: [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9), Apache-2.0.
  - FLUX.2 [dev]: "32 billion parameter", guidance-distilled, 50 steps (28 trade-off), guidance 4 ([card L28](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/e8af80c9dd820ab07ee4171d024f03e5c4b3024b/README.md#L28), [L34](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/e8af80c9dd820ab07ee4171d024f03e5c4b3024b/README.md#L34), [L86-L87](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/e8af80c9dd820ab07ee4171d024f03e5c4b3024b/README.md#L86-L87)).
  - MiniMax-H3-Image:
    - 6 steps, no CFG ([card L69](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/README.md#L69)); `num_inference_steps` "Omit or 6"; default minimum generation area 2048×2048 ([L220-L221](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/README.md#L220-L221)).
    - The card says the text encoder is loaded from `MiniMaxAI/MiniMax-H3` ([L130](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/README.md#L130)), but a 33.36 B `text_encoder/` is present at this commit.

### Licenses and the Reference pack

- **Qwen-Image-2.1 / Turbo**, [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE) (identical file in the Turbo repo).
  - "Non-Commercial" means "for research or evaluation purposes only" (§1.i). The grant is for non-commercial purposes only; commercial use needs a separate license from Qwen (§2).
  - The license has no clause restricting the publication of outputs. "Built with Qwen" is required only when outputs train a distributed model (§4.b).
  - A published Reference pack used to evaluate Fits is evaluation use [INFERENCE]. A user running a Fit commercially needs Qwen's commercial license.
- **Z-Image-Turbo** and **FLUX.2 [klein] 4B**: Apache-2.0 ([Z-Image card L2](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/f332072aa78be7aecdf3ee76d5c247082da564a6/README.md#L2), [klein card L31](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/README.md#L31)). A Reference pack may be published without restriction.
- **Krea 2**, [Krea 2 Community License](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf):
  - The grant covers using, distributing and "generate Outputs from" the model (§2.1).
  - Commercial use only under US$1 M company revenue (§2.3).
  - Deployments must run content filters (§4.2).
  - "You own all Outputs you generate" (§5.3).
  - Repos are gated (`gated: auto`), so downloading needs an HF token even after accepting.
- **FLUX 3 Image Klein**, FLUX Kommunity License:
  - Non-commercial use; commercial use of outputs for "Qualifying Users" under US$5 M revenue (§2.a, §1.e).
  - Outputs must not be used to improve other models (§4.b).
- **MiniMax-H3-Image**: the license's "Excluded Territories" are "the European Union, the United Kingdom, the Republic of Korea and the United States of America" ([LICENSE L10](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/LICENSE#L10)). Use or display of the model or its outputs there is not authorized (§IV.4). That rules it out as a reference.

## Fit arithmetic for the two picks

All figures are weights only; activations and VAE decode come on top. 12.71 GB is the limit.

| Pipeline | All BF16 | All 8-bit | Example mixes (weights) |
|---|---|---|---|
| Qwen-Image-2.1-Turbo | 32.44 GB | 16.22 GB | DiT `Q8_0` 7.64 + TE `Q8_0` 8.71 + VAE BF16 0.68 = **17.0 GB, no**. DiT `Q8_0` 7.64 + TE `Q4_K_M` 5.03 + VAE 0.68 = 13.35 GB, no. DiT `Q6_K` 6.27 + TE `Q4_K_M` 5.03 + VAE 0.68 = 11.98 GB, fits. Or unload the TE after encoding: DiT `Q8_0` + VAE = 8.3 GB [INFERENCE: arithmetic only] |
| Z-Image-Turbo | 20.52 GB | 10.26 GB | DiT `Q8_0` 7.22 (unsloth) or 6.58 (leejet) + Qwen3-4B `Q8_0` 4.28 + VAE 0.17 = 11.0–11.7 GB, fits. mflux `q8` build total 10.98 GB, fits. BF16 TE + `Q8_0` DiT = 15.4 GB, no |

- Sizes: [unsloth Turbo GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/tree/bd53e520a64472adaff780da0b2f2d7eb1605e30), [Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/f982a07559d4a2f6c8744d840bf6fccab30eea96), [Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b).
- For text-to-image the Qwen3-VL vision tower (`mmproj`, 1.16 GB F16) is not needed; editing needs it.
- Measured figures from mflux on an M5 Max ([qwen21 README](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/README.md)):
  - BF16 peak memory ~46 GB, ~1.5 s/step at 1024² (L28-L33).
  - A q8 save, text encoder included, is 17 GB (L96).
  - q8 vs BF16 at 1024², seed 42: PSNR 29.5 dB (L97).
  - q8 peak memory 27 GB with ControlNet at 1024² (L127).
  - These confirm that 8-bit Qwen-Image-2.1 does not fit 16 GB.
- Token count grows with resolution. Qwen-Image-2.1's VAE compresses 16× with DiT `patch_size` 1 ([transformer config](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/transformer/config.json); "one latent token per 16×16 pixel tile", [mflux L11](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/README.md#L11)). So 2048² is 16,384 image tokens and 1024² is 4,096. Z-Image-Turbo at 1024² is 4,096 tokens (8× VAE, `all_patch_size` 2).

## Runtime support

Checked in each runtime's own repo at these commits (2026-11-24/25). ✅ = a model implementation or official recipe exists; ❌ = none found.

| Runtime @ commit | Qwen-Image-2.1 (+Turbo) | Z-Image-Turbo | Krea 2 Turbo | FLUX.2 [klein] 4B | FLUX 3 Image Klein | Qwen-Image-2512 | FLUX.2 [dev] | MiniMax-H3-Image |
|---|---|---|---|---|---|---|---|---|
| [diffusers](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8) `3d33683` | ✅ [`pipelines/qwenimage21`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/qwenimage21) | ✅ [`pipelines/z_image`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/z_image) | ✅ [`pipelines/krea2`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/krea2) | ✅ [`pipelines/flux2`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/flux2) | ✅ [`modular_pipelines/flux3`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/modular_pipelines/flux3) | ✅ [`pipelines/qwenimage`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/pipelines/qwenimage) | ✅ `pipelines/flux2` | ✅ [`modular_pipelines/minimax_h3_image`](https://github.com/huggingface/diffusers/tree/3d336834a7855493cb665b20db99bdcf0be8c7e8/src/diffusers/modular_pipelines/minimax_h3_image) |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py) `a2103ae` | ✅ [`QwenImage21`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L2204) | ✅ [`ZImage`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L1240) | ✅ [`Krea2`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L2028) | ✅ [`Flux2`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L802) | ✅ [`Flux3`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L848) | ✅ [`QwenImage`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L2175) | ✅ `Flux2` | ✅ [`MiniMaxH3Image`](https://github.com/Comfy-Org/ComfyUI/blob/a2103ae69d98d8ef94c150f5a472b89775eed929/comfy/supported_models.py#L1015) |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/52491bd9f1fc15281791723bdff13baab7fd6819) `52491bd` | ✅ [txt2img](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/qwen_image_2.1/qwen-image-2-1-txt2img.dgl), [edit](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/qwen_image_2.1/qwen-image-2-1-edit.dgl) | ✅ [recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/z_image/z-image-turbo-txt2img.dgl) | ✅ [recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/krea2/krea2-raw-txt2img.dgl) (Raw) | ✅ [recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/flux2/flux2-klein-4b-txt2img.dgl) | ✅ [recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/flux3/flux3-image-klein-txt2img.dgl) | ✅ [recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/qwen_image/qwen-image-txt2img.dgl) | ✅ [edit recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/flux2/flux2-dev-edit.dgl) | ✅ [recipe](https://github.com/leejet/stable-diffusion.cpp/blob/52491bd9f1fc15281791723bdff13baab7fd6819/docs/dgl/minimax_h3_image/minimax-h3-image-txt2img.dgl) |
| [mflux (MLX)](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/README.md#L122-L144) `3d1d17a` | ✅ [`models/qwen21`](https://github.com/mflux-community/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21), Turbo included | ✅ [`models/z_image`](https://github.com/mflux-community/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/z_image) | ✅ [`models/krea2`](https://github.com/mflux-community/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/krea2) | ✅ [`models/flux2`](https://github.com/mflux-community/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/flux2) | ✅ [`models/flux3`](https://github.com/mflux-community/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/flux3) | ✅ [`models/qwen`](https://github.com/mflux-community/mflux/tree/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen) | ❌ klein only | ❌ |
| [Draw Things engine](https://github.com/drawthingsai/draw-things-community/tree/cb2d4c61d1b58816ce1843481e74293a240e5bf4/Libraries/SwiftDiffusion/Sources/Models) `cb2d4c6` | ✅ `QwenImage2_1.swift` | ✅ `ZImage.swift` | ✅ `Krea2.swift` | ✅ `Flux2.swift` | ✅ `Flux3.swift` | ✅ `QwenImage.swift` | ✅ `Flux2.swift` [INFERENCE: same family file] | ✅ `MiniMaxH3.swift` [INFERENCE: video and image share it] |
| [InvokeAI](https://github.com/invoke-ai/InvokeAI/tree/1bd3954850c1eb1efac041854f8f0ba098f1b47a/invokeai/app/invocations) `1bd3954` | ✅ `qwen_image_2_1/` | ✅ `z_image/` | ✅ `krea2/` | ✅ `flux2_klein_*` | ✅ `flux3/` | ✅ `qwen_image/` | ✅ `flux2_dev_*` | ✅ `minimax_h3*` |

Precision, offload and seed-determinism controls per runtime belong to the runtime ticket ("Which runtimes run current text-to-image pipelines on a 16 GB Apple Silicon Mac, with which per-component precisions, offload and seed-determinism controls?"). Points relevant to the picks:

- **Per-component precision.**
  - sd.cpp loads the DiT, text encoder and VAE from separate files, each with its own precision. Its Qwen-Image-2.1 recipe pairs an int8 DiT with a `Q4_K_M` Qwen3-VL-8B GGUF; its Z-Image-Turbo recipe pairs a `Q3_K` DiT with a `Q4_K_M` Qwen3-4B (recipes above).
  - mflux quantizes the Qwen-Image-2.1 text encoder only in the edit command and in pre-quantized checkpoints ([qwen21 README L31-L33](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/README.md#L31-L33), [L96](https://github.com/mflux-community/mflux/blob/3d1d17a93ee45707f0e987eed2c494a3665296b7/src/mflux/models/qwen21/README.md#L96)).
- **ComfyUI-GGUF.** Its arch list at [`6ea2651`](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/loader.py#L12-L13) (last commit 2026-01-12) has `lumina2` and `qwen_image` but no `krea2` or `flux3`. unsloth's Z-Image-Turbo GGUF declares `general.architecture = lumina2`, and its Qwen-Image-2.1-Turbo GGUF declares `qwen_image` (read from the GGUF headers). Both pass that check.

## Community builds (GGUF, MLX)

Commits as of 2026-11-25; download counts are for the last 30 days.

**Qwen-Image-2.1 / Turbo** (DiT only unless noted):

- GGUF:
  - [unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/tree/bd53e520a64472adaff780da0b2f2d7eb1605e30) `bd53e52`: Q2_K 2.47 … Q8_0 7.64, F16 14.23 GB; 2.12 M dl.
  - [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/tree/2c31ccd392b367a6637841a143813320a02dff55) `2c31ccd`: same quant set; 1.46 M dl.
  - [leejet/Qwen-Image-2.1-GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/0bca76da6edd8a18211e9717454615c1bbaaf6ed) `0bca76d`: Q2_K 2.56 … Q8_0 7.69 GB.
  - Text encoder: [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/f982a07559d4a2f6c8744d840bf6fccab30eea96) (Q4_K_M 5.03, Q8_0 8.71 GB, mmproj F16 1.16 GB).
  - Whether these GGUFs are bit-identical to the pipeline's `text_encoder` was not checked.
- MLX (mflux format, full pipeline):
  - [OsaurusAI/Qwen-Image-2.1-Turbo-mflux-8bit](https://huggingface.co/OsaurusAI/Qwen-Image-2.1-Turbo-mflux-8bit/tree/b03f3e21956b3f492f59f16299ec901c237bbc4e) `b03f3e2`: 18.45 GB, text encoder 8-bit.
  - OsaurusAI Turbo 4-bit: 10.62 GB.
  - [mlx-community/Qwen-Image-2.1-mflux-q4](https://huggingface.co/mlx-community/Qwen-Image-2.1-mflux-q4/tree/746a58556820933a2df5c75887a2570f1ad200c0) `746a585`: 9.59 GB.
- Draw Things: [drawthingsai/Qwen-Image-2.1-Turbo](https://huggingface.co/drawthingsai/Qwen-Image-2.1-Turbo/tree/f927ee61f6b672a9112312c3706b5d93af30a6f9) `f927ee6`: DiT `i8x` 7.17 GB, `i4x` 4.12 GB; Qwen3-VL-8B `i8x` 8.78 GB; VAE f16 0.66 GB.

**Z-Image-Turbo:**

- GGUF (DiT):
  - [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/tree/b40f712c569ecfff42f2307dcc43399ae1d71318) `b40f712`: Q2_K 2.31 … Q8_0 7.22, BF16 12.31 GB; 1.12 M dl.
  - [leejet/Z-Image-Turbo-GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/c61c0e422dc8b541b7548cf33a4ef8302b0f8085) `c61c0e4`: Q2_K 2.59 … Q8_0 6.58 GB; published 2025-11-30, 5 days after release.
  - [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/tree/6e2dff8e3abd78051c1211fb5833a03818a2f3cc) `6e2dff8`: published 2025-11-27.
  - Text encoder: [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b) (Q8_0 4.28 GB). sd.cpp's recipe uses Qwen3-4B-Instruct-2507 instead; whether that matches the pipeline's encoder weights was not checked.
- MLX (mflux, full pipeline):
  - [mflux-community/z-image-turbo-mflux-q8](https://huggingface.co/mflux-community/z-image-turbo-mflux-q8/tree/4ccff28917346aa9daae49cd3c477cf5260a242c) `4ccff28`: 10.98 GB.
  - q6 8.44 GB, [q4](https://huggingface.co/mflux-community/z-image-turbo-mflux-q4/tree/f427e257d8e6ffa03edd4d9ac554a05809da456c) 5.89 GB, bf16 20.51 GB.
  - [filipstrand/Z-Image-Turbo-mflux-4bit](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit/tree/b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b) `b3a8f31`: 5.89 GB, published 2025-12-02.

**Runners-up and others:**

- Krea 2 Turbo:
  - [vantagewithai/Krea-2-Turbo-GGUF](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF/tree/390847e2835c16bdf82303731e3d35b874760d89) `390847e`: Q2_K 4.89 … Q6_K 10.58, Q8_0 13.71 GB.
  - [mflux-community/krea-2-turbo-mflux-q8](https://huggingface.co/mflux-community/krea-2-turbo-mflux-q8/tree/5f94d7734c8873daddf803721f6423148619f63c) `5f94d77`.
  - The mflux q4 build is 15.76 GB, because it keeps the text encoder at BF16.
- FLUX.2 [klein] 4B:
  - [unsloth/FLUX.2-klein-4B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/tree/a09d5bb51ac4000ccd9643d10e9db3ed14200508) `a09d5bb`: Q8_0 4.30 GB.
  - [mflux-community/flux2-klein-4b-mflux-q8](https://huggingface.co/mflux-community/flux2-klein-4b-mflux-q8/tree/755dc213d6767d4a4eae9291dd3f6e829454177e) `755dc21`.
  - BFL also publishes official FP8 and NVFP4 builds.
- FLUX 3 Image Klein: [unsloth/flux-3-image-klein-GGUF](https://huggingface.co/unsloth/flux-3-image-klein-GGUF/tree/12ef13ce920d53f45fe805093910c899668b9e76) (Q8_0 6.09 GB); mflux-community q4/q6/q8; BFL ships FP8 and NVFP4 variants in the repo.
- Qwen-Image-2512: [unsloth/Qwen-Image-2512-GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/tree/899597deea5b6e8593a021684b4571e06b59ebea) (Q4_0 11.85, Q8_0 22.35 GB); [mlx-community/Qwen-Image-2512-8bit](https://huggingface.co/mlx-community/Qwen-Image-2512-8bit/tree/5f756d3e8cc11878c6c131571d1b904202b788b7) (19.8 GB).
- FLUX.2 [dev]: [unsloth/FLUX.2-dev-GGUF](https://huggingface.co/unsloth/FLUX.2-dev-GGUF/tree/d9105cfeb7c59853ba7b1857c7f53d1371e38ade) (Q2_K 12.86, Q4_K_M 19.96 GB).
- MiniMax-H3-Image: [unsloth/MiniMax-H3-Image-GGUF](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF/tree/0d25c7349e007529691bbc0f4426b579fce7de01) ships only a "pruned" DiT (Q4_K_M 12.72 GB).

## Why these two

- **Large.**
  - Qwen-Image-2.1 is the most-used text-to-image release of the last three months: the Comfy-Org repack has 9.0 M downloads in 30 days, and the unsloth GGUFs 3.6 M combined.
  - It sits where a 16 GB Fit is hard but possible. 8-bit everywhere is 16.2 GB, too much, but a 6-bit DiT with a 4-bit text encoder fits under 12.71 GB, and so does unloading the text encoder [INFERENCE: arithmetic].
  - Its two big components are close in size (8.8 B text encoder, 7.1 B DiT). So the Fit plan's choice of which component shrinks first is a real decision, not a foregone one.
  - Editing in the same checkpoint covers the map's editing scope.
  - All six runtimes support it, and an 8-step distilled checkpoint exists. That matters for "seconds per image" on an M4 [INFERENCE].
  - Why the runner-ups lose:
    - Krea 2 Turbo is the closest alternative. It loses on its gate, its revenue-capped license and its lack of an editing model.
    - Qwen-Image-2512, FLUX.2 [dev] and MiniMax-H3-Image would push every Fit into 2–3-bit territory or cannot fit at all.
- **Small.**
  - Z-Image-Turbo is the most-liked authors' text-to-image repo of the last 12 months and is Apache-2.0 and ungated, so a published Reference pack has no license question.
  - At 8-bit it fits with ~1–2 GiB to spare, depending on the build, so a near-lossless option can be measured. At BF16 it does not fit, so the plan still decides something.
  - All six runtimes support it, with GGUF builds 1–5 days after release.
  - Its gap is editing: none is released.
  - FLUX.2 [klein] 4B is the runner-up. It has Apache-2.0, built-in editing and more headroom, but fewer users of the authors' own repo.

## Facts later tickets depend on

- **Large reference:** `Qwen/Qwen-Image-2.1-Turbo` @ `fc13ba1986fcb905854dc8de2f52eee395f49b5b`. Base `Qwen/Qwen-Image-2.1` @ `d26bb61231c349cf6b7896fa83353113880e1ba3` for a non-distilled check.
  - Components: Qwen3-VL-8B 8.767 B, DiT 7.115 B, VAE 0.338 B.
  - BF16 32.4 GB (base 33.1 GB with its FP32 VAE).
  - 8 steps, CFG 1, presets 1536–2752 px.
  - Qwen Research License.
- **Small reference:** `Tongyi-MAI/Z-Image-Turbo` @ `f332072aa78be7aecdf3ee76d5c247082da564a6`.
  - Components: Qwen3-4B 4.022 B, DiT 6.155 B (**published FP32**), VAE 0.084 B.
  - BF16 20.5 GB; 8-bit 10.3 GB.
  - 1024², 9 steps = 8 NFEs, guidance 0.
  - Apache-2.0; no editing variant.
- **Source precision is not uniform.** Z-Image-Turbo's DiT and the base Qwen-Image-2.1 VAE are published in FP32, everything else in BF16. A Reference pack "at Source precision" has to say whether those components are run at FP32 or at the BF16 that the cards load them in.
- **Runners-up:** Krea 2 Turbo (`98e0fe1`, gated, 17.4 B) and FLUX.2 [klein] 4B (`e7b7dc2`, Apache-2.0, 7.98 B, with editing).
- **Disqualified:** MiniMax-H3-Image, whose license excludes the EU, UK, Korea and US. FLUX 3 Image Klein's card makes a 9 B prompt upsampler part of every request.
