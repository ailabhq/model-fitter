# Which runtimes run Qwen3.8-27B at BF16 and Inkling 2 Mini at native NVFP4/MXFP8 from disk, with parallel sequences and MTP drafting?

Map ticket: "Which runtimes can run Qwen3.8-27B at BF16 and Inkling 2 Mini at native NVFP4/MXFP8 on Apple Silicon when the weights exceed memory, with parallel sequences and MTP speculative decoding?" Researched 2026-11-25 from source at these commits. Nothing was downloaded or run; sizes come from the Hugging Face tree API and the Ollama registry manifests.

| Project | Commit | Date |
|---|---|---|
| [ggml-org/llama.cpp][L] | `5220a2d` (contains `51831d47d`, Metal NVFP4) | 2026-11-25 |
| [ml-explore/mlx-lm][ML] | `6fe0ce4` | 2026-11-24 |
| [Blaizzy/mlx-vlm][MV] | `e8035ae` | 2026-11-24 |
| [jundot/omlx][OM] | `9ff073c` | 2026-11-25 |
| [ollama/ollama][OL] | `924cc28` | 2026-11-23 |

`[INFERENCE]` marks a conclusion no source states directly. Earlier research is reused where it settles a point: "How llama.cpp, MLX and Ollama run…" ([runtime-support][rs]), "Which kimi-k3-in-c techniques carry over…" ([kimi][kimi]), "Running Inkling's NVFP4 experts on Apple GPUs" ([nvfp4-metal][nv]) and the Inkling-on-Metal-NVFP4 measurement ([inkling-nvfp4][inv]).

## Answer

- **Only llama.cpp runs either model from disk when the weights exceed memory.** It mmaps GGUFs by default, and `--fit` puts what does not fit the Metal budget in mmapped CPU buffers that the OS pages from SSD ([runtime-support][rs]). The GGUF can sit on any volume (`-m /Volumes/…`). MLX runtimes (mlx-lm, mlx-vlm, oMLX, Ollama's MLX runner) materialize every weight in memory. Their SSD expert offloads do not cover Inkling 2, and nothing in MLX streams a dense model. Ollama's GGUF runner turns mmap **off** by default on Metal when the model does not fit, so it needs `use_mmap: true`.
- **Qwen3.8-27B BF16, llama.cpp:** download [ggml-org's `Qwen3.8-27B-BF16.gguf`][gq] (53.81 GB, no MTP inside) plus its [`mtp-Qwen3.8-27B-Q8_0.gguf`][gq] (3.16 GB) or `Q4_0` (1.68 GB). Run them with `--spec-type draft-mtp -md mtp-….gguf`. The draft is quantized and the target stays BF16. `-np N` decodes N sequences, and the drafter is built per sequence. No conversion is needed: about 57 GB on disk.
- **Inkling 2 Mini, llama.cpp:** the upstream converter always dequantizes NVFP4 and MXFP8 to float ([inkling2.py L116-L185][l-ink-dq]). No published GGUF keeps every expert native. A GGUF *can* hold the NVFP4 experts as is: `GGML_TYPE_NVFP4` has the same 16-weight E4M3-scaled layout, and the per-expert outer scale goes in an `ffn_*_exps.scale` F32 tensor. But only a custom repack writes that today. **ggml has no exact MXFP8 type**: `F8_E4M3` has no block scale (`blck_size 1`) and no Metal matmul. So the MXFP8 layers 20–23 must be widened to BF16, which is exact. The ready path is [ggml-org's BF16 upcast][gi] (52.23 GB) plus [unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf`][ui] (0.50 GB) via `--spec-type draft-dspark`. Caveat: the converter folds the per-expert FP32 outer scale into each weight before BF16 rounding, so the NVFP4 part of that upcast is **not bit-exact** `[INFERENCE]`.
- **Inkling's "MTP head" is a DSpark drafter** (`mtp.safetensors`, a DFlash block drafter with a Markov head), not a NextN layer. llama.cpp runs it as `draft-dspark`, mlx-vlm and oMLX load it directly, and Ollama's MLX runner quantizes it to 8-bit at load.
- **Native NVFP4/MXFP8 Inkling runs in mlx-vlm, oMLX and Ollama-MLX, but only in memory.** It needs about 20 GB resident, so ≥ 32 GB Macs. All three batch sequences together with the draft active.
- **Default paths that change precision:**
  - llama.cpp's CPU path multiplies NVFP4 weights against **Q8_0-quantized activations** (`vec_dot_type = Q8_0`). Any NVFP4 expert that `--fit`/`--cpu-moe` leaves on the CPU breaks Source precision. BF16 tensors on the CPU use BF16 activations.
  - The default KV cache is unquantized everywhere (llama.cpp F16, MLX model dtype). KV quantization is opt-in only: `-ctk/-ctv`, `--kv-bits`, `OLLAMA_KV_CACHE_TYPE`, oMLX TurboQuant.
- **Disk:**
  - Prebuilt files need no Source safetensors: Qwen ~57 GB, Inkling BF16 ~53 GB, Inkling native (MLX/Ollama) ~20 GB.
  - Converting locally keeps Source and output side by side: Qwen ~110 GB peak, or ~54 GB with the converter's experimental `--remote` (reads safetensors over HTTP).
  - mlx-vlm `moe_offload` repack demands 2× the Source free.

## Per sub-question

### 1. Weights larger than memory; weights on another volume

| Runtime | Larger than memory? | Other volume |
|---|---|---|
| llama.cpp | **Yes.** `--load-mode auto` = mmap; Metal wraps host memory directly; `--fit` (default on) fills the GPU back to front and leaves the rest, or MoE experts first, in system memory ([runtime-support][rs]). On the reference M4, `--fit` already spilled Inkling experts to mmapped CPU buffers and stayed Responsive at 40 GiB page-ins ([inkling-nvfp4][inv]). A dense 54 GB Qwen streams most layers every token: the map's rough 8–19 s/token estimate ([kimi][kimi]). | `-m` takes any path; mmap is file-backed, so page-ins read that volume. |
| mlx-lm | **No.** No offload. Weights are wired up to the recommended working set, then left in pageable memory with a warning ([runtime-support][rs], [kimi][kimi]). Beyond RAM that becomes swap, which lives on the boot volume `[INFERENCE]`. | Local path or `HF_HUB_CACHE`. |
| mlx-vlm | **Only for MoE via `moe_offload`**: a repacked per-expert store, mmapped with a byte-budgeted LRU, served with `--expert-cache-gb` ([moe_offload.py L1-L13][mv-off], [server/cli.py L271-L277][mv-cli]). The repack checks for 2× the Source size free ([L76-L93][mv-disk]). Its tests cover `inkling` (v1) `InklingSwitchGLU`, not `inkling2` ([test_moe_offload.py L17, L464][mv-test]): untested on Inkling 2. No path for a dense model. | Same as mlx-lm; the offload store's `--out` can be any volume. |
| oMLX | **Not for these models.** Its `moe_expert_offload` streams experts from the checkpoint's own safetensors with no copy and preserves routing by construction ([moe_expert_offload.py L1-L28][om-off]). But `inkling2` is not in `_SUPPORTED_TYPES` ([moe_offload_compat.py L12-L25][om-types]), and NVFP4 experts with a global scale stay resident ([L1061-L1062, L1298][om-nvfp4]). The SSD tier stores KV blocks, not weights. | `--model-dir`. |
| Ollama | **GGUF runner: yes, but mmap is off by default.** On Metal, when the predicted VRAM exceeds what is available, the scheduler sets `use_mmap=false` ("metal_partial_offload"; [sched.go L1171-L1208][ol-mmap]), and llama-server then gets `--load-mode none` ([llama_server.go L723-L725][ol-load]). A 55 GB model would then load into anonymous memory. Setting `use_mmap: true` in the request options or a Modelfile skips that default. **MLX runner: no** ([runtime-support][rs]). | `OLLAMA_MODELS` ([envconfig L111-L114][ol-env]). |

### 2. Native NVFP4/MXFP8 for Inkling, or exact BF16 upcast

- **llama.cpp converter.** `Inkling2Model.dequant_model` decodes every `w13`/`w2` expert bank:
  - NVFP4: E2M1 × E4M3 block scale × `amax / outer_scale_denominator`.
  - MXFP8: E4M3 × 2^(E8M0−127).

  Both are written as float32 chunks ([inkling2.py L33-L65, L116-L185][l-ink-dq]). The generic ModelOpt NVFP4 repack into `GGML_TYPE_NVFP4` ([base.py L1311-L1339, L1454][l-base-nvfp4]) is bypassed because Inkling overrides `dequant_model`. The converter also skips the vision and MTP tensors ([L103, L188][l-ink-dq]).
- **Can a GGUF keep NVFP4 as is?** Yes in format, no in upstream tooling.
  - `block_nvfp4` stores the same E2M1 nibbles with one UE4M3 scale per 16 weights ([nvfp4-metal][nv]).
  - The loader reads per-expert F32 `ffn_{gate,up,down}_exps.scale` tensors ([llama-model.cpp L1998-L2010][l-scale]), and the graph applies them after `mul_mat_id`.
  - unsloth's GGUFs use exactly this to keep NVFP4 down-projections native ([nvfp4-metal][nv]).
  - `llama-quantize` from a BF16 upcast does **not** restore the Source bits. `quantize_row_nvfp4_ref` picks its own scale `amax/6` per block with no outer scale ([ggml-quants.c L381-L413][l-q]), so the nibbles and scales differ from the Source `[INFERENCE]`.
  - A native GGUF needs a repack script that copies nibbles and scales and writes `.scale` tensors.
  - Metal runs NVFP4 since `51831d47d`, as fast as other 4-bit types ([nvfp4-metal][nv]).
- **Exact MXFP8 type in ggml? No.**
  - `GGML_TYPE_F8_E4M3` is per element (`blck_size 1`, [ggml.c L777-L783][l-f8]) and has no E8M0 block scale.
  - The converter keeps FP8 as F8_E4M3 only for a single per-tensor scale (`prod(scale.shape) != 1 → skip`, [base.py L730-L745][l-base-f8]).
  - Metal excludes F8_E4M3 from matmul and `get_rows` ([ggml-metal-device.m L1528-L1537, L1997-L1999][l-metal]).
  - So layers 20–23 must be widened to BF16. E4M3 × 2^k is exact in BF16, and the cost is +3.9 GB (4.08 G weights at 2 bytes instead of ~1.03) `[INFERENCE]`.
- **Is the BF16 upcast exact?** For MXFP8, yes. For NVFP4, E2M1 × E4M3 fits BF16 exactly, but the converter multiplies in the FP32 per-expert factor `amax/denominator` before the BF16 cast. BF16 then rounds to 8 significant bits, up to 2⁻⁹ relative per weight `[INFERENCE]`. An exact upcast needs `--outtype f32` for the experts, or the outer scale kept in the `.scale` tensor. The map's standing choice calls ggml-org's upcast exact; the measured KLD between it and the native model has not been checked.
- **CPU fallback quantizes activations.** In ggml-cpu, `NVFP4` and `MXFP4` have `vec_dot_type = GGML_TYPE_Q8_0`; `BF16` uses `BF16`; `F16` and `F8_E4M3` use F32 ([ggml-cpu.c L293-L305, L407][l-cpu]). A native-NVFP4 GGUF is therefore Source-precision only while every NVFP4 tensor runs on Metal. On 16 GB, `--fit` already spilled part of the experts of 29 of 49 layers ([inkling-nvfp4][inv]). On the GPU, the default mat-mat kernel `kernel_mul_mm_nvfp4_f32` stages tiles as `half`. There are `_prec_bf16`/`_prec_f32` variants, selected by a per-tensor precision policy that no CLI flag exposes ([mul_mm.metal L680-L737][l-mm], [ggml.h L1285-L1300][l-prec]). BF16 weights use `bfloat` tiles. `[INFERENCE]` F16 staging is kernel arithmetic, not quantization, but it is not the reference's arithmetic either.
- **MLX family: native.**
  - mlx-vlm loads the native checkpoint directly: "The loader preserves mixed BF16/FP32 parameters and packed NVFP4/MXFP8 experts" ([inkling2/README.md L7-L9][mv-ink]). It adds: "Inference uses floating-point activations, not the reference runtime's activation quantization" (L105-L107).
  - oMLX's Inkling loader keeps `nvfp4`/`mxfp8` per layer ([patches/inkling2/loader.py L49, L569][om-ink]).
  - Ollama's MLX runner keeps the expert banks (`inkling-2-mini:26b-a5b-nvfp4`, 20.16 GB of tensor blobs, registry manifest).
  - mlx-lm has no `inkling2` model at `6fe0ce4`.

### 3. Several sequences decoded together

| Runtime | Flag | With a draft? |
|---|---|---|
| llama.cpp | `llama-server -np N` (`--parallel`, [arg.cpp L2519-L2530][l-arg]); `-c` is split across slots unless `--kv-unified` | Yes. `common_speculative_init(params, n_parallel)` builds the drafter for `n_seq = n_parallel` ([server-context.cpp L1497-L1519][l-srv]). Recurrent Qwen uses checkpoints for rollback (`SEQ_RM_TYPE_FULL`, same lines). |
| mlx-lm | `mlx_lm.server --decode-concurrency 32 --prompt-concurrency 8` (defaults, [server.py L2014-L2025][ml-srv]) | **No.** `is_batchable = draft_model is None` ([L475][ml-batch]); `--kv-bits` also disables batching (L2043-L2049). |
| mlx-vlm | Continuous batching via `BatchGenerator`; `--max-num-seqs` (default unbounded, [server/cli.py L302-L310][mv-cli]) | Yes: `draft_model`/`draft_kind` are passed into `BatchGenerator` ([generation.py L1911-L1934][mv-gen]); MTP/DFlash rows keep per-row state (L2022-L2028). |
| oMLX | `max_concurrent_requests` default 8 (`--max-concurrent-requests`, `OMLX_MAX_CONCURRENT_REQUESTS`; [settings.py L352, L1331][om-set]) | Yes for Inkling DSpark ("wrap it for the batched verify path", [dspark.py L364-L390][om-dsp]); Lightning MTP has late-joining mixed-batch tests ([TESTING.md L211][om-test]). |
| Ollama | `OLLAMA_NUM_PARALLEL` (0 = backend default, [envconfig L276-L277][ol-env]); GGUF: passed as `-np` ([llama_server.go L434-L437][ol-np]); MLX: scheduler logs `"drafting", "shared-verification"` ([mlxrunner/scheduler.go L229][ol-mlxsched]) | Yes on both runners. |

### 4. MTP draft, quantized, with the target at Source precision

- **Qwen3.8-27B (one NextN layer, `mtp_num_hidden_layers: 1`).**
  - **llama.cpp.** The converter can export the target without MTP (`--no-mtp`) and the head alone (`--mtp`), "to publish target and draft as two files" ([convert_hf_to_gguf.py L125-L126][l-conv]). `draft-mtp` runs on the target's own weights when the head is embedded, or on a separate file given with `-md` ([speculative.cpp L3236-L3290][l-spec]; [common.cpp L1246-L1265][l-common]).
    - ggml-org publishes `mtp-Qwen3.8-27B-{BF16,Q8_0,Q4_0}.gguf` (5.95 / 3.16 / 1.68 GB) beside a 53.81 GB BF16 target without MTP ([tree][gq]). unsloth publishes `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB).
    - So the target stays BF16 and the draft is Q8_0 or Q4_0. The draft gets its own KV types (`-ctkd/-ctvd`) and placement (`-ngld`, `-otd`, `--spec-draft-cpu-moe`) ([arg.cpp L4161-L4337][l-arg-d]).
  - **mlx-lm:** no. MTP weights are dropped ([qwen3_5/checkpoint.py L56-L58][ml-ck]).
  - **mlx-vlm:** yes. `--draft-model <same path> --draft-kind mtp` ("Qwen3.5/3.6/3.8 checkpoints with bundled MTP weights can use the same path as --model", [server/cli.py L279-L294][mv-cli]). Quantized: `python -m mlx_vlm.split_mtp --q-bits N` writes a quantized standalone drafter ([split_mtp.py L67-L79][mv-split]).
  - **oMLX:** "Lightning MTP" (`mtp_enabled` per model; [model_settings.py L52, L109][om-ms]). Its published Qwen builds are oQ-quantized, with the MTP head at ≥ 4 bit ([oQ doc L93][om-oq]). A BF16 target with a quantized head is not documented `[INFERENCE]`.
  - **Ollama:** embedded MTP. Tags `qwen3.8:27b-bf16` and `27b-mtp-bf16` are both one 54.66 GB GGUF blob, so the MTP head is BF16 inside it (53.81 + 0.85). The runner adds `--spec-type draft-mtp --spec-draft-n-max <draft_num_predict>` ([llama_server.go L888-L908, L992-L999][ol-draft]). A Modelfile `DRAFT <path>` line can attach a separate quantized MTP GGUF instead ([parser.go L83-L104][ol-parser]; `externalDraftType` detects MTP). `[INFERENCE]` That path is untested here.
- **Inkling 2 Mini (DSpark drafter in `mtp.safetensors`, 0.64 GB BF16).**
  - **llama.cpp:** `convert_hf_to_gguf.py <dir> --dspark` → `llama-server -md Inkling-2-Mini-DSpark.gguf --spec-type draft-dspark --spec-draft-n-max 3` ([docs/speculative.md L135-L142][l-specdoc]). unsloth publishes a Q8_0 DSpark draft (0.50 GB) whose `Mainline_Support/` copy targets current master ([tree][ui]). Native NVFP4 is not needed for the draft, because the map allows a quantized draft.
  - **mlx-vlm:** `--draft-model thinkingmachines/Inkling-2-Mini` loads the bundled drafter with no conversion ([README L24-L25, L83-L100][mv-ink]).
  - **oMLX:** `patches/inkling2/dspark.py` loads `mtp.safetensors` for batched verify.
  - **Ollama MLX:** the drafter is quantized at load to affine 8-bit, with a 4-bit vocabulary head ("Proposals only steer verification", [inkling2/draft.go L20-L27][ol-ink-draft]).

### 5. Default quantization of KV cache or activations

| Runtime | KV default | Activation quantization by default |
|---|---|---|
| llama.cpp | F16 K and V ([common.h L364-L365][l-common-h]); `bf16` is allowed (`-ctk bf16 -ctv bf16`, [arg.cpp L281-L284][l-arg-kv]) | **CPU NVFP4/MXFP4 matmul: Q8_0 activations.** GPU: none; prefill tiles are staged as F16 for NVFP4 and BF16 for BF16 weights (see section 2). |
| mlx-lm | Full precision; `--kv-bits` opt-in ("Default: None (full precision)", [server.py L2043-L2049][ml-srv]) | None |
| mlx-vlm | Full precision; `--kv-bits` / TurboQuant opt-in ([server/cli.py L214-L265][mv-cli]) | None ("floating-point activations", [README][mv-ink]) |
| oMLX | Full precision unless TurboQuant is enabled per model `[INFERENCE from settings normalization; not traced to a default]` | None known |
| Ollama | `OLLAMA_KV_CACHE_TYPE` "default: f16" ([envconfig L319][ol-env]) | GGUF: as llama.cpp (pinned `b12651`) |

`[INFERENCE]` F16 KV for a BF16 model narrows the exponent range, but it is not quantization. Use `-ctk bf16 -ctv bf16` where strictness matters.

### 6. Disk from Source to runnable weights

| Path | Download | Peak disk on the chosen volume |
|---|---|---|
| Qwen, llama.cpp, published files | BF16 GGUF 53.81 + MTP Q8_0 3.16 (or Q4_0 1.68) + mmproj 0.93 if vision | **~57–58 GB**; no safetensors |
| Qwen, llama.cpp, own conversion | safetensors 55.56 | ~110 GB (Source + GGUF side by side). With `--remote` (experimental, reads safetensors over HTTP ranges, [convert_hf_to_gguf.py L113-L115][l-conv]), ~54 GB `[INFERENCE: not tested on qwen35 + --mtp]` |
| Qwen, MLX (mlx-vlm with MTP) | safetensors 55.56 | 55.6 GB; loads as is |
| Qwen, Ollama | `qwen3.8:27b-bf16` 54.66 + 0.93 projector, or `27b-mlx-bf16` 55.56 | ~56 GB under `OLLAMA_MODELS` |
| Inkling, llama.cpp BF16 upcast | ggml-org BF16 52.23 + DSpark Q8_0 0.50 | **~53 GB** |
| Inkling, llama.cpp native-NVFP4 GGUF (custom repack) | Source 20.2 | Source 20.2 + GGUF ≈ 23.5 (NVFP4 experts 11.5, MXFP8 experts as BF16 8.2, rest ≈ 3.8) ≈ **44 GB peak** `[INFERENCE]` |
| Inkling, MLX (mlx-vlm / oMLX) | native 19.5 + mtp 0.64 | **20.2 GB**, no conversion |
| Inkling, mlx-vlm `moe_offload` | native 20.2 | repack demands 2× Source free = ~40 GB ([moe_offload.py L76-L93][mv-disk]) |
| Inkling, Ollama | `inkling-2-mini:26b-a5b-nvfp4` 20.16 | ~20 GB |

The reference Mac has ~21 GB free, so every Qwen path and every Inkling GGUF path needs another volume.

## Candidate read paths for the measurement tasks

| # | Model | Runtime | Weight format | Where the weights live | Parallel sequences | MTP / draft | Known blockers |
|---|---|---|---|---|---|---|---|
| Q1 | Qwen3.8-27B | llama.cpp master ≥ `51831d47d` | ggml-org `Qwen3.8-27B-BF16.gguf` (53.81 GB) | external volume, mmapped; `--fit` (default) puts ~11 GB of layers on Metal and pages the rest | `llama-server -np N` | `--spec-type draft-mtp -md mtp-Qwen3.8-27B-Q8_0.gguf --spec-draft-n-max 2..3` (or Q4_0) | Dense: a token reads ~51 GB of weights ([kimi][kimi]), ~40 GB of it from SSD `[INFERENCE]` → seconds per token; the volume's read speed rules. Parallel sequences and accepted drafts share each read. Keep the default `--load-mode auto`. |
| Q2 | Qwen3.8-27B | Ollama (GGUF runner, llama.cpp `b12651`) | `qwen3.8:27b-bf16` (54.66 GB, MTP embedded BF16) | `OLLAMA_MODELS` on the external volume | `OLLAMA_NUM_PARALLEL=N` | automatic `draft-mtp`, BF16 head; a quantized head only via a Modelfile `DRAFT` | **mmap is disabled by default on Metal partial offload**: must pass `use_mmap: true`; otherwise the load goes to anonymous memory and swap. Adds nothing over Q1. |
| Q3 | Qwen3.8-27B | mlx-vlm | BF16 safetensors (55.6 GB) | any path | `--max-num-seqs N` | `--draft-model <split_mtp --q-bits 8 output> --draft-kind mtp` | **Must be resident** (no streaming): only on Macs with ≥ 64 GB. Not a candidate on 16 GB. |
| I1 | Inkling 2 Mini | llama.cpp master ≥ `51831d47d` | ggml-org `Inkling-2-Mini-BF16.gguf` (52.23 GB, BF16 upcast) | external volume, mmapped; `--fit` keeps experts on CPU first (MoE path) | `-np N` | `--spec-type draft-dspark -md dspark-Inkling-2-Mini-Q8_0.gguf --spec-draft-n-max 3` | 2.6× the bytes per expert of native NVFP4. NVFP4 part not bit-exact (outer scale folded before BF16 rounding). CPU experts use BF16 activations, which is fine. |
| I2 | Inkling 2 Mini | llama.cpp master ≥ `51831d47d` | **custom** GGUF: NVFP4 experts native + `.scale` F32, MXFP8 experts → BF16, rest BF16/F32 (~23.5 GB) | external volume, mmapped | `-np N` | as I1 | **No tool writes it**: needs a repack script (copy nibbles/E4M3 scales, outer scale → `.scale`). **Any NVFP4 expert left on the CPU runs with Q8_0 activations**, which is not Source precision. On 16 GB, `--fit` spills experts; keep NVFP4 on Metal (`-ot` the BF16 MXFP8 layers 20–23 to CPU instead) or accept the breach. |
| I3 | Inkling 2 Mini | mlx-vlm | native checkpoint (NVFP4 + MXFP8, 20.2 GB) | any path, resident | `--max-num-seqs N` | `--draft-model thinkingmachines/Inkling-2-Mini` (bundled DSpark) | **Resident only** (~20 GB + KV): ≥ 32 GB Macs. `moe_offload` is untested on `inkling2` and its repack needs ~40 GB free. |
| I4 | Inkling 2 Mini | oMLX | native checkpoint | `--model-dir`, resident | `--max-concurrent-requests N` (default 8) | bundled DSpark, batched verify | Resident only; expert offload excludes `inkling2` and NVFP4-with-global-scale. |
| I5 | Inkling 2 Mini | Ollama (MLX runner) | `inkling-2-mini:26b-a5b-nvfp4` (20.16 GB) | `OLLAMA_MODELS`, resident | `OLLAMA_NUM_PARALLEL=N` | DSpark drafter auto-quantized to 8-bit at load | Resident only. |

On the 16 GB reference only **Q1, I1 and I2** read weights larger than memory. Q2 is Q1 behind Ollama, with `use_mmap: true`. Q3, I3, I4 and I5 are the paths for larger Macs, where the weights fit.

### Facts later tickets depend on

- llama.cpp is the only from-disk runtime. MLX runtimes need the weights resident.
- Qwen BF16 + quantized MTP is off the shelf in llama.cpp: two ggml-org files, `-md … --spec-type draft-mtp`, `-np N`.
- Native-NVFP4 Inkling GGUF needs a custom repack. MXFP8 has no exact ggml type and becomes BF16. NVFP4 on the CPU quantizes activations to Q8_0.
- ggml-org's Inkling BF16 upcast folds an FP32 outer scale before BF16 rounding, so it is not bit-exact `[INFERENCE]`.
- Ollama on Metal disables mmap for oversized GGUFs unless `use_mmap: true` is set.
- mlx-lm cannot batch while drafting, and it drops Qwen's MTP head.

[L]: https://github.com/ggml-org/llama.cpp/tree/5220a2d4040aa75f52678ebf52d77a0fba85de9f
[ML]: https://github.com/ml-explore/mlx-lm/tree/6fe0ce48c90cec11f2c305830ceba461051c33b2
[MV]: https://github.com/Blaizzy/mlx-vlm/tree/e8035ae3b9740d5e3fc6cb0de4894132d60c8682
[OM]: https://github.com/jundot/omlx/tree/9ff073ce2cd1835ccf52e6c1425306697c68c5df
[OL]: https://github.com/ollama/ollama/tree/924cc286cc592f49c481fa84e15fa60dbc75989e
[rs]: https://github.com/ailabhq/model-fitter/blob/research/runtime-support/research/runtime-support.md
[kimi]: https://github.com/ailabhq/model-fitter/blob/research/kimi-k3-techniques/research/kimi-k3-techniques.md
[nv]: https://github.com/ailabhq/model-fitter/blob/research/nvfp4-metal/research/nvfp4-metal.md
[inv]: https://github.com/ailabhq/model-fitter/blob/research/inkling-nvfp4/research/inkling-nvfp4.md
[gq]: https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/main
[gi]: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/main
[ui]: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main
[l-ink-dq]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/conversion/inkling2.py#L33-L188
[l-base-nvfp4]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/conversion/base.py#L1311-L1339
[l-base-f8]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/conversion/base.py#L730-L745
[l-scale]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/src/llama-model.cpp#L1998-L2010
[l-q]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/ggml/src/ggml-quants.c#L381-L413
[l-f8]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/ggml/src/ggml.c#L777-L783
[l-metal]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/ggml/src/ggml-metal/ggml-metal-device.m#L1528-L1537
[l-cpu]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/ggml/src/ggml-cpu/ggml-cpu.c#L293-L305
[l-mm]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/ggml/src/ggml-metal/kernels/mul_mm.metal#L680-L737
[l-prec]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/ggml/include/ggml.h#L1285-L1300
[l-arg]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/common/arg.cpp#L2519-L2530
[l-arg-d]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/common/arg.cpp#L4161-L4337
[l-arg-kv]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/common/arg.cpp#L281-L284
[l-srv]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/tools/server/server-context.cpp#L1497-L1519
[l-conv]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/convert_hf_to_gguf.py#L113-L138
[l-spec]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/common/speculative.cpp#L3236-L3290
[l-common]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/common/common.cpp#L1246-L1265
[l-common-h]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/common/common.h#L364-L365
[l-specdoc]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/docs/speculative.md#L135-L142
[ml-srv]: https://github.com/ml-explore/mlx-lm/blob/6fe0ce48c90cec11f2c305830ceba461051c33b2/mlx_lm/server.py#L2014-L2049
[ml-batch]: https://github.com/ml-explore/mlx-lm/blob/6fe0ce48c90cec11f2c305830ceba461051c33b2/mlx_lm/server.py#L475-L486
[ml-ck]: https://github.com/ml-explore/mlx-lm/blob/6fe0ce48c90cec11f2c305830ceba461051c33b2/mlx_lm/models/qwen3_5/checkpoint.py#L56-L58
[mv-off]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/moe_offload.py#L1-L13
[mv-disk]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/moe_offload.py#L76-L93
[mv-test]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/tests/test_moe_offload.py#L17
[mv-cli]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/server/cli.py#L214-L310
[mv-gen]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/server/generation.py#L1911-L1934
[mv-ink]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/models/inkling2/README.md
[mv-split]: https://github.com/Blaizzy/mlx-vlm/blob/e8035ae3b9740d5e3fc6cb0de4894132d60c8682/mlx_vlm/split_mtp.py#L67-L79
[om-off]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/patches/moe_expert_offload.py#L1-L28
[om-types]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/patches/moe_offload_compat.py#L12-L25
[om-nvfp4]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/patches/moe_expert_offload.py#L1061-L1062
[om-ink]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/patches/inkling2/loader.py#L569
[om-dsp]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/patches/inkling2/dspark.py#L364-L390
[om-set]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/settings.py#L352
[om-test]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/docs/TESTING.md#L211
[om-ms]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/omlx/model_settings.py#L52-L109
[om-oq]: https://github.com/jundot/omlx/blob/9ff073ce2cd1835ccf52e6c1425306697c68c5df/docs/oQ_Quantization.md#L93
[ol-mmap]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/server/sched.go#L1171-L1208
[ol-load]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/llm/llama_server.go#L723-L725
[ol-np]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/llm/llama_server.go#L434-L437
[ol-draft]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/llm/llama_server.go#L888-L908
[ol-env]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/envconfig/config.go#L276-L335
[ol-parser]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/parser/parser.go#L83-L104
[ol-mlxsched]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/mlxrunner/scheduler.go#L229
[ol-ink-draft]: https://github.com/ollama/ollama/blob/924cc286cc592f49c481fa84e15fa60dbc75989e/mlxrunner/model/inkling2/draft.go#L20-L27
