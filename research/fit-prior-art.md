# Prior art: tools that fit models to hardware automatically

Ticket: [#5](https://github.com/ailabhq/model-fitter/issues/5) · Researched 2026-11-23 · Terms as in [GLOSSARY.md](../GLOSSARY.md)

## Question

What existing tools take a model and a machine, then pick quantization and offload and run the result (or recommend one)? For each tool: what it does, what it lacks (measured Quality loss, disk streaming, new architectures, CLI/automation), its license, and how active it is.

## Answer in brief

A lot of tools solve part of this. None of them does what model-fitter needs end to end.

- **Placement is solved.** llama.cpp's `--fit` is on by default. It shrinks context and moves tensors (dense tensors first, then MoE experts) until a *pre-quantized GGUF* fits in free device memory. Ollama and the new `Llama` app from the llama.cpp team build on it.
- **Recommendation is mostly solved, from catalogs.** llmfit, llm-checker, the Llama app, LM Studio and the Hugging Face Hub "Hardware compatibility" panel all pick *a published quant file that fits*. llmfit and Llama pick "the highest-precision quant that fits".
- **The rest is unsolved.** No tool starts from a new **Source model** and produces a **Fit plan** with precision per part of the model. No tool targets a speed or context bar like **Usable**. None measures **Quality loss** for the Fit it made. None plans **disk streaming** within a disk budget. All of them wait days to weeks for upstream runtimes to support a **new architecture**. On 2026-11-23, Inkling 2 Mini support was still an open draft PR in both llama.cpp and mlx-lm.

## What each tool does

Activity numbers come from the GitHub API on 2026-11-23 (stars, date of last push to any branch).

| Tool | Kind | Input | Picks precision? | Picks placement / context? | Runs it? | License | Activity |
|---|---|---|---|---|---|---|---|
| **llama.cpp `--fit` / `llama-fit-params`** | Runtime option + CLI | One GGUF you already have | No | **Yes**: context ≥ `--fit-ctx`, GPU layers, tensor split, per-tensor overrides (MoE experts to CPU first) | Yes | MIT | 136k★, pushed 2026-11-23, release v1.0.0 (2026-11-13) |
| **Ollama scheduler** | Server | Ollama library tag or imported GGUF/safetensors | No (the user picks the tag; `create -q` only when importing safetensors) | Yes: default context from VRAM size, pre-flight estimate, eviction, retries with less context on OOM; GGUF placement is handed to llama-server `--fit` | Yes | MIT | 188k★, pushed 2026-11-22, v0.40.17 |
| **Llama app** (llama.cpp team + HF) | macOS/Windows app + `llama` CLI | Curated catalog or any HF GGUF | **Yes**: "highest precision that fits in memory" from the published quants | Yes: context "only sizes that fit", batch size by RAM; writes `models.ini` | Yes (llama-server) | MIT | Llama-macOS 1.9k★, pushed 2026-11-17 |
| **LM Studio** (app + `lms`) | Desktop app + CLI | Hub GGUF/MLX quants | No (the user picks; shows "Full GPU Offload Possible") | Yes: automatic GPU offload; "guardrails" block loads that are estimated not to fit; `lms load --estimate-only` | Yes | App: proprietary EULA; `lms` CLI: MIT | `lms` 5.4k★, pushed 2026-11-17 |
| **Hugging Face Hub "Hardware compatibility"** | Web panel | GGUF/MLX repos on the Hub | Advisory: says which quant files "will run" on the hardware you saved | No | No (links out to Local Apps) | Closed service (Hub); hardware list in MIT `huggingface.js` | Live; docs current |
| **`accelerate estimate-memory`** / HF "Model Memory Usage" Space | CLI + Space | Any transformers/timm repo | No: table of int4…fp32 sizes | No | No | Apache-2.0 | accelerate 10k★, pushed 2026-11-23 |
| **llmfit** | Rust TUI/CLI/REST | Built-in catalog of HF models (custom models can be added) | **Yes**: walks Q8_0→Q2_K, then halves context | Reports a run mode (GPU / MoE offload / CPU+GPU / CPU) | No (`bench` measures a provider that is already running) | MIT | 39k★, pushed 2026-11-23, v1.1.21 (2026-11-23) |
| **llm-checker** | Node CLI (Ollama-centric) | Ollama catalog + registry of 55k artifacts | Yes, by score | No | Through Ollama (`ai-run`) | **NPDL-1.0** (custom, no paid distribution) | 3.1k★, pushed 2026-11-16 |
| **gguf-parser-go** (GPUStack) | Go CLI | Local or remote GGUF (parses only the header) | No | Estimates RAM/VRAM per offload setting, finds the right offload layer count, max TPS from device metrics you supply | No | MIT | 314★, pushed 2026-11-20 |
| **mlx-lm** (`convert`, `dynamic_quant`, `dwq`, `awq`) | Python CLI | **HF safetensors Source model** | Partly: the user sets bits or a mixed recipe; `dynamic_quant` gives sensitive layers more bits to hit a target bits per weight | No; warns when the model is ≥ 90% of Metal's recommended working set | Yes (`mlx_lm.generate/server`) | MIT | 7.7k★, pushed 2026-11-21, v0.32.2 |
| **mlx-my-repo / gguf-my-repo** (HF Spaces) | Hosted converters | HF repo | No: the user picks FP16 or Q2–Q8 from a dropdown | No | No | Space code, no license declared | Running; updated 2026-04 / 2026-06 |
| **can-it-run-llm** (HF Space) | Web calculator | HF repo + GPU | No | No | No | GPL-3.0 | Last modified 2025-02-04 |
| **Unsloth Dynamic quants** | Quant producer (not a fitter) | Their own picks of popular models | Per-layer mixed precision, chosen offline per model | No | No | Weights on HF | Dynamic v3.1, Nov 2026 |
| **SSD-streaming engines** (llama.cpp PR #25294, moe-stream, AirLLM, …) | Runtimes / forks | Specific models | No | Stream experts or layers from disk; moe-stream switches between resident, hybrid and streaming modes by model size vs RAM | Yes | MIT / Apache-2.0 | PR still open; forks have few stars |

## Gaps by tool

✔ = does it · ~ = partly · ✘ = no

| Tool | Measures speed on this machine | Measures Quality loss of its own output | Plans disk streaming | Supports new architectures right away | Headless / scriptable |
|---|---|---|---|---|---|
| llama.cpp `--fit` | ~ (`llama-bench` is a separate step; it now accepts `-fitt/-fitc`) | ✘ (`llama-perplexity --kl-divergence` exists but is separate) | ~ (mmap; `--lazy-mode` for single large tensors; MoE streaming PR still open) | ✘ (needs a per-architecture PR; Inkling 2 Mini still a draft) | ✔ |
| Ollama | ~ (`ollama ps` shows the CPU/GPU split, not tok/s) | ✘ | ✘ (turns mmap *off* for partial Metal offload) | ✘ (follows llama.cpp; Inkling bump is an open PR) | ✔ |
| Llama app | ✘ | ✘ | ✘ | ✘ (catalog + llama.cpp) | ~ (config file; app-driven) |
| LM Studio | ✘ | ✘ | ✘ | ✘ | ~ (can't override guardrails headlessly; issue open) |
| HF Hardware compatibility | ✘ | ✘ | ✘ | ~ (any GGUF/MLX repo, but someone has to publish quants first) | ✘ (UI only; the hardware itself can be read via API) |
| accelerate estimate-memory | ✘ | ✘ | ✘ | ✔ (meta-device load of any transformers model) | ✔ (weights only, no inference) |
| llmfit | ~ (formula; `bench` replaces it with local or community measurements) | ✘ (Quality = heuristic score) | ✘ | ✘ (catalog compiled into the binary) | ✔ (JSON, REST) |
| llm-checker | ~ (tok/s while running through Ollama) | ✘ | ✘ | ✘ | ✔ |
| gguf-parser-go | ~ (max TPS estimate) | ✘ | ✘ | ~ (needs a GGUF) | ✔ |
| mlx-lm | ✘ (tok/s printed per run) | ~ (`dynamic_quant --report-ppl`; `mlx_lm.perplexity`) | ✘ (convert needs the whole checkpoint on disk; RFC open) | ~ (new archs land fast as PRs; Inkling 2 Mini still a draft) | ✔ |
| Unsloth | ✘ | ~ (publishes KLD for its own quants, on its own eval sets) | ✘ | ~ (only models they pick) | n/a |
| SSD-streaming engines | ✔ (manual) | ✘ / bit-exact claims | ✔ | ✘ (hand-written per model) | ~ |

## Notes per tool

### llama.cpp `--fit` (the strongest prior art for placement)

- The `llama_params_fit` API was added in [PR #16653](https://github.com/ggml-org/llama.cpp/pull/16653). It "adjusts the provided `llama_model_params` and `llama_context_params`" so the model fits in free device memory. Binaries expose it as `-fit/--fit`, and `llama-fit-params` prints the chosen CLI arguments ([tools/fit-params/README.md](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/fit-params/README.md)).
- The algorithm, per its author: check whether the model fits as-is; if not, reduce context; then move tensors to RAM, "Dense tensors are prioritized for better MoE performance"; layers may "overflow" to the next device or to system memory. It uses "virtual test allocations" instead of heuristics. The author also says Ollama's and KoboldCpp's heuristics "tend to be inaccurate" ([Discussion #18049](https://github.com/ggml-org/llama.cpp/discussions/18049)).
- Defaults: `fit_params = true`, minimum context 4096, 1024 MiB margin per device ([common/common.h L504–L509](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.h#L504-L509); flags in [common/arg.cpp L2863–L2922](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2863-L2922)). Fitting is all or nothing: if the user sets `-ngl`, `-ts` or `-ot`, the program stops managing memory allocation ([Discussion #18049](https://github.com/ggml-org/llama.cpp/discussions/18049)).
- What it doesn't do: it never changes precision; it only places the GGUF it was given. It targets memory, not tok/s. Measuring speed is a separate `llama-bench` run. `llama-bench` did not support fitting at launch ([Discussion #18049](https://github.com/ggml-org/llama.cpp/discussions/18049)) but now takes `-fitt/-fitc` ([llama-bench.cpp L443–L444](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/llama-bench/llama-bench.cpp#L443-L444)). The only disk-related controls are `--load-mode` (mmap/mlock/DirectIO) and `--lazy-mode`, which reads rows of certain large tensors (for example per-layer embeddings) from disk on demand ([common/arg.cpp L2665–L2696](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2665-L2696)). General MoE expert streaming from SSD is [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294), open since 2026-07-04. It reports 1.8–2.2 tok/s decode for a 254 GB model on a 128 GB GB10.
- Quality measurement exists, but as a separate tool: `llama-perplexity --kl-divergence-base … --kl-divergence` reports KLD, ΔPPL and "Same top p" against a reference model's logits ([tools/perplexity/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)).

### Ollama

- For GGUF models, Ollama now starts upstream `llama-server`, which "auto-detects GPU layers". The scheduler only runs a "full-fit preflight" before that ([llm/llama_server.go L1142–L1146](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/llm/llama_server.go#L1142-L1146); `--fit` is mentioned at [L1215](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/llm/llama_server.go#L1215)). The preflight estimate is file size plus an f16 KV-cache estimate, which the code calls "intentionally conservative" ([L2915–L2934](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/llm/llama_server.go#L2915-L2934)). A model is evicted when the estimate exceeds 80% of free memory ([server/sched.go L589–L615](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/server/sched.go#L589-L615)). When the context was chosen automatically, Ollama retries with a smaller one after an OOM ([L808](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/server/sched.go#L808)). Safetensors models go to a separate MLX runner on Metal ([L630–L631](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/server/sched.go#L630-L631)).
- Default context depends on VRAM: under 24 GiB you get 4k ([docs/context-length.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/context-length.mdx)). On a 16 GB Mac that is below the 8K that **Usable** requires.
- Precision is the user's choice. Ollama "does not quantize GGUF models during import" ([docs/import.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/import.mdx)). `ollama create --quantize` works only for safetensors imports (e.g. `nvfp4`) ([cmd/cmd.go L380–L381, L2594](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/cmd/cmd.go#L2594)).
- Disk: when it predicts a partial offload on Metal, it *disables* mmap ([server/sched.go L1191–L1209](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/server/sched.go#L1191-L1209)), which is the opposite of streaming. On macOS it reads memory-compression growth as a pressure signal ([discover/memory_darwin.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/discover/memory_darwin.go)). That is the closest any tool comes to checking "machine stays responsive".
- New architectures: Inkling 2 Mini is waiting on a llama.cpp bump ([issue #19826](https://github.com/ollama/ollama/issues/19826), [PR #19861](https://github.com/ollama/ollama/pull/19861), both open).

### Llama app (ggml-org, with Hugging Face)

- A menu bar/tray app plus a `llama` CLI. It "checks your Mac and sets up each model": Quantization "Highest precision that fits in memory", Context "Only sizes that fit in memory", batch size "Larger on Macs with 32 GB or more" ([llama.app](https://llama.app)). It writes a generated `models.ini` that users can override ([Llama-macOS README](https://github.com/ggml-org/Llama-macOS/blob/4e975ae3e7f25fd16eb9230a29e72a9c977e4193/README.md)).
- The budget is Metal's `recommendedMaxWorkingSetSize`, not physical RAM. Apple sets it "around three quarters of RAM on smaller Macs", and it is "a recommendation, not a hard cap… trading speed for swap" ([SystemMemory.swift L21–L43](https://github.com/ggml-org/Llama-macOS/blob/4e975ae3e7f25fd16eb9230a29e72a9c977e4193/Llama/System/SystemMemory.swift#L21-L43)). On a 16 GB M4 that means roughly 12 GB [INFERENCE from the 0.75 fallback in that file].
- Recommendations come from a curated web catalog. Each size is "resolved to the highest-precision quant that fits", and "lower quants are strictly fallbacks" ([Catalog.swift L4–L21](https://github.com/ggml-org/Llama-macOS/blob/4e975ae3e7f25fd16eb9230a29e72a9c977e4193/Llama/Catalog/Catalog.swift#L4-L21)). This is the simplest baseline Fit plan, and model-fitter should beat it.

### LM Studio

- The model loader offloads to the GPU automatically ("If not specified, LM Studio will automatically determine optimal GPU usage"). `lms load --estimate-only` prints estimated GPU and total memory and whether "resource guardrails" would allow the load ([docs: lms load](https://github.com/lmstudio-ai/docs/blob/8c7159be3ffe364af10aa8ae6f5a7570e9402089/3_cli/0_local-models/load.md)). The Discover view shows a per-quant badge such as "Full GPU Offload Possible" (seen in the first-party tracker, [issue #2324](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2324)). The docs only tell users to "Choose a 4-bit option or higher if your machine is capable enough" ([docs: download-model](https://github.com/lmstudio-ai/docs/blob/8c7159be3ffe364af10aa8ae6f5a7570e9402089/0_app/1_basics/download-model.md)).
- Known weak spots: in an open issue, the guardrail estimated 22.92 GB for a model that used about 11.5 GB in llama.cpp, and "There is no way to override or change the guardrails setting headlessly" (staff reply, [issue #1631](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1631)).
- License: the desktop app is proprietary. Its EULA forbids reverse engineering and creating derivative works ([App Terms](https://lmstudio.ai/app-terms)). The `lms` CLI and `mlx-engine` are MIT (GitHub API).

### Hugging Face Hub

- Users save their hardware (GPU/CPU/Apple Silicon and its memory). Then "on model pages that offer GGUF or MLX files, a **Hardware compatibility** panel estimates whether each quantization will run on your saved hardware" ([docs: Hardware](https://huggingface.co/docs/hub/hardware)). The estimation method isn't documented, and the Hub's server code isn't public. The SKU table (TFLOPS, memory) is in [huggingface.js `packages/tasks/src/hardware.ts`](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/hardware.ts) (MIT). Local Apps then turn the chosen quant into a run command ([docs: Local Apps](https://huggingface.co/docs/hub/local-apps)).
- `accelerate estimate-memory` loads the model on the `meta` device and reports size per dtype. It covers "purely load[ing] the model in, *not*… inference" and suggests adding "up to an additional 20%" ([docs](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator)).

### llmfit

- Detects hardware (Apple Silicon: "VRAM = system RAM") and keeps a model DB "embedded at compile time". It walks "a hierarchy from Q8_0 … down to Q2_K, picking the highest quality that fits", and if nothing fits "it tries again at half context". Its "Quality" dimension is "Parameter count, model family reputation, quantization penalty, task alignment", which is a heuristic and not a measurement. Its speed estimate is `(bandwidth / model_size) × 0.55`. Every number carries a confidence tag (`measured_local`, `measured_community`, `calibrated`, `estimated`) ([docs/how-it-works.md](https://github.com/AlexsJones/llmfit/blob/495ef55e92478b7a6f04d3f3234ce99560e7e186/docs/how-it-works.md)).
- Automation: `llmfit recommend --json`, `llmfit fit`, `llmfit bench`, a REST server, and a JSON output contract ([README](https://github.com/AlexsJones/llmfit/blob/495ef55e92478b7a6f04d3f3234ce99560e7e186/README.md)). The "MoE" run mode puts inactive experts in *RAM*, not on disk.

### llm-checker

- "Hardware-aware model selection". It scores on Quality, Speed, Fit and Context, with a "bytes-per-parameter formula validated against real Ollama sizes", and pulls and runs models through Ollama ([README](https://github.com/signerless/llm-checker)). llmfit's README says it "doesn't support MoE… all models are treated as dense" ([llmfit README § Alternatives](https://github.com/AlexsJones/llmfit/blob/495ef55e92478b7a6f04d3f3234ce99560e7e186/README.md#alternatives)). The license is a custom "No Paid Distribution License (NPDL-1.0)" ([LICENSE](https://github.com/signerless/llm-checker/blob/main/LICENSE)), which is not OSI-approved, so we cannot reuse its code in an MIT/Apache project without care.

### gguf-parser-go

- Reads only the GGUF header, even from a remote URL. It estimates RAM/VRAM per offload setting, with a `UMA` column for Apple unified memory, and "Get Proper Offload Layers". It claims results "usually deviate from the actual usage by about 100MiB" and can compute max TPS from device metrics you supply ([README](https://github.com/gpustack/gguf-parser-go/blob/main/README.md)). Only useful after a GGUF exists.

### mlx-lm and MLX community converters

- `mlx_lm.convert` takes a Source model and quantizes it with fixed `--q-bits` or a `--quant-predicate` recipe (`mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`) ([convert.py L80](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/convert.py#L80)). `mlx_lm.dynamic_quant` "estimate[s] the sensitivity for each quantizable layer" and gives sensitive layers higher precision to hit `--target-bpw`. `--report-ppl` prints base vs quantized perplexity ([LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/LEARNED_QUANTS.md), [dynamic_quant.py L161–L172, L223–L254](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/quant/dynamic_quant.py#L161-L254)). This is the closest existing step to per-part precision with a quality check, but it doesn't look at the device at all.
- At run time it only *warns* when the model is > 90% of `max_recommended_working_set_size` ([generate.py L262–L274](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py#L262-L274)) and points users to `sudo sysctl iogpu.wired_limit_mb=N` ([README § Large Models](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/README.md#large-models)).
- Disk: "`mlx_lm.convert` requires the full source checkpoint on disk before conversion starts" ([RFC #1842](https://github.com/ml-explore/mlx-lm/issues/1842), open). Both reference Source models are about 53–54 GB in BF16 (26.4B / 27B × 2 bytes) [INFERENCE: arithmetic]. That is well over the reference Mac's 21 GB of free disk, so today's converters cannot run on the reference device without streaming the conversion.
- The hosted [mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) and gguf-my-repo Spaces just call `convert(..., q_bits=…)` with a quant the user picks from a dropdown ([app.py](https://huggingface.co/spaces/mlx-community/mlx-my-repo/blob/main/app.py)).

### Quant producers (Unsloth)

- Unsloth picks precision per layer offline for each model ("Each model now uses a custom-tailored quantization scheme"). It reports KL divergence and "Divergence-300 @32" against BF16, and argues "Using perplexity is incorrect" ([Dynamic 3.0 docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)). This is measured Quality loss, but for *their* quants on *their* eval sets, not for a Fit made on a particular device.

### SSD-streaming runtimes

- These exist only as per-model engines or forks: the llama.cpp [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) above; [moe-stream](https://github.com/GOBA-AI-Labs/moe-stream), which has "3 inference modes… auto-selected based on model size vs system RAM" and reports ~2.1 tok/s for an 80B MoE on a 24 GB M4 Pro; and [AirLLM](https://github.com/lyogavin/airllm) (Apache-2.0), which streams layers and lists Qwen3.8-27B support on CUDA. None of them chooses precision or measures Quality loss. The runtime side belongs to other tickets; it appears here only for the disk-streaming column.

## Gap analysis against model-fitter

| Requirement (GLOSSARY) | Best existing coverage | Remaining gap |
|---|---|---|
| Start from a new **Source model** | mlx-lm convert / dynamic_quant; Ollama `create -q` (safetensors) | Every fitter (llama.cpp, Ollama, Llama, LM Studio, HF, llmfit) needs a quant someone already published. Converters need the full checkpoint on disk (about 2.5× the reference Mac's free disk). |
| **Fit plan** with precision per part, residency and runtime | llama.cpp `--fit` (residency for one GGUF); dynamic_quant (bits per layer to hit a bits-per-weight target) | Nobody combines the two. No tool searches precision and placement together under the device's memory **and** disk budget, and none treats looped layers, the vision tower or MoE experts as separate parts with their own precision. |
| **Usable**: ≥ 5 tok/s, ≥ 8K context, machine responsive | Speed formulas (llmfit, gguf-parser), `llama-bench`, Ollama's memory-pressure signal | Every fitter optimises for "fits in memory", not for a speed or context floor. Defaults go below 8K on 16 GB machines (Ollama 4k; llama.cpp will cut down to 4096). Nothing measures responsiveness. |
| Measured **Quality loss** on every Fit | `llama-perplexity --kl-divergence`, `mlx_lm.dynamic_quant --report-ppl`, Unsloth's published KLD | Measurement tools exist but no fitter runs them. A Fit never comes with its own measured loss, and the reference logits needed (from a BF16 model larger than RAM and disk) are not produced on-device. |
| **Disk streaming** chosen by the plan | mmap, `--lazy-mode`, the open MoE-stream PR, one-off SSD engines | No fitter decides *what* to stream or checks that disk read speed can sustain ≥ 5 tok/s. Ollama turns mmap off in exactly the partial-offload case. |
| **New architectures** on release day | mlx-lm and llama.cpp take community PRs within days | On 2026-11-23, Inkling 2 Mini (released 2026-11-20) was still a draft PR in llama.cpp ([#32695](https://github.com/ggml-org/llama.cpp/pull/32695)) and mlx-lm ([#2086](https://github.com/ml-explore/mlx-lm/pull/2086)), and Ollama was waiting on a bump. Catalog tools (llmfit compiled-in DB, Llama/LM Studio catalogs) lag further behind. |
| CLI / automation | llama.cpp `llama-fit-params`, llmfit `--json`, `lms --estimate-only`, gguf-parser | Fine as building blocks. LM Studio guardrails can't be overridden headlessly; the HF panel is UI-only. |

## Implications for model-fitter

- **Don't rebuild placement for GGUF.** Call `llama_params_fit` / `llama-fit-params` as the residency step for llama.cpp Fits and read its memory breakdown. For MLX, the budget to respect is `recommendedMaxWorkingSetSize` (the same number llama.cpp's Metal backend and the Llama app use).
- **Beat the baseline.** The Llama app and llmfit both use "highest-precision published quant that fits". A Fit plan has to beat that on Quality loss at equal speed to justify itself.
- **Report measured, not estimated.** Take llmfit's per-number confidence labels as the pattern and always report measured values. Use KLD against Source-model logits (llama.cpp `--kl-divergence` or an MLX equivalent) as the Quality loss metric, which is what Unsloth and llama.cpp both use.
- **Streamed conversion is a prerequisite** on the reference device, since the BF16 Source model doesn't fit on disk. mlx-lm RFC #1842 describes the approach (shard-by-shard download, quantize, delete).

## Sources

All links checked on 2026-11-23. Code links are pinned to the commit that was read.

- llama.cpp: [PR #16653](https://github.com/ggml-org/llama.cpp/pull/16653), [Discussion #18049](https://github.com/ggml-org/llama.cpp/discussions/18049), [fit-params README](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/fit-params/README.md), [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2665-L2922), [common/common.h](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.h#L504-L509), [common/fit.cpp](https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/fit.cpp), [perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md), [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294), [PR #32695](https://github.com/ggml-org/llama.cpp/pull/32695)
- Ollama: [server/sched.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/server/sched.go), [llm/llama_server.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/llm/llama_server.go), [cmd/cmd.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/cmd/cmd.go), [discover/memory_darwin.go](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/discover/memory_darwin.go), [docs/context-length.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/context-length.mdx), [docs/faq.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/faq.mdx), [docs/import.mdx](https://github.com/ollama/ollama/blob/318ee284aa3b5558de57efefff730ea33c4a9bef/docs/import.mdx), [issue #19826](https://github.com/ollama/ollama/issues/19826), [PR #19861](https://github.com/ollama/ollama/pull/19861)
- Llama app: [llama.app](https://llama.app), [Llama-macOS README](https://github.com/ggml-org/Llama-macOS/blob/4e975ae3e7f25fd16eb9230a29e72a9c977e4193/README.md), [SystemMemory.swift](https://github.com/ggml-org/Llama-macOS/blob/4e975ae3e7f25fd16eb9230a29e72a9c977e4193/Llama/System/SystemMemory.swift), [Catalog.swift](https://github.com/ggml-org/Llama-macOS/blob/4e975ae3e7f25fd16eb9230a29e72a9c977e4193/Llama/Catalog/Catalog.swift)
- LM Studio: [lms load docs](https://github.com/lmstudio-ai/docs/blob/8c7159be3ffe364af10aa8ae6f5a7570e9402089/3_cli/0_local-models/load.md), [lms get docs](https://github.com/lmstudio-ai/docs/blob/8c7159be3ffe364af10aa8ae6f5a7570e9402089/3_cli/0_local-models/get.md), [download-model docs](https://github.com/lmstudio-ai/docs/blob/8c7159be3ffe364af10aa8ae6f5a7570e9402089/0_app/1_basics/download-model.md), [0.3.14 blog](https://lmstudio.ai/blog/lmstudio-v0.3.14), [issue #1631](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1631), [issue #2324](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2324), [App Terms](https://lmstudio.ai/app-terms)
- Hugging Face: [Hardware docs](https://huggingface.co/docs/hub/hardware), [Local Apps docs](https://huggingface.co/docs/hub/local-apps), [hardware.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/hardware.ts), [accelerate estimate-memory](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator), [can-it-run-llm Space](https://huggingface.co/spaces/Vokturz/can-it-run-llm), [mlx-my-repo Space](https://huggingface.co/spaces/mlx-community/mlx-my-repo), [gguf-my-repo Space](https://huggingface.co/spaces/ggml-org/gguf-my-repo)
- llmfit: [README](https://github.com/AlexsJones/llmfit/blob/495ef55e92478b7a6f04d3f3234ce99560e7e186/README.md), [how-it-works](https://github.com/AlexsJones/llmfit/blob/495ef55e92478b7a6f04d3f3234ce99560e7e186/docs/how-it-works.md)
- llm-checker: [README](https://github.com/signerless/llm-checker), [LICENSE](https://github.com/signerless/llm-checker/blob/main/LICENSE)
- gguf-parser-go: [README](https://github.com/gpustack/gguf-parser-go/blob/main/README.md)
- mlx-lm: [convert.py](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/convert.py), [dynamic_quant.py](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/quant/dynamic_quant.py), [generate.py](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py#L262-L274), [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/LEARNED_QUANTS.md), [README § Large Models](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/README.md#large-models), [RFC #1842](https://github.com/ml-explore/mlx-lm/issues/1842), [PR #2086](https://github.com/ml-explore/mlx-lm/pull/2086)
- Unsloth: [Dynamic 3.0 GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)
- Streaming engines: [moe-stream](https://github.com/GOBA-AI-Labs/moe-stream), [AirLLM](https://github.com/lyogavin/airllm)
- Repo metadata (stars, license, last push, latest release): GitHub REST API `GET /repos/{owner}/{repo}`, `/releases/latest`, queried 2026-11-23.
