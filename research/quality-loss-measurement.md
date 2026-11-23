# Measuring Quality loss when the Source model cannot run on the device

Ticket: [#6](https://github.com/ailabhq/model-fitter/issues/6). Researched 2026-11-23.
Terms follow [GLOSSARY.md](../GLOSSARY.md). `[INFERENCE]` marks a conclusion of this note that no source states directly. Every other claim links to the source that owns it.

## Answer

The reference Mac cannot hold either Source model, so **compute the reference off the device, once per Source revision, and ship only a small reference pack to the Mac.** The Mac then runs the Fit over the same tokens and compares.

- **Primary metric: KL-divergence (KLD) plus top-token agreement** between the Fit and Source next-token distributions on a fixed corpus. llama.cpp implements this (`llama-perplexity --kl-divergence-base / --kl-divergence`). Published work finds that KLD tracks answer "flips" well, and that flips can stay hidden when benchmark accuracy looks unchanged ([Dutta et al. 2024][dutta]).
- **The reference does not have to be full-vocabulary.** llama.cpp's full-vocab logit file costs about 0.4–0.5 MB per scored token for these vocabularies, which is too much for 21 GB of free disk. A top-k reference (k=20) costs about 160 B per token. Thinking Machines' Tinker API returns exactly that (`topk_prompt_logprobs`) for **both** reference Source models, at $0.15 (Inkling 2 Mini) or $0.70 (Qwen3.8-27B) per million prompt tokens ([pricing][tinker-price], [SamplingClient][tinker-sc]). vLLM `prompt_logprobs` on a rented GPU is the self-hosted alternative ([vLLM][vllm-sp]).
- **Secondary checks:** greedy multi-token agreement (Unsloth's "Divergence@32") and a small paired task eval scored by flips. Report perplexity only as context: it is not comparable across implementations, it is misleading for chat models and long context, and errors in it can cancel out ([llama.cpp README][ppl-readme], [Fang et al.][longppl], [Unsloth][unsloth-ud]).
- **Published quantized results** (authors' FP8, Unsloth's KLD tables) are useful sanity bounds. They say nothing about a custom Fit unless the Fit uses the exact same files.

## The two reference Source models, as published

| | Inkling 2 Mini | Qwen3.8-27B |
|---|---|---|
| Published precision | **Already quantized.** Mixed NVFP4 (experts in layers 0–19) and MXFP8 (experts in layers 20–23); embeddings, attention and unembedding in BF16 ([hf_quant_config.json][ink-quant]). The card calls it "a single checkpoint of about 20GB, in mixed NVFP4 and MXFP8 precisions" ([card][ink-card]). | BF16, 18 shards ([repo][qwen-card]); official FP8 sibling `Qwen3.8-27B-FP8` ([card][qwen-fp8]) |
| Checkpoint size (HF API) | 19.5 GB model + 0.64 GB MTP | 55.6 GB BF16; 30.9 GB FP8 |
| Text vocab (`config.json`) | `vocab_size` 201024 (`unpadded_vocab_size` 200058) ([config][ink-config]) | `vocab_size` 248320 ([config][qwen-config]) |
| Authors' hardware note | "Apple silicon devices with 32GB or more memory" ([card][ink-card]) | serving via SGLang / vLLM / TokenSpeed ([card][qwen-card]) |
| Hosted with prompt logprobs | Tinker `thinkingmachines/Inkling-2-Mini`, prefill $0.15/M, sample $0.50/M ([pricing][tinker-price]) | Tinker `Qwen/Qwen3.8-27B`, prefill $0.6975/M, sample $5.595/M ([pricing][tinker-price]) |

Consequences:

- For Inkling 2 Mini, "the Source" is the NVFP4/MXFP8 checkpoint. No higher-precision weights are published, so Quality loss can only be measured against that checkpoint. Unsloth builds a lossless reference GGUF (`UD-Q8_K_XL`, 22.87 GB) that "keeps the original NVFP4 expert weights and stores the remaining weights in BF16", and "upcasts MXFP8 to BF16 so KLD = 0" ([Unsloth Inkling guide][unsloth-ink]). That reference is larger than both the 16 GB of memory and the ~21 GB of free disk.
- For Qwen3.8-27B, even the FP8 sibling (30.9 GB) exceeds free disk. On the reference Mac, every Quality-loss reference has to come from somewhere else. `[INFERENCE]`

## Methods

Device costs assume the reference Mac (M4, 10-core GPU, 16 GB, ~21 GB free disk). The only primary throughput figure for this chip is llama.cpp's Apple Silicon table: LLaMA 7B Q4_0 prompt processing (pp512) at **221 t/s** on the M4 10-core ([llama.cpp #4167][m4-perf]). Time estimates below scale that figure by active parameters, so they are `[INFERENCE]`. Budget them as order-of-magnitude numbers until the benchmark ticket measures real ones.

| # | Method | Reference needed / where computed | Disk on the Mac | Time on the Mac (est.) | Runtime support | Known correlation with downstream quality |
|---|---|---|---|---|---|---|
| 1 | **Perplexity (PPL) on a fixed corpus**, compared with Source PPL | One Source PPL number per corpus, runtime and settings. Compute it off-device (cloud GPU, or Tinker `compute_logprobs`, which returns per-token logprobs ([SamplingClient][tinker-sc])). | Corpus only (MBs) | Fit forward pass over the corpus. Example: 16k scored tokens at `-c 512` means 32k evaluated tokens, because llama.cpp scores only the second half of each chunk ([source][ppl-first]). That is ~10 min for 27B dense and a few minutes for Inkling's 5.4B effective active parameters `[INFERENCE]`. | llama.cpp `llama-perplexity` ([README][ppl-readme]); mlx-lm `mlx_lm.perplexity`, which reports PPL ± SE, by default over 256×512 tokens of `allenai/tulu-3-sft-mixture` ([perplexity.py][mlx-ppl]); lm-eval `loglikelihood_rolling` tasks such as wikitext ([lm-eval][lmeval]) | Weak to moderate. Jin et al. find PPL "can serve as a proxy metric for quantized LLMs on most benchmarks" ([2402.16775][jin]). llama.cpp warns that PPL "is **not** directly comparable between models", that its numbers are "not directly comparable to those of other projects", and that finetunes raise PPL even when human-rated quality improves ([README][ppl-readme]). PPL is unreliable for long context because it averages over the key tokens ([2410.23771][longppl]). Unsloth: "Using perplexity is incorrect since output token values can cancel out" ([Unsloth][unsloth-ud]). |
| 2 | **Full-vocabulary KLD** vs Source logits | A Source logit file over the same tokens. llama.cpp writes it with `--kl-divergence-base FILE` while running the reference model ([README][ppl-readme]); args at ([arg.cpp][ppl-args]). Must run on a machine that holds the Source: a cloud GPU, or a larger Mac. | **Large.** The file stores, per scored token, `2*((n_vocab+1)/2)+4` uint16 log-probs ([source][ppl-fmt]). That is ≈ 497 KB/token for Qwen3.8-27B (248320 vocab) and ≈ 402 KB/token for Inkling (201024) `[INFERENCE: assumes GGUF n_vocab = config vocab_size]`. llama.cpp's own examples: "11 GiB for LLaMA 2 or 37 GiB for LLaMA 3" on Wikitext-2 ([README][ppl-readme]). A 4 GB budget holds only ~8k scored Qwen tokens. | Same as #1: one forward pass of the Fit | llama.cpp only: `llama-perplexity -m fit.gguf --kl-divergence-base ref.kld --kl-divergence`. It reports mean/percentile KLD, PPL ratio, Δp percentiles, RMS Δp and "Same top p" ([README][ppl-readme]). mlx-lm has no KLD tool: its `perplexity.py` takes no reference argument ([perplexity.py][mlx-ppl]). | **Best available proxy.** Dutta et al. show that quantized models with near-equal accuracy still flip many answers and score "significantly worse" on MT-Bench. They propose KLD and flips and "show that they are well correlated" ([2407.09141][dutta]). Unsloth calls KLD "one of the gold standards for reporting quantization errors" ([Unsloth][unsloth-ud]). |
| 3 | **Top-k (sparse) KLD** vs Source top-k logprobs | Per position, the Source's top-k token ids and logprobs. **Tinker** returns `topk_prompt_logprobs=k` for every prompt position in one prefill (`max_tokens=1`) for both Source models ([SamplingClient][tinker-sc]). Tinker's own docs say the API "does not expose full-vocabulary logits" and that top-K=20 matched full-vocab KL in their distillation ablation ([SDFT recipe][tinker-sdft], [cross-entropy][tinker-ce]). **vLLM** returns `prompt_logprobs=k`; `-1` means the full vocabulary, but the default server cap is `max_logprobs=20` ([sampling_params.py][vllm-sp], [config][vllm-max]). | **Small.** k=20 × (int32 id + float32 logprob) ≈ 160 B/token, so 100k tokens ≈ 16 MB `[INFERENCE]` | Same as #1. The Fit side needs full logits locally, which both llama.cpp and MLX produce. The metric code (~50 lines) does not exist in either runtime and must be written `[INFERENCE]`. | Same metric family as #2. Grouping all tokens outside the Source's top-k into one "tail" bucket gives a **lower bound** on the true KL, because merging outcomes cannot increase KL (data-processing inequality) `[INFERENCE: standard result]`. It is exact for the dominant mass. |
| 4 | **Top-token agreement** ("Same top p" / top-1) | Comes with #2 or #3: the Source argmax per position (only token ids are needed: 4 B/token) | Negligible | Same as #1 | llama.cpp `--kl-divergence` prints "Same top p" ([README][ppl-readme]) | Token-level analogue of flips ([2407.09141][dutta]). llama.cpp's Llama-3-8B table shows it falling with bit width: q8_0 97.7 %, q4_K_M 91.9 %, q2_K 71.1 % ([README][ppl-readme]). Unsloth's Inkling table: UD-Q4_K_XL KLD 0.0107 / 98.0 %, UD-IQ2_XXS 0.0726 / 90.6 % ([Unsloth Inkling][unsloth-ink]). |
| 5 | **Greedy trajectory agreement** ("Divergence@N") | The Source's greedy continuation (N tokens) for a fixed prompt set: token ids only. Generate on Tinker or vLLM. | KBs | Prefill of prompts plus N×prompts decode tokens at the Fit's tok/s. For 300 prompts × 32 tokens, that is 9.6k decode tokens: ~30 min at 5 tok/s `[INFERENCE]`. | Any runtime with temperature 0 (llama.cpp, mlx-lm). There is no built-in comparator. | Unsloth's "Divergence-300 @32": 300 held-out prompts, greedy decoding of 32 tokens, compared with BF16. They call it "a better indicator for actual inference workloads" than top-1. On 1-bit quants: "A Top-1 recovery of 77% is not a replacement for Divergence-300 @32 at 8%" ([Unsloth][unsloth-ud]). Caveat: GPU non-determinism makes greedy references noisy ([Unsloth Qwen3.8][unsloth-qwen]). |
| 6 | **Small task evals**, paired with the Source | Source answers or scores on the **same items, same harness and same prompts**. Generate them off-device: lm-eval `local-completions` against vLLM/Tinker-like servers ([lm-eval][lmeval]). Model-card numbers do not qualify (different harness, thinking settings, sampling). | Datasets (MBs) | Loglikelihood multiple-choice tasks are prefill-only: minutes to ~1 h per few hundred items. Generative or thinking tasks at Usable speed (5 tok/s) take hours per hundred items `[INFERENCE]`. | `mlx_lm.evaluate` wraps lm-eval (`loglikelihood`, `loglikelihood_rolling`, `generate_until`; `--limit`, `--num-shots`, chat-template flags) ([evaluate.py][mlx-eval]). lm-eval `gguf` drives `llama-server` (loglikelihood and generation; no rolling PPL) ([lm-eval][lmeval]). `llama-perplexity --hellaswag / --winogrande / --multiple-choice` ([arg.cpp][ppl-args]). | Directly measures capability but is **noisy at small n**. Unsloth found single benchmark runs misleading, averaged 14 trials, and states "benchmarks should be run multiple times" ([Unsloth Qwen3.8][unsloth-qwen]). Aggregate accuracy hides flips, so score the paired per-item disagreement ([2407.09141][dutta]). 4-bit losses concentrate in long-context and non-English tasks, up to 59 % drops ([2505.20276][longq]). |
| 7 | **Compare with published quantized results** | None to compute. Use the authors' or a quantizer's published numbers. | None | None | n/a | Indicative only. Qwen says its FP8 is "nearly identical" to the original, with no numbers ([FP8 card][qwen-fp8]). Kurtic et al. find FP8 "effectively lossless" and INT4 weight-only "competitive" on Llama-3.1 ([2411.02355][kurtic]). Unsloth publishes per-quant KLD/top-1 for Inkling 2 Mini ([guide][unsloth-ink]), and per-domain KLD for Qwen3.8 NVFP4 vs BF16, e.g. chat 0.058 / 92.2 %, code 0.026 / 96.7 % ([guide][unsloth-qwen]). These apply only to the exact published files, run on their runtimes. |
| 8 | *(Feasibility)* **On-device exact reference by layer streaming** | The full Source checkpoint, streamed layer by layer (AirLLM-style). AirLLM states that it first "decompose[s]" the model to disk layer-wise and needs "sufficient disk space" ([AirLLM][airllm]). | ≥ checkpoint size: 55.6 GB (Qwen BF16), or about 20–23 GB for Inkling. **Exceeds ~21 GB free.** A custom pass that downloads, applies and deletes one shard at a time could fit, but needs architecture-specific code `[INFERENCE]`. | Dominated by download plus a full-weight read per corpus pass `[INFERENCE]` | AirLLM (macOS via MLX) ([AirLLM][airllm]) | Same as #2 (exact full-vocab KLD), but not practical on the reference device `[INFERENCE]` |

## Runtime support summary

| Capability | llama.cpp | mlx-lm (v0.32.2) | lm-eval-harness |
|---|---|---|---|
| PPL | `llama-perplexity` ([README][ppl-readme]) | `mlx_lm.perplexity` ([perplexity.py][mlx-ppl]) | `loglikelihood_rolling` tasks; not via the `gguf` backend ([README][lmeval]) |
| Save reference logits | `--kl-divergence-base` (full vocab, uint16) ([source][ppl-fmt]) | none | none |
| KLD / top-token agreement | `--kl-divergence` ([README][ppl-readme]) | none built-in (custom code over model logits) | none |
| Task evals | `--hellaswag`, `--winogrande`, `--multiple-choice` ([arg.cpp][ppl-args]); lm-eval via `llama-server` | `mlx_lm.evaluate` (lm-eval adapter) ([evaluate.py][mlx-eval]); MMLU-Pro example table for bf16→q4 ([BENCHMARKS.md][mlx-bench]) | Backends: `local-completions` (loglikelihood needs logprobs), `vllm`, `gguf` ([README][lmeval]) |
| Off-device reference source | any GPU box running `llama-perplexity` on a BF16/Q8 GGUF | n/a | `vllm` / `local-completions` |

Architecture support for Inkling's looped layers in each runtime is out of scope here (ticket on runtime support). Unsloth notes that it needs "a build with looped-layer support and per-expert NVFP4 scales" ([guide][unsloth-ink]).

## Recommendation for model-fitter `[INFERENCE]`

1. **Reference pack per Source revision** (keyed by HF commit sha). Contents: a fixed tokenized corpus; for every scored position, the Source's top-20 ids and logprobs plus the sampled-token logprob; greedy 32-token continuations for ~300 prompts; and optionally Source per-item answers for one small paired eval. Build it once off-device: on Tinker for both reference Sources (cost is cents for ~100k prefill tokens at the [listed prices][tinker-price]), or with vLLM on a rented GPU. Spot-check a slice against a second implementation, because Tinker does not document its serving precision. The pack is tens of MB and can be published, so users never need the Source locally.
2. **Corpus.** Use the Source's own chat-templated outputs across domains (chat, code, multilingual, tool calls), since the Sources are chat/thinking models and Wikitext PPL is a base-model convention ([README][ppl-readme]). Include at least one **8K-context** slice, because Usable requires ≥ 8K context and long-context loss is where 4-bit quants fail ([2505.20276][longq]). Score with the Fit's actual KV-cache type, so that KV quantization counts too.
3. **Report**, for every Fit: mean and 99th-percentile KLD, top-1 agreement, Divergence@32, and PPL ratio. Optionally add a paired-flip rate on a small eval, with its binomial uncertainty.
4. **Multimodal gap.** None of the text-only tools above measure vision-tower quantization. Image prompts need their own corpus through a multimodal runtime (Tinker supports image inputs for Inkling ([Inkling cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/images/index.md))).

## Open questions

- What is the maximum `topk_prompt_logprobs` k and prompt length per Tinker request? The SDK reference only says the server bounds it ([SamplingClient][tinker-sc]).
- Does the GGUF `n_vocab` equal `config.vocab_size` for both models? This affects the 0.4–0.5 MB/token estimate.
- Measured pp/tg rates for each Fit on the M4 (belongs to the benchmark ticket).

[ppl-readme]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/perplexity/README.md
[ppl-fmt]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/perplexity/perplexity.cpp#L144-L172
[ppl-first]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/perplexity/perplexity.cpp#L519-L622
[ppl-args]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2447-L2500
[m4-perf]: https://github.com/ggml-org/llama.cpp/discussions/4167
[mlx-ppl]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/perplexity.py
[mlx-eval]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/evaluate.py
[mlx-bench]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/BENCHMARKS.md
[lmeval]: https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/README.md
[vllm-sp]: https://github.com/vllm-project/vllm/blob/6aeedab553dd85d6bc3f9f4deb3bb264bce0cb7c/vllm/sampling_params.py#L342-L352
[vllm-max]: https://github.com/vllm-project/vllm/blob/6aeedab553dd85d6bc3f9f4deb3bb264bce0cb7c/vllm/config/model.py#L305-L309
[tinker-sc]: https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md
[tinker-price]: https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md
[tinker-sdft]: https://tinker-docs.thinkingmachines.ai/cookbook/recipes/sdft/index.md
[tinker-ce]: https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/index.md
[ink-card]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/README.md
[ink-quant]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/hf_quant_config.json
[ink-config]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/config.json
[qwen-card]: https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md
[qwen-config]: https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/config.json
[qwen-fp8]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8
[unsloth-ink]: https://unsloth.ai/docs/models/inkling-2-mini
[unsloth-qwen]: https://unsloth.ai/docs/models/qwen3.8
[unsloth-ud]: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
[dutta]: https://arxiv.org/abs/2407.09141
[jin]: https://arxiv.org/abs/2402.16775
[kurtic]: https://arxiv.org/abs/2411.02355
[longppl]: https://arxiv.org/abs/2410.23771
[longq]: https://arxiv.org/abs/2505.20276
[airllm]: https://github.com/lyogavin/airllm
