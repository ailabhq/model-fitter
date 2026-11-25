# Draft Reference packs: Quality loss of the Usable builds on the M4 16 GB

Map ticket: "Draft Reference packs: what Quality loss do Qwen UD-IQ3_XXS and Inkling oQ2.5e have on the M4 16 GB?"

## Answer

**Under the rules as decided, all four builds are Rejected.** That verdict is driven by the user's prompt tokens, where even two full-precision references disagree beyond the Near-lossless threshold. **Scored on the Source's own response tokens only, the three llama.cpp builds are Lossy and MLX oQ2.5e is Rejected.**

| Build (runtime) | Whole corpus: mean KLD / top-1 (band) | 8K item, whole | Response tokens only (band) | Prompt tokens only | Scoring wall time on the Mac |
|---|---|---|---|---|---|
| Qwen UD-IQ3_XXS (llama.cpp `--fit`, `-b 128`) | 0.864 / 83.7% (**Rejected**) | 1.914 / 75.4% | **0.060 / 91.8% (Lossy)** | 1.798 / 74.3% | 6.7 min (405 s) |
| Inkling UD-IQ3_XXS (llama.cpp `--fit`, Metal NVFP4) | 0.078 / 92.9% (**Rejected**: 8K item) | 0.144 / 90.7% | **0.020 / 95.4% (Lossy)** | 0.153 / 89.7% | 8.6 min (515 s) |
| Inkling UD-Q2_K_XL (llama.cpp `--fit`, Metal NVFP4) | 0.126 / 90.1% (**Rejected**) | 0.208 / 88.0% | **0.049 / 92.7% (Lossy)** | 0.226 / 86.6% | 3.5 min (211 s) |
| Inkling oQ2.5e (MLX, 128-token prefill chunks) | 0.243 / 86.2% (**Rejected**) | 0.359 / 84.3% | **0.105 / 88.9% (Rejected)** | 0.421 / 82.6% | 3.1 min (183 s) |

Bands use the provisional thresholds (Near-lossless ≤ 0.02 / ≥ 97%, Lossy ≤ 0.10 / ≥ 88%), the worse of whole corpus and 8K item. Every run passed the tokenizer check and was Responsive, behind the quiet gate. The response-only 8K item has only 512 positions; it lands at 0.004–0.019 KLD for every build, so it never decides the band.

### Why the prompt tokens can't decide the band

- **The Sources barely model user text.** Over the 8K document, Qwen's mean log-prob for the token that actually follows is −6.43 nats (−6.43 in vLLM too). Its own top token is right on only 40% of positions, against 99% on its own response. Inkling does better on the document (−1.73, 70%), but its short user prompts sit at −2 to −7. Three independent implementations agree on this: transformers, vLLM and llama.cpp (where the Fit's top token matches the next document token on 38% of positions).
- **Two full-precision references disagree there.** transformers against vLLM, same GPU and same tokens:

  | | Response tokens | Prompt tokens |
  |---|---|---|
  | Qwen (vLLM in BF16) | KLD 0.0005, top-1 99.96% | KLD 0.060, top-1 96.8% |
  | Inkling (vLLM's own NVFP4 path, `quantization: inkling2`) | KLD 0.0025, top-1 98.5% | KLD 0.088, top-1 93.3% |

  On prompt tokens the reference noise alone is already past the Near-lossless KLD threshold (0.02) and, for Inkling, past the 97% top-1 threshold.
- **The Fits blow up there.** Qwen's prompt-token KLD reaches p99 23. Positions where the Source puts 0.99+ on one token, often wrong about the actual text (it "completes" its own default system prompt and the document's model names differently), get ~0 from the quantized Fit. These are distributions the Source was never trained to get right, so small weight perturbations move them a lot.
- **On response tokens the measurement is stable.** The two references agree to 0.0005–0.0025, about 10× below the smallest Fit loss measured. That is roughly the noise floor the Qwen2.5-0.5B smoke test showed for a BF16 reference.

`[INFERENCE]` Scoring only the Source's response tokens (prompt as context) is what makes the band mean something. That reverses part of "How are Reference packs built and distributed…" ("Scoring covers both prompt and response"), so it is left to its own ticket.

### Facts later tickets depend on

- **Qwen UD-IQ3_XXS is not Rejected on response tokens** (Lossy, 0.060 / 91.8%), so the only Usable Qwen build stays a candidate, with headroom of 0.04 KLD and 3.8 points of top-1 to the Rejected line. Its loss is spread evenly: chat 0.078, multilingual 0.074, code 0.043, tool calls 0.031.
- **For Inkling, llama.cpp beats MLX on quality as well as being simpler**: UD-IQ3_XXS (0.020 / 95.4%) < UD-Q2_K_XL (0.049 / 92.7%) < oQ2.5e (0.105 / 88.9%, Rejected). MLX oQ2.5e is the fastest build (22 tok/s) and the most lossy. Inkling UD-IQ3_XXS misses Near-lossless only on top-1 (95.4% vs 97%; KLD 0.0203 vs 0.02).
- **Inkling's reference noise on response top-1 is 98.5%** (vLLM's NVFP4 kernels vs transformers' dequantized BF16). The pack cross-check tolerance (top-1 ≥ 99%) and a 97% Near-lossless top-1 threshold leave little room above that noise.
- **The scorer's own memory counts against the Responsive rule.** Keeping full-vocabulary logits for every position costs about `n_batch × vocab × 4 B` twice (host and Metal), ~0.5 GB each at `-b 512` for Qwen's 248K vocabulary. At `-b 512` the Qwen run tripped the kill-switch 21 s in (window check stuck 793 ms, peak wired 13.35 GiB, 4.5 s critical; kept in `results/aborted/`). At `-b 128 -ub 128` it was Responsive (peak wired 12.76 GiB, no critical pressure). Inkling UD-IQ3_XXS at `-b 512` passed but swapped 32.7 GiB out, against 2.7 GiB in its speed benchmark. Prefill batch size barely changes the numbers; placement can change slightly (`--fit` sees a smaller compute buffer).
- **Packs are small and quick to build.** 7 MB each (21–22K positions, top-20). On the H100: download 1–2 min, generation 2.6–4.8 min with transformers' slow fallback kernels, scoring 5–7 s. The whole pod session (setup, both packs, both vLLM checks, one rerun) cost about $2.40 at $4.39/h. vLLM needed `max_num_seqs` ≤ 251 for Qwen's linear-attention cache.
- **Pack generation ran with transformers' reference PyTorch kernels** for Qwen's gated delta net (`flash-linear-attention` and `causal-conv1d` not installed). vLLM agrees with it to 0.0005 on responses, so the slow path is correct, just slower.

## Method

### Prompt set v0 (draft)

[`draft-packs/prompt-set-v0/prompts.json`](draft-packs/prompt-set-v0/prompts.json). It is enough for a measurement, not the curated v1:

- 16 short items, 4 per domain: chat, code (Python, JS, SQL, C), multilingual (French, Chinese, Spanish, Japanese) and tool calls (one, two and three tools, plus one turn with a tool result). All are maintainer-written (CC0-1.0). The decided v1 takes chat and code from permissively licensed post-cutoff datasets instead.
- One 8K item: llama.cpp `docs/speculative.md` lines 1–473 at `b6d5ce4bc` (MIT), with a question that needs the whole document. It is 7,642 prompt tokens for Inkling and 6,738 for Qwen, plus up to 512 response tokens, so it fits the 8,704-token context the builds run with.
- Each item is rendered with the Source's own chat template in its default thinking mode (Inkling: effort 0.9; Qwen: thinking on, with the template's default system prompt). The Source's temperature-0 response is appended, capped at 1,024 tokens (512 for the 8K item). Every position is scored. The report splits the positions into prompt and response (rows from `n_prompt − 1` on predict the Source's own response).

### Reference packs

- Built by [`pack/build_pack.py`](draft-packs/pack/build_pack.py) with transformers 5.24.0 (torch 2.15, CUDA 13.2) on one rented RunPod H100 80 GB ([`pack/pod_run.sh`](draft-packs/pack/pod_run.sh), [`pack/runpod_pod.py`](draft-packs/pack/runpod_pod.py)). Weights are BF16 (Inkling's NVFP4/MXFP8 dequantized on load). Log-probs are an FP32 `log_softmax` over the full vocabulary. Each item is scored in one unbatched forward pass with no cache. Responses are generated greedily in a left-padded batch.
- Sources: `Qwen/Qwen3.8-27B` at `37561816`, and `thinkingmachines/Inkling-2-Mini` at `c9d02830` (the commit after `7a67ae23` only adds evaluation results; the weights are unchanged).
- Per position the pack keeps the top-20 ids and log-probs and the log-prob of the token that follows. The tokens' text is stored with the Source tokenizer's ids for that text, for the tokenizer check.
- Second implementation: [`pack/cross_check.py`](draft-packs/pack/cross_check.py) runs vLLM `0.34.1rc1.dev100` `prompt_logprobs=20` on the same GPU. It ran on **every** item (`--extra-tokens 100000`) so that prompt and response positions could be compared. vLLM's own top-20 is saved in `packs/<source>/cross_check/`. Tolerance: mean KLD ≤ 0.005 and top-1 ≥ 99% (provisional). Over all positions both packs fail (Qwen 0.028 / 98.5%, Inkling 0.040 / 96.3%). Over response positions Qwen passes (0.0005 / 99.96%), and Inkling passes on KLD (0.0025) but misses on top-1 (98.5%).

### Scoring on the Mac

- llama.cpp: `llama-mf-score` ([source](draft-packs/score/llama-mf-score.cpp)), a ~150-line tool on llama.cpp's common library, built in llama.cpp `master` `22b99122e` (it contains `51831d47d`, Metal NVFP4). It takes the same flags as the server, so the same `--fit` placement, `-c 8704 -fa on` and f16 KV cache apply. It prefills each item in a cleared context with logits on every position, and writes the log-prob the Fit gives each of the pack's top-20 tokens plus the Fit's own top token. It also tokenizes the item text (special tokens parsed, no BOS) for the tokenizer check. Inkling ran at `-b 512`, Qwen at `-b 128 -ub 128` (see the memory fact above).
- MLX: [`score/mlx_score.py`](draft-packs/score/mlx_score.py) runs oMLX 0.8.3's Inkling 2 code through mlx-lm with one prompt cache per item, prefilling in chunks of 128 tokens. That is the safe step from "Can MLX hold Inkling oQ2.5e at 8K…"; the logits of a chunk add to the peak (11.33 GiB here).
- All four runs went through [`score/score_fit.py`](draft-packs/score/score_fit.py), driven by [`score/run_all.sh`](draft-packs/score/run_all.sh): `bench.py`'s quiet start gate, its Monitor (Responsive probes, kill-switch, contamination retry), then the report. Builds: unsloth GGUFs as in the Baseline and "How fast is Inkling 2 Mini on llama.cpp with Metal NVFP4…" (Qwen at `4af51b35`, Inkling at `8078d149`), and `mlx-community/Inkling-2-Mini-oQ2.5e-mtp` at `d4a4f3f5`. Each was downloaded, scored and deleted in turn. Results: [`draft-packs/results/`](draft-packs/results/), with the raw per-position output (`*.scores.bin`) for re-analysis.

### Metrics

[`pack/metrics.py`](draft-packs/pack/metrics.py):

- **KLD(Source ‖ Fit)** over a coarsened distribution: the Source's top-20 tokens plus one bucket for the rest of the vocabulary. Coarsening never increases KL, so this is a lower bound on full-vocabulary KLD. Where the Source is confident the rest bucket is small; on user-text positions it can be large, so the bound is looser there. The reference-vs-reference check coarsens to the ids both sides list plus the rest.
- **Top-1 agreement** counts a position as agreeing when the Fit's top token is one of the Source's top tokens. Exact ties are common in BF16 logits. Without this, a BF16 model scored against its own pack agreed on only 98.9% of positions, at KLD 0.
- The **band** is the worse of the whole corpus and the 8K item, under the provisional thresholds (Near-lossless ≤ 0.02 / ≥ 97%, Lossy ≤ 0.10 / ≥ 88%).

### Validation of the pipeline (Qwen2.5-0.5B-Instruct, CPU-built pack)

| Scored model | Mean KLD | Top-1 |
|---|---|---|
| transformers FP32 (vs the BF16 pack) | 0.0023 | – |
| llama.cpp F32 GGUF (converted from the same weights) | 0.0018 | 99.5% |
| MLX BF16 (converted from the same weights) | 0.0013 | 99.1% |
| llama.cpp Q4_K_M (Qwen's official GGUF) | 0.074 | 90.7% |
| MLX 4-bit (mlx-community) | 0.20 | 81.5% |

So both scorers read the pack correctly, and a BF16 reference sets a noise floor of about 0.002 KLD.
