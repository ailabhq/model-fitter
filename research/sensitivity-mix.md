# Does a sensitivity-weighted mix lose no more than Unsloth's UD-IQ3_XXS at the same size?

Map ticket: "Does a sensitivity-weighted mix lose no more than Unsloth's UD-IQ3_XXS at the same size, for both reference models?" (map: "Map: sarto, Usable Fits of new models on small Apple Silicon Macs"). This branch is never merged.

## Answer

**Not answered: the measurement as decided can't produce sₜ.** In a BF16 transformers forward pass, swapping *any* Qwen tensor costs about 0.0005 KLD on response tokens, even noise below BF16's own rounding. A single tensor's IQ3_XXS error adds only 0–0.0004 on top of that. So one real swap per tensor at about 3 bpw reads mostly the floor, and sₜ = KLDₜ / eₜ would be floor ÷ table error, which says nothing about the tensor. No mix was solved, converted or scored. The run was stopped after 71 of Qwen's 402 probes and a dose-response check, to save the GPU budget for a method that works.

## The floor

Dose-response on Qwen3.8-27B: 4 tensors, each swapped at four ggml types and once with Gaussian noise of relative size 1e-3 (the control), over 40 calibration sequences (38,257 positions, 29,839 of them response positions). Mean KLD(Source ‖ swapped) on response tokens:

| Tensor | Noise 1e-3 (rel 0.0008) | Q8_0 (0.006) | Q4_K (0.07) | IQ3_XXS (0.19) | IQ2_XXS (0.36) |
|---|---|---|---|---|---|
| `blk.0.ffn_down` (89 M) | 0.000533 | +0.000003 | +0.000014 | **+0.000071** | +0.000213 |
| `blk.0.ssm_beta_alpha` (0.5 M) | 0.000532 | +0.000003 | +0.000001 | **+0.000034** | +0.000019 |
| `blk.11.attn_k` (5 M) | 0.000481 | −0.000001 | +0.000012 | **+0.000045** | +0.000143 |
| `blk.32.ffn_down` (89 M) | 0.000368 | +0.000007 | +0.000096 | **+0.000425** | +0.001259 |

The control column is absolute; the other columns are the excess over it. Nothing swapped gives exactly 0.

- **Any change re-rolls about 0.0004–0.0005.** The 1e-3 noise is below BF16's rounding step (about 0.4%), yet it costs as much as Q8_0. `[INFERENCE]` Changing any weight changes the BF16 rounding of everything downstream, and over 64 layers that noise grows to about 0.0005 KLD. It is the same size as transformers vs vLLM on Qwen's responses (0.0005, draft packs ticket). The floor is spread over all positions: no response position exceeds 1 nat, and the median moves with it.
- **The floor depends on where the swap is:** 0.00037 at layer 32, 0.00053 at layer 0. One global floor can't be subtracted; it has to be matched per tensor.
- **Above the floor the signal is real for large tensors.** It grows with the error, roughly as its 1.6th power (`blk.32.ffn_down`: relative error 0.073 → 0.185 → 0.349 for Q4_K → IQ3_XXS → IQ2_XXS gives excess KLD 1 : 4.4 : 13). Depth shows: layer 32's `ffn_down` is about 6× layer 0's at IQ3_XXS, the direction of Unsloth's depth ramp.
- **Small tensors are lost in the floor's own jitter.** Repeat swaps differ by about ±1e-5 (control vs Q8_0), and `ssm_beta_alpha` reads 0.000034 at IQ3_XXS but 0.000019 at IQ2_XXS. Unsloth's types imply a weight of about 196 for it ("How does sarto's allocator weigh one tensor's quantization error against another's…"); this measurement can neither confirm nor refute that.

### What the 71 completed probes show

All 71 Qwen probes (layers 0, 1 and 10–19, every role there) read 0.00047–0.00101 KLD on response tokens and 0.0026–0.0111 over every position. The per-sequence KLD of two unrelated tensors (`blk.0.ssm_beta_alpha`, `blk.0.ffn_down`) correlates at 0.95. Over every position the probe is dominated by prompt positions: they are 24% of the positions and carry 86–94% of the KLD, mostly in a few positions of 16–40 nats, as when the draft packs were scored. Data: `results/qwen-iq3-probes.partial.jsonl`.

## What does work, for the next decision

- The pipeline is sound. With nothing swapped, KLD is exactly 0, from layer 0 and from a stored layer-32 input. Logits recomputed from stored hidden states equal the model's own. All 402 Qwen probe tensors map to their checkpoint tensors (498 parts compared value for value).
- A matched control (the same tensor perturbed below BF16 rounding, or at Q8_0) removes the floor. For large tensors, the excess is a usable signal.
- Probe cost: a whole-model pass of Qwen over 131,598 tokens takes 19.5 s on one H100 SXM; from a stored layer-32 input, 11 s. A per-tensor probe with a matched control is therefore about 2 × 15 s × 402 ≈ 3.4 h for Qwen, more than this ticket's estimate.
- `[INFERENCE]` Ways around the floor: (a) floor-subtracted probes with a matched control per tensor; (b) a larger probe dose (about 2 bpw), which needs the per-tensor slope; (c) probing groups (one role × a block of layers, about 16× the signal) and spreading sₜ within a group; (d) an FP32 forward pass, which removes the cause but needs more than 80 GB for either model (Qwen ≈ 108 GB) and is about 15× slower without TF32. Each changes the method decided in "How does sarto's allocator weigh one tensor's quantization error against another's…".

## Other facts

- **Calibration stand-in** (`calib/`): 224 prompts, 56 per domain, from OpenAssistant/oasst2 (Apache-2.0; English roots for chat, 12 other languages for multilingual), ise-uiuc/Magicoder-OSS-Instruct-75K (MIT; 9 languages) and glaiveai/glaive-function-calling-v2 (Apache-2.0; first user turn with its function schemas). None shares a prompt or an 8-word run (16 characters for CJK) with Prompt set v0 (`overlap.json`).
- **The Sources' responses** (greedy, default thinking, 1,024-token cap) with vLLM `0.34.1rc1.dev209` on the H100: Qwen 202,816 tokens in 88 s, Inkling 188,445 tokens in 34 s. Most chat, multilingual and code responses hit the cap mid-thinking, as in the Reference pack; tool calls finish (56/56, short). Token ids are in `calib/{qwen,inkling}.jsonl`. The probes take whole items round-robin across domains up to about 131K tokens (Qwen: 145 sequences, 131,598 tokens).
- **The weighted solver reproduces the table-mix ticket's mixes.** Run without weights, `solve_weighted.py` gives both type maps from "Does sarto's table-allocated mix lose less quality than Unsloth's UD-IQ3_XXS at the same size…" tensor for tensor, and their file sizes to the byte (Qwen 10,934,850,240, Inkling 11,663,310,368). The inputs come from `layout.py` on the reference conversions: header 10,990,016 / 11,977,760; other tensors 10,582,016 / 147,523,584; Qwen MTP fixed at 351,008,768.
- **The NVFP4 copy works on the converted file.** `splice_nvfp4.py` swaps a `llama-quantize` output's NVFP4 experts for the Source's bytes plus `.scale`. Applied to the Inkling reference conversion, all 20 `ffn_down_exps` (layers 0–19) match Unsloth's file byte for byte, `.scale` included, in under 6 min on the NAS.
- **Probe types.** IQ3_XXS for 400 Qwen and 200 Inkling tensors; Q3_K for `output` and `token_embd` (no imatrix). Inkling's 24 `ffn_down_exps` have rows of 1152, which allow neither, so they would take IQ4_NL, the type their mixes land on.
- **Cost:** one H100 SXM (secure cloud, $4.39/h; no community H100 was available) for about 44 min, $3.20: setup, both calibration texts, 71 probes and the dose-response check. The pod was terminated.

## Method

- `pod/probe.py`, `pod/units.py`, `pod/ggmlq.py`: one probe per GGUF tensor of the pinned converter (fused `attn_qkv_gate`, `ssm_beta_alpha`; one layer's routed experts). Each part is quantized and dequantized with `ggml_quantize_chunk` and the type's `to_float` from libggml-base at `22b99122e`, with Unsloth's imatrix (Qwen's adapted to the fused names), in GGUF column order (Qwen `ssm_out`'s V-head reorder, checked against the converter's own `_reorder_v_heads`). Probed values are rounded to BF16. KLD is fp32 `log_softmax` over the full vocabulary, at every position, with response positions also reported apart. Qwen reruns from the stored input of every 4th layer; layers before it are swapped for a stub.
- `pod/diag.py`: the dose-response check. `pod/smoke.py`: a CPU smoke test on tiny random Qwen3.5 and Inkling 2 Loop models. `pod/gen_calib.py`, `pod/pod_run.sh`, `pod/runpod_pod.py`: the pod.
- transformers 5.24.0, torch 2.15.0+cu132, flash-linear-attention 0.6.0, BF16 weights, TF32 off.

## Files

- `calib/`: prompt builder, prompts, overlap check, both Sources' responses (token ids).
- `pod/`: probes, dose-response check, smoke test, pod scripts.
- `solve_weighted.py`, `layout.py`, `splice_nvfp4.py`: solver, file layout, NVFP4 copy.
- `results/qwen-iq3-probes.partial.jsonl`: the 71 probes. `results/diag-qwen/`: the dose-response summary and per-position KLD (`*.npy`, float32, the 38,257 positions of the 40 sequences).
