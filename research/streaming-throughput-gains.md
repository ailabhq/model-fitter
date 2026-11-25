# How much do batched and speculative decoding multiply throughput when weights are read from SSD, and where do the gains stop?

Map ticket: "How much do batched decoding and speculative decoding multiply throughput when weights are read from SSD, and where do the gains stop?" Researched 2026-11-25. Desk research: nothing was downloaded or run.

Every tok/s and tok/h figure below is an **estimate** from the cost model in [`streaming-throughput-gains/model.py`](streaming-throughput-gains/model.py), not a measurement. Measured numbers are quoted from their sources.

## Answer

- **When weights stream from SSD, decoding is bound by the read, not by compute.** One decode step reads the streamed weights once, however many positions it evaluates. A position is one token the target evaluates: N sequences × (1 + k drafted tokens). The M4 GPU only becomes the limit at about **1,400 positions per step at 2.2 GB/s (≈ 620 at 5 GB/s)**. A 16 GB Mac holds at most ~125 positions (Qwen) or ~90 (Inkling). So up to the memory limit, both levers multiply almost for free.
- **Dense Qwen3.8-27B BF16.** A step reads 51.25 GB.
  - **Single stream: ~0.05 tok/s (≈ 190 tok/h) at 2.2 GB/s, 0.12 tok/s (≈ 430 tok/h) at 5 GB/s.**
  - **Batching** scales throughput linearly. It stops at memory: each sequence needs 0.29 GB at 2K context, 0.69 GB at 8K and 2.3 GB at 32K, and it takes that RAM from the pinned weights. On a ~10 GB resident budget that is **N ≈ 34 / 14 / 4** sequences. Best batch-only: **≈ 5,300 / 2,200 / 630 tok/h** at 2.2 GB/s, ×2.27 at 5 GB/s.
  - **MTP drafting** multiplies by the mean tokens per step: 1.87 / 2.55 / 3.14 / 3.64 for k = 1–4, measured on vLLM. With llama.cpp's new GDN replay (merged 2026-11-23), drafting adds only ~3.4 MiB per position to each sequence. Batching and drafting then compose: **≈ 14,000 / 6,200 / 1,700 tok/h at 2K / 8K / 32K (k = 4)**, 9–75× the single stream. Older snapshot rollback stores one 150 MiB state per draft position, and its best is ≈ 5,300 / 3,400 / 1,200 tok/h.
  - Drafting stops at acceptance. Measured data end at k = 4, where each extra drafted token adds ~0.5 tokens per step.
- **MoE Inkling 2 Mini, native NVFP4/MXFP8.** Here a step's bytes grow with the number of positions, because each position routes to its own experts.
  - **Single stream: ~2.1 tok/s (≈ 7,500 tok/h) at 2.2 GB/s, 4.7 tok/s at 5 GB/s**, with a 6.3 GB expert cache and uniform routing.
  - **Batching alone gains only ~1.2×** (≈ 9,000 tok/h at 8K). Each extra sequence brings mostly new experts, and its KV evicts expert cache.
  - **The gain returns once ~45 positions saturate the union.** At that point every step reads all 15.67 GB of experts, and the model behaves like a dense 15.7 GB model. Drafting is then I/O-free. Best at 8K: **≈ 21,000 tok/h on code** (N = 11, k = 4), 14,000 on reasoning and 9,000 on prose, where drafting doesn't pay. ggml-org's BF16 upcast reads 3.1× the expert bytes: it gets ~⅓ of these batched numbers and ~⅕ of the single stream (0.43 tok/s).
- **The macOS page cache makes mmap streaming hostile to Responsive.**
  - XNU keeps a file-cache floor of ~37 % of available memory. Above it, the pageout scan still takes **two anonymous pages for every file page** `[INFERENCE from the kernel source]`. That means other apps' memory gets compressed and swapped.
  - Page-fault read-ahead on an SSD is 16–128 KiB per fault.
  - On this Mac, mmap paging of ~1–1.5 GiB per token already measured **124 GiB swapped out, a 2.4 s process-spawn stall and ~0.3–0.55 GiB/s effective reads**.
  - A Source-precision Fit that streams should read with `F_NOCACHE` into a fixed ring. It should wire a fixed resident set and put the weights on a volume other than the boot (swap) disk.

## Device and model inputs

| Input | Value | Source |
|---|---|---|
| Memory bandwidth | 120 GB/s peak; **84 GB/s** planning (70 %) | [Apple tech specs][apple-spec]; planning figure from [kimi-k3 research][kimi] |
| SSD | **2.2 GB/s** internal planning, **5 GB/s** faster volume | [kimi-k3 research][kimi] (AmorphousDiskMark QD1 2,192 MB/s on the 256 GB M4 mini) |
| Resident budget R | **10 GB** for pinned weights + all per-sequence state | map's "~10 GB resident"; WindowServer froze at a 14 GB wired limit ([baseline][base]) |
| M4 GPU compute | **3.1 TFLOPS effective**: LLaMA-7B F16 prefill 230 tok/s on a 10-core M4 | [llama.cpp #4167][d4167] (6.74 B params × 2 FLOP × 230) |
| Qwen step bytes | **51.25 GB**: 64 layers 48.71 + lm_head 2.54 | [kimi-k3 research][kimi] |
| Qwen per-sequence state | 149.6 MiB Gated DeltaNet state + **64 KiB/token** KV (16 full-attention layers) | [runtime-support][rs]; [llama.cpp #31669][pr31669] |
| Qwen draft | MTP layer + lm_head of `mtp-Qwen3.8-27B-Q8_0.gguf`, **1.8 GB** resident `[INFERENCE from the 3.16 GB file, whose token embedding is read by row]` | [source-precision-runtimes][spr] |
| Inkling experts | 15.67 GB native: 4.48 MB/expert in layers 0–19, 8.21 MB in 20–23; 128 per layer, top-8; layers 6–17 run 3 passes | [kimi-k3 research][kimi]; [config.json][ink-cfg] |
| Inkling resident trunk | 3.22 GB (trunk 1.37 + embed/unembed 1.85) | [kimi-k3 research][kimi] |
| Inkling per-sequence state | 32 KiB/token global KV + 240 MiB sliding (llama.cpp cells) + 3.7 MiB conv | [runtime-support][rs] |
| Inkling draft | DSpark Q8_0, **0.37 GB** | [llama.cpp #32773][pr32773] |
| Inkling compute | **~6 ms per position** `[ESTIMATE]` from measured 104–188 tok/s prefill on this Mac | [inkling-nvfp4][inv], [mlx-8k][mlx8k] |

## The cost model

One decode step reads the streamed weights once. It then evaluates N × (1 + k) positions and emits N × E(k) tokens. E(k) is the mean tokens per step (1 + accepted drafts).

- **Step time** = max(SSD bytes ÷ SSD speed, DRAM bytes ÷ 84 GB/s + positions × compute per position). This assumes reads overlap compute, as a ring-buffer streaming engine does. Without overlap, the Q3 totals drop by 1–9 % at 2.2 GB/s and by up to 18 % at 5 GB/s (2K, k = 4), because compute is small next to I/O.
- **Memory**: pinned weights + draft weights + N × per-sequence state ≤ R. For Qwen, every GB taken by sequences is a GB more to stream. For Inkling, it is a GB less expert cache.
- **Inkling union**: under uniform, independent routing `[ASSUMPTION]`, a step needs a fraction 1 − (1 − 8/128)^(positions × passes) of each layer's experts. Cache hit rate = cached share of the experts.
- **Single-stream tok/s** = E(k) ÷ step time. **Batch throughput** = 3600 × N × E(k) ÷ step time (tokens per hour).

When sequences fill the budget, the step reads every streamed byte. Batch throughput then reduces to a planning formula:

$$\text{tok/h} \approx 3600 \cdot \frac{S}{B} \cdot N_{\max}(k, c) \cdot E(k)$$

Here S is the SSD speed, and B is 51.25 GB for Qwen, or the 15.67 GB expert pool for Inkling once ≥ ~45 positions saturate it. N_max = (R − draft) ÷ per-sequence bytes at context c.

## 1. Dense Qwen3.8-27B BF16: batching and drafting

### Q1: batching, 8K context, no drafting

| N | pinned GB | step s @2.2 | tok/s per stream @2.2 | tok/h @2.2 | step s @5 | tok/s per stream @5 | tok/h @5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.3 | 19.1 | 0.052 | 189 | 8.4 | 0.119 | 429 |
| 2 | 8.6 | 19.4 | 0.052 | 372 | 8.5 | 0.117 | 844 |
| 4 | 7.2 | 20.0 | 0.050 | 720 | 8.8 | 0.114 | 1,635 |
| 8 | 4.5 | 21.3 | 0.047 | 1,354 | 9.4 | 0.107 | 3,077 |
| 14 | 0.3 | 23.2 | 0.043 | 2,176 | 10.2 | 0.098 | 4,945 |

Each sequence slows every stream by only ~1.5 %. It moves 0.69 GB from pinned weights to the streamed set, which costs 0.3 s per step at 2.2 GB/s. Throughput is nearly linear in N, and N = 14 is where the 10 GB budget runs out.

### Q2: drafting, one stream, 8K context

| k | tokens/step | per-seq GB | tok/s @2.2 | tok/h @2.2 | tok/s @5 | tok/h @5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.69 | 0.052 | 189 | 0.119 | 429 |
| 1 | 1.87 | 0.73 | 0.094 | 338 | 0.214 | 769 |
| 2 | 2.55 | 0.74 | 0.128 | 461 | 0.291 | 1,048 |
| 3 | 3.14 | 0.74 | 0.158 | 568 | 0.359 | 1,291 |
| 4 | 3.64 | 0.75 | 0.183 | 658 | 0.416 | 1,496 |

The k + 1 verify positions cost 17–80 ms of GPU time against a 19 s read. Drafting therefore converts acceptance into speed one-for-one. Resident models don't get that: vLLM's 2.65× at one request falls to +19 % at 48 requests in flight, once the GPU is busy ([tokenwatt][tw]).

### Q3: sequences at the memory limit, by context and draft length

| context | k | N max (replay) | tok/h @2.2 | tok/h @5 | tok/s per stream @2.2 | N max (snapshots) | tok/h @2.2 (snapshots) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0 | 34 | 5,265 | 11,966 | 0.043 | 34 | 5,265 |
| 2K | 1 | 26 | 7,547 | 17,153 | 0.081 | 17 | 4,955 |
| 2K | 2 | 26 | 10,273 | 23,348 | 0.110 | 13 | 5,146 |
| 2K | 3 | 26 | 12,627 | 28,698 | 0.135 | 10 | 4,900 |
| 2K | 4 | 25 | 14,137 | 32,129 | 0.157 | 8 | 4,570 |
| 8K | 0 | 14 | 2,176 | 4,945 | 0.043 | 14 | 2,176 |
| 8K | 1 | 11 | 3,186 | 7,242 | 0.080 | 9 | 2,613 |
| 8K | 2 | 11 | 4,342 | 9,868 | 0.110 | 7 | 2,809 |
| 8K | 3 | 11 | 5,342 | 12,141 | 0.135 | 6 | 2,970 |
| 8K | 4 | 11 | 6,188 | 14,064 | 0.156 | 6 | 3,380 |
| 32K | 0 | 4 | 628 | 1,427 | 0.044 | 4 | 628 |
| 32K | 1 | 3 | 882 | 2,004 | 0.082 | 3 | 874 |
| 32K | 2 | 3 | 1,202 | 2,732 | 0.111 | 2 | 832 |
| 32K | 3 | 3 | 1,480 | 3,364 | 0.137 | 2 | 1,018 |
| 32K | 4 | 3 | 1,715 | 3,899 | 0.159 | 2 | 1,172 |

The N drop between k = 0 and k = 1 is the 1.8 GB draft taking budget.

### Where the Qwen gains stop

- **Per-sequence memory, first.** The 48 linear-attention layers keep a constant state of 149.6 MiB per sequence. The 16 full-attention layers add 64 KiB per token. At 2K the state is half the cost; from 8K on the KV dominates. Quantizing the KV would roughly double N, but a Source-precision Fit quantizes nothing at run time (map standing choice).
- **Rollback state, in older builds.** At llama.cpp `a0677c3` a drafting context allocates `mem_size × (1 + n_rs_seq)` recurrent rows, with `n_rs_seq = draft.n_max` ([llama-memory-recurrent.cpp L102][l-rec102], [common.h L417-L428][l-nrs], [common.cpp L1748][l-nrs-set]). For Qwen3.8-27B that is 598.5 MiB per sequence at n_rs_seq 3. [PR #31669][pr31669] ("GDN: use replay instead of rollback", merged 2026-11-23 as `d35dc4d`) keeps one state plus the last ubatch's inputs: 163.2 MiB at 3, 176.8 MiB at 7. It is on for `qwen35` ([llama-context.cpp L193-L199 @ 5220a2d][l-replay]). SGLang reserves (S + D) state slots per request, where D is the number of draft tokens ([SGLang Qwen3.8-27B cookbook][sgl]). **A Fit plan must check which rollback its runtime build uses**, because it changes N by 2–3× at short context.
- **Acceptance.** Measured mean tokens per step for Qwen3.8-27B's MTP head:
  - vLLM, FP8 target: 1.87, 2.55, 3.14, 3.64 at k = 1–4. 84 %, 71 % and 59 % of first, second and third drafts were accepted. Prose gained 1.88×, code 2.65×, JSON 2.96× at k = 3 ([tokenwatt][tw]).
  - llama.cpp on an M4 Pro at n_max 2: 67–81 % acceptance, mean accepted length 2.34–2.61 ([stridenote][sn]).

  The Qwen card says only "MTP: trained with multiple steps" ([model card L52][qwen-card]). No acceptance is published for a BF16 target `[INFERENCE: should be similar]`. Past k = 4 there are no measurements. If the per-position decline seen at k = 1–4 continues (×0.84 per position), E would be ≈ 5 at k = 8 `[INFERENCE]`.
- **Compute: not on the GPU.** At 16.5 ms per position (51 GFLOP at 3.1 TFLOPS; consistent with the 58 tok/s Qwen prefill measured on this Mac, [baseline][base]), a 23.3 s step is compute-bound only above ~1,400 positions. The 16 GB budget holds at most ~125 (N = 25, k = 4). `[INFERENCE]` **This holds only if the streamed layers are computed on the GPU.** llama.cpp computes `CPU_Mapped` layers on the CPU, and its decode faults their pages in without overlap. At an assumed 0.1–0.3 TFLOPS of CPU BF16 `[ASSUMPTION]`, compute equals the read at 45–140 positions, which is inside the range above.
- **Prefill** follows the same rule with ubatch size u: one read per ubatch. At u = 512, an 8K prompt takes 16 reads ≈ 6 min at 2.2 GB/s. Above u ≈ 1,400 it becomes compute-bound at ~60 tok/s. A larger u costs compute-buffer memory (Inkling: 1,047 MiB at u 512, 523 at 256, measured in [inkling-nvfp4][inv]), and that memory competes with N.

## 2. MoE Inkling 2 Mini: the union of experts

### I1: batching, 8K context, no drafting, native experts

| N | expert cache GB | hit | union of experts | SSD GB/step | tok/s per stream @2.2 | tok/h @2.2 | tok/h @5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.3 | 40% | 11% | 1.06 | 2.08 | 7,487 | 17,015 |
| 2 | 5.7 | 37% | 21% | 2.08 | 1.06 | 7,628 | 17,337 |
| 4 | 4.7 | 30% | 36% | 4.00 | 0.55 | 7,914 | 17,986 |
| 8 | 2.6 | 17% | 57% | 7.48 | 0.29 | 8,466 | 19,241 |
| 12 | 0.5 | 3% | 70% | 10.60 | 0.21 | 8,964 | 20,373 |

### I2: batching × drafting, 8K context, code acceptance, tok/h at 2.2 GB/s

| k \ N | 1 | 2 | 4 | 8 | 11 |
|---|---:|---:|---:|---:|---:|
| 0 | 7,487 | 7,628 | 7,914 | 8,466 | 8,846 |
| 2 | 7,220 | 8,275 | 10,283 | 13,671 | 15,717 |
| 3 | 7,048 | 8,480 | 11,189 | 15,689 | 18,402 |
| 4 | 7,047 | 8,845 | 12,222 | 17,789 | 21,150 |

### I3: best (N, k) by context and content

E(k) = 1 + acceptance × k, from llama.cpp's DSpark on a 16 GB card:

- code: 81 / 73 / 68.5 % acceptance at k = 2 / 3 / 4;
- reasoning: 51 / 40 / 36.5 %;
- prose: 16.5 / 12.5 / 10 %.

Source: [PR #32773][pr32773]. Unsloth's guide says the third draft token "is rarely accepted" on its build ([unsloth][unsloth]).

| experts | context | content | single stream tok/h @2.2 | best N, k | tok/h @2.2 | tok/h @5 (same N, k) |
|---|---:|---|---:|---|---:|---:|
| native | 2K | code | 7,650 | 18, 4 | 34,123 | 77,553 |
| native | 2K | reasoning | 7,650 | 18, 4 | 22,445 | 51,011 |
| native | 2K | prose | 7,650 | 18, 4 | 12,773 | 29,031 |
| native | 8K | code | 7,487 | 11, 4 | 21,150 | 48,068 |
| native | 8K | reasoning | 7,487 | 11, 4 | 13,912 | 31,617 |
| native | 8K | prose | 7,487 | 12, 0 | 8,964 | 20,373 |
| native | 32K | code | 6,897 | 4, 4 | 9,310 | 21,160 |
| native | 32K | reasoning | 6,897 | 1, 0 | 6,897 | 15,675 |
| native | 32K | prose | 6,897 | 1, 0 | 6,897 | 15,675 |
| bf16 | 2K | code | 1,564 | 18, 4 | 10,919 | 24,817 |
| bf16 | 2K | reasoning | 1,564 | 18, 4 | 7,182 | 16,323 |
| bf16 | 2K | prose | 1,564 | 18, 4 | 4,087 | 9,290 |
| bf16 | 8K | code | 1,556 | 11, 4 | 6,759 | 15,362 |
| bf16 | 8K | reasoning | 1,556 | 11, 4 | 4,446 | 10,104 |
| bf16 | 8K | prose | 1,556 | 12, 0 | 2,723 | 6,190 |
| bf16 | 32K | code | 1,528 | 4, 4 | 2,869 | 6,521 |
| bf16 | 32K | reasoning | 1,528 | 4, 4 | 1,887 | 4,289 |
| bf16 | 32K | prose | 1,528 | 5, 0 | 1,812 | 4,118 |

"native" means the NVFP4/MXFP8 experts as shipped: 15.67 GB, the custom-GGUF or MLX path. "bf16" means ggml-org's 52.23 GB upcast, the ready llama.cpp path ([source-precision-runtimes][spr]). That upcast stores 15.9 MB per expert, so its pool is 48.9 GB.

### How MoE differs from dense

- **The step's bytes grow with positions until the union saturates.**
  - Dense: every position needs the same 51 GB, so position 2 is free.
  - Inkling: a position needs 8 of 128 experts per layer, three times over in the 12 looped layers. Under uniform routing one position touches 11 % of the expert pool (1.76 GB), 8 positions 57 %, ~45 positions ≥ 95 %. Until then, extra sequences add reads almost in proportion.
- **Sequences and the expert cache share the same RAM.** Each 8K sequence (0.52 GB) evicts 0.52 GB of expert cache, about 3 points of hit rate. So batching alone gains only 1.2× at 8K (I1).
- **Past saturation, MoE behaves like dense.** The step reads the whole pool (15.67 GB native, 7.1 s at 2.2 GB/s), and more positions are free again. That is when drafting multiplies (I2). The draft is a dense 3-layer DSpark head that stays resident, so drafted tokens read no experts `[INFERENCE from the mtp_config: 3 dense layers, intermediate 9216]`. This is unlike kimi-k3-in-c's int8 self-draft, which measured 53 s/token against 19.8 s for plain decode because "every drafted token still reads ~25.8 GB of experts" ([int8-draft-container.md L56-L84][k-int8]).
- **A single stream should not draft on a streamed MoE.** Verifying k + 1 positions widens the union just as k + 1 sequences would. At N = 1, k = 4 reads ~4× the experts for 3.7 tokens, which is slightly slower than no drafting (I2, first column).
- **Skewed routing helps the small-N regime most.** kimi-k3-in-c measured chunk-union prefill cutting expert bytes 1.96× on a 40-token prompt ([speed-2026-08.md L40-L43][k-union]). Uniform routing over K3's 896 experts, top-16, predicts 1.39× `[INFERENCE: computed]`. So tokens of one prompt share more experts than uniform routing predicts. Inkling's routing skew is unmeasured. Skew would raise cache hit rates and slow the union's growth at small N. Past saturation, it changes nothing.
- **Compute is far away.** At ~6 ms per position, a saturated 7.1 s step is compute-bound only above ~1,200 positions (~520 at 5 GB/s). The 16 GB budget holds ~55–90.

## 3. The macOS page cache under a file larger than memory, and what it means for Responsive

**Reclaim policy.** Reading XNU `xnu-12377.1.9`, the macOS 26 kernel source:

- **File-cache floor.** On macOS, `vm_page_filecache_min = AVAILABLE_NON_COMPRESSED_MEMORY × 10 / 27`, about 37 % of active + inactive + free + speculative pages ([vm_pageout.c L2233-L2242][x-fcmin], [L5280][x-div], [vm_page.h L1528][x-avail]). While file-backed pages are below that floor, the pageout scan takes anonymous pages: other apps' memory goes to the compressor and swap ([L2621-L2645][x-choose]).
- **Above the floor, it is still two to one.** It takes anonymous pages whenever they exceed a small minimum, at most `ANONS_GRABBED_LIMIT` = 2 in a row before one file page ([L2646][x-choose], [L2679-L2728][x-want], [L1611][x-limit]). `[INFERENCE from the code; not traced on a running system]`
- **Read-ahead goes first.** Only read-ahead pages that were never touched and have aged are reclaimed before both ([L2551-L2565][x-spec]).

Streaming 51 GB per step through the cache therefore keeps pushing other apps' memory out while the weights' own pages cycle. The kernel's LRU gets ~0 % reuse on a cyclic scan larger than memory ([kimi-k3 research][kimi], row 6).

**Read-ahead.**

- **Page faults.** A fault on an SSD-backed mapping reads a cluster of 16–128 KiB: the 1 MiB `preheat_max_bytes` is divided by 8 on SSD. An explicit cluster size, as from `MADV_WILLNEED` or wiring, allows up to 1 MiB ([vm_object.c L6758-L6763, L6825-L6867][x-preheat], [memory_object_types.h L475][x-upl]).
- **Buffered `read()`.** Read-ahead goes up to 192 MiB, in I/Os of at most 512 KiB on SSD ([vfs_cluster.c L289-L299][x-cluster]).
- **Small reads are slow.** On the M4 mini's SSD, QD1 random 4K reads run at 64 MB/s ([kimi-k3 research][kimi]). LLM in a Flash needed ≥ 32 KiB reads across 32 threads to approach useful throughput: 1.25 GiB/s scattered vs 6.1 GiB/s sequential on an M1 Max ([arXiv:2312.11514 §2.2, Table 2, §4.1][flash]).
- **llama.cpp issues no read-ahead during decode.** It calls `posix_madvise(WILLNEED)` only at load ([llama-mmap.cpp L726-L746][l-mmap]). Its only runtime prefetch is for rows of lazily loaded tensors ([llama-impl.cpp L22-L40][l-prefetch]). So during decode, `CPU_Mapped` weights fault in cluster by cluster `[INFERENCE: no other caller found at a0677c3]`.

**What was measured on this Mac.**

- **Qwen UD-Q3_K_XL, 13.15 GB (baseline).** `--fit` left 1,577 MiB `CPU_Mapped`. The largest buffer, 648 MiB, is most likely the token embedding, which is read by row `[INFERENCE]`. Decode ran at **0.36 tok/s**, with **194.6 GiB of page-ins and 124.1 GiB of swap-outs**, pressure critical for 63 s, a 1.3 s timer stall and a **2.4 s process spawn** ([baseline][base]; `qwen-q3kxl-llamacpp-fit.json`). Re-reading ~0.9–1.5 GiB per token at 0.36 tok/s means ~0.3–0.55 GiB/s effective, a quarter of the 2.2 GB/s planning figure or less `[INFERENCE]`.
- **Inkling UD-IQ3_XXS with `--fit` spill.** It stayed Responsive at 40 GiB of page-ins in 254 s and 2.7 GiB of swap-outs. The spilled share was small: 1.42 GiB ([inkling-nvfp4][inv]).
- **kimi-k3-in-c**, on Linux: pinning 51 GB on a capped machine ran slower than pinning nothing ([speed-2026-08.md L38-L39][k-union]).

**Apple's own guidance.** Don't map a file that is "several hundred megabytes or more"; "for large sequential read operations, disable disk caching and read the file into a small memory buffer instead" ([Mapping Files Into Memory][apple-mmap]). `F_NOCACHE` "turns data caching off" for a descriptor ([fcntl.2 L189-L192][x-fcntl]).

**What this means for Responsive** `[INFERENCE]`:

- **mmap paging (or any cached read) of a Source-precision Fit can't be both fast and Responsive on 16 GB.** XNU defends its file cache by evicting desktop memory. Fault-sized reads also run at a fraction of SSD speed. The runtimes' own mmap path is the one model-fitter would try first (map standing choice). It can pass only when the spilled share is small, as with Inkling's 1.4 GiB.
- **The streaming read path should be:**
  - `F_NOCACHE` `pread`s of ≥ 1 MiB chunks, issued in parallel, into a fixed ring;
  - the resident set wired (Metal-resident), not left to the page cache;
  - in short, kimi-k3-in-c's design ([kimi-k3 research][kimi], rows 2 and 6).

  Then the streamed bytes add no page-cache pressure. Memory use is fixed at R + ring, so model-fitter can admit it up front.
- **N trades directly against Responsive.** Every sequence adds wired memory. The budget R has to come from the Responsive test, not from maximizing N.
- **Put the weights on a volume other than the boot disk.** Overnight, the reader keeps an SSD saturated (2.2 GB/s ≈ 7.9 TB/hour). On the boot volume it competes with swap-ins and app launches. An external 5 GB/s volume is faster and also leaves the boot SSD to the desktop.
- **Heat.** LLM in a Flash saw no thermal slowdown over 1,000 tokens ([§5.1][flash]). Hours of sustained reads are unmeasured.

## Published measurements used

| Claim | Measured where | Source |
|---|---|---|
| Speculative decoding composes with flash loading: λ = 4 gave 1.4× vs 1.58× for plain speculative decoding; the paper studies single-sequence inference only | OPT-6.7B, RTX 4090 | [LLM in a Flash §5.2, §8][flash] |
| Resident batching scales sublinearly: S_TG 41.6 → 465 tok/s for B = 1 → 32 (11.2×) | LLaMA-7B F16, llama-batched-bench sample (hardware not stated) | [batched-bench README L38-L51][l-bb] |
| Drafting gain shrinks under load when resident: 2.65× → +101 % / +49 % / +19 % at 4 / 16 / 48 requests | Qwen3.8-27B FP8 + MTP, vLLM 0.33, H100 NVL | [tokenwatt][tw] |
| Qwen3.8-27B MTP acceptance in llama.cpp: 67–81 %, mean 2.34–2.61 at n_max 2 | M4 Pro 48 GB | [stridenote][sn] |
| Inkling DSpark acceptance: code 68–81 %, reasoning 24–55 %, prose 8–21 %; n_max 7 collapsed on Vulkan because the graph is rebuilt when the batch size changes | Inkling GGUF + DSpark Q8_0, RX 9070 XT 16 GB | [llama.cpp #32773][pr32773] |
| Rollback memory: snapshots 598.5 MiB vs replay 163.2 MiB per sequence at n_rs_seq 3; 32 sequences at 7: 37.4 vs 5.5 GiB | Qwen3.8-27B Q8_0, RTX PRO 6000 | [llama.cpp #31669][pr31669] |
| MoE union: chunk-union prefill read 1.96× fewer expert bytes | Kimi K3, 40-token prefill | [kimi-k3-in-c speed-2026-08.md L40-L43][k-union] |
| A self-draft that streams experts is slower than plain decode (53 vs 19.8 s/token); cache-only routing dropped acceptance to 12.5 % | Kimi K3, 64 GB cap | [int8-draft-container.md L56-L84][k-int8] |

## What this means for the Fit plan

`[INFERENCE]` For the estimating ticket ("How does model-fitter estimate single-stream speed and Batch throughput before converting, and how does it choose the number of parallel sequences?"):

- **Choose N by memory, not compute.** Divide the Responsive budget, minus draft weights, by per-sequence state at the batch's longest context: recurrent state + KV per token × context. On a dense model, use the largest N that fits. On an MoE, the same holds once N × (1 + k) ≥ ~45; below that, a single stream with a large expert cache is about as good.
- **Choose k by the batch's content, and draft on streamed models.** In the streaming regime drafting costs only memory: ~3.4 MiB per position with replay, a full state copy per position with snapshots. Its gain is the acceptance itself. Code- or JSON-like batches favor k = 3–4. For prose on Inkling, drafting doesn't pay. Acceptance should come from a short measured run on the user's own prompts, because it ranges from 10 % to 80 % with content.
- **Check the rollback mode in the runtime build before planning k > 0 on Qwen:** llama.cpp ≥ `d35dc4d` replays; older builds snapshot.
- **Plan with the read path's real speed.** Planning figures assume large uncached reads. A runtime that relies on mmap faults measured ~0.3–0.55 GiB/s here, and it also fails Responsive.

## Not answered here

- Inkling's real routing skew, and how much batching gains below union saturation. That needs a recorded trace (kimi-style replay).
- Acceptance with a BF16 target and a Q8_0 or Q4_0 draft on this Mac, and E(k) past k = 4.
- Whether llama.cpp's mmap path can be made Responsive for a 52–54 GB file at all. This belongs to "Which runtime and read path run each reference model's Source-precision Fit, and is a kimi-k3-in-c-style streaming engine needed?"; the model-specific runs belong to the two "How fast is … at Source precision / at BF16 … single-stream and batched" tickets.

## Sources and versions

- llama.cpp `a0677c3` (2026-11-23) and `5220a2d` (2026-11-25); kimi-k3-in-c `b26ee82`; XNU `xnu-12377.1.9` (`f6217f8`).
- Earlier research in this repo, reused rather than redone:
  - [kimi-k3 techniques][kimi]
  - [runtime support][rs]
  - [baseline benchmarks][base]
  - [Inkling NVFP4 on llama.cpp][inv]
  - [MLX Inkling 8K][mlx8k]
  - [Source-precision runtimes][spr]

[kimi]: https://github.com/ailabhq/model-fitter/blob/research/kimi-k3-techniques/research/kimi-k3-techniques.md
[rs]: https://github.com/ailabhq/model-fitter/blob/research/runtime-support/research/runtime-support.md
[base]: https://github.com/ailabhq/model-fitter/blob/research/baseline-benchmarks/research/baseline-benchmarks.md
[inv]: https://github.com/ailabhq/model-fitter/blob/research/inkling-nvfp4/research/inkling-nvfp4.md
[mlx8k]: https://github.com/ailabhq/model-fitter/blob/research/mlx-inkling-8k/research/mlx-8k.md
[spr]: https://github.com/ailabhq/model-fitter/blob/research/source-precision-runtimes/research/source-precision-runtimes.md
[apple-spec]: https://support.apple.com/en-us/121555
[apple-mmap]: https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemAdvancedPT/MappingFilesIntoMemory/MappingFilesIntoMemory.html
[flash]: https://arxiv.org/abs/2312.11514
[d4167]: https://github.com/ggml-org/llama.cpp/discussions/4167
[tw]: https://www.tokenwatt.io/blog/qwen38-27b-mtp-speculative-decoding.html
[sn]: https://stridenote.net/speculative-decoding-vs-mtp/
[sgl]: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
[unsloth]: https://unsloth.ai/docs/models/inkling-2-mini
[qwen-card]: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md#L52
[ink-cfg]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/config.json
[pr31669]: https://github.com/ggml-org/llama.cpp/pull/31669
[pr32773]: https://github.com/ggml-org/llama.cpp/pull/32773
[l-bb]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/tools/batched-bench/README.md#L38-L51
[l-rec102]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-memory-recurrent.cpp#L102
[l-nrs]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.h#L417-L428
[l-nrs-set]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.cpp#L1748
[l-replay]: https://github.com/ggml-org/llama.cpp/blob/5220a2d4040aa75f52678ebf52d77a0fba85de9f/src/llama-context.cpp#L193-L199
[l-mmap]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-mmap.cpp#L726-L746
[l-prefetch]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-impl.cpp#L22-L40
[k-union]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/data/speed-2026-08.md#L36-L43
[k-int8]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/notes/int8-draft-container.md#L56-L84
[x-fcmin]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c#L2211-L2247
[x-div]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c#L5276-L5282
[x-avail]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_page.h#L1528
[x-choose]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c#L2621-L2646
[x-want]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c#L2679-L2728
[x-limit]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c#L1611
[x-spec]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c#L2551-L2565
[x-preheat]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_object.c#L6758-L6867
[x-upl]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/mach/memory_object_types.h#L475
[x-cluster]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vfs/vfs_cluster.c#L289-L299
[x-fcntl]: https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/man/man2/fcntl.2#L189-L192
