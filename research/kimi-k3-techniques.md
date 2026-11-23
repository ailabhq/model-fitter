# Which kimi-k3-in-c techniques carry over, and which do existing runtimes already provide?

Ticket: [#4](https://github.com/ailabhq/model-fitter/issues/4). Researched 2026-11-23.

## Answer

Most of kimi-k3-in-c's *mechanisms* are general, but its *tuning conclusions* hold only for Kimi K3. The mechanisms are a routed-expert/trunk split, explicit expert streaming with page-cache bypass, a slot cache for experts, a pinned set plus ring buffer for the trunk, packed-FP4 matmul without dequantising, memory admission and presets. The tuning conclusions are "fill the trunk before the expert cache" and "the expert cache does nothing below ~36 GB". They depend on Kimi's 109 GB bf16 trunk and its deliberately flat router.

On the reference Mac mini, llama.cpp already covers part of this:

- packed MXFP4 on Metal;
- mmap-backed weights, so experts page in from SSD on demand;
- expert placement with `-ot`/`--cpu-moe`;
- an experimental frequency-aware MoE cache;
- `--fit` memory fitting;
- speculative decoding, including MTP.

MLX has MXFP4/MXFP8/NVFP4 kernels and gathered expert matmuls, but it has no SSD streaming. It reads every weight into memory.

Neither runtime has Kimi's explicit expert reads with `F_NOCACHE`, its pinned-set trunk ring, or its replay of a recorded expert trace to size the cache.

Rough estimates, explained below:

- **Inkling 2 Mini** (5.4B active MoE) can plausibly be Usable with experts streamed, but only if roughly ≥ 80 % of expert reads hit RAM.
- **Qwen3.8-27B** (dense) cannot be Usable while any meaningful share of its weights streams from this SSD. It must be quantised small enough to be fully resident.

## Sources and versions

Every code link is pinned to the commit read on 2026-11-23:

- kimi-k3-in-c @ [`b26ee82`][K] (2026-11-16, v2.0.0 + unreleased)
- llama.cpp @ [`a0677c3`][L] (2026-11-23)
- MLX @ [`b835fb3`][M] (2026-11-23)
- mlx-lm @ [`1bf8da0`][ML] (2026-11-21)

Model shapes and byte counts come from each model's `config.json` and from the safetensors headers. The headers were read with HTTP range requests; no weights were downloaded:

- [Inkling-2-Mini config](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/config.json), [index](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/model.safetensors.index.json)
- [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json), [index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json)

[K]: https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/b26ee8231bfa4390545ef2aec3576fd834b92ba7
[L]: https://github.com/ggml-org/llama.cpp/tree/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830
[M]: https://github.com/ml-explore/mlx/tree/b835fb313e14d08de17d43cd59ce2dbfd3422569
[ML]: https://github.com/ml-explore/mlx-lm/tree/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0

## What kimi-k3-in-c does, in one paragraph

Kimi K3 is 2.78T parameters, ~104B active per token, 93 layers, with 896 routed experts per layer of which 16 are picked ([ARCHITECTURE.md L8-20][k-arch-model]). The engine sorts every tensor into one of four groups ([ARCHITECTURE.md L30-41][k-arch-mem]):

| Part | Size | Residency |
|---|---:|---|
| Routed experts | 1.45 TB of MXFP4 | Never resident, streamed |
| Dense trunk | 108.81 GB of bf16 | "the dial": resident or streamed |
| Embed + lm_head | 4.70 GB | Resident |
| Recurrent state | 626 MB | Resident |

It runs from 8 GB to 224 GB with byte-identical output ([PERFORMANCE.md L16-37][k-perf-ladder]). At every budget it spends 41–61 % of wall time waiting on disk ([README L3415-3428][k-readme-storage]). Its best sustained speed is ~10.7–11.8 s/token on a 124-core EPYC with 3.2 GB/s NVMe ([PERFORMANCE.md L191-201][k-perf-longer]). It is a "make it run at all" engine, not a "make it Usable" one.

## Technique table

Columns:

- **Scope**: whether the technique is tied to Kimi K3's architecture or numbers, or is a general mechanism.
- **MoE / dense**: which kind of Source model it helps.
- **llama.cpp / MLX**: the closest existing equivalent, or "none".

| # | Technique (kimi-k3-in-c source) | Scope | MoE / dense | llama.cpp equivalent | MLX / mlx-lm equivalent |
|---|---|---|---|---|---|
| 1 | **Classify tensors as routed expert vs trunk; routed experts never resident.** Shared experts count as trunk ([README L855-874][k-readme-classify], [ARCHITECTURE.md L30-41][k-arch-mem]) | General (any MoE); the name pattern is per checkpoint | MoE only | Yes. `-ot/--override-tensor`, `--cpu-moe`, `--n-cpu-moe` place expert tensors in host buffers ([arg.cpp L2728-2750][l-arg-ot]). With the default mmap load mode those host buffers are file-backed, so the OS pages experts in from SSD on demand ([arg.cpp L2665-2683][l-arg-load]). [INFERENCE: the mmap backing of host-placed tensors comes from the load mode, not from a documented expert-specific guarantee] | None. MLX reads every tensor into memory with `pread` through `ParallelFileReader` and does not mmap ([load.cpp L513-550][m-load]). Weights must fit in RAM. |
| 2 | **Explicit expert streaming**: one coalesced `pread` per expert, page cache bypassed (`O_DIRECT`; on macOS `fcntl(F_NOCACHE)`), reads sorted by disk offset and issued in parallel up to QD16, split into 1 MiB chunks ([README L2678-2757][k-readme-cache], [k3_portable_io.h L6-9, L98-104][k-pio], [CHANGELOG L227-231][k-cl-110]) | General | MoE (any sparsely read weights) | Partial. Implicit through mmap page faults. `posix_madvise` sets `WILLNEED`/`RANDOM` hints on ranges ([llama-mmap.cpp L704-746][l-mmap]). `--load-mode dio` exists, but only for loading ([arg.cpp L2665-2683][l-arg-load]). There are no explicit per-expert reads and no page-cache bypass during decode. | None |
| 3 | **Expert slot cache**: LRU over whole-expert slots with EMPTY/INFLIGHT/key states, batch reserve → parallel read → publish, prefetch ([README L2648-2724][k-readme-cache]). The policy is plain LRU. | Mechanism is general. Its value depends on the model: K3's Quantile-Balanced router gives 0 % retention below ~36 GB ([PERFORMANCE.md L39-69][k-perf-two-caches], [TUNING.md L69-81][k-tune-weak]). | MoE | Partial and experimental. `--moe-cache-mib` + `--moe-cache-cpu-share` keep "recently and repeatedly used" experts of host-resident layers in a *device* buffer ([arg.cpp L2751-2770][l-arg-moecache], [llama.h L409-410][l-h-moecache], [llama-moe-cache.h L33-39][l-moecache-h]). The policy is a ClassLRU: recency plus frequency, decayed with a half-life of 128 tokens ([llama-moe-cache.cpp L21-28][l-moecache-policy]). It is a host→GPU cache; the SSD→RAM layer is still the OS page cache. Only layers whose experts sit in host (CPU) buffers take part ([llama-moe-cache.cpp L313-427][l-moecache-host]), and the Metal shared/mapped buffer types report `is_host = false` ([ggml-metal.cpp L359-363, L515-519][l-metal-ishost]). On a Mac it therefore applies only with `--cpu-moe`/`-ot` placing experts in mmapped CPU buffers. [INFERENCE: on unified memory this copies hot experts a second time, so its value on a 16 GB Mac is unclear; measure it.] | None |
| 4 | **Trace recording + offline replay** of `(layer, expert)` requests under LRU / Belady / pin+LRU at any capacity, to size the cache without re-running the model ([README L2775-2849][k-readme-trace], [PERFORMANCE.md L71-107][k-perf-capacity]) | General tooling | MoE | Partial. The MoE cache logs hits, misses and hit rate ([llama-moe-cache.cpp L870-876][l-moecache-log]). No trace dump, no replay. | None |
| 5 | **Packed trunk**: each layer's non-expert tensors copied into one contiguous, 4096-aligned run, so loading a layer is one read ([ARCHITECTURE.md L130-138][k-arch-trunk], [README L2301-2361][k-readme-pack]) | General | Both (all of a dense model is "trunk") | Format-level. GGUF stores tensors contiguously in one mmapped file. llama.cpp does not stream it layer by layer; it relies on mmap ([llama-mmap.cpp L704-746][l-mmap]). | None (whole-model load) |
| 6 | **Trunk streaming = fixed pinned set (largest layers first) + ring buffer, prefetching layer L+1, parallel 64 MiB chunk reads.** Deliberately *not* LRU, because a cyclic layer scan makes LRU hit exactly 0 % ([README L2363-2423][k-readme-pin], [CHANGELOG L232-242][k-cl-110]) | General | Both. This is the only way a dense model larger than RAM runs at all. | None explicit. mmap + page cache is the implicit equivalent. [INFERENCE: whether macOS's page reclaim behaves like the cyclic-scan LRU pathology Kimi describes is unmeasured; it is the risk to test]. `--load-mode mlock`/`mmap+mlock` pins the whole model or nothing ([arg.cpp L2665-2683][l-arg-load], [llama-mmap.cpp L920-930][l-mlock]). Metal wraps mmapped memory with `newBufferWithBytesNoCopy` ([ggml-metal-device.m L2240-2283][l-metal-map]) and keeps residency sets wired ([ggml-metal-device.m L927-940][l-metal-rset]). | None. mlx-lm wires the model up to `max_recommended_working_set_size` and only *warns* above 90 % of it ([generate.py L249-283][ml-wired], [README L267-289][ml-readme-large]). |
| 7 | **"Fill the trunk before the expert cache"** allocation rule (1.69× at a fixed 128 GB) ([TUNING.md L11-44][k-tune-split], [PERFORMANCE.md L109-144][k-perf-alloc]) | **Kimi-specific in direction.** It holds because the trunk is 4.2× the per-token expert bytes. For Inkling 2 Mini the trunk is 1.37 GB (see below), so the rule inverts: the trunk is trivially resident and all spare RAM should go to experts. The general rule is "spend RAM where the per-token bytes saved are largest". | Both | No equivalent policy; the user chooses placement | No equivalent |
| 8 | **Presets + `--preset auto` + `k3-doctor.sh`**: size the budgets from free RAM, trunk first ([TUNING.md L83-108][k-tune-presets], [README L601-638][k-readme-presets], [CHANGELOG L316-319][k-cl-100]) | The pattern is general. The numbers come from K3's measured ladder. | Both | `--fit on`, `--fit-target MiB` adjust unset arguments to fit device memory ([arg.cpp L2863-2895][l-arg-fit]) | mlx-lm sets the wired limit to the recommended working set ([utils.py L84-93][ml-utils-wired]); no presets |
| 9 | **Memory admission**: sum every allocation up front (trunk, model, cache, state, buffers, KV) and refuse to start above 95 % of available memory. On macOS "available" is free + inactive + purgeable pages from `host_statistics64`. Peak RSS is reported afterwards as the number to quote ([README L2430-2453][k-readme-admit], [k3_sysmem.h L25-27][k-sysmem]). The KV cache is also checked up front ([TUNING.md L112-118][k-tune-incr]). | General | Both | Partial: `--fit` fits rather than refuses ([arg.cpp L2863-2895][l-arg-fit]) | Partial: warning only ([generate.py L262-274][ml-wired]) |
| 10 | **Packed 4-bit matmul that never dequantises** (OCP MXFP4, E2M1 nibbles + E8M0 scale per 32), plus NEON paths ([ARCHITECTURE.md L110-128][k-arch-mxfp4], [CHANGELOG L170-171][k-cl-110a]) | General. The format is an OCP standard; only the kernel is K3's. | Both | MXFP4 yes (`GGML_TYPE_MXFP4` with Metal kernels: [ggml.h L258-259][l-ggml-types], [dequantize.h L348][l-metal-mxfp4]). **NVFP4 and F8_E4M3 matmul are not supported on Metal at this commit** ([ggml-metal-device.m L1525-1535][l-metal-nofp4]). [INFERENCE: such ops fall back to CPU] | MXFP4, MXFP8 and NVFP4 quantisation modes with Metal kernels ([ops.cpp L4856-4919][m-ops-quant], [primitives.h L155][m-prim]); experts use `gather_qmm` ([moe.py L100-112][ml-moe]) |
| 11 | **No trunk quantisation** (bf16 only). Reason: int4 gave ~17 % mean weight error, int8 ~1 %, and K3's experts were QAT'd for MXFP4 while the trunk was not ([README L3469-3524][k-readme-noquant], [ROADMAP L71-77][k-roadmap]) | **Kimi-specific decision.** For model-fitter, trunk/dense quantisation is the main lever (owned by the low-bit quantisation ticket). The lesson that carries over: measure per-tensor error before choosing bits. | Both | Many quant types, mixed per tensor | Affine 2–8 bit, mixed recipes `mixed_2_6`…`mixed_4_6` ([convert.py L20-33][ml-mixed]) |
| 12 | **Ultra-low-memory**: stream embedding rows and lm_head chunks, reuse one recurrent-state slot ([README L430-444][k-readme-memopts], [CHANGELOG L165-169][k-cl-110a]) | General | Both | Partial: `--lazy-mode` reads rows of selected large tensors (e.g. per-layer embeddings) from disk on demand, auto only for tensors > 4 GiB ([arg.cpp L2684-2696][l-arg-load]) | None |
| 13 | **Chunk-union prefill**: each unique routed expert fetched once per prompt chunk instead of once per token, ~½ the expert bytes ([CHANGELOG L320-322][k-cl-100]) | General | MoE | [INFERENCE] Implicit: a batched `ggml_mul_mat_id` touches each selected expert once per ubatch. The MoE cache separately logs "large batches" ([llama-moe-cache.cpp L870-876][l-moecache-log]) | Tokens are sorted by expert when ≥ 64 routing slots (`_gather_sort`, [moe.py L18-24, L203-216][ml-moe-sort]) |
| 14 | **Incremental decode, state save/resume, chat prefix reuse** ([CHANGELOG L323-326, L185-188][k-cl-100]) | General | Both | `--prompt-cache`, `--slot-save-path`, `--cache-reuse`, `--cache-ram` ([arg.cpp L1699, L1851, L3631, L3663][l-arg-cache]) | `--prompt-cache-file`, `--max-kv-size`, `--kv-bits` ([generate.py L186-230][ml-gen-args]) |
| 15 | **Speculative decoding**: n-gram `--spec N` (exact by construction). The int8 self-draft and cache-only-routing draft were built and **measured not to pay** when experts stream, because the draft step still reads ~25.8 GB of experts ([CHANGELOG L327-328][k-cl-100], [int8-draft-container.md L56-84][k-int8]) | General. The negative result transfers to any streamed MoE: a draft that streams the same experts saves nothing. | Both (gain is larger when the model is resident) | n-gram (simple / map-k / map-k4v / mod), draft model, Eagle3, **MTP**, DFlash variants ([common.h L177-188][l-spec]) | `--draft-model`, `--num-draft-tokens` ([generate.py L230][ml-gen-args]) |
| 16 | **Hybrid attention with bounded state** (69 KDA linear-attention layers with constant state + 24 Gated MLA layers with a latent KV cache) ([ARCHITECTURE.md L76-90][k-arch-attn]) | **Architecture-specific** (it is K3's design, not an engine trick). Both reference models have analogues: Qwen3.8-27B has 48 `linear_attention` + 16 `full_attention` layers; Inkling 2 Mini has 20 of 24 layers as 1024-token sliding-window (`local_layer_ids`) (configs above). | n/a | Per-architecture graph code | Per-architecture model code (`mlx_lm/models/qwen3_5` exists) |
| 17 | **2 MB hugepages for O_DIRECT arenas** ([README L2425-2428, L3441-3462][k-readme-storage]) | Linux-specific (a no-op stub on other OSes, [k3_portable_io.h L284-291][k-pio]); the measured +5.4 % is inside the noise | Both | — | — |
| 18 | **Lossless trunk compression** (Huffman on the bf16 exponent byte, 1.45× smaller). Built, then **shelved**: the decoder ran at 0.31 GB/s/core, too slow to feed a ~3 GB/s laptop SSD ([compressed-trunk.md L5-30][k-huf]) | General idea, negative result | Both | None | None |
| 19 | **Threads = physical cores** (OpenMP's SMT default measured ~23 % slower) ([TUNING.md L141-152][k-tune-threads], [CHANGELOG L190-191][k-cl-110a]) | General; Apple Silicon has no SMT, so this is moot there | Both | `-t/--threads` ([arg.cpp L1500][l-arg-threads]) | GPU-scheduled |
| 20 | **Correctness ladder**: op → layer → tiny oracle → full-model logits vs torch; deterministic kernels; no BLAS so results are bit-exact across machines ([ARCHITECTURE.md L140-149][k-arch-correct], [README L694-704][k-readme-noblas]) | General practice, not a speed technique | Both | — | — |
| 21 | **Split storage**: trunk and experts on separate devices, then size the ring just above the overlap threshold ([TUNING.md L46-67][k-tune-split-dev]) | General | MoE | — | — |

Not applicable to the reference device: row 17 (Linux only), row 19 (no SMT on Apple Silicon), row 21 (one internal SSD; an external Thunderbolt 4 SSD is the only second device; [Apple spec][apple-spec]).

## What model-fitter would have to build vs. reuse

- **Reuse:**
  - packed FP4 / FP8 kernels: MLX for NVFP4/MXFP8, llama.cpp for MXFP4 on Metal;
  - expert placement and mmap paging (llama.cpp);
  - prompt/KV cache and speculative decoding (both). Inkling 2 Mini ships an `mtp_config` and Qwen3.8 has `mtp_num_hidden_layers: 1`. llama.cpp has an MTP draft type.
- **Build, or contribute upstream, if a Fit plan needs SSD-streamed experts:**
  - an explicit expert reader with `F_NOCACHE`, so streamed bytes do not also pressure the page cache. [INFERENCE: this matters for the "machine stays responsive" part of Usable on a 16 GB machine]
  - a RAM expert cache with a measured policy;
  - Kimi's trace-and-replay tool, to choose cache size and policy per model offline.
- **Do not port as-is:** the trunk-first allocation rule, K3's preset numbers and the "expert cache is useless" finding. They follow from K3's flat router and 109 GB trunk.

## Expected tok/s on the reference device (ESTIMATES)

Everything in this section is a back-of-envelope **estimate**, not a measurement. Benchmarks belong to a separate ticket.

### Device inputs

- **Memory bandwidth.** M4 has 120 GB/s; the 16 GB base Mac mini ships with a 256 GB SSD ([Apple Mac mini (2024) tech specs][apple-spec]). I assume 70 % of peak is achievable for decode, i.e. **84 GB/s [ASSUMPTION]**.
- **SSD.** Apple does not publish SSD throughput on that page. Third-party measurements of the M4 Mac mini 256 GB ([AAPL Ch.][aaplch-m4], reported by [AppleInsider][ai-ssd]):
  - Blackmagic read 2,792.5 MB/s;
  - AmorphousDiskMark sequential read 2,789 MB/s at QD8, 2,192 MB/s at QD1;
  - random 4K reads 666 MB/s at QD64, 64 MB/s at QD1.

  The same article says larger configurations exceed 5 GB/s (1 TB). Apple's own *LLM in a Flash* paper measured >6 GiB/s for large sequential reads on an M1 Max. It reports scattered small reads dropping to ~1.25 GiB/s and says throughput needs large chunks ([arXiv:2312.11514 §2.2, §4][flash]).
- **Planning numbers.** About 21 GB free disk implies the 256 GB model [INFERENCE]. I plan with **2.2 GB/s** effective for MB-sized expert reads on it (≈ the QD1 sequential figure) [ASSUMPTION], and with 5.0 GB/s for a ≥ 1 TB configuration.
- **Weight budget.** I assume ~10 GB of the 16 GB is available to weights while the machine stays responsive [ASSUMPTION; the device-profile ticket should replace this].

### Inkling 2 Mini (26.4B MoE, ~5.4B active)

Facts from the safetensors headers:

- **Total:** 19.53 GB as shipped.
- **Routed experts:** 15.67 GB.
  - Layers 0–19 store experts as NVFP4: U8 packed nibbles + F8_E4M3 scale per 16, so **4.48 MB/expert**.
  - Layers 20–23 store them as FP8 E4M3 with a U8 scale per 32, so **8.21 MB/expert**.
  - Each layer stacks 128 experts in one tensor. One expert is therefore 4 contiguous runs (w13, w13-scale, w2, w2-scale) of 0.17–2.65 MB, not one run as in Kimi K3. A Kimi-style "one `pread` per expert" needs a repack.
- **Non-expert trunk layers:** only **1.37 GB** bf16.
- **Other:** embed + unembed 1.85 GB; vision 0.64 GB.
- **Routing and loop:** `num_experts_per_tok: 8` of `n_routed_experts: 128`. `loop_config` runs layers 6–17 for 3 passes, so a token makes 48 layer evaluations [INFERENCE from `core_start: 6, core_end: 18, num_passes: 3`].

Per-token expert bytes on a full miss are **0.98 GB** if the three loop passes pick the same experts, and **1.84 GB** if they pick disjoint ones. Per-token DRAM traffic is ~5.5 GB: the looped bf16 trunk ~2.8 GB, unembed 0.93 GB, expert compute 1.84 GB, so ~0.066 s at 84 GB/s.

With the trunk resident (~3.2 GB with embeddings), a ~7 GB expert cache holds ~45 % of the expert pool.

| Expert RAM hit rate *h* | SSD bytes/token | tok/s @ 2.2 GB/s (256 GB SSD) | tok/s @ 5.0 GB/s (≥ 1 TB SSD) |
|---:|---:|---:|---:|
| 0 % | 0.98–1.84 GB | 1.1–2.0 | 2.3–3.8 |
| 50 % | 0.49–0.92 GB | 2.1–3.5 | 4.0–6.1 |
| 80 % | 0.20–0.37 GB | 4.3–6.4 | 7.2–9.5 |
| 95 % | 0.05–0.09 GB | 9.3–11.3 | 11.8–13.2 |

Model: s/token = SSD bytes ÷ SSD bandwidth + DRAM bytes ÷ 84 GB/s, with no I/O-compute overlap (pessimistic).

Reading of the table:

- **Usable needs h ≳ 80 % on the base SSD.** At 5 tok/s the SSD budget is ≈ 0.29 GB/token at 2.2 GB/s and ≈ 0.67 GB/token at 5 GB/s.
- **Uniform routing would cap h near 45 %.** If routing were as uniform as K3's, the hit rate would be about the cached fraction (~45 %), which misses the bar on the base SSD. K3 itself saw 0 % at 1.6 % of experts cached and 36 % at 4.4 % in a full-recompute trace ([PERFORMANCE.md L39-104][k-perf-two-caches]).
- **Skew could make it Usable.** Inkling has `use_gate_bias: true`, a bias-steered router like K3's [INFERENCE: probably load-balancing, so expect little skew]. It also reuses expert weights across 3 loop passes within a token. Only a recorded routing trace (row 4) can settle this.
- **Shrinking the experts removes streaming entirely.** Requantising the experts to ~3 bits would bring the whole model to ~11–12 GB, near the weight budget. That trade-off belongs to the low-bit quantisation and Quality-loss tickets.

### Qwen3.8-27B (dense)

Facts from the safetensors headers: 55.56 GB bf16 in total, of which the 64 decoder layers are 48.71 GB (0.74–0.77 GB each), lm_head 2.54 GB, embed 2.54 GB, vision 0.92 GB and MTP 0.85 GB. A token reads ~25.6B parameters (layers + lm_head).

Every token reads the whole trunk, so there is no sparsity to exploit. Kimi's dense-trunk streaming (rows 5–6) is the only mechanism, and its cost is linear in the streamed bytes.

| Weight precision (effective bpw) | Bytes/token | Streamed beyond a 10 GB resident set | tok/s @ 2.2 GB/s | tok/s @ 5.0 GB/s |
|---|---:|---:|---:|---:|
| ~4.5 (4-bit + group scales) | 14.4 GB | 4.4 GB | ~0.5 | ~1.0 |
| ~3.5 | 11.2 GB | 1.2 GB | ~1.5 | ~2.8 |
| ~3.0 (fully resident) | 9.6 GB | 0 | ~8.7 (bandwidth ceiling) | ~8.7 |

Reading of the table:

- **Usable leaves almost no room for streaming.** At ~3 bpw, DRAM time alone is ~0.11 s/token. The 5 tok/s budget (0.2 s) leaves only ~0.19 GB/token of SSD reads at 2.2 GB/s and ~0.43 GB/token at 5 GB/s.
- **So streaming a dense model is Slow mode only.** Kimi's trunk ring makes Qwen3.8-27B *run* above 10 GB, at roughly 0.5–3 tok/s. A Usable Fit must be fully resident, which means ≲ 3 bpw on this machine; the Quality-loss ticket must price that.
- **Context memory stays small.** Qwen3.8's 48 linear-attention layers keep a constant state, so only 16 layers grow a KV cache (config). That is what keeps an 8K context affordable next to a ~10 GB weight set.

## Open questions this raises (for other tickets, not answered here)

- Inkling 2 Mini routing skew and cross-pass expert reuse: record a trace and replay it, Kimi-style.
- Whether llama.cpp's `--cpu-moe` + mmap + `--moe-cache-mib` on Metal keeps the machine responsive at 16 GB, compared with an explicit `F_NOCACHE` reader.
- Does llama.cpp support Inkling 2 Mini's NVFP4/FP8 experts and looped architecture on Metal? NVFP4/F8 matmul is not supported on Metal at the commit read. Runtime support is covered by its own ticket.

[k-arch-model]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ARCHITECTURE.md#L8-L20
[k-arch-mem]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ARCHITECTURE.md#L30-L41
[k-arch-attn]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ARCHITECTURE.md#L76-L90
[k-arch-mxfp4]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ARCHITECTURE.md#L110-L128
[k-arch-trunk]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ARCHITECTURE.md#L130-L138
[k-arch-correct]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ARCHITECTURE.md#L140-L149
[k-perf-ladder]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/PERFORMANCE.md#L16-L37
[k-perf-two-caches]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/PERFORMANCE.md#L39-L104
[k-perf-capacity]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/PERFORMANCE.md#L71-L107
[k-perf-alloc]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/PERFORMANCE.md#L109-L144
[k-perf-longer]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/PERFORMANCE.md#L191-L201
[k-tune-split]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/TUNING.md#L11-L44
[k-tune-split-dev]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/TUNING.md#L46-L67
[k-tune-weak]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/TUNING.md#L69-L81
[k-tune-presets]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/TUNING.md#L83-L108
[k-tune-incr]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/TUNING.md#L112-L118
[k-tune-threads]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/TUNING.md#L141-L152
[k-readme-memopts]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L430-L452
[k-readme-presets]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L601-L638
[k-readme-noblas]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L694-L704
[k-readme-classify]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L855-L874
[k-readme-pack]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L2301-L2361
[k-readme-pin]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L2363-L2423
[k-readme-admit]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L2430-L2453
[k-readme-cache]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L2648-L2757
[k-readme-trace]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L2775-L2849
[k-readme-storage]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L3415-L3462
[k-readme-noquant]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/README.md#L3469-L3524
[k-roadmap]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/ROADMAP.md#L71-L77
[k-cl-110]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/CHANGELOG.md#L225-L242
[k-cl-110a]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/CHANGELOG.md#L163-L191
[k-cl-100]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/CHANGELOG.md#L314-L337
[k-int8]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/notes/int8-draft-container.md#L56-L84
[k-huf]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/docs/notes/compressed-trunk.md#L5-L30
[k-pio]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/src/io/k3_portable_io.h#L1-L104
[k-sysmem]: https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/b26ee8231bfa4390545ef2aec3576fd834b92ba7/src/cli/k3_sysmem.h#L18-L27
[l-arg-threads]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L1499-L1504
[l-arg-cache]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L1698-L1702
[l-arg-load]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2665-L2696
[l-arg-ot]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2728-L2750
[l-moecache-host]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-moe-cache.cpp#L313-L427
[l-arg-moecache]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2751-L2770
[l-arg-fit]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/arg.cpp#L2863-L2895
[l-spec]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/common/common.h#L177-L188
[l-h-moecache]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/include/llama.h#L409-L410
[l-moecache-h]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-moe-cache.h#L33-L39
[l-moecache-policy]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-moe-cache.cpp#L21-L28
[l-moecache-log]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-moe-cache.cpp#L870-L876
[l-mmap]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-mmap.cpp#L704-L746
[l-mlock]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/src/llama-mmap.cpp#L920-L930
[l-ggml-types]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/include/ggml.h#L258-L259
[l-metal-mxfp4]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/kernels/dequantize.h#L348
[l-metal-nofp4]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal-device.m#L1525-L1535
[l-metal-ishost]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal.cpp#L359-L363
[l-metal-map]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal-device.m#L2240-L2283
[l-metal-rset]: https://github.com/ggml-org/llama.cpp/blob/a0677c341ba6bbfbbf74fa3c966f4dba74e1c830/ggml/src/ggml-metal/ggml-metal-device.m#L927-L940
[m-load]: https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/mlx/io/load.cpp#L513-L550
[m-ops-quant]: https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/python/src/ops.cpp#L4856-L4919
[m-prim]: https://github.com/ml-explore/mlx/blob/b835fb313e14d08de17d43cd59ce2dbfd3422569/mlx/primitives.h#L155
[ml-wired]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py#L249-L283
[ml-gen-args]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py#L186-L235
[ml-utils-wired]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/utils.py#L84-L93
[ml-readme-large]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/README.md#L267-L289
[ml-moe]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/layers/moe.py#L100-L112
[ml-moe-sort]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/layers/moe.py#L18-L24
[ml-mixed]: https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/convert.py#L20-L33
[apple-spec]: https://support.apple.com/en-us/121555
[aaplch-m4]: https://applech2.com/archives/20241110-mac-mini-2024-ssd-speed.html
[ai-ssd]: https://appleinsider.com/articles/26/09/23/base-mac-mini-still-has-slower-storage-than-any-other-configuration
[flash]: https://arxiv.org/abs/2312.11514
