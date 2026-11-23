# MLX at 8K: Inkling 2 Mini oQ2.5e on the M4 16 GB

Map ticket: "Can MLX hold Inkling oQ2.5e at 8K context on the M4 16 GB with prefill and memory-guard settings tuned?"

## Answer

**Yes.** On the default GPU limit (11.84 GiB), MLX runs Inkling oQ2.5e at 8K once prefill is cut into chunks of **≤ 256 tokens**: **22 tok/s generating at 8K depth**, 25 tok/s at short context, 160–190 tok/s prefill (an 8K prompt takes 43–57 s to the first token). The machine stays **Responsive** under the map's rule in every run that worked, with 1.0–3.6 s of critical pressure (0.7–3.1% of the run), no hangs, and 1.7–3.6 GiB of other apps swapped out. That makes it **Usable**. Quality loss is still unmeasured.

| Lever | Result |
|---|---|
| **Prefill step** (the one that matters) | 2048 and 512: Metal out-of-memory. 256: works, peak MLX memory 11.70 GiB (0.14 GiB below the limit). 128: works, peak 11.26 GiB (0.58 GiB headroom), prefill 160 vs 174–191 tok/s, about 9 s more to the first token. |
| **KV-cache quantization** | Not a lever. The 8K KV cache is only 0.24 GB: just 8 of the 48 layer visits are full attention (with the loop, the 12 core layers run 3 times), and the sliding ones keep 1,024 keys. mlx-lm's `kv_bits=4` crashes in oMLX's Inkling attention (`QuantizedKVCache` hands back tuples to `banded_sink_attention`). oMLX's `affine4` reports itself enabled, but peak memory and the first 300 characters of output are unchanged, so there's no sign it reaches Inkling's attention (inferred). |
| **mlx-vlm directly** | Not possible. Stock mlx-vlm 0.7.11 and 0.8.0 have no looped Inkling 2 (`inkling2_loop`); 0.8.0's `inkling2` is the non-looped 48-layer model. The only MLX code for Inkling 2 Mini is oMLX's. "MLX direct" below is mlx-lm's `stream_generate` over that code, without oMLX's server. |
| **oMLX** | Stock oMLX can't do it: the prefill step is hard-coded at 2048, with no flag or setting. With the step patched to 256 it still runs out of memory, because by default oMLX also loads the bundled DSpark/MTP drafter (11.78 GB resident vs 10.25–10.30 GB). With the step patched to 256 **and** `mtp_enabled: false` it works, about as well as MLX direct: 22.0 tok/s at 8K, ~188 tok/s prefill. Text-only engine (`model_type_override: llm`) makes no difference. |
| **Memory guard** | Doesn't help, and has to be off. A custom 12 GB ceiling refuses the load with MTP on (12.17 GB projected) and aborts mid-prefill with MTP off (hard watermark 11.6 GB, then Metal out-of-memory): it doesn't shrink the chunks enough. 13 GB is refused by oMLX's own Metal cap check (12.02 GB). |
| **MLX buffer cache** (`mx.set_cache_limit(0)`) | Same peak, same speed, a little less swap (1.73 vs 2.39 GiB). Not needed. |
| **`iogpu.wired_limit_mb` 13312** | Lets prefill step 512 fit (peak 12.2 GiB): 208 tok/s prefill, 44 s to the first token, same 22 tok/s generation. It costs the desktop: 3.56 GiB swapped out, a 406 ms process launch, peak wired 14.3 GiB. It still passes, but with the least margin. Not worth it. |

### What this means for the map

- On MLX, Inkling at 2.5 bits fits fully resident at 8K with no expert streaming. Streaming Inkling's experts from SSD is now only a llama.cpp question, and only if the Metal NVFP4 run leaves experts on the CPU.
- The binding constraint is prefill scratch memory against the GPU working-set limit, not the weights or the KV cache. Peak above the resident weights (10.16 GiB) at 8K: +1.1 GiB at step 128, +1.5 GiB at 256, about +2.0 GiB at 512. A Fit plan has to choose the prefill chunk from the context length and the headroom. Step 256 leaves only 0.14 GiB, so step 128 is the safe choice on this Mac. Above 8K the scratch grows (not measured).
- oMLX needs two non-default settings to work, one of them a code patch. It also sends Inkling's text to the client as a single event at the end of each request, so there's no live streaming, and its usage timings count the whole request as prefill. Its decode speed here is derived (see Method notes).

## Setup

- Mac mini M4, 16 GB, macOS as in each result's `machine`. Usual desktop and agent sessions left open; nothing was quieted. `iogpu.wired_limit_mb` default (0 → 11.84 GiB GPU working set) unless a row says otherwise.
- Model: [`mlx-community/Inkling-2-Mini-oQ2.5e-mtp`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ2.5e-mtp) at `d4a4f3f5`, 11.6 GB on disk (10.16 GiB resident after load without the MTP head).
- Runtimes: oMLX 0.8.3 (mlx 0.32.4, mlx-lm 0.32.3.dev8, mlx-vlm 0.7.11), and "MLX direct": mlx-lm's `stream_generate` in a worker process ([`harness/mlx_direct.py`](baseline/harness/mlx_direct.py)).
- Workload as in the Baseline: 8,000-token prompt from llama.cpp's docs at `b6d5ce4`, 256 tokens generated twice, then a 69-token prompt with 128 tokens generated. Greedy, EOS suppressed. Cold load (`purge` first).
- Harness: the Baseline's [`bench.py`](baseline/harness/bench.py), changed to judge **Responsive** by the map's rule: a `CGWindowListCopyWindowInfo` round-trip every 0.5 s (own thread, with a watchdog for a call that never returns), the 20 ms sleep timer, `/usr/bin/true` launches each second and memory pressure every 0.5 s. A run fails on any check ≥ 500 ms, window or timer p99 > 100 ms, or critical pressure > 5% of the run or ≥ 10 s in one stretch. The kill-switch applies the same rule live and SIGKILLs the runtime (critical time is budgeted against a planned 300 s run). The start gate only waits for normal pressure, CPU ≤ 40% and no other runtime or download.

## Method notes

- MLX direct reports prefill and decode speed from mlx-lm's `GenerationResponse`, and peak memory from `mx.get_peak_memory()` ("Peak MLX GiB"). Its second 8K request runs a fresh prefill; there's no prompt cache.
- oMLX decode speed = (n − 1) / (wall(n) − wall(1)), using the same prefix-cached prompt asked for 1 token. Cold prefill = the cold request's wall time minus that decode time. The harness reads the stream unbuffered (`iter_lines(chunk_size=1)`), so the single text event is oMLX's behaviour, not the client's.
- oMLX ran with an isolated `--base-path` (empty settings and prefix cache each run). Prefill step 256 was set by patching `GlobalSettings.to_scheduler_config` (`OMLX_PREFILL_PATCH` in `bench.py`); the server log confirms `prefill_step_size = 256`.
- Output check: greedy generation, and the first 300 characters are identical across MLX direct and oMLX at 8K. The text continues the documentation coherently.
- Runs that failed (out-of-memory, refused, crashed) aren't judged for responsiveness: they are a few seconds long, and a single critical blip is a large share of them.

## All runs

| Config | Wired MB | Load s | TTFT @8K s | Prefill tok/s @8K | Gen tok/s @8K | Gen tok/s short | Peak MLX GiB | Peak wired GiB | Swap-outs GiB | Critical s (share, longest) | Window max ms | Timer p99 / max ms | Launch max ms | Responsive | Usable | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ink-oq25e-mlx-p256 | default | 11.31 | 48.35 | 173.98 | 22.3 | 25.02 | 11.7 | 13.45 | 2.39 | 3.1 (2.3%, 3.1) | 237.0 | 10.8 / 240.2 | 20.6 | yes | **yes** | ok |
| ink-oq25e-mlx-p2048 | default | 7.69 | – | – | – | – | – | 13.59 | 1.93 | 2.1 (11.4%, 2.1) | 11.4 | 11.0 / 223.7 | 255.2 | – | no | RuntimeError: RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory ( |
| ink-oq25e-mlx-p512 | default | 7.55 | – | – | – | – | – | 13.55 | 1.18 | 0.0 (0.0%, 0.0) | 147.2 | 10.9 / 257.4 | 17.0 | – | no | RuntimeError: RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory ( |
| ink-oq25e-mlx-p256-kv4 | default | 7.52 | – | – | – | – | – | 13.04 | 1.25 | 0.9 (5.5%, 0.9) | 138.9 | 12.0 / 241.3 | 20.2 | – | no | RuntimeError: AttributeError: 'list' object has no attribute 'shape' |
| ink-oq25e-mlx-p256-cache0 | default | 7.65 | 47.89 | 191.47 | 22.04 | 24.79 | 11.7 | 13.44 | 1.73 | 1.0 (0.7%, 1.0) | 19.7 | 10.7 / 247.2 | 19.4 | yes | **yes** | ok |
| ink-oq25e-omlx-guard-off-p256 | default | 16.2 | – | – | – | – | – | 13.28 | 2.8 | 2.6 (6.4%, 2.6) | 9.6 | 10.9 / 231.4 | 20.9 | – | no | RuntimeError: stream error: {'message': '[METAL] Command buffer execution failed: Insuffic |
| ink-oq25e-omlx-guard-12gb | default | – | – | – | – | – | – | 1.71 | 0.0 | 0.0 (0.0%, 0.0) | 5.6 | 13.2 / 13.5 | 19.1 | – | no | RuntimeError: HTTP 507: {"error":{"message":"Cannot load Inkling-2-Mini-oQ2.5e-mtp: projec |
| ink-oq25e-omlx-guard-13gb | default | – | – | – | – | – | – | 1.85 | 0.0 | 0.0 (0.0%, 0.0) | 2.5 | 10.1 / 10.1 | 11.2 | – | no | RuntimeError: HTTP 507: {"error":{"message":"Cannot load Inkling-2-Mini-oQ2.5e-mtp: projec |
| ink-oq25e-mlx-p128 | default | 7.65 | 57.23 | 160.04 | 21.95 | 24.69 | 11.26 | 13.14 | 2.28 | 2.1 (1.4%, 2.1) | 242.2 | 10.7 / 248.2 | 245.9 | yes | **yes** | ok |
| ink-oq25e-mlx-p512@wired13312 | 13312 | 8.17 | 44.07 | 208.14 | 22.08 | 24.92 | 12.2 | 14.3 | 3.56 | 3.6 (2.9%, 3.6) | 14.3 | 10.4 / 278.4 | 406.2 | yes | **yes** | ok |
| ink-oq25e-omlx-guard-off-p256-nomtp | default | 13.67 | 42.59 | 187.84 | 22.0 | 22.09 | – | 13.54 | 2.53 | 2.1 (1.8%, 2.1) | 5.1 | 10.1 / 209.8 | 20.0 | yes | **yes** | ok |
| ink-oq25e-omlx-guard-off-p256-nomtp-llm | default | 11.77 | 43.41 | 184.29 | 21.74 | 22.64 | – | 13.51 | 1.93 | 2.1 (1.8%, 2.1) | 158.4 | 10.3 / 241.8 | 67.4 | yes | **yes** | ok |
| ink-oq25e-omlx-guard-off-p256-nomtp-kv4 | default | 14.01 | 42.86 | 186.65 | 21.63 | 22.76 | – | 13.58 | 2.93 | 1.6 (1.3%, 1.6) | 144.8 | 10.1 / 224.6 | 35.2 | yes | **yes** | ok |
| ink-oq25e-omlx-guard-12gb-nomtp | default | 13.62 | – | – | – | – | – | 13.37 | 2.07 | 2.1 (8.1%, 2.1) | 137.0 | 10.1 / 209.3 | 73.7 | – | no | RuntimeError: stream error: {'message': 'oMLX memory guard aborted this request mid-prefil |
