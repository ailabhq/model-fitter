# Can a per-tensor streaming converter build both reference models' GGUFs within 20 GB of disk on the M4, identical to llama.cpp's own conversion?

Map ticket: "Can a per-tensor streaming converter build both reference models' GGUFs within 20 GB of disk on the M4, identical to llama.cpp's own conversion?" (map: "Map: sarto, Usable Fits of new models on small Apple Silicon Macs").

## Answer

Yes, for both reference models.

- **Disk.** Peak disk is the output file plus 25 MB of config: 10.2 GiB for Qwen3.8-27B and 10.9 GiB for Inkling 2 Mini. No Source byte touches the Mac's disk.
- **Memory and responsiveness.** Peak process memory is 2.4 GiB and memory pressure stays normal, so both runs stay Responsive.
- **Identity.** The output is tensor-for-tensor identical to `convert_hf_to_gguf.py --outtype f32` + `llama-quantize`:
  - Qwen on the same machine: 770 of 770 tensors;
  - Inkling from the M4 against the x86 reference: 704 of 704 tensors.
- **Identity across platforms.** Qwen's M4 output differs from the x86 reference only where the platform itself differs, and each cause is reproduced below:
  - clang's FMA contraction on ARM, removed by building ggml with `-ffp-contract=off`;
  - macOS's unstable `qsort` in IQ1_S/IQ1_M quantization;
  - torch's `exp` in one converter transform.
- **Wall time.** It is set by the network: 90 min for Qwen (54.7 GB read) and 41 min for Inkling (18.9 GB read) at about 8–10 MB/s from a LAN mirror.

## Setup

- llama.cpp `22b99122e` (the scorer's tree), with `conversion/`, `gguf-py/` and `ggml/` unmodified. The Mac builds shared `libggml-base` and `llama-quantize` from it.
- `stream_convert.py`:
  - The pinned converter's own model classes map the names and apply every transform, as `convert_hf_to_gguf.py --outtype f32` does.
  - Each tensor is quantized while the GGUFWriter writes it, with `ggml_quantize_chunk` and llama-quantize's imatrix arithmetic: `in_sum2 / lround(counts)`, per expert, with chunks never crossing an expert.
  - It has three source paths:
    - plain upcasts of a stored tensor are read by byte range in row slabs of 256 MiB of F32;
    - transformed tensors are materialized whole (largest: 320 MiB of F32);
    - Inkling's packed NVFP4/MXFP8 experts are read one expert at a time and dequantized by llama.cpp's own `_dequant_expert_chunk`.
  - `--plan-only` gives the exact output size from headers alone, before any tensor data is read.
- **Target mix:** the per-tensor types of Unsloth's published UD-IQ3_XXS (read from the GGUF header only), with Unsloth's `imatrix_unsloth.gguf` as the stand-in imatrix.
- **Reference:**
  - `convert_hf_to_gguf.py --outtype f32`, then `llama-quantize --imatrix … --tensor-type-file …` naming every tensor (an unnamed tensor would get the IQ3_XXS ftype's own rules).
  - It runs on an x86 NAS in docker (`nas/Dockerfile`, `GGML_NATIVE=OFF`), because it needs 176 GB of disk for Qwen: Source, F32 intermediate and output.
  - `stream_convert.py` also runs there, for a same-machine comparison.
- **Mac runs:**
  - M4, 16 GB, on Wi-Fi, reading the pinned Source revisions by HTTP range from a LAN mirror of the Hugging Face files. The Mac's own internet link measured 0.24 MB/s.
  - ggml is linked from a `-ffp-contract=off` build of `libggml-base`.
- **Responsive:** `bench.py`'s Monitor and kill-switch, using the rule from "What measurable test decides that the machine stays responsive for a Usable Fit?", behind the quiet start gate (6 GiB floor). Other processes' CPU is recorded but never stops a conversion.

## Results on the M4

|  | Qwen3.8-27B | Inkling 2 Mini |
|---|---|---|
| Source revision | `37561816` (18 BF16 shards, 55.56 GB) | `c9d02830` (one NVFP4/MXFP8 file, 19.53 GB) |
| Tensors written | 770 | 704 |
| Output bytes | 10,937,802,944 | 11,663,287,840 |
| Published UD-IQ3_XXS bytes | 10,934,860,704 | 11,663,310,784 |
| Peak disk used (output dir) | 10.21 GiB | 10.89 GiB |
| Free disk before / minimum | 19.65 / 9.10 GiB | 19.29 / 8.71 GiB |
| Peak process RSS | 2.35 GiB | 1.97 GiB |
| Memory pressure | normal throughout | normal throughout |
| Swap-outs during run | 0 | 0 |
| Window frame max / timer lateness max / spawn max | 23 / 104 / 31 ms | 21 / 79 / 30 ms |
| Responsive | yes | yes |
| Wall time | 5,418 s (90 min) | 2,434 s (41 min) |
| Bytes read from Source | 54,735,767,552 (2,320 range requests) | 18,894,601,376 (13,069 range requests) |
| Mean read rate | 10.1 MB/s | 7.8 MB/s |

- **Wall time** is set by the network. The NAS read Qwen from its own disk with `stream_convert.py` (a Pentium Gold 8505 limited to 3 CPUs) and took 4,936 s, so throughput there was probably CPU-bound (inferred, not profiled).
- **Reference cost on the NAS:** for Qwen, the F32 convert took 1,960 s and wrote a 109 GB intermediate, and `llama-quantize` took 5,477 s. For Inkling, the F32 convert took 941 s for a 103 GB intermediate, and `llama-quantize` took 4,829 s.
- **Inkling's output is 23 KB smaller than the published file.** Every tensor we write has the published file's name and type. The published file also has 40 small F32 tensors that the pinned toolchain never writes: `blk.N.ffn_{gate_up,down}_exps.scale`, for its 20 NVFP4 expert tensors. So it was built by a path that keeps NVFP4's per-tensor second-level scale. Here, both the reference and this converter quantize the dequantized F32 back to NVFP4 without it.

## Identity

| Comparison | Qwen3.8-27B | Inkling 2 Mini |
|---|---|---|
| Same machine (x86): ours vs reference | **770 / 770 identical** | not run (the M4 output already matches) |
| M4 (`-ffp-contract=off`) vs x86 reference | 717 / 770 identical | **704 / 704 identical** |

Identical means same name, type, shape and data bytes. Inkling's mix has no IQ1 types, and none of its tensors differed across platforms. That includes its 48 packed expert tensors, dequantized from NVFP4/MXFP8 by llama.cpp's own code one expert at a time. In the Qwen same-machine comparison (`results/qwen-nas-compare.json`), two kinds of difference do not involve tensors:

- **Metadata:** `general.name` (the converter derives it from the repo id, the reference from the directory name), and `general.basename` plus the `quantize.imatrix.*` keys that `llama-quantize` adds.
- **Tensor order:** `llama-quantize` writes tensors in a different order, so whole-file hashes differ even when every tensor is identical.

Every cross-platform difference was reproduced and attributed:

| Tensors | Cause | Evidence |
|---|---|---|
| Most k-quant and i-quant tensors on a stock Mac build (Qwen3-0.6B smoke: 122/311) | clang contracts multiply-adds into FMA on ARM | With ggml built with `-ffp-contract=off`, Mac and x86 are 311/311 identical (`results/smoke/qwen3-0.6b-cross-platform.json`) |
| 5 IQ1_S / IQ1_M tensors in Qwen | `iq1_sort_helper` compares values only. macOS `qsort` is not stable, while glibc 2.36's is a stable merge sort, so ties are ordered differently | With a stable merge sort interposed for `qsort` (`qsort-check/`), the Mac output is byte-identical to x86 for `blk.0.ffn_up` (IQ1_S, 1 of 348,160 blocks differed) and `blk.13.ffn_gate` (IQ1_M, 7,127 blocks differed) |
| 48 `ssm_a` (F32) in Qwen | The converter's transform `-torch.exp(A_log)` | `-exp` of all 48 stored `A_log` tensors gives different bytes from the Mac's torch 2.15 (arm64) and the reference image's torch 2.11 (x86) (`exp_check.py`) |

## Findings for the map

1. **Quantized bytes depend on the platform and its libraries, not only on the llama.cpp commit.**
   - A Fit's bytes reproduce on another machine only with the same ggml build flags, a libc whose `qsort` breaks ties the same way, and the same torch for converter-side transforms.
   - Comparing a Fit's hash across machines, or a golden check after a pin bump, must pin these too, or else compare within one platform.
2. **Published mixes do not carry over across llama.cpp versions.**
   - The pinned converter fuses Qwen3.5's `attn_qkv`+`attn_gate` into `attn_qkv_gate` and `ssm_beta`+`ssm_alpha` into `ssm_beta_alpha`. Unsloth's file keeps them separate.
   - `llama-quantize` itself rejects the published mix (`Missing importance matrix for tensor blk.0.ssm_beta_alpha.weight`).
   - `adapt_mix.py` maps the fused tensors: the type of the largest part, and the imatrix entry of the parts. The parts' entries were identical because they share an input.
   - A mix keyed by tensor role, not by tensor name, avoids this.
3. **The output size is known exactly before any download.** `--plan-only` reads only headers and config: Qwen in 35 s, Inkling in 10 s. It agreed with the final files to the byte, apart from metadata.
4. **Upstream remote mode is not usable as is.**
   - The pinned Inkling converter calls `element_size()` on each lazy tensor while indexing. A lazy tensor answers non-tensor methods by materializing itself, which downloads the whole Source once before converting.
   - `stream_convert.py` answers `element_size`, `dim`, `numel` and similar calls from the meta tensor.
5. **Conversion time is download time.** At the Mac's own measured 0.24 MB/s internet link, Qwen's 54.7 GB would take about 63 hours. On a typical home link, the Source download, not the Mac, sets how long a Fit takes.
6. **Built-in target size.** `llama-quantize` at this pin has `--target-bpw` and `--target-size`. This is relevant to "How does sarto's converter choose a type for each tensor to hit a target file size?".

## Files

- `streaming-converter/stream_convert.py`: the converter.
- `streaming-converter/run_convert.py`, `mac_run.sh`, `gate.py`: monitored Mac runs.
- `streaming-converter/nas/`: reference image and job.
- `streaming-converter/compare.py`, `tensor_hashes.py`: comparisons.
- `streaming-converter/mix_types.py`, `adapt_mix.py`: published mix to per-tensor types.
- `streaming-converter/qsort-check/`, `exp_check.py`: the cross-platform cause checks.
- `streaming-converter/smoke.py`: small-model checks.
- `streaming-converter/results/`: all measurements.
