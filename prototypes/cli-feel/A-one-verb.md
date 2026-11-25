# PROTOTYPE variant A: one verb

`sarto fit <hf-id>` is the product: it runs the whole plan run and leaves a Fit. Everything else looks at
Fits or starts them. Rerunning the same `sarto fit` resumes; it is the only way to make progress.

## `sarto --help`

```text
sarto: fit a newly released model to this Mac, measured, not guessed.

Usage: sarto <command> [options]

Commands:
  fit <hf-id>      Find the best-quality Fit that is Usable on this Mac, and write it
  serve <fit>      Start a Fit with the exact llama-server command it was measured with
  show <fit>       Print a Fit's report: speed, Responsive, Quality loss, every candidate
  list             List Fits on this Mac
  rm <fit>         Delete a Fit and its weights
  device           Print this Mac's Device profile

Run 'sarto <command> --help' for options and examples.
```

## `sarto fit --help`

```text
Usage: sarto fit <hf-id> [options]

Walks Community builds largest first, benchmarks each on this Mac (speed at the requested context,
Responsive), scores Quality loss against the Source's Reference pack, then tries one sarto
conversion. Keeps the Usable candidate with the least Quality loss. Rerun the same command to resume.

Options:
  --revision <commit>   Source revision (default: the repo's main at first run, then pinned in the run)
  -c, --context <n>     Context length to be Usable at (default: 8192)
  --slow                Accept a Responsive, not-Rejected candidate below 5 tok/s if nothing is Usable
  --degraded            Accept a Usable, Rejected candidate if nothing better passes
  --build <repo>        Also try this Community build (any publisher; allowlisted ones are always tried)
  --no-convert          Skip sarto's own conversion (it downloads the full Source again)
  --dry-run             List candidates, drops, download sizes and time estimates; download nothing
  -y, --yes             Don't ask before downloading
  --json                One JSON object per event on stdout; the final one is the Fit or the refusal

Examples:
  sarto fit Qwen/Qwen3.8-27B
  sarto fit Qwen/Qwen3.8-27B --dry-run
  sarto fit thinkingmachines/Inkling-2-Mini -c 16384 --yes
  sarto fit Qwen/Qwen3.8-27B --slow --degraded --no-convert
```

## Session: Qwen3.8-27B on the Mac mini M4 16 GB

```text
$ sarto fit Qwen/Qwen3.8-27B
sarto 0.1.0 · llama.cpp 22b99122e (pinned; built on first use, 3m10s)

Device   Mac mini M4 · 16 GB · GPU limit 11.84 GiB · 21.3 GB free disk · pressure normal
Source   Qwen/Qwen3.8-27B @ 3f1c2a9 · dense 27B · 18 shards, 55.56 GB
Pack     ailabhq/sarto-packs · prompts v1 · calibration v1 · 7 MB · cross-checked (vLLM) ✓
Goal     Usable (≥ 5 tok/s at 8192 context, Responsive), then least Quality loss

Candidates: 24 Community builds (unsloth, bartowski) + 1 sarto conversion
  dropped before download (llama-fit-params on each header):
    18 builds 12.32–15.48 GB    dense: layers would land on the CPU at 8192 context
  to measure, largest first (stops at the first pass, then up to 2 more within 1 GB below):
    1  unsloth    IQ3_S         12.04 GB
    2  unsloth    UD-IQ3_XXS    10.90 GB
    3  bartowski  IQ3_XXS       10.71 GB
    4  unsloth    UD-Q2_K_XL    10.15 GB
    …  2 more if the first ones fail
    +  sarto conversion, sized between the pass and the failure above it (downloads the Source, 55.56 GB)
  disk   one candidate at a time, peak ~16 GB of 21.3 GB free
  time   ~2–4 h, mostly downloads; Ctrl-C is safe, rerun to resume

Download about 100 GB, one candidate at a time, and benchmark on this Mac? Keep your usual apps open; don't start heavy jobs. [Y/n] y

[1] unsloth IQ3_S 12.04 GB
    download   12.04 GB  4m02s  sha256 ✓
    benchmark  cold load… killed at 0:19: memory pressure critical for 10 s
               → memory failure. Larger candidates of this placement: none left.
[2] unsloth UD-IQ3_XXS 10.90 GB
    download   10.90 GB  3m38s  sha256 ✓
    benchmark  killed at 4:12: window system paused 931 ms during generation; pressure never critical
               → Stall. Running it once more.
    benchmark  7.0 tok/s at 8192 · 8K prefill 71 s · Responsive (critical 0.4% of run, longest 1.1 s) · 4.2 GiB of other apps swapped out
    score      KLD 0.060 · top-1 91.8% · 8K slice KLD 0.011 · 6m12s
    ✓ Usable · Lossy · best so far (kept on disk: the next candidate fits beside it)
[3] bartowski IQ3_XXS 10.71 GB
    benchmark  7.3 tok/s · Responsive
    score      KLD 0.072 · top-1 90.6% → Lossy, worse than [2]
[4] unsloth UD-Q2_K_XL 10.15 GB
    benchmark  7.9 tok/s · Responsive
    score      KLD 0.094 · top-1 88.4% → Lossy, worse than [2]
[5] sarto conversion, target 11.40 GB (between [2] 10.90 ✓ and [1] 12.04 ✗)
    mix        IQ3_S / Q3_K, attention Q4_K, imatrix from pack · llama-fit-params: all layers on the GPU ✓
    convert    18 shards, one on disk at a time · 55.56 GB downloaded · 38m
    benchmark  6.4 tok/s · Responsive (critical 0.9%, longest 2.0 s) · 3.1 GiB swapped out
    score      KLD 0.041 · top-1 93.5% · 8K slice 0.014 → Lossy, better than [2]

Fit qwen3.8-27b.m4-16gb
  weights   sarto conversion, 11.40 GB
  speed     6.4 tok/s at 8192 context · Responsive
  quality   Lossy · KLD 0.041 · top-1 93.5% (8K slice 0.014)
  took      2h41m · 99.4 GB downloaded
  start     sarto serve qwen3.8-27b.m4-16gb       (OpenAI-compatible on 127.0.0.1:8080)
  report    sarto show qwen3.8-27b.m4-16gb
```

Run again, already done:

```text
$ sarto fit Qwen/Qwen3.8-27B
Fit qwen3.8-27b.m4-16gb is current (Source 3f1c2a9, pack p1.c1, llama.cpp 22b99122e). Nothing to do.
  --redo to measure again from scratch.
```

Interrupted and rerun:

```text
$ sarto fit Qwen/Qwen3.8-27B
Resuming plan run qwen3.8-27b.m4-16gb: [1]–[3] measured, continuing at [4] unsloth UD-Q2_K_XL.
```

## `sarto show qwen3.8-27b.m4-16gb`

```text
Fit qwen3.8-27b.m4-16gb   Usable · Lossy
Source     Qwen/Qwen3.8-27B @ 3f1c2a9
Device     Mac mini M4 · 16 GB (Mac16,10)
Weights    sarto conversion · 11.40 GB · IQ3_S / Q3_K, attention Q4_K
Runtime    llama.cpp 22b99122e · --fit on -c 8192 -ub 512

Speed      6.4 tok/s at 8192 · 8K prefill 74 s (111 tok/s)
Responsive yes · window p99 18 ms · timer p99 9 ms · launch max 61 ms · critical 0.9% (longest 2.0 s) · 3.1 GiB swapped out
Quality    Lossy (pack ailabhq p1.c1, published, cross-checked)
             mean KLD 0.041 · p99 KLD 0.62 · top-1 93.5%
             chat 0.037 · code 0.029 · multilingual 0.058 · tool calls 0.040
             8K slice 0.014 · 97.1%
             band edges (provisional): Near-lossless ≤ 0.02 & ≥ 97% · Lossy ≤ 0.10 & ≥ 88%

Candidates
  ✓ [5] sarto conversion   11.40 GB   6.4 tok/s   KLD 0.041   winner
    [2] unsloth UD-IQ3_XXS 10.90 GB   7.0 tok/s   KLD 0.060   Lossy; passed on 2nd run (1st: Stall, window 931 ms)
    [3] bartowski IQ3_XXS  10.71 GB   7.3 tok/s   KLD 0.072   Lossy
    [4] unsloth UD-Q2_K_XL 10.15 GB   7.9 tok/s   KLD 0.094   Lossy
    [1] unsloth IQ3_S      12.04 GB   –           –           memory failure at 0:19 (cold load)
    18 builds 12.32–15.48 GB dropped before download: layers on the CPU at 8192 context
```
