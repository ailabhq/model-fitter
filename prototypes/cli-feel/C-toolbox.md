# PROTOTYPE variant C: toolbox

Every step of a plan run is its own `<noun> <verb>` command that works alone and prints what it measured.
`sarto fit build` is the walk that chains them, and every line it prints names the primitive that did it,
so a person or the agent skill can redo any one step by hand: bench one build again, rescore a Fit after
the thresholds move, convert to an exact size.

## `sarto --help`

```text
sarto: fit a newly released model to this Mac, measured, not guessed.

Usage: sarto <noun> <verb> [options]

  fit        build · show · serve · list · rm     The Fit: the walk that makes one, and using it
  candidate  list · bench · score                 One set of weights measured on this Mac
  convert    <hf-id> --size <GB>                  sarto's own per-tensor conversion to a target size
  pack       get · status · build                 Reference packs (build runs on a CUDA host)
  device     show                                 This Mac's Device profile
  runtime    show · install                       The pinned llama.cpp

Start here:  sarto fit build <hf-id>
Run 'sarto <noun> --help' for verbs, 'sarto <noun> <verb> --help' for options and examples.
```

## `sarto candidate --help`

```text
Usage: sarto candidate <verb> <hf-id> [<candidate>] [options]

Verbs:
  list    Community builds of the Source and what llama-fit-params says about each on this Mac
  bench   Download one candidate and benchmark it: speed at the context, Responsive. Kill-switch always on.
  score   Quality loss of one benchmarked candidate against the Source's Reference pack

<candidate> is <publisher>:<quant> (unsloth:UD-IQ3_XXS), a repo:file, or a local .gguf.

Examples:
  sarto candidate list Qwen/Qwen3.8-27B
  sarto candidate bench Qwen/Qwen3.8-27B unsloth:UD-IQ3_XXS
  sarto candidate bench Qwen/Qwen3.8-27B ./my-quant.gguf -c 16384
  sarto candidate score Qwen/Qwen3.8-27B unsloth:UD-IQ3_XXS --json
```

## Session: Qwen3.8-27B on the Mac mini M4 16 GB

The front door:

```text
$ sarto fit build Qwen/Qwen3.8-27B
device show            Mac mini M4 · 16 GB · GPU limit 11.84 GiB · 21.3 GB free disk
pack get               ailabhq/sarto-packs p1.c1 for 3f1c2a9 · 7 MB · cross-checked ✓
candidate list         24 builds · 18 dropped (layers on CPU at 8192) · 6 to measure, largest first
                       about 100 GB of downloads, ~3–4 h. Go ahead? [Y/n] y
candidate bench   [1]  unsloth:IQ3_S 12.04 GB        memory failure at 0:19
candidate bench   [2]  unsloth:UD-IQ3_XXS 10.90 GB   Stall (window 931 ms), second run 7.0 tok/s · Responsive
candidate score   [2]                                KLD 0.060 · top-1 91.8% · Lossy ✓ first pass
candidate bench   [3]  bartowski:IQ3_XXS 10.71 GB    7.3 tok/s · Responsive
candidate score   [3]                                KLD 0.072 · Lossy
candidate bench   [4]  unsloth:UD-Q2_K_XL 10.15 GB   7.9 tok/s · Responsive
candidate score   [4]                                KLD 0.094 · Lossy
convert                --size 11.40GB                IQ3_S / Q3_K, attention Q4_K · 55.56 GB read · 38m
candidate bench   [5]  sarto:11.40GB                 6.4 tok/s · Responsive
candidate score   [5]                                KLD 0.041 · top-1 93.5% · Lossy ✓ best
fit build              qwen3.8-27b.m4-16gb written (2h41m)
                       start: sarto fit serve qwen3.8-27b.m4-16gb
```

Using the pieces alone, e.g. from the agent skill or after the map's threshold recalibration:

```text
$ sarto candidate list Qwen/Qwen3.8-27B --json | jq -r '.[] | select(.fit_params.ok) | .id'
unsloth:IQ3_S
unsloth:UD-IQ3_XXS
bartowski:IQ3_XXS
unsloth:UD-Q2_K_XL
bartowski:Q2_K
unsloth:UD-IQ2_M

$ sarto candidate bench Qwen/Qwen3.8-27B bartowski:Q2_K
download   9.93 GB  3m20s  sha256 ✓
benchmark  8.4 tok/s at 8192 · 8K prefill 62 s · Responsive (critical 0.2%) · 2.6 GiB swapped out
saved      ~/.sarto/runs/qwen3.8-27b.m4-16gb/candidates/bartowski-Q2_K/bench.json
next       sarto candidate score Qwen/Qwen3.8-27B bartowski:Q2_K

$ sarto convert Qwen/Qwen3.8-27B --size 11.0GB --dry-run
mix        IQ3_XXS / Q3_K, attention Q4_K · 10.98 GB
fit-params all layers on the GPU at 8192 ✓
disk       peak 15.0 GB (output + one 3.99 GB shard) of 21.3 GB free
download   Source 18 shards, 55.56 GB, ~19 min

$ sarto fit show qwen3.8-27b.m4-16gb --rescore
Quality bands recalibrated in sarto 0.2.0 (Near-lossless ≤ 0.03): no rerun needed, the scores are kept.
  band  Lossy → Lossy (KLD 0.041)
```

Errors point to the next primitive:

```text
$ sarto candidate score Qwen/Qwen3.8-27B bartowski:Q2_K
error: bartowski:Q2_K has no benchmark on this Device profile with llama.cpp 22b99122e.
  Quality loss is only scored on a candidate that was Responsive.
  sarto candidate bench Qwen/Qwen3.8-27B bartowski:Q2_K
exit 1
```
