# PROTOTYPE variant B: plan, then apply

Two steps with a file between them. `sarto plan` reads only headers and metadata: no weights download,
no benchmark, done in under a minute. It writes a plan file a person or agent looks at, and may trim.
`sarto apply` carries the plan out without asking anything: the plan file is the go-ahead.
The Fit plan of the glossary becomes a thing you can hold.

## `sarto --help`

```text
sarto: fit a newly released model to this Mac, measured, not guessed.

Usage: sarto <command> [options]

Plan a Fit:
  plan <hf-id>       Write a Fit plan for this Mac: candidates, drops, downloads, time. Downloads no weights.
  apply [<plan>]     Carry out a plan: download, benchmark, score, write the Fit. Rerun to resume.
  status [<plan>]    Where a plan run is: done, running, next candidate

Use a Fit:
  serve <fit>        Start a Fit with the exact llama-server command it was measured with
  show <fit>         A Fit's report
  list               Fits and plans on this Mac
  rm <fit|plan>      Delete one

Run 'sarto <command> --help' for options and examples.
```

## `sarto plan --help`

```text
Usage: sarto plan <hf-id> [options]

Lists the candidates sarto would measure on this Mac, in order, with what llama-fit-params already rules
out, and the downloads and time it would take. Writes the plan to ./<source>.<device>.plan.toml
(or -o). Downloads only GGUF headers and the Reference pack.

Options:
  --revision <commit>   Source revision (default: the repo's main now; pinned into the plan)
  -c, --context <n>     Context length to be Usable at (default: 8192)
  --slow                Plan to accept a Responsive, not-Rejected candidate below 5 tok/s
  --degraded            Plan to accept a Usable, Rejected candidate
  --build <repo>        Add this Community build to the candidates
  --no-convert          Leave out sarto's own conversion
  -o, --out <file>      Where to write the plan
  --json                Print the plan as JSON instead of writing TOML

Examples:
  sarto plan Qwen/Qwen3.8-27B
  sarto plan Qwen/Qwen3.8-27B --no-convert -o qwen.plan.toml && sarto apply qwen.plan.toml
  sarto plan thinkingmachines/Inkling-2-Mini -c 16384 --json
```

## Session: Qwen3.8-27B on the Mac mini M4 16 GB

```text
$ sarto plan Qwen/Qwen3.8-27B
Device   Mac mini M4 · 16 GB · GPU limit 11.84 GiB · 21.3 GB free disk
Source   Qwen/Qwen3.8-27B @ 3f1c2a9 · dense 27B
Pack     ailabhq/sarto-packs p1.c1 · cross-checked ✓ (downloaded, 7 MB)
Headers  24 Community builds read (unsloth, bartowski) · 41 s

 #  candidate                 size      fit-params        download   status
 1  unsloth    IQ3_S          12.04 GB  all layers GPU    ~4 min     measure
 2  unsloth    UD-IQ3_XXS     10.90 GB  all layers GPU    ~4 min     measure if 1 fails
 3  bartowski  IQ3_XXS        10.71 GB  all layers GPU    ~4 min     measure if within 1 GB of the first pass
 4  unsloth    UD-Q2_K_XL     10.15 GB  all layers GPU    ~3 min     measure if within 1 GB of the first pass
 5  bartowski  Q2_K           9.93 GB   all layers GPU    ~3 min     measure if 1–4 all fail
 6  unsloth    UD-IQ2_M       9.40 GB   all layers GPU    ~3 min     measure if 1–5 all fail
 +  sarto conversion          sized later                 ~19 min    measure last (Source, 55.56 GB)
 –  18 builds 12.32–15.48 GB  layers on CPU               –          dropped

Worst case 101 GB downloaded, ~3–4 h. Peak disk ~16 GB of 21.3 GB.
Plan written: ./qwen3.8-27b.m4-16gb.plan.toml
Next: sarto apply qwen3.8-27b.m4-16gb.plan.toml
```

The plan file, trimmable by hand:

```toml
# sarto Fit plan. Delete a [[candidate]] to skip it; apply refuses a plan whose order or rules were edited.
source = "Qwen/Qwen3.8-27B"
revision = "3f1c2a9"
device = "Mac16,10 M4 16GB"
context = 8192
slow = false
degraded = false
llama_cpp = "22b99122e"
pack = { repo = "ailabhq/sarto-packs", prompt_set = 1, calibration_set = 1 }

[[candidate]]
repo = "unsloth/Qwen3.8-27B-GGUF"
commit = "b91e04f"
file = "Qwen3.8-27B-IQ3_S.gguf"
size = 12_040_000_000

[[candidate]]
repo = "unsloth/Qwen3.8-27B-GGUF"
commit = "b91e04f"
file = "Qwen3.8-27B-UD-IQ3_XXS.gguf"
size = 10_900_000_000

# … 4 more …

[convert]
enabled = true
```

```text
$ sarto apply qwen3.8-27b.m4-16gb.plan.toml
[1] unsloth IQ3_S 12.04 GB      memory failure at 0:19 (critical 10 s, cold load)
[2] unsloth UD-IQ3_XXS 10.90 GB Stall (window 931 ms) → second run: 7.0 tok/s · Responsive · KLD 0.060 · Lossy ✓ first pass
[3] bartowski IQ3_XXS 10.71 GB  7.3 tok/s · Responsive · KLD 0.072 · Lossy
[4] unsloth UD-Q2_K_XL 10.15 GB 7.9 tok/s · Responsive · KLD 0.094 · Lossy
[5] bartowski Q2_K              skipped: a candidate passed
[6] unsloth UD-IQ2_M            skipped: a candidate passed
[+] sarto conversion 11.40 GB   6.4 tok/s · Responsive · KLD 0.041 · Lossy ✓ best
Fit qwen3.8-27b.m4-16gb written (2h41m, 99.4 GB downloaded)
  start   sarto serve qwen3.8-27b.m4-16gb
  report  sarto show qwen3.8-27b.m4-16gb
```

Each `[n]` line updates in place while it runs (`download 41% · 2m10s left`, `benchmark: generating, 5.9 tok/s so far`).

A stale plan:

```text
$ sarto apply qwen3.8-27b.m4-16gb.plan.toml
error: the plan is 9 days old and the candidates changed:
  new   unsloth/Qwen3.8-27B-GGUF UD-IQ3_S 11.31 GB (published 2026-11-29)
  Source main moved to 7a20e11; this plan stays on 3f1c2a9.
  Re-plan to include it:  sarto plan Qwen/Qwen3.8-27B
  Apply as written:       sarto apply qwen3.8-27b.m4-16gb.plan.toml --as-written
exit 1
```

`sarto status` while it runs, from another terminal or the agent skill:

```text
$ sarto status
qwen3.8-27b.m4-16gb.plan.toml   running 1h12m · [3] bartowski IQ3_XXS benchmarking · best so far [2] KLD 0.060
```
