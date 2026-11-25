# PROTOTYPE: how sarto's CLI feels, from `--help` to a finished Fit

Throwaway. Lives only on the `prototype/cli-feel` branch, never merged. It answers
[How should the CLI feel from `--help` to a finished Fit?](https://github.com/ailabhq/sarto/issues/12)
by giving three structurally different command shapes to react to, all running the same
plan run: **Qwen/Qwen3.8-27B on the Mac mini M4 16 GB**.

| Variant | Shape | Feels like |
|---|---|---|
| [A: one verb](A-one-verb.md) | `sarto fit <hf-id>` does the whole plan run; a few commands to look at and start Fits | `brew install` |
| [B: plan, then apply](B-plan-apply.md) | `sarto plan` writes a plan file without downloading anything; `sarto apply` carries it out | `terraform plan/apply` |
| [C: toolbox](C-toolbox.md) | `<noun> <verb>` primitives (`device`, `pack`, `candidate`, `convert`, `fit`); `sarto fit build` chains them | `gh`, `kubectl` |

What the three share, so only the shape differs:

- Everything the decided tickets fix: candidates walked largest first, `llama-fit-params` drops before download,
  benchmark then score, Stall retried once, memory failure skips larger candidates, one sarto conversion last,
  Slow and Degraded mode only at the end, no Fit without a Reference pack, resume by rerunning.
- Every flag has a non-interactive form; the only prompt is the go-ahead before downloads, skipped by `--yes`.
- `--json` on every command, for the agent skill.
- Exit codes: `0` Fit written, `2` no acceptable candidate (report still written), `3` no Reference pack,
  `4` machine not ready (memory pressure above normal at start, or not enough free disk), `1` anything else.

**Numbers:** the device, Source sizes, pack size and the Qwen UD-IQ3_XXS result (7.0 tok/s, KLD 0.060,
top-1 91.8%) come from the research tickets. Every other candidate's result, every timing and every
commit hash is made up to show the flow.

## The Fit on disk (same in all three variants)

```text
~/.sarto/                                    # $SARTO_HOME
├── llama.cpp/22b99122e/                     # pinned build: llama-server, llama-fit-params, scorer
├── packs/Qwen--Qwen3.8-27B/3f1c2a9/p1.c1/   # Reference pack for Source commit 3f1c2a9, Prompt set v1, Calibration set v1 (7 MB)
├── runs/qwen3.8-27b.m4-16gb/                # plan-run state; a rerun resumes here
│   ├── state.json                           # candidates, order, which are done
│   └── candidates/
│       ├── 1-unsloth-IQ3_S/                 # bench.json, monitor.jsonl (0.5 s samples), llama.log
│       ├── 2-unsloth-UD-IQ3_XXS/            # + score.json; run-1/ (Stall) and run-2/ kept apart
│       └── …
└── fits/qwen3.8-27b.m4-16gb/                # the Fit
    ├── fit.json                             # identity and recipe (below)
    ├── model.gguf                           # the chosen candidate's weights, 11.40 GB, moved here, never copied
    ├── serve.sh                             # the exact llama-server command line that was measured
    ├── report.md                            # winner first, then every candidate and why it lost
    └── report.json
```

`fit.json` holds no path outside the Fit directory, no hostname and no user name, so a Fit can be shared
later without scrubbing:

```json
{
  "fit": "qwen3.8-27b.m4-16gb",
  "sarto": "0.1.0",
  "source": { "repo": "Qwen/Qwen3.8-27B", "commit": "3f1c2a9" },
  "device_profile": { "model": "Mac16,10", "chip": "M4", "memory_gb": 16, "gpu_limit_gib": 11.84 },
  "weights": {
    "file": "model.gguf",
    "sha256": "…",
    "origin": { "kind": "sarto-conversion", "mix": "IQ3_S/Q3_K, attn Q4_K", "imatrix": "pack p1.c1" }
  },
  "runtime": {
    "llama_cpp": "22b99122e",
    "args": ["--fit", "on", "-c", "8192", "-ub", "512"]
  },
  "reference_pack": { "repo": "ailabhq/sarto-packs", "prompt_set": 1, "calibration_set": 1, "origin": "published" },
  "result": {
    "usable": true, "mode": "normal",
    "tok_s_8k": 6.4, "prefill_8k_s": 74, "responsive": true, "swapped_out_gib": 3.1,
    "quality_band": "Lossy", "kld_mean": 0.041, "kld_p99": 0.62, "top1": 0.935, "kld_8k_slice": 0.014
  }
}
```

`serve.sh`:

```sh
#!/bin/sh
# sarto Fit qwen3.8-27b.m4-16gb: measured with exactly this command. Changing an argument voids the report.
cd "$(dirname "$0")"
exec "${SARTO_HOME:-$HOME/.sarto}/llama.cpp/22b99122e/bin/llama-server" \
  -m model.gguf --fit on -c 8192 -ub 512 --host 127.0.0.1 --port "${PORT:-8080}" "$@"
```

## Refusals (same in all three variants)

No Reference pack:

```text
$ sarto fit someorg/Brand-New-30B
error: no Reference pack for someorg/Brand-New-30B @ 9e0d771 (Prompt set v1, Calibration set v1)
  ailabhq hasn't published one yet. Without a pack, Quality loss can't be measured, so sarto makes no Fit.
  Build one yourself on a CUDA host (~$3–5 on one H100 80 GB); it stays on that machine:
    sarto pack build someorg/Brand-New-30B --revision 9e0d771
  Or wait for ailabhq: sarto pack status someorg/Brand-New-30B
exit 3
```

Nothing acceptable (Qwen on an 8 GB MacBook Air):

```text
no Fit: no candidate is Usable and not Rejected on this Mac (8 GB, GPU limit 5.33 GiB)
  14 dropped before download   dense model would put layers on the CPU at 8K
   2 measured                  bartowski IQ2_XS 7.9 GB: memory failure · unsloth UD-IQ1_M 6.19 GB: 5.6 tok/s, Rejected (KLD 0.31)
  rerun with --degraded to accept unsloth UD-IQ1_M (Usable, Rejected: KLD 0.31, top-1 71%)
  --slow would not help: no Responsive candidate was too slow.
  report: ~/.sarto/runs/qwen3.8-27b.mba-8gb/report.md
exit 2
```

## What to react to

1. **Shape**: A, B, C, or parts of several (e.g. A's front door with C's primitives underneath).
2. **The go-ahead**: a prompt before multi-hour, 10–70 GB downloads (A, C), or a plan file you look at first (B).
3. **Where the Fit lives**: `~/.sarto/fits/<name>/` with the GGUF inside, or a directory the user picks (`-o ./my-fit`).
4. **Fit names**: `<source>.<device>` (`qwen3.8-27b.m4-16gb`), so a second plan run on the same Mac replaces it; or one name per run.
5. **Starting a Fit**: `sarto serve <fit>` (just `exec`s `serve.sh`), or only `serve.sh` and sarto stays out of the way.
6. **Progress**: one line per step (shown here), or a live full-screen view while a run takes hours.
