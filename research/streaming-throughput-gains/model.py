"""Cost model behind research/streaming-throughput-gains.md.

Every output of this script is an ESTIMATE, not a measurement. Run with
`python3 model.py`; it prints the Markdown tables used in the write-up.

Units: GB = 1e9 bytes, seconds, tokens. A "position" is one token the target
model evaluates in a decode step: N sequences x (1 + k drafted tokens).
"""

DRAM_GBPS = 84.0  # 70 % of the M4's 120 GB/s (planning figure)
BUDGET_GB = 10.0  # resident set for weights + per-sequence state (map's ~10 GB)
CONTEXTS = (2048, 8192, 32768)
MIB = 2**20 / 1e9

# ---------------------------------------------------------------- Qwen3.8-27B BF16
Q_STEP_GB = 48.71 + 2.54  # 64 decoder layers + lm_head, read once per decode step
Q_STATE_GB = 149.6 * MIB  # Gated DeltaNet state of one sequence (48 layers, f32)
Q_KV_PER_TOK = 16 * 4096 / 1e9  # 16 full-attention layers x 4 KiB (16-bit KV)
Q_MTP_KV_PER_TOK = 4096 / 1e9  # the MTP layer's own KV
Q_DRAFT_GB = 1.8  # MTP layer + lm_head of mtp-Qwen3.8-27B-Q8_0.gguf [INFERENCE from file size]
Q_GPU_S_PER_TOK = 2 * 25.6e9 / 3.1e12  # 2 FLOP/param at 3.1 TFLOPS (M4 F16, llama.cpp #4167)
# Mean tokens per step (1 + accepted drafts), vLLM, Qwen3.8-27B MTP, H100 NVL (tokenwatt.io)
Q_E = {0: 1.0, 1: 1.87, 2: 2.55, 3: 3.14, 4: 3.64}


def qwen_state_gb(k, rollback):
    """Recurrent state per sequence while drafting k tokens.

    replay   = llama.cpp >= d35dc4d: one state + the last ubatch's inputs
               (163.2 MiB at k=3, 176.8 MiB at k=7, PR #31669)
    snapshot = one state copy per draft position (598.5 MiB at k=3, PR #31669)
    """
    if rollback == "snapshot":
        return Q_STATE_GB * (1 + k)
    return Q_STATE_GB + (3.4 * MIB * (k + 1) if k else 0.0)


def qwen_seq_gb(k, ctx, rollback="replay"):
    kv = Q_KV_PER_TOK * ctx + (Q_MTP_KV_PER_TOK * ctx if k else 0)
    return qwen_state_gb(k, rollback) + kv


def qwen(n, k, ctx, ssd, rollback="replay"):
    draft = Q_DRAFT_GB if k else 0.0
    pinned = BUDGET_GB - draft - n * qwen_seq_gb(k, ctx, rollback)
    if pinned < 0:
        return None
    pinned = min(pinned, Q_STEP_GB)
    t_io = (Q_STEP_GB - pinned) / ssd
    t_gpu = (pinned + n * Q_KV_PER_TOK * ctx + k * Q_DRAFT_GB) / DRAM_GBPS
    t_gpu += n * (k + 1) * Q_GPU_S_PER_TOK
    step = max(t_io, t_gpu)  # reads overlap compute (streaming engine)
    return dict(n=n, k=k, pinned=pinned, step=step, per=Q_E[k] / step,
                tph=3600 * n * Q_E[k] / step, positions=n * (k + 1))


def qwen_nmax(k, ctx, rollback="replay"):
    return int((BUDGET_GB - (Q_DRAFT_GB if k else 0.0)) // qwen_seq_gb(k, ctx, rollback))


# ---------------------------------------------------------------- Inkling 2 Mini
# GB per expert by layer: native = NVFP4 layers 0-19, MXFP8 layers 20-23;
# bf16 = ggml-org's BF16 upcast (3 x 2304 x 1152 x 2 B). Layers 6-17 run three
# times per token (loop_config), so they get three routing draws per token.
I_EXPERT_GB = {
    "native": [4.48e-3 if l < 20 else 8.21e-3 for l in range(24)],
    "bf16": [3 * 2304 * 1152 * 2 / 1e9] * 24,
}
I_PASSES = [3 if 6 <= l < 18 else 1 for l in range(24)]
I_FIXED_GB = 1.37 + 1.85  # bf16 trunk + embed/unembed, always resident
I_DRAFT_GB = 0.37  # dspark-Inkling-2-Mini-Q8_0.gguf (PR #32773)
I_GPU_S_PER_TOK = 0.006  # [ESTIMATE] from measured 104-188 tok/s prefill on this Mac
I_DRAM_GB_PER_STEP = 2.8 + 0.93  # looped trunk + unembed
# Mean tokens per step = 1 + acceptance x k, llama.cpp DSpark on a 16 GB card, PR #32773
I_E = {
    "code": {0: 1.0, 2: 1 + 0.81 * 2, 3: 1 + 0.73 * 3, 4: 1 + 0.685 * 4},
    "reasoning": {0: 1.0, 2: 1 + 0.51 * 2, 3: 1 + 0.40 * 3, 4: 1 + 0.365 * 4},
    "prose": {0: 1.0, 2: 1 + 0.165 * 2, 3: 1 + 0.125 * 3, 4: 1 + 0.10 * 4},
}


def ink_experts_gb(prec):
    return sum(128 * b for b in I_EXPERT_GB[prec])


def ink_seq_gb(k, ctx):
    glob = 8 * 4096 * ctx  # 8 global layer visits
    local = 40 * min(ctx, 1536) * 4096  # 40 sliding visits, llama.cpp default cells
    conv = 3.7 * 2**20 * (1 + k)
    draft_kv = (4096 * ctx + 2 * min(ctx, 1024) * 4096) if k else 0
    return (glob + local + conv + draft_kv) / 1e9


def ink_union_gb(positions, prec):
    """Expert bytes one step needs if routing is uniform and independent [ASSUMPTION]."""
    return sum(128 * b * (1 - (1 - 8 / 128) ** (positions * p))
               for b, p in zip(I_EXPERT_GB[prec], I_PASSES))


def inkling(n, k, ctx, ssd, content="code", prec="native"):
    total = ink_experts_gb(prec)
    draft = I_DRAFT_GB if k else 0.0
    cache = BUDGET_GB - I_FIXED_GB - draft - n * ink_seq_gb(k, ctx)
    if cache < 0:
        return None
    cache = min(cache, total)
    hit = cache / total  # uniform routing: hit rate = cached share
    union = ink_union_gb(n * (k + 1), prec)
    ssd_gb = (1 - hit) * union
    t_io = ssd_gb / ssd
    t_gpu = (I_DRAM_GB_PER_STEP + hit * union + k * I_DRAFT_GB) / DRAM_GBPS
    t_gpu += n * (k + 1) * I_GPU_S_PER_TOK
    step = max(t_io, t_gpu)
    e = I_E[content][k]
    return dict(n=n, k=k, cache=cache, hit=hit, union=union / total, ssd_gb=ssd_gb,
                step=step, per=e / step, tph=3600 * n * e / step, positions=n * (k + 1))


def ink_nmax(k, ctx):
    return int((BUDGET_GB - I_FIXED_GB - (I_DRAFT_GB if k else 0.0)) // ink_seq_gb(k, ctx))


def best_ink(ctx, ssd, content, prec="native"):
    rows = [inkling(n, k, ctx, ssd, content, prec)
            for k in (0, 2, 3, 4) for n in range(1, ink_nmax(k, ctx) + 1)]
    return max((r for r in rows if r), key=lambda r: r["tph"])


def main():
    print("Qwen per-sequence GB (k=0 / k=3 replay / k=3 snapshot):",
          {c: (round(qwen_seq_gb(0, c), 2), round(qwen_seq_gb(3, c), 2),
               round(qwen_seq_gb(3, c, "snapshot"), 2)) for c in CONTEXTS})
    print("Inkling per-sequence GB (k=0 / k=3):",
          {c: (round(ink_seq_gb(0, c), 2), round(ink_seq_gb(3, c), 2)) for c in CONTEXTS})
    print("Inkling expert pool GB:", {p: round(ink_experts_gb(p), 2) for p in I_EXPERT_GB})
    print()
    print("## Q1 Qwen, 8K context, no drafting")
    print("| N | pinned GB | step s @2.2 | tok/s per stream @2.2 | tok/h @2.2 | step s @5 | tok/s per stream @5 | tok/h @5 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in (1, 2, 4, 8, qwen_nmax(0, 8192)):
        a, b = qwen(n, 0, 8192, 2.2), qwen(n, 0, 8192, 5.0)
        print(f"| {n} | {a['pinned']:.1f} | {a['step']:.1f} | {a['per']:.3f} | {a['tph']:,.0f} "
              f"| {b['step']:.1f} | {b['per']:.3f} | {b['tph']:,.0f} |")
    print()
    print("## Q2 Qwen, one stream, 8K context, MTP draft length k")
    print("| k | tokens/step | per-seq GB | tok/s @2.2 | tok/h @2.2 | tok/s @5 | tok/h @5 |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for k in range(5):
        a, b = qwen(1, k, 8192, 2.2), qwen(1, k, 8192, 5.0)
        print(f"| {k} | {Q_E[k]:.2f} | {qwen_seq_gb(k, 8192):.2f} | {a['per']:.3f} | {a['tph']:,.0f} "
              f"| {b['per']:.3f} | {b['tph']:,.0f} |")
    print()
    print("## Q3 Qwen, N at the memory limit, by context and k")
    print("| context | k | N max (replay) | tok/h @2.2 | tok/h @5 | tok/s per stream @2.2 | N max (snapshots) | tok/h @2.2 (snapshots) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in CONTEXTS:
        for k in range(5):
            n = qwen_nmax(k, c)
            a, b = qwen(n, k, c, 2.2), qwen(n, k, c, 5.0)
            ns = qwen_nmax(k, c, "snapshot")
            s = qwen(ns, k, c, 2.2, "snapshot")
            print(f"| {c // 1024}K | {k} | {n} | {a['tph']:,.0f} | {b['tph']:,.0f} | {a['per']:.3f} "
                  f"| {ns} | {s['tph']:,.0f} |")
    print()
    print("## I1 Inkling native, 8K context, no drafting")
    print("| N | expert cache GB | hit | union of experts | SSD GB/step | tok/s per stream @2.2 | tok/h @2.2 | tok/h @5 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in (1, 2, 4, 8, ink_nmax(0, 8192)):
        a, b = inkling(n, 0, 8192, 2.2), inkling(n, 0, 8192, 5.0)
        print(f"| {n} | {a['cache']:.1f} | {a['hit']:.0%} | {a['union']:.0%} "
              f"| {a['ssd_gb']:.2f} | {a['per']:.2f} | {a['tph']:,.0f} | {b['tph']:,.0f} |")
    print()
    print("## I2 Inkling native, 8K context, code acceptance, tok/h @2.2 by k and N")
    ns = (1, 2, 4, 8, ink_nmax(4, 8192))
    print("| k \\ N | " + " | ".join(str(n) for n in ns) + " |")
    print("|---|" + "---:|" * len(ns))
    for k in (0, 2, 3, 4):
        cells = []
        for n in ns:
            r = inkling(n, k, 8192, 2.2, "code")
            cells.append(f"{r['tph']:,.0f}" if r else "–")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print()
    print("## I3 Inkling, best (N, k) by context and content")
    print("| experts | context | content | single stream tok/h @2.2 | best N, k | tok/h @2.2 | tok/h @5 (same N, k) |")
    print("|---|---:|---|---:|---|---:|---:|")
    for prec in ("native", "bf16"):
        for c in CONTEXTS:
            for content in ("code", "reasoning", "prose"):
                one = inkling(1, 0, c, 2.2, content, prec)
                a = best_ink(c, 2.2, content, prec)
                b = inkling(a["n"], a["k"], c, 5.0, content, prec)
                print(f"| {prec} | {c // 1024}K | {content} | {one['tph']:,.0f} | {a['n']}, {a['k']} "
                      f"| {a['tph']:,.0f} | {b['tph']:,.0f} |")


if __name__ == "__main__":
    main()
