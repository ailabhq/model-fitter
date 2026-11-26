#!/usr/bin/env python3
"""Group sensitivity probes: one role across one eighth of the model's depth at a time, minus a matched control.

The measurement of "How does sarto measure a tensor's sensitivity when one BF16 swap reads mostly noise?":
- A group is every GGUF tensor of one role whose layer falls in one eighth of the depth (Qwen: 8 layers,
  Inkling: 3 physical layers); fewer than 8 layers give one layer per block. A fused tensor is one member;
  `output` and `token_embd` are single-tensor groups.
- Swap: every member replaced by ggml's quantize-then-dequantize at its probe type (probe.dequant), rounded to BF16.
- Control: every member plus Gaussian noise of relative size 1e-3 (σ = 1e-3 × the member's RMS), over the same tokens.
- Only the Source's response positions are scored. Per sequence, the KLD sums of swap and control are kept, so the
  excess and its paired standard error are computed afterwards (sens.py).
- One block per eighth also gets a second control with another seed, to check the paired standard error against
  the control-to-control spread.
- `--swap-only TAG_OF_FIRST_RUN`: a second probe type over the same groups, reusing the first run's controls.

Qwen reruns each group from the stored input of its block's first layer (probe.Prober.hidden); Inkling runs the
whole model. A group of `output` needs only the head.

Usage: group_probe.py ARCH SOURCE_REPO REVISION CALIB_JSONL TABLE_TSV_GZ IMATRIX_GGUF OUT_DIR
                      [--types iq3_xxs,q3_K,iq4_nl] [--tokens 131072] [--swap-only] [--only REGEX] [--limit N]
Writes OUT_DIR/groups.jsonl (one line per group, resumable) and OUT_DIR/manifest.json.
"""
import argparse
import json
import os
import platform
import queue
import re
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import transformers

import ggmlq
import probe as P
import units as U

NOISE = 1e-3


def role_of(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "", name)


def make_groups(us: list) -> list:
    layers = [P.unit_layer(u["gguf"]) for u in us]
    n_layers = max(L for L in layers if L is not None) + 1
    per = n_layers / 8 if n_layers >= 8 else 1
    by: dict = {}
    for u, L in zip(us, layers):
        block = None if L is None else int(L // per)
        by.setdefault((role_of(u["gguf"]), block), []).append(u)
    out = []
    for (role, block), members in sorted(by.items(), key=lambda kv: (kv[0][1] is None, kv[0][1] or 0, kv[0][0])):
        members.sort(key=lambda u: P.unit_layer(u["gguf"]) or 0)
        gid = role if block is None else f"{role}@{block}"
        out.append({"id": gid, "role": role, "block": block, "members": members,
                    "first_layer": None if block is None else P.unit_layer(members[0]["gguf"]),
                    "nelem": sum(u["nelem"] for u in members)})
    return out


class Run:
    def __init__(self, pr: P.Prober, seqs: list, ckpts: list):
        self.pr, self.seqs, self.ckpts = pr, seqs, ckpts

    def start_of(self, grp: dict):
        if grp["id"] == "output.weight":
            return None
        if grp["first_layer"] is None or not self.ckpts:
            return 0
        return max([c for c in self.ckpts if c <= grp["first_layer"]], default=0)

    @torch.inference_mode()
    def response_kld(self, start, chunk: int = 1024) -> np.ndarray:
        """Per sequence: [KLD sum over response positions, response positions, top-1 agreements]."""
        pr, rows = self.pr, []
        for s, it in enumerate(self.seqs):
            if start is None:
                hp = pr.base_h[s]
            elif start == 0:
                hp = pr.hidden(it["ids"])
            else:
                hp = pr.hidden(it["ids"], start, pr.ckpt[start][s])
            T, npr = len(it["ids"]), it["n_prompt"]
            hb, hp = pr.base_h[s][npr - 1:T - 1], hp[npr - 1:T - 1]  # rows predicting the Source's response
            kl, t1 = 0.0, 0
            for c in range(0, hb.shape[0], chunk):
                lb = torch.log_softmax(pr.logits(hb[c:c + chunk], base=True), -1)
                lp = torch.log_softmax(pr.logits(hp[c:c + chunk]), -1)
                kl += float((lb.exp() * (lb - lp)).sum())
                t1 += int((lb.argmax(-1) == lp.argmax(-1)).sum())
            rows.append([kl, T - npr, t1])
        return np.array(rows, dtype=np.float64)

    def with_weights(self, new: dict, start) -> np.ndarray:
        """Response KLD with params replaced by `new` {name: tensor}, restored afterwards."""
        backup = {}
        try:
            for name, d in new.items():
                p = self.pr.params[name]
                backup[name] = p.detach().to("cpu", copy=True)
                p.data.copy_(d.to(p.device, p.dtype))
            return self.response_kld(start)
        finally:
            for name, b in backup.items():
                self.pr.params[name].data.copy_(b.to(self.pr.params[name].device))

    @torch.inference_mode()
    def noised(self, grp: dict, seed: int) -> dict:
        """Every member's params plus N(0, (1e-3 · RMS)²), computed on the GPU, rounded to the param's dtype."""
        out = {}
        for u in grp["members"]:
            for tname, _c, _perm in u["parts"]:
                p = self.pr.params[tname]
                gen = torch.Generator(device=p.device).manual_seed(seed + zlib.crc32(tname.encode()))
                flat = p.detach().reshape(-1)
                sq = sum(float(flat[i:i + 2**26].float().square().sum()) for i in range(0, flat.numel(), 2**26))
                sigma = NOISE * (sq / flat.numel()) ** 0.5
                d = torch.empty_like(p, device="cpu")
                dflat = d.reshape(-1)
                for i in range(0, flat.numel(), 2**26):
                    x = flat[i:i + 2**26].to(torch.float32, copy=True)
                    x += torch.randn(x.shape, generator=gen, device=p.device) * sigma
                    dflat[i:i + 2**26].copy_(x.to(p.dtype))
                out[tname] = d
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=sorted(U.ARCH))
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("calib")
    ap.add_argument("table")
    ap.add_argument("imatrix", help="imatrix GGUF, or 'ones' (smoke tests only)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--model-path")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--types", default="iq3_xxs,q3_K,iq4_nl")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--swap-only", action="store_true", help="second probe type: no controls")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--experts-impl", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log = lambda **k: print(json.dumps(k, default=float), flush=True)

    seqs = P.select([json.loads(x) for x in open(a.calib)], a.tokens)
    g = ggmlq.GGML(threads=a.threads or None)
    pr = P.Prober(a)
    us = U.units(a.arch, a.table, pr.tcfg, tuple(a.types.split(",")))
    groups = make_groups(us)
    if a.only:
        groups = [x for x in groups if re.search(a.only, x["id"])]
    if a.limit:
        groups = groups[:a.limit]
    log(event="loaded", load_s=round(pr.load_s), seqs=len(seqs), tokens=sum(len(s["ids"]) for s in seqs),
        response_positions=sum(len(s["ids"]) - s["n_prompt"] for s in seqs), units=len(us), groups=len(groups),
        gpu_gib=round(torch.cuda.memory_allocated() / 2**30, 1) if a.device == "cuda" else None)
    check = pr.check_units(us)
    log(event="mapping", **{k: v if not isinstance(v, list) else v[:8] for k, v in check.items()})
    imat = None if a.imatrix == "ones" else ggmlq.load_imatrix(a.imatrix)

    # Qwen: a checkpoint at the first layer of every block, so a group reruns from its own block
    starts = sorted({x["first_layer"] for x in groups if x["first_layer"]})
    ckpts = starts if a.arch == "qwen" else []
    t0 = time.perf_counter()
    head = pr.base(seqs, ckpts)
    base_s = time.perf_counter() - t0
    log(event="base", s=round(base_s, 1), checkpoints=ckpts, **head)
    run = Run(pr, seqs, ckpts)
    nulls = {}
    for start in [0] + ckpts[len(ckpts) // 2:len(ckpts) // 2 + 1]:
        t0 = time.perf_counter()
        r = run.response_kld(start)
        nulls[str(start)] = float(r[:, 0].sum() / r[:, 1].sum())
        log(event="null", start=start, kld_response=nulls[str(start)], s=round(time.perf_counter() - t0, 1))
        assert nulls[str(start)] == 0.0, "nothing swapped must give exactly 0"

    # the repeated-seed control: per block, the group with the most weights
    repeat = set()
    if not a.swap_only:
        for b in {x["block"] for x in groups if x["block"] is not None}:
            repeat.add(max((x for x in groups if x["block"] == b), key=lambda x: (x["nelem"], x["id"]))["id"])

    out = a.out / "groups.jsonl"
    done = {json.loads(x)["group"] for x in open(out)} if out.exists() else set()
    todo = [x for x in groups if x["id"] not in done]
    q: queue.Queue = queue.Queue(maxsize=1)

    def producer():
        for grp in todo:
            t = time.perf_counter()
            try:
                parts = [p for u in grp["members"] for p in P.dequant(pr, g, imat, u)]
                q.put((grp, {n: d.to(torch.bfloat16) for n, d, _ in parts}, [r for *_, r in parts], time.perf_counter() - t, None))
            except Exception as e:  # surfaced in the main loop
                q.put((grp, None, None, 0, e))
                return

    threading.Thread(target=producer, daemon=True).start()
    t_all = time.perf_counter()
    with open(out, "a") as f:
        for k in range(len(todo)):
            grp, swapped, rel, q_s, err = q.get()
            if err is not None:
                raise err
            start = run.start_of(grp)
            t = time.perf_counter()
            sw = run.with_weights(swapped, start)
            del swapped
            rec = {"group": grp["id"], "role": grp["role"], "block": grp["block"], "start_layer": start,
                   "members": [u["gguf"] for u in grp["members"]], "probe_types": [u["probe_type"] for u in grp["members"]],
                   "table_errors": [u["table_error"] for u in grp["members"]], "nelem": grp["nelem"], "rel_rms": rel,
                   "swap": sw[:, 0].tolist(), "n_resp": sw[:, 1].astype(int).tolist(), "swap_top1": sw[:, 2].astype(int).tolist()}
            if not a.swap_only:
                seed = a.seed * 1_000_003 + zlib.crc32(grp["id"].encode())
                ct = run.with_weights(run.noised(grp, seed), start)
                rec |= {"control": ct[:, 0].tolist(), "control_top1": ct[:, 2].astype(int).tolist()}
                if grp["id"] in repeat:
                    c2 = run.with_weights(run.noised(grp, seed + 7919), start)
                    rec["control2"] = c2[:, 0].tolist()
            rec |= {"quant_s": round(q_s, 1), "run_s": round(time.perf_counter() - t, 1)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_resp = sum(rec["n_resp"])
            el = time.perf_counter() - t_all
            log(event="group", n=k + 1, of=len(todo), group=grp["id"], members=len(grp["members"]),
                swap=sum(rec["swap"]) / n_resp, control=sum(rec.get("control", [0])) / n_resp, start=start,
                run_s=rec["run_s"], quant_s=rec["quant_s"], eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))

    manifest = {"arch": a.arch, "source": {"repo": a.repo, "revision": a.revision}, "probe_types": a.types.split(","),
                "imatrix": os.path.basename(a.imatrix), "table": os.path.basename(a.table), "swap_only": a.swap_only,
                "noise_relative": NOISE, "seed": a.seed, "repeat_control_groups": sorted(repeat),
                "calibration": {"file": os.path.basename(a.calib), "items": [s["id"] for s in seqs],
                                "tokens": sum(len(s["ids"]) for s in seqs),
                                "response_positions": sum(len(s["ids"]) - s["n_prompt"] for s in seqs)},
                "checkpoints": ckpts, "null_probes": nulls, "mapping_check": check, "head": head,
                "base_s": round(base_s, 1), "load_s": round(pr.load_s),
                "backend": {"transformers": transformers.__version__, "torch": torch.__version__,
                            "cuda": torch.version.cuda, "python": platform.python_version(),
                            "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else a.device,
                            "experts_implementation": a.experts_impl or "default",
                            "weights": "bf16; probed values ggml-dequantized, then rounded to bf16",
                            "kld": "KLD(Source || probed), fp32 log_softmax over the full vocabulary, response positions only"}}
    (a.out / ("manifest.json" if not a.swap_only else "manifest-swap.json")).write_text(json.dumps(manifest, indent=1, default=float))
    log(event="done", groups=len(todo), total_min=round((time.perf_counter() - t_all) / 60, 1))


if __name__ == "__main__":
    main()
