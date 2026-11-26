#!/usr/bin/env python3
"""Dose-response check of the probe signal: does a tensor's KLD grow with its quantization error?

For a few tensors, swap each at several types (and, as a numerical-noise control, a perturbation of relative
size 1e-3) and keep the KLD at every position. Reports, per swap: relative RMS error, mean / median / 99%-trimmed
mean KLD over response and prompt positions, the number of positions above 1 nat, and how much of the mean those
positions carry. Also reports how far the flip positions repeat across swaps.

Usage: diag.py ARCH REPO REVISION CALIB_JSONL TABLE IMATRIX OUT_DIR [--seqs 40] [--units RX] [--types q8_0,q4_K,iq3_xxs,iq2_xxs]
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import ggmlq
import probe as P
import units as U


def main():
    ap = argparse.ArgumentParser()
    for k in ("arch", "repo", "revision", "calib", "table", "imatrix"):
        ap.add_argument(k)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seqs", type=int, default=40)
    ap.add_argument("--units", default=r"^blk\.(0\.ssm_beta_alpha|0\.ffn_down|32\.ffn_down|11\.attn_k)\.weight$")
    ap.add_argument("--types", default="q8_0,q4_K,iq3_xxs,iq2_xxs")
    ap.add_argument("--experts-impl", default="")
    a = ap.parse_args()
    a.device, a.model_path, a.ckpt_dir = "cuda", None, None
    a.out.mkdir(parents=True, exist_ok=True)
    seqs = P.select([json.loads(x) for x in open(a.calib)], 10**9)[:a.seqs]
    g = ggmlq.GGML()
    pr = P.Prober(a)
    imat = ggmlq.load_imatrix(a.imatrix)
    us = [u for u in U.units(a.arch, a.table, pr.tcfg, ("iq3_xxs", "q3_K", "iq4_nl")) if re.search(a.units, u["gguf"])]
    pr.base(seqs, [])
    resp_mask = []
    for it in seqs:
        T, npr = len(it["ids"]), it["n_prompt"]
        m = np.zeros(T - 1, bool)
        m[npr - 1:] = True
        resp_mask.append(m)
    resp_mask = np.concatenate(resp_mask)

    @torch.inference_mode()
    def kl_positions():
        out = []
        for s, it in enumerate(seqs):
            hp, hb, T = pr.hidden(it["ids"]), pr.base_h[s], len(it["ids"])
            for c in range(0, T - 1, 1024):
                e = min(c + 1024, T - 1)
                lb = torch.log_softmax(pr.logits(hb[c:e], base=True), -1)
                lp = torch.log_softmax(pr.logits(hp[c:e]), -1)
                out.append((lb.exp() * (lb - lp)).sum(-1).float().cpu())
        return torch.cat(out).numpy()

    def stats(kl):
        r = {}
        for name, m in (("response", resp_mask), ("prompt", ~resp_mask)):
            x = np.sort(kl[m])
            big = x > 1
            r[name] = {"mean": float(x.mean()), "median": float(np.median(x)),
                       "trimmed99": float(x[:int(len(x) * 0.99)].mean()), "n_over_1": int(big.sum()),
                       "share_over_1": float(x[big].sum() / max(x.sum(), 1e-30))}
        return r

    null = kl_positions()
    rows, flips = [], {}
    print(json.dumps({"event": "null", **stats(null)}), flush=True)
    for u in us:
        variants = [(t, None) for t in a.types.split(",")] + [("noise1e-3", 1e-3)]
        for t, noise in variants:
            if noise is None and t not in U.load_table(a.table)[u["gguf"]]["types"]:
                continue
            backup, rels = {}, []
            for tname, _c, perm in u["parts"]:
                p = pr.params[tname]
                backup[tname] = p.detach().clone()
                w = p.detach().float().cpu().numpy()
                shape, n = w.shape, w.shape[-1]
                w2 = w.reshape(-1, n)
                if noise is not None:
                    d = w2 + np.random.default_rng(0).standard_normal(w2.shape).astype(np.float32) * noise * np.sqrt((w2 ** 2).mean())
                else:
                    ww = w2[:, perm] if perm is not None else w2
                    im = imat.get(u["gguf"]) if u["imatrix"] else None
                    if im is None and g.requires_imatrix(t):
                        continue
                    dd = g.roundtrip(t, ww, im, shape[-2] if len(shape) == 3 else None)
                    if perm is not None:
                        d = np.empty_like(dd)
                        d[:, perm] = dd
                    else:
                        d = dd
                bf = torch.from_numpy(d.reshape(shape)).to(p.dtype)
                rels.append(float(((bf.float().numpy().reshape(-1, n) - w2) ** 2).sum() ** 0.5 / (w2 ** 2).sum() ** 0.5))
                p.data.copy_(bf.to(p.device))
            t0 = time.perf_counter()
            kl = kl_positions()
            for tname, b in backup.items():
                pr.params[tname].data.copy_(b)
            flips[(u["gguf"], t)] = set(np.nonzero(kl > 1)[0].tolist())
            row = {"unit": u["gguf"], "type": t, "rel_rms_bf16": rels, "s": round(time.perf_counter() - t0, 1), **stats(kl)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.save(a.out / f"kl-{u['gguf']}-{t}.npy", kl.astype(np.float32))
    keys = list(flips)
    inter = set.intersection(*flips.values()) if flips else set()
    union = set.union(*flips.values()) if flips else set()
    summary = {"null": stats(null), "rows": rows, "flip_positions_in_every_swap": len(inter), "flip_positions_any": len(union),
               "positions": int(len(resp_mask)), "response_positions": int(resp_mask.sum())}
    (a.out / "diag.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"event": "done", "flip_in_every": len(inter), "flip_any": len(union)}), flush=True)


if __name__ == "__main__":
    main()
