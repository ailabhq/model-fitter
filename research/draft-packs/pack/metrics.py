"""Quality loss metrics against a top-k Reference pack (numpy only; used on the GPU host and on the Mac).

KLD(Source || Fit) is computed on a coarsened distribution: the Source's top-k tokens plus one "rest"
bucket holding everything else. Coarsening never increases KL, so this is a lower bound on the
full-vocabulary KLD; with k = 20 the gap is small because the rest bucket carries little Source mass.
"""
import json
from pathlib import Path

import numpy as np

# Provisional Quality bands ("What Quality loss report does every Fit carry ...").
BANDS = [("Near-lossless", 0.02, 0.97), ("Lossy", 0.10, 0.88)]
EPS = 1e-10


def load_pack(d: Path) -> dict:
    d = Path(d)
    m = json.loads((d / "manifest.json").read_text())
    items = [json.loads(l) for l in (d / "items.jsonl").read_text().splitlines()]
    return {"manifest": m, "items": items, "ids": np.load(d / "ids.npy"), "topk_ids": np.load(d / "topk_ids.npy"),
            "topk_lp": np.load(d / "topk_lp.npy"), "next_lp": np.load(d / "next_lp.npy")}


def item_rows(pack: dict, i: int) -> tuple:
    it = pack["manifest"]["items"][i]
    ids = pack["ids"][it["token_offset"]:it["token_offset"] + it["n_tokens"]]
    rows = slice(it["row_offset"], it["row_offset"] + it["n_tokens"] - 1)
    return ids, rows


def kld_topk(ref_lp: np.ndarray, fit_lp_at_ref: np.ndarray) -> np.ndarray:
    """Per-position KL over (Source top-k ids + rest). ref_lp, fit_lp_at_ref: [N, k] natural-log probs."""
    p = np.exp(ref_lp.astype(np.float64))
    q = np.exp(fit_lp_at_ref.astype(np.float64))
    p_rest = np.clip(1.0 - p.sum(-1), 0.0, 1.0)
    q_rest = np.clip(1.0 - q.sum(-1), EPS, 1.0)
    head = (p * (ref_lp - fit_lp_at_ref)).sum(-1)
    rest = np.where(p_rest > EPS, p_rest * (np.log(np.maximum(p_rest, EPS)) - np.log(q_rest)), 0.0)
    return np.maximum(head + rest, 0.0)


def top1_agree(ref_ids: np.ndarray, ref_lp: np.ndarray, fit_top: np.ndarray) -> np.ndarray:
    """The Fit's top token is one of the Source's top tokens; exact ties are common in BF16 logits."""
    return ((ref_ids == fit_top[:, None]) & (ref_lp == ref_lp[:, :1])).any(-1)


def kld_two_topk(a_ids, a_lp, b_ids, b_lp) -> tuple:
    """KL(a || b) when both sides only have their own top-k: coarsen to (shared ids + rest). Also top-1 match."""
    kl = np.empty(len(a_ids))
    for n in range(len(a_ids)):
        bm = dict(zip(b_ids[n].tolist(), b_lp[n].tolist()))
        shared = [j for j, t in enumerate(a_ids[n].tolist()) if t in bm]
        pa = np.exp(a_lp[n, shared].astype(np.float64))
        qb = np.exp(np.array([bm[a_ids[n, j]] for j in shared], np.float64))
        pr, qr = max(1 - pa.sum(), 0.0), max(1 - qb.sum(), EPS)
        kl[n] = max((pa * (np.log(pa) - np.log(qb))).sum() + (pr * (np.log(max(pr, EPS)) - np.log(qr)) if pr > EPS else 0.0), 0.0)
    return kl, top1_agree(a_ids, a_lp, b_ids[:, 0]) | top1_agree(b_ids, b_lp, a_ids[:, 0])


def summary(kld: np.ndarray, top1: np.ndarray) -> dict:
    return {"positions": int(len(kld)), "mean_kld": float(kld.mean()), "p99_kld": float(np.percentile(kld, 99)),
            "median_kld": float(np.median(kld)), "top1": float(top1.mean())}


def band(s: dict) -> str:
    for name, kmax, tmin in BANDS:
        if s["mean_kld"] <= kmax and s["top1"] >= tmin:
            return name
    return "Rejected"


def report(pack: dict, per_item: dict) -> dict:
    """per_item: id -> (kld [n], top1 [n]). Whole corpus, per domain, the 8K slice, and the band (worse of whole and 8K)."""
    by_dom: dict = {}
    for it in pack["manifest"]["items"]:
        if it["id"] in per_item:
            by_dom.setdefault(it["domain"], []).append(per_item[it["id"]])
    cat = lambda xs, j: np.concatenate([x[j] for x in xs])
    allx = [v for vs in by_dom.values() for v in vs]
    whole = summary(cat(allx, 0), cat(allx, 1))
    doms = {d: summary(cat(v, 0), cat(v, 1)) for d, v in by_dom.items()}
    long_ = doms.get("long_8k")
    order = ["Near-lossless", "Lossy", "Rejected"]
    b = max([band(whole)] + ([band(long_)] if long_ else []), key=order.index)
    return {"whole": whole, "domains": doms, "band_whole": band(whole), "band_8k": band(long_) if long_ else None, "band": b}
