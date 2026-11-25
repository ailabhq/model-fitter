#!/usr/bin/env python3
"""Second-implementation check of a Reference pack with vLLM prompt_logprobs.

Usage: cross_check.py PACK_DIR [--extra-tokens 2000] [vLLM engine args as JSON]

Scores the 8K item plus whole items from the other domains (round-robin, until about --extra-tokens
more positions) and compares vLLM's top-k with the pack's: KL over the shared ids plus a rest bucket,
and top-1 agreement. Passes at mean KLD <= 0.005 and top-1 >= 99% (provisional). Writes cross_check.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from metrics import kld_two_topk, load_pack, item_rows

TOL = {"mean_kld": 0.005, "top1": 0.99}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("--extra-tokens", type=int, default=2000)
    ap.add_argument("--engine", default="{}")
    a = ap.parse_args()
    pack = load_pack(a.pack)
    m = pack["manifest"]
    k = m["k"]
    items = m["items"]

    chosen = [i for i, it in enumerate(items) if it["domain"] == "long_8k"]
    by_dom = {}
    for i, it in enumerate(items):
        if it["domain"] != "long_8k":
            by_dom.setdefault(it["domain"], []).append(i)
    extra = 0
    while extra < a.extra_tokens and any(by_dom.values()):
        for d in list(by_dom):
            if by_dom[d] and extra < a.extra_tokens:
                i = by_dom[d].pop(0)
                chosen.append(i)
                extra += items[i]["n_tokens"] - 1

    t0 = time.perf_counter()
    llm = LLM(model=m["source"]["repo"], revision=m["source"]["revision"], max_logprobs=k,
              max_model_len=max(items[i]["n_tokens"] for i in chosen) + 16, enable_prefix_caching=False,
              **json.loads(a.engine))
    load_s = time.perf_counter() - t0
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=k)
    prompts = []
    for i in chosen:
        ids, _ = item_rows(pack, i)
        prompts.append(TokensPrompt(prompt_token_ids=ids.tolist()))
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    gen_s = time.perf_counter() - t0

    per, kl_all, t1_all, saved = {}, [], [], {"ids": [], "lp": [], "next": [], "items": []}
    for i, o in zip(chosen, outs):
        ids, rows = item_rows(pack, i)
        plp = o.prompt_logprobs[1:]  # entry j + 1 is the distribution that predicted token j + 1
        b_ids = np.zeros((len(plp), k), np.int64)
        b_lp = np.full((len(plp), k), -np.inf)
        b_next = np.array([d[int(t)].logprob for d, t in zip(plp, ids[1:])], np.float32)
        for n, d in enumerate(plp):
            top = sorted(((v.rank, t, v.logprob) for t, v in d.items() if v.rank is not None and v.rank <= k))[:k]
            for j, (_, t, lp) in enumerate(top):
                b_ids[n, j], b_lp[n, j] = t, lp
        kl, t1 = kld_two_topk(pack["topk_ids"][rows], pack["topk_lp"][rows], b_ids, b_lp)
        per[items[i]["id"]] = {"positions": len(kl), "mean_kld": float(kl.mean()), "top1": float(t1.mean()),
                               "mean_next_lp": float(b_next.mean()), "pack_mean_next_lp": float(pack["next_lp"][rows].mean())}
        kl_all.append(kl)
        t1_all.append(t1)
        saved["ids"].append(b_ids.astype(np.int32))
        saved["lp"].append(b_lp.astype(np.float32))
        saved["next"].append(b_next)
        saved["items"].append(items[i]["id"])
    out = a.pack / "cross_check"
    out.mkdir(exist_ok=True)
    for key in ("ids", "lp", "next"):
        np.save(out / f"{key}.npy", np.concatenate(saved[key]))
    (out / "items.json").write_text(json.dumps(saved["items"]))
    kl, t1 = np.concatenate(kl_all), np.concatenate(t1_all)
    res = {"backend": {"name": "vllm", "version": vllm.__version__, "engine": json.loads(a.engine),
                       "dtype": str(llm.llm_engine.model_config.dtype),
                       "quantization": llm.llm_engine.model_config.quantization},
           "positions": int(len(kl)), "mean_kld": float(kl.mean()), "p99_kld": float(np.percentile(kl, 99)),
           "top1": float(t1.mean()), "tolerance": TOL,
           "pass": bool(kl.mean() <= TOL["mean_kld"] and t1.mean() >= TOL["top1"]),
           "items": per, "timing_s": {"load": round(load_s), "score": round(gen_s)}}
    (a.pack / "cross_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "items"}), flush=True)


if __name__ == "__main__":
    main()
