#!/usr/bin/env python3
"""Build a draft Reference pack with transformers on a CUDA GPU.

Usage: build_pack.py SOURCE_REPO REVISION PROMPT_SET_DIR OUT_DIR [--k 20] [--batch 16] [--device cuda]

1. Render every Prompt set item with the Source's own chat template (default thinking mode).
2. Generate the Source's temperature-0 response (greedy, capped at the item's max_new_tokens).
3. Run prompt + response through the model once, unbatched, BF16 weights and FP32 logits, and keep
   for every position the top-k next-token ids and log-probs, plus the log-prob of the token that follows.

Pack layout (OUT_DIR):
  manifest.json   Source, commit, Prompt set version and hash, backend versions and precision, per-item index
  ids.npy         int32 [T]       all items' token ids, concatenated
  topk_ids.npy    int32 [R, k]    R = sum(len(item) - 1): row j of an item predicts its token j + 1
  topk_lp.npy     float32 [R, k]  log-probs over the full vocabulary, descending
  next_lp.npy     float32 [R]     log-prob of the token that actually follows
  items.jsonl     id, domain, text (decoded with special tokens) and the Source tokenizer's ids for that text

No credentials, account names or paths are written. Pre-quantized Sources (Inkling's NVFP4/MXFP8) are
dequantized on load by transformers and computed in BF16: that is the reference.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


def render(tok, item, prompt_dir: Path) -> list:
    msgs = json.loads(json.dumps(item["messages"]))
    if "document" in item:
        doc = (prompt_dir / item["document"]).read_text()
        msgs[0]["content"] = msgs[0]["content"].replace("{document}", doc)
    text = tok.apply_chat_template(msgs, tools=item.get("tools"), tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


@torch.inference_mode()
def generate(model, tok, prompts: list, max_new: list, eos: list, batch: int) -> list:
    """Greedy responses. Items are batched by equal max_new_tokens, left-padded."""
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: (max_new[i], len(prompts[i])))
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) == batch or max_new[cur[0]] != max_new[i]):
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    for g in groups:
        width = max(len(prompts[i]) for i in g)
        ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in g], device=model.device)
        mask = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in g], device=model.device)
        t = time.perf_counter()
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new[g[0]], do_sample=False,
                             temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=pad)
        for row, i in zip(gen[:, width:].tolist(), g):
            resp = []
            for x in row:
                resp.append(x)
                if x in eos:
                    break
            out[i] = resp
        print(json.dumps({"generated": [len(out[i]) for i in g], "s": round(time.perf_counter() - t, 1)}), flush=True)
    return out


@torch.inference_mode()
def score(model, ids: list, k: int, chunk: int = 1024):
    """Top-k next-token log-probs at every position of one sequence (no padding, no cache)."""
    x = torch.tensor([ids], device=model.device)
    logits = model(input_ids=x, use_cache=False).logits[0, :-1]  # row j predicts token j + 1
    nxt = x[0, 1:]
    tk_ids, tk_lp, nx_lp = [], [], []
    for s in range(0, logits.shape[0], chunk):
        lp = torch.log_softmax(logits[s:s + chunk].float(), dim=-1)
        v, i = lp.topk(k, dim=-1)
        tk_ids.append(i.int().cpu())
        tk_lp.append(v.cpu())
        nx_lp.append(lp.gather(-1, nxt[s:s + chunk, None])[:, 0].cpu())
    del logits
    return torch.cat(tk_ids).numpy(), torch.cat(tk_lp).numpy(), torch.cat(nx_lp).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("prompt_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")  # cpu only for smoke tests of the pipeline
    a = ap.parse_args()

    prompts_path = a.prompt_dir / "prompts.json"
    ps = json.loads(prompts_path.read_text())
    ps_hash = hashlib.sha256(b"".join((a.prompt_dir / f).read_bytes() for f in
                                      sorted({"prompts.json"} | {it["document"] for it in ps["items"] if "document" in it}))).hexdigest()

    t0 = time.perf_counter()
    path = snapshot_download(a.repo, revision=a.revision)
    dl_s = time.perf_counter() - t0
    cfg = AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.perf_counter()
    model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map=a.device).eval()
    load_s = time.perf_counter() - t0
    gcfg = model.generation_config
    eos = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
    print(json.dumps({"loaded": cls.__name__, "download_s": round(dl_s), "load_s": round(load_s), "eos": eos,
                      "gpu_mem_gib": round(torch.cuda.memory_allocated() / 2**30, 1) if a.device == "cuda" else None}), flush=True)

    items = ps["items"]
    prompt_ids = [render(tok, it, a.prompt_dir) for it in items]
    t0 = time.perf_counter()
    responses = generate(model, tok, prompt_ids, [it["max_new_tokens"] for it in items], eos, a.batch)
    gen_s = time.perf_counter() - t0

    a.out.mkdir(parents=True, exist_ok=True)
    all_ids, all_tki, all_tkl, all_nxt, index = [], [], [], [], []
    row = tok_off = 0
    t0 = time.perf_counter()
    with open(a.out / "items.jsonl", "w") as f:
        for it, p, r in zip(items, prompt_ids, responses):
            ids = p + r
            tki, tkl, nxl = score(model, ids, a.k)
            text = tok.decode(ids, skip_special_tokens=False)
            text_ids = tok(text, add_special_tokens=False).input_ids
            f.write(json.dumps({"id": it["id"], "domain": it["domain"], "text": text, "text_ids": text_ids},
                               ensure_ascii=False) + "\n")
            index.append({"id": it["id"], "domain": it["domain"], "n_tokens": len(ids), "n_prompt": len(p),
                          "n_response": len(r), "finished": bool(r and r[-1] in eos), "token_offset": tok_off,
                          "row_offset": row, "text_ids_match": text_ids == ids,
                          "mean_next_lp": round(float(nxl.mean()), 4), "top1_is_next": round(float((tki[:, 0] == np.array(ids[1:])).mean()), 4)})
            all_ids.append(np.array(ids, np.int32))
            all_tki.append(tki)
            all_tkl.append(tkl)
            all_nxt.append(nxl)
            row += len(ids) - 1
            tok_off += len(ids)
            print(json.dumps(index[-1]), flush=True)
    score_s = time.perf_counter() - t0
    np.save(a.out / "ids.npy", np.concatenate(all_ids))
    np.save(a.out / "topk_ids.npy", np.concatenate(all_tki))
    np.save(a.out / "topk_lp.npy", np.concatenate(all_tkl).astype(np.float32))
    np.save(a.out / "next_lp.npy", np.concatenate(all_nxt).astype(np.float32))

    manifest = {
        "kind": "reference-pack", "draft": True,
        "source": {"repo": a.repo, "revision": a.revision},
        "prompt_set": {"version": ps["prompt_set"], "sha256": ps_hash},
        "k": a.k, "positions": row, "tokens": tok_off,
        "backend": {"name": "transformers", "transformers": transformers.__version__, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else a.device,
                    "python": platform.python_version(),
                    "class": cls.__name__, "weights": "bf16 (pre-quantized tensors dequantized on load)",
                    "logits": "fp32 log_softmax over the full vocabulary", "attention": model.config._attn_implementation,
                    "generation": "greedy, batched with left padding; scoring unbatched without cache"},
        "vocab": int(model.get_output_embeddings().weight.shape[0]),
        "timing_s": {"download": round(dl_s), "load": round(load_s), "generate": round(gen_s), "score": round(score_s)},
        "items": index,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"done": str(a.out.name), "positions": row, "generate_s": round(gen_s), "score_s": round(score_s)}), flush=True)


if __name__ == "__main__":
    main()
