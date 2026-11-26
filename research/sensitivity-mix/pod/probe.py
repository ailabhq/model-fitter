#!/usr/bin/env python3
"""Per-tensor sensitivity probes: one real swap per GGUF tensor of the pinned converter.

For each probe unit (units.py), the unit's transformers parameters are replaced by ggml's own
quantize-then-dequantize at the probe type, with the same imatrix the error tables were built with. The
calibration text is run through, and the final-logit KLD(Source || probed) is taken at every position. The
parameters are then restored. sₜ = KLDₜ / eₜ(probe type) is computed afterwards from probes.jsonl and the tables.

- Base pass: the unchanged Source's final hidden states for every sequence stay on the GPU; base and probed
  logits are both recomputed from hidden states with the model's own head, so no logits are stored.
- Qwen (a plain residual stream): the base pass also keeps every `--ckpt-every`-th layer's input, and a probe in
  layer L reruns only from the checkpoint at or below L. Layers before it are swapped for a stub that hands over
  the stored hidden state, so masks and positions are still built by the model itself.
- Inkling (block attention residuals and a looped core) runs the whole forward pass for every probe.
- A probe of `output.weight` needs no forward pass: only the head changes.
- Null probes (nothing swapped, from layer 0 and from a checkpoint) measure the noise floor first.

Usage: probe.py ARCH SOURCE_REPO REVISION CALIB_JSONL TABLE_TSV_GZ IMATRIX_GGUF OUT_DIR
                [--types iq3_xxs,q3_K,iq4_nl] [--tokens 131072] [--ckpt-every 4] [--limit N]
                [--experts-impl eager] [--device cuda] [--threads N]
Writes OUT_DIR/probes.jsonl (one line per unit, resumable), OUT_DIR/manifest.json.
"""
import argparse
import json
import os
import platform
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn

import ggmlq
import units as U


class Stub(nn.Module):
    """A decoder layer that is skipped: returns its input, or the stored input of the first layer that runs."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        for k in ("layer_type", "attention_type", "layer_idx"):
            if hasattr(layer, k):
                setattr(self, k, getattr(layer, k))
        self.payload = None

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states if self.payload is None else self.payload


def select(calib: list, tokens: int) -> list:
    """Whole items, round-robin over domains in file order, until `tokens` tokens."""
    by = {}
    for it in calib:
        by.setdefault(it["domain"], []).append(it)
    out, n = [], 0
    while n < tokens and any(by.values()):
        for d in list(by):
            if by[d] and n < tokens:
                it = by[d].pop(0)
                out.append(it)
                n += len(it["ids"])
    return out


class Prober:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device)
        path = a.model_path or __import__("huggingface_hub").snapshot_download(a.repo, revision=a.revision)
        self.path = path
        cfg = transformers.AutoConfig.from_pretrained(path)
        cls = getattr(transformers, cfg.architectures[0])
        kw = {"experts_implementation": a.experts_impl} if a.experts_impl else {}
        t0 = time.perf_counter()
        self.model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map=a.device, **kw).eval()
        self.load_s = time.perf_counter() - t0
        self.tcfg = getattr(cfg, "text_config", cfg).to_dict()
        self.params = dict(self.model.named_parameters())
        self.text = getattr(self.model.model, "language_model", self.model.model)
        self.layers = self.text.layers
        self.head0 = self.model.lm_head.weight.detach().clone()

    # ------------------------------------------------------------------ mapping checks
    def check_units(self, us: list) -> dict:
        """Every part exists; part sizes add up to the table's; non-packed parts equal their checkpoint tensor."""
        from safetensors import safe_open
        ck = Path(self.a.ckpt_dir or self.path)
        idx = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
        handles, checked, report = {}, 0, {"units": len(us), "size_mismatch": [], "value_checked": 0, "value_skipped": []}
        for u in us:
            n = 0
            for tname, cname, _ in u["parts"]:
                p = self.params[tname]
                n += p.numel()
                f = idx.get(cname)
                if f is None:
                    raise KeyError(f"{u['gguf']}: checkpoint has no {cname}")
                h = handles.get(f) or handles.setdefault(f, safe_open(str(ck / f), "pt"))
                sl = h.get_slice(cname)
                if sl.get_dtype() not in ("BF16", "F16", "F32"):
                    report["value_skipped"].append(cname)  # packed NVFP4/MXFP8 experts: dequantized on load
                    continue
                ref = h.get_tensor(cname)
                if tuple(ref.shape) != tuple(p.shape) or not torch.equal(ref.to(p.dtype), p.detach().cpu()):
                    raise ValueError(f"{u['gguf']}: {tname} differs from checkpoint {cname}")
                checked += 1
            if n != u["nelem"]:
                report["size_mismatch"].append({"unit": u["gguf"], "params": n, "table": u["nelem"]})
        report["value_checked"] = checked
        return report

    # ------------------------------------------------------------------ forward passes
    def logits(self, h: torch.Tensor, base: bool = False) -> torch.Tensor:
        """The model's own logits from final hidden states; base=True uses the unchanged head."""
        x = torch.nn.functional.linear(h, self.head0 if base else self.model.lm_head.weight).float()
        sc = getattr(self.model, "logit_softcap", None)
        return sc(x) if sc is not None else x

    @torch.inference_mode()
    def hidden(self, ids: list, start: int = 0, payload: torch.Tensor | None = None) -> torch.Tensor:
        """Final (normed) hidden states of one sequence, running layers from `start` with `payload` as its input."""
        x = torch.tensor([ids], device=self.dev)
        saved = {}
        try:
            for i in range(start):
                saved[i] = self.layers[i]
                self.layers[i] = Stub(saved[i])
            if start:
                self.layers[start - 1].payload = payload.to(self.dev, non_blocking=True)
            return self.model.model(input_ids=x, use_cache=False).last_hidden_state[0]
        finally:
            for i, layer in saved.items():
                self.layers[i] = layer

    @torch.inference_mode()
    def base(self, seqs: list, ckpt_layers: list):
        self.base_h, self.ckpt = [], {c: [] for c in ckpt_layers}
        hooks, cur = [], {}
        for c in ckpt_layers:
            def pre(mod, args, kwargs, c=c):
                hs = args[0] if args else kwargs["hidden_states"]
                cur[c] = hs.detach().to("cpu", copy=True)
            hooks.append(self.layers[c].register_forward_pre_hook(pre, with_kwargs=True))
        try:
            for it in seqs:
                self.base_h.append(self.hidden(it["ids"]))
                for c in ckpt_layers:
                    self.ckpt[c].append(cur.pop(c))
        finally:
            for h in hooks:
                h.remove()
        x = torch.tensor([seqs[0]["ids"]], device=self.dev)
        full = self.model(input_ids=x, use_cache=False).logits[0].float()
        return {"head_matches_model": bool(torch.equal(full, self.logits(self.base_h[0], base=True)))}

    @torch.inference_mode()
    def compare(self, seqs: list, start: int | None, chunk: int = 1024) -> dict:
        """KLD(base || current weights) and top-1 agreement over every position; start None = head only."""
        per = []
        for s, it in enumerate(seqs):
            if start is None:
                hp = self.base_h[s]
            elif start == 0:
                hp = self.hidden(it["ids"])
            else:
                hp = self.hidden(it["ids"], start, self.ckpt[start][s])
            hb, T, npr = self.base_h[s], len(it["ids"]), it["n_prompt"]
            kl_all, t1_all = [], []
            for c in range(0, T - 1, chunk):
                e = min(c + chunk, T - 1)
                lb = torch.log_softmax(self.logits(hb[c:e], base=True), -1)
                lp = torch.log_softmax(self.logits(hp[c:e]), -1)
                kl_all.append((lb.exp() * (lb - lp)).sum(-1))
                t1_all.append(lb.argmax(-1) == lp.argmax(-1))
            kl, t1 = torch.cat(kl_all), torch.cat(t1_all)
            r = slice(npr - 1, T - 1)  # row j predicts token j + 1: these rows predict the Source's response
            per.append([float(kl.sum()), T - 1, float(kl[r].sum()), T - npr, int(t1.sum()), int(t1[r].sum()),
                        float(kl.max())])
        return summarize(per, seqs)


def summarize(per: list, seqs: list) -> dict:
    p = np.array(per, dtype=np.float64)
    out = {"kld_all": p[:, 0].sum() / p[:, 1].sum(), "kld_response": p[:, 2].sum() / p[:, 3].sum(),
           "kld_prompt": (p[:, 0] - p[:, 2]).sum() / (p[:, 1] - p[:, 3]).sum(),
           "top1_all": p[:, 4].sum() / p[:, 1].sum(), "top1_response": p[:, 5].sum() / p[:, 3].sum(),
           "kld_max": p[:, 6].max(), "by_domain": {}}
    dom = np.array([it["domain"] for it in seqs])
    for d in sorted(set(dom)):
        m = dom == d
        out["by_domain"][d] = {"kld_all": p[m, 0].sum() / p[m, 1].sum(), "kld_response": p[m, 2].sum() / p[m, 3].sum()}
    out["per_seq"] = [[round(x, 6) if isinstance(x, float) else x for x in row] for row in per]
    return out


def unit_layer(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m[1]) if m else None



def dequant(pr: "Prober", g: ggmlq.GGML, imat: dict | None, u: dict, qtype: str | None = None) -> list:
    """[(param name, dequantized values in the param's layout, relative RMS error)] for one unit at `qtype`
    (default: the unit's probe type). imat None = all-ones imatrix (smoke tests only)."""
    qtype = qtype or u["probe_type"]
    parts = []
    for tname, _cname, perm in u["parts"]:
        w = pr.params[tname].detach().float().cpu().numpy()
        shape, n = w.shape, w.shape[-1]
        w = w.reshape(-1, n)
        if perm is not None:
            w = w[:, perm]  # GGUF column order
        if not u["imatrix"]:
            im = None
        elif imat is None:
            im = np.ones((1, n), np.float32)
        else:
            im = imat[u["gguf"]]
        d = g.roundtrip(qtype, w, im, shape[-2] if len(shape) == 3 else None)
        rel = float(np.sqrt(((d - w) ** 2).sum() / (w ** 2).sum()))
        if perm is not None:
            back = np.empty_like(d)
            back[:, perm] = d
            d = back
        parts.append((tname, torch.from_numpy(d.reshape(shape)), rel))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=sorted(U.ARCH))
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("calib")
    ap.add_argument("table")
    ap.add_argument("imatrix", help="imatrix GGUF, or 'ones' (smoke tests only)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--model-path", help="local model directory instead of the Hub")
    ap.add_argument("--ckpt-dir", help="checkpoint (Source names) for the mapping check, if not the model directory")
    ap.add_argument("--types", default="iq3_xxs,q3_K,iq4_nl")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--ckpt-every", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="regex: probe only matching units")
    ap.add_argument("--experts-impl", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    seqs = select([json.loads(x) for x in open(a.calib)], a.tokens)
    g = ggmlq.GGML(threads=a.threads or None)
    pr = Prober(a)
    tcfg = pr.tcfg
    us = U.units(a.arch, a.table, tcfg, tuple(a.types.split(",")))
    if a.only:
        us = [u for u in us if re.search(a.only, u["gguf"])]
    if a.limit:
        us = us[:a.limit]
    log = lambda **k: print(json.dumps(k, default=float), flush=True)
    log(event="loaded", load_s=round(pr.load_s), seqs=len(seqs), tokens=sum(len(s["ids"]) for s in seqs), units=len(us),
        gpu_gib=round(torch.cuda.memory_allocated() / 2**30, 1) if a.device == "cuda" else None)
    check = pr.check_units(us)
    log(event="mapping", **{k: v if not isinstance(v, list) else v[:8] for k, v in check.items()})
    imat = None if a.imatrix == "ones" else ggmlq.load_imatrix(a.imatrix)

    n_layers = len(pr.layers)
    ckpts = list(range(a.ckpt_every, n_layers, a.ckpt_every)) if a.arch == "qwen" and a.ckpt_every else []
    t0 = time.perf_counter()
    head = pr.base(seqs, ckpts)
    base_s = time.perf_counter() - t0
    log(event="base", s=round(base_s, 1), checkpoints=ckpts, **head)

    def start_of(u):
        if u["gguf"] == "output.weight":
            return None
        L = unit_layer(u["gguf"])
        if L is None or not ckpts:
            return 0
        return max([c for c in ckpts if c <= L], default=0)

    nulls = {}
    for start in [0] + ckpts[len(ckpts) // 2:len(ckpts) // 2 + 1]:
        t0 = time.perf_counter()
        r = pr.compare(seqs, start)
        nulls[str(start)] = {k: r[k] for k in ("kld_all", "kld_response", "top1_all", "kld_max")} | {"s": round(time.perf_counter() - t0, 1)}
        log(event="null", start=start, **nulls[str(start)])

    out = a.out / "probes.jsonl"
    done = {json.loads(x)["unit"] for x in open(out)} if out.exists() else set()
    todo = [u for u in us if u["gguf"] not in done]

    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for u in todo:
            t = time.perf_counter()
            try:
                q.put((u, dequant(pr, g, imat, u), time.perf_counter() - t, None))
            except Exception as e:  # surfaced in the main loop
                q.put((u, None, 0, e))
                return
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t_all = time.perf_counter()
    with open(out, "a") as f:
        for k in range(len(todo)):
            item = q.get()
            u, parts, q_s, err = item
            if err is not None:
                raise err
            backup = {}
            for tname, d, _ in parts:
                p = pr.params[tname]
                backup[tname] = p.detach().clone()
                p.data.copy_(d.to(p.dtype))
            start = start_of(u)
            t = time.perf_counter()
            try:
                r = pr.compare(seqs, start)
            finally:
                for tname, b in backup.items():
                    pr.params[tname].data.copy_(b)
            run_s = time.perf_counter() - t
            rec = {"unit": u["gguf"], "probe_type": u["probe_type"], "table_error": u["table_error"], "start_layer": start,
                   "quant_s": round(q_s, 2), "run_s": round(run_s, 2), "rel_rms": [x[2] for x in parts], **r}
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            el = time.perf_counter() - t_all
            log(event="probe", n=k + 1, of=len(todo), unit=u["gguf"], type=u["probe_type"], kld=r["kld_all"],
                kld_resp=r["kld_response"], start=start, run_s=round(run_s, 1), eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))

    manifest = {"arch": a.arch, "source": {"repo": a.repo, "revision": a.revision}, "probe_types": a.types.split(","),
                "imatrix": os.path.basename(a.imatrix), "table": os.path.basename(a.table),
                "calibration": {"file": os.path.basename(a.calib), "items": [s["id"] for s in seqs],
                                "tokens": sum(len(s["ids"]) for s in seqs), "positions": sum(len(s["ids"]) - 1 for s in seqs)},
                "checkpoints": ckpts, "null_probes": nulls, "mapping_check": check, "head": head,
                "base_s": round(base_s, 1), "load_s": round(pr.load_s),
                "backend": {"transformers": transformers.__version__, "torch": torch.__version__,
                            "cuda": torch.version.cuda, "python": platform.python_version(),
                            "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else a.device,
                            "experts_implementation": a.experts_impl or "default", "weights": "bf16; probed values ggml-dequantized, then rounded to bf16",
                            "kld": "KLD(Source || probed), fp32 log_softmax over the full vocabulary, every position"}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    log(event="done", units=len(todo), total_min=round((time.perf_counter() - t_all) / 60, 1))


if __name__ == "__main__":
    main()
