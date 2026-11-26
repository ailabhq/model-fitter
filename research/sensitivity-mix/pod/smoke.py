#!/usr/bin/env python3
"""CPU smoke test of probe.py on tiny random Qwen3.5 and Inkling 2 Loop models (no weights, no GPU).

Checks: the column permutation equals the pinned converter's own `_reorder_v_heads`; the mapping check passes
against a checkpoint written under Source names; a stubbed rerun from a checkpoint layer gives the same hidden
states as a full forward, with and without a swapped tensor; null probes are 0; probes are > 0.

Usage: MF_LLAMA=<llama.cpp at the pin> MF_GGML_LIB=<its libggml-base> smoke.py WORKDIR
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), os.path.join(os.environ["MF_LLAMA"], "gguf-py"), os.environ["MF_LLAMA"]]
import units as U  # noqa: E402

CFG = Path(os.environ.get("MF_CFG", Path.home() / "mf-bench/sens/cfg"))


def tiny_qwen() -> transformers.PretrainedConfig:
    c = json.loads((CFG / "qwen/config.json").read_text())
    t = c["text_config"]
    t.update(hidden_size=256, intermediate_size=512, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
             head_dim=64, linear_num_key_heads=2, linear_num_value_heads=8, linear_key_head_dim=64,
             linear_value_head_dim=64, vocab_size=1024, mtp_num_hidden_layers=0,
             layer_types=["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(8)])
    t.get("rope_parameters", {}).update(mrope_section=[11, 11, 10])
    v = c["vision_config"]
    v.update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256)
    return transformers.AutoConfig.for_model(**c)


def tiny_inkling() -> transformers.PretrainedConfig:
    c = json.loads((CFG / "inkling/config.json").read_text())
    t = c["text_config"]
    t.update(hidden_size=256, num_hidden_layers=6, vocab_size=1024, unpadded_vocab_size=1000, num_attention_heads=4,
             num_key_value_heads=2, head_dim=64, swa_head_dim=64, swa_num_attention_heads=4, swa_num_key_value_heads=2,
             d_rel=16, rel_extent=64, sliding_window_size=32, local_layer_ids=[0, 2, 3], n_layers_per_attn_res_block=2,
             n_routed_experts=8, num_experts_per_tok=2, intermediate_size=128,
             loop_config={"core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 16, "is_varloop": False})
    c["vision_config"].update(num_hidden_layers=1, hidden_size=64, num_attention_heads=2, projection_hidden_dim=64,
                              text_hidden_size=256)
    c["mtp_config"].update(hidden_size=256, num_hidden_layers=1, intermediate_size=128, local_layer_ids=[0],
                           target_layer_ids=[1], num_attention_heads=4, num_key_value_heads=2, head_dim=64)
    c.pop("quantization_config")
    return transformers.AutoConfig.for_model(**c)


def gguf_names(arch: str, n_layers: int, tcfg: dict, params: dict) -> list:
    roles = {"qwen": ["attn_qkv_gate", "ssm_beta_alpha", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output",
                      "ffn_gate", "ffn_up", "ffn_down"],
             "inkling": ["attn_q", "attn_k", "attn_v", "attn_gate", "attn_output", "attn_rel", "attn_sink_proj",
                         "attn_k_conv_proj", "attn_v_conv_proj", "ffn_gate_up_exps", "ffn_down_exps"]}[arch]
    out = ["token_embd.weight", "output.weight"]
    for L in range(n_layers):
        for r in roles:
            n = f"blk.{L}.{r}.weight"
            try:
                parts = U.ARCH[arch](n, tcfg)
            except KeyError:
                continue
            if all(p[0] in params for p in parts):
                out.append(n)
    return out


def build(arch: str, work: Path) -> tuple:
    torch.manual_seed(0)
    cfg = tiny_qwen() if arch == "qwen" else tiny_inkling()
    model = getattr(transformers, cfg.architectures[0])._from_config(cfg, dtype=torch.bfloat16)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
    mdir, cdir = work / f"{arch}-model", work / f"{arch}-ckpt"
    model.save_pretrained(mdir)
    params = dict(model.named_parameters())
    tcfg = cfg.text_config.to_dict()
    names = gguf_names(arch, cfg.text_config.num_hidden_layers, tcfg, params)
    ck, rows = {}, []
    for n in names:
        parts = U.ARCH[arch](n, tcfg)
        nel = sum(params[t].numel() for t, _, _ in parts)
        width = params[parts[0][0]].shape[-1]
        for t, c, _ in parts:
            ck[c] = params[t].detach().clone().contiguous()
        imat = n not in ("output.weight", "token_embd.weight")
        types = (["iq3_xxs"] if imat and width % 256 == 0 else []) + (["q3_K"] if width % 256 == 0 else []) + ["iq4_nl", "q8_0"]
        for ty in types:
            rows.append("\t".join(map(str, [n, ty, nel, 1.0, 1.0, 0, nel, int(imat)])))
    cdir.mkdir(parents=True, exist_ok=True)
    save_file(ck, cdir / "ckpt.safetensors")
    (cdir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "ckpt.safetensors" for k in ck}}))
    import gzip
    with gzip.open(work / f"{arch}.tsv.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    rng = np.random.default_rng(0)
    with open(work / f"{arch}-calib.jsonl", "w") as f:
        for i, d in enumerate(["chat", "code", "multilingual", "tool_calls"] * 2):
            T = int(rng.integers(40, 90))
            f.write(json.dumps({"id": f"s{i}", "domain": d, "n_prompt": T // 3, "n_response": T - T // 3, "finished": True,
                                "ids": rng.integers(0, 999, T).tolist()}) + "\n")
    return mdir, cdir, model


def check_perm():
    from conversion.qwen import _LinearAttentionVReorderBase as V
    t = torch.arange(3 * 16 * 3 * 128, dtype=torch.float64).reshape(3, 16 * 3 * 128)
    ref = V._reorder_v_heads(t, 1, 16, 3, 128)
    perm = U._v_head_perm(16, 3, 128)
    assert torch.equal(ref, t[:, perm]), "column permutation differs from the converter's"
    print(json.dumps({"check": "v_head_perm equals converter", "ok": True}))


def check_skip(work: Path):
    """Stubbed rerun from a checkpoint equals the full forward (Qwen), with a swapped tensor in a later layer."""
    import probe as P

    class A:
        device, experts_impl, model_path, ckpt_dir, repo, revision = "cpu", "", str(work / "qwen-model"), None, "", ""
    pr = P.Prober(A())
    seqs = [json.loads(x) for x in open(work / "qwen-calib.jsonl")][:3]
    pr.base(seqs, [4])
    with torch.no_grad():
        p = pr.params["model.language_model.layers.5.mlp.down_proj.weight"]
        p.add_(torch.randn_like(p) * 0.05)
        for s, it in enumerate(seqs):
            full = pr.hidden(it["ids"])
            skip = pr.hidden(it["ids"], 4, pr.ckpt[4][s])
            assert torch.equal(full, skip), "stubbed rerun differs from the full forward"
            assert not torch.equal(full, pr.base_h[s])
    print(json.dumps({"check": "checkpoint rerun equals full forward", "ok": True}))


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    check_perm()
    for arch in ("qwen", "inkling"):
        mdir, cdir, _ = build(arch, work)
        out = work / f"{arch}-out"
        cmd = [sys.executable, str(HERE / "probe.py"), arch, "local", "none", str(work / f"{arch}-calib.jsonl"),
               str(work / f"{arch}.tsv.gz"), "ones", str(out), "--model-path", str(mdir), "--ckpt-dir", str(cdir),
               "--device", "cpu", "--tokens", "100000", "--ckpt-every", "4"]
        subprocess.run(cmd, check=True)
        man = json.loads((out / "manifest.json").read_text())
        recs = [json.loads(x) for x in open(out / "probes.jsonl")]
        assert all(v["kld_all"] == 0 for v in man["null_probes"].values()), man["null_probes"]
        assert man["head"]["head_matches_model"], man["head"]
        assert all(r["kld_all"] > 0 for r in recs), [r["unit"] for r in recs if r["kld_all"] <= 0]
        print(json.dumps({"arch": arch, "units": len(recs), "mapping": {k: man["mapping_check"][k] for k in ("value_checked", "size_mismatch")},
                          "null": man["null_probes"], "kld_range": [min(r["kld_all"] for r in recs), max(r["kld_all"] for r in recs)],
                          "types": sorted({r["probe_type"] for r in recs})}))
        gout = work / f"{arch}-groups"
        gcmd = [sys.executable, str(HERE / "group_probe.py"), arch, "local", "none", str(work / f"{arch}-calib.jsonl"),
                str(work / f"{arch}.tsv.gz"), "ones", str(gout), "--model-path", str(mdir), "--ckpt-dir", str(cdir),
                "--device", "cpu", "--tokens", "100000"]
        subprocess.run(gcmd, check=True)
        g2 = work / f"{arch}-groups-q8"
        subprocess.run([*gcmd[:8], str(g2), *gcmd[9:], "--types", "q8_0", "--swap-only"], check=True)
        gm = json.loads((gout / "manifest.json").read_text())
        grs = [json.loads(x) for x in open(gout / "groups.jsonl")]
        assert all(v == 0 for v in gm["null_probes"].values()), gm["null_probes"]
        assert sum(len(r["members"]) for r in grs) == len(recs), "every probed tensor is in exactly one group"
        assert all(sum(r["swap"]) > 0 and sum(r["control"]) > 0 for r in grs)
        assert len({r["group"] for r in grs if "control2" in r}) == len({r["block"] for r in grs if r["block"] is not None})
        print(json.dumps({"arch": arch, "groups": len(grs), "blocks": sorted({r["block"] for r in grs if r["block"] is not None}),
                          "members_max": max(len(r["members"]) for r in grs)}))
        subprocess.run([sys.executable, str(HERE.parent / "sens.py"), str(gout / "groups.jsonl"), "--second", str(g2 / "groups.jsonl"),
                        "--out", str(work / f"{arch}-sens")], check=True)
    check_skip(work)


if __name__ == "__main__":
    main()
