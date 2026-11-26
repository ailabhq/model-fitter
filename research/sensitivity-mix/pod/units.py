"""Probe units: one per GGUF tensor of the pinned converter, mapped to the transformers parameters it is made of.

A unit's parts are the transformers parameters (and their Source checkpoint names) that the pinned converter
turns into that one GGUF tensor. ggml quantizes each row on its own, against a per-column imatrix, so a part can
be quantized by itself as long as its columns are in the GGUF tensor's column order. Row order and the split
into parts don't change any row's bytes. The only column reorder in either reference model is Qwen's
`ssm_out` (the V-head reorder of `_LinearAttentionVReorderBase`), given here as `col_perm`.
"""
import gzip
import re

import numpy as np

QWEN_LM = "model.language_model."
INK_LM = "model.language_model."


def _v_head_perm(n_k: int, v_per_k: int, head_dim: int) -> np.ndarray:
    """GGUF column j holds HF column perm[j] (converter: reshape (n_k, v_per_k, d), swap the first two)."""
    return np.arange(n_k * v_per_k * head_dim).reshape(n_k, v_per_k, head_dim).transpose(1, 0, 2).reshape(-1)


def qwen_parts(gguf_name: str, cfg: dict) -> list | None:
    """[(transformers name, checkpoint name, col_perm or None)] for one Qwen3.5 GGUF tensor."""
    if gguf_name == "token_embd.weight":
        return [(QWEN_LM + "embed_tokens.weight",) * 2 + (None,)]
    if gguf_name == "output.weight":
        return [("lm_head.weight", "lm_head.weight", None)]
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf_name)
    if not m or int(m[1]) >= cfg["num_hidden_layers"]:
        return None  # MTP (blk.64, nextn) stays outside the allocation
    p = f"{QWEN_LM}layers.{m[1]}."
    la, sa, mlp = p + "linear_attn.", p + "self_attn.", p + "mlp."
    one = {"attn_q": sa + "q_proj.weight", "attn_k": sa + "k_proj.weight", "attn_v": sa + "v_proj.weight",
           "attn_output": sa + "o_proj.weight", "ffn_gate": mlp + "gate_proj.weight", "ffn_up": mlp + "up_proj.weight",
           "ffn_down": mlp + "down_proj.weight"}
    role = m[2]
    if role in one:
        return [(one[role], one[role], None)]
    if role == "attn_qkv_gate":
        return [(la + "in_proj_qkv.weight",) * 2 + (None,), (la + "in_proj_z.weight",) * 2 + (None,)]
    if role == "ssm_beta_alpha":
        return [(la + "in_proj_b.weight",) * 2 + (None,), (la + "in_proj_a.weight",) * 2 + (None,)]
    if role == "ssm_out":
        n_k, n_v = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        return [(la + "out_proj.weight",) * 2 + (_v_head_perm(n_k, n_v // n_k, cfg["linear_value_head_dim"]),)]
    raise KeyError(gguf_name)


def inkling_parts(gguf_name: str, cfg: dict) -> list | None:
    if gguf_name == "token_embd.weight":
        return [(INK_LM + "embed_tokens.weight", "model.llm.embed.weight", None)]
    if gguf_name == "output.weight":
        return [("lm_head.weight", "model.llm.unembed.weight", None)]
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf_name)
    if not m:
        raise KeyError(gguf_name)
    p, c = f"{INK_LM}layers.{m[1]}.", f"model.llm.layers.{m[1]}."
    table = {
        "attn_q": ("self_attn.q_proj.linear.weight", "attn.wq_du.weight"),
        "attn_k": ("self_attn.k_proj.linear.weight", "attn.wk_dv.weight"),
        "attn_v": ("self_attn.v_proj.linear.weight", "attn.wv_dv.weight"),
        "attn_gate": ("self_attn.o_proj.gate_proj.weight", "attn.gate_attn.weight"),
        "attn_output": ("self_attn.o_proj.linear.weight", "attn.wo_ud.weight"),
        "attn_rel": ("self_attn.relative_position_bias.linear.weight", "attn.wr_du.weight"),
        "attn_sink_proj": ("self_attn.sink_proj.weight", "attn.attn_sink_proj.weight"),
        "attn_k_conv_proj": ("self_attn.k_proj.sconv.weight_proj.weight", "attn.k_sconv.weight_proj"),
        "attn_v_conv_proj": ("self_attn.v_proj.sconv.weight_proj.weight", "attn.v_sconv.weight_proj"),
        "ffn_gate_up_exps": ("mlp.experts.gate_up_proj", "mlp.experts.w13_weight"),
        "ffn_down_exps": ("mlp.experts.down_proj", "mlp.experts.w2_weight"),
    }
    t, k = table[m[2]]
    return [(p + t, c + k, None)]


ARCH = {"qwen": qwen_parts, "inkling": inkling_parts}


def load_table(path: str) -> dict:
    """name -> {"nelem", "imatrix", "types": {type: (bytes, err_a)}} from the table-mix error tables."""
    t: dict = {}
    for line in gzip.open(path, "rt"):
        name, typ, nbytes, err_a, _err_b, _rows, nelem, imat = line.rstrip("\n").split("\t")
        e = t.setdefault(name, {"nelem": int(nelem), "imatrix": imat == "1", "types": {}})
        e["types"][typ] = (int(nbytes), float(err_a))
    return t


def probe_type(entry: dict, preference: tuple) -> str:
    """The first type in `preference` that the tensor's row width (and imatrix) allows.

    IQ3_XXS, else Q3_K where the row width doesn't allow IQ types or there is no imatrix (output, token_embd).
    Inkling's down experts (rows of 1152) allow neither; they take IQ4_NL, the type their mixes land on.
    """
    for t in preference:
        if t in entry["types"]:
            return t
    raise KeyError(f"no probe type among {preference}")


def units(arch: str, table_path: str, cfg: dict, preference: tuple) -> list:
    tab = load_table(table_path)
    out = []
    for name in sorted(tab):
        parts = ARCH[arch](name, cfg)
        if parts is None:
            continue
        e = tab[name]
        pt = probe_type(e, preference)
        out.append({"gguf": name, "parts": parts, "probe_type": pt, "table_error": e["types"][pt][1],
                    "imatrix": e["imatrix"], "nelem": e["nelem"]})
    return out
