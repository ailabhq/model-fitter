#!/usr/bin/env python3
"""The Source's own temperature-0 responses to the calibration prompts, with vLLM on the GPU host.

Each prompt is rendered with the Source's own chat template in its default thinking mode (tools passed to the
template for tool calls), as the Reference pack builder renders the Prompt set. The response is capped at the
item's max_new_tokens. Writes one JSON line per item: id, domain, n_prompt, n_response, finished, ids.

Usage: gen_calib.py SOURCE_REPO REVISION PROMPTS_JSONL OUT_JSONL [--engine JSON]
"""
import argparse
import json
import time

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--engine", default="{}")
    a = ap.parse_args()
    path = snapshot_download(a.repo, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(path)
    gcfg = GenerationConfig.from_pretrained(path)
    eos = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
    items = [json.loads(line) for line in open(a.prompts)]
    prompts = []
    for it in items:
        text = tok.apply_chat_template(it["messages"], tools=it.get("tools"), tokenize=False, add_generation_prompt=True)
        prompts.append(tok(text, add_special_tokens=False).input_ids)
    t0 = time.perf_counter()
    llm = LLM(model=path, max_model_len=max(map(len, prompts)) + 1100, enable_prefix_caching=False,
              **json.loads(a.engine))
    load_s = time.perf_counter() - t0
    sp = [SamplingParams(temperature=0.0, max_tokens=it["max_new_tokens"], stop_token_ids=eos) for it in items]
    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    gen_s = time.perf_counter() - t0
    with open(a.out, "w") as f:
        for it, p, o in zip(items, prompts, outs):
            r = list(o.outputs[0].token_ids)
            f.write(json.dumps({"id": it["id"], "domain": it["domain"], "n_prompt": len(p), "n_response": len(r),
                                "finished": bool(r and r[-1] in eos), "ids": p + r}) + "\n")
    print(json.dumps({"items": len(items), "load_s": round(load_s), "generate_s": round(gen_s),
                      "tokens": sum(len(p) + len(o.outputs[0].token_ids) for p, o in zip(prompts, outs))}), flush=True)


if __name__ == "__main__":
    main()
