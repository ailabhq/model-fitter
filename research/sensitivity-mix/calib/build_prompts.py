#!/usr/bin/env python3
"""Draw the stand-in calibration prompts from public, permissively licensed datasets.

The Calibration set itself is still open; this is the stand-in of "Does a sensitivity-weighted mix lose no more
than Unsloth's UD-IQ3_XXS at the same size, for both reference models?". Four domains, as in the Prompt set:

  chat          OpenAssistant/oasst2 (Apache-2.0): English root prompts
  multilingual  OpenAssistant/oasst2 (Apache-2.0): non-English root prompts, round-robin over languages
  code          ise-uiuc/Magicoder-OSS-Instruct-75K (MIT): `problem`, round-robin over languages
  tool_calls    glaiveai/glaive-function-calling-v2 (Apache-2.0): the first user turn with its function schemas

Rows come from the Hugging Face datasets server at pinned dataset commits, in dataset order, after fixed
filters, so the draw is reproducible. Every prompt is checked against Prompt set v0: no identical prompt and no
shared run of 8 words (or 16 characters for CJK). Writes prompts.jsonl and overlap.json next to this file.

Usage: build_prompts.py PROMPT_SET_JSON [--per-domain 56]
"""
import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://datasets-server.huggingface.co"
SOURCES = {
    "OpenAssistant/oasst2": {"license": "apache-2.0", "revision": "179dd21f"},
    "ise-uiuc/Magicoder-OSS-Instruct-75K": {"license": "mit", "revision": "5f839b1f"},
    "glaiveai/glaive-function-calling-v2": {"license": "apache-2.0", "revision": "e7f4b645"},
}
ML_LANGS = ["es", "de", "fr", "ru", "zh", "pt-BR", "ja", "it", "ca", "uk", "pl", "ko", "vi", "id"]
CODE_LANGS = ["python", "cpp", "java", "typescript", "rust", "shell", "php", "csharp", "swift", "javascript"]


def get(path: str, **q) -> dict:
    url = f"{API}/{path}?" + urllib.parse.urlencode(q)
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:
            if attempt == 7:
                raise
            time.sleep(5 * 2 ** attempt)


def rows(dataset: str, where: str | None = None, start: int = 0, page: int = 100):
    off = start
    while True:
        if where:
            j = get("filter", dataset=dataset, config="default", split="train", where=where, offset=off, length=page)
        else:
            j = get("rows", dataset=dataset, config="default", split="train", offset=off, length=page)
        rs = j.get("rows", [])
        if not rs:
            return
        for r in rs:
            yield r["row_idx"], r["row"]
        off += len(rs)


def ok_text(t: str, lo: int = 40, hi: int = 2000) -> bool:
    return lo <= len(t) <= hi and "http" not in t


def oasst(langs: list, n: int) -> list:
    """Root prompts, first n per the language list's round-robin."""
    per = {lang: [] for lang in langs}
    need = {lang: 2 * -(-n // len(langs)) if len(langs) > 1 else n for lang in langs}  # spare for thin languages
    for lang in langs:
        where = f"\"parent_id\" IS NULL AND \"lang\"='{lang}' AND \"deleted\"=false AND \"review_result\"=true"
        for idx, r in rows("OpenAssistant/oasst2", where):
            if ok_text(r["text"]):
                per[lang].append((idx, r))
            if len(per[lang]) >= need[lang]:
                break
    out = []
    for i in range(max(map(len, per.values()), default=0)):
        for lang in langs:
            if i < len(per[lang]) and len(out) < n:
                idx, r = per[lang][i]
                out.append({"source": "OpenAssistant/oasst2", "row": idx, "source_id": r["message_id"], "lang": lang,
                            "messages": [{"role": "user", "content": r["text"].strip()}]})
    return out


def magicoder(n: int) -> list:
    per = {lang: [] for lang in CODE_LANGS}
    need = 2 * -(-n // len(CODE_LANGS))  # spare for thin languages
    for lang in CODE_LANGS:
        for idx, r in rows("ise-uiuc/Magicoder-OSS-Instruct-75K", f"\"lang\"='{lang}'"):
            if ok_text(r["problem"], 200, 2500):
                per[lang].append((idx, r))
            if len(per[lang]) >= need:
                break
    out = []
    for i in range(need):
        for lang in CODE_LANGS:
            if i < len(per[lang]) and len(out) < n:
                idx, r = per[lang][i]
                out.append({"source": "ise-uiuc/Magicoder-OSS-Instruct-75K", "row": idx, "source_id": r["index"], "lang": lang,
                            "messages": [{"role": "user", "content": r["problem"].strip()}]})
    return out


def glaive_tools(system: str) -> list:
    body = system.split("Use them if required -", 1)
    if len(body) != 2:
        return []
    s, dec, tools = body[1], json.JSONDecoder(), []
    i = 0
    while True:
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            return tools
        try:
            obj, i = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict) or "name" not in obj:
            return []
        tools.append({"type": "function", "function": obj})


def glaive(n: int) -> list:
    out, seen = [], set()
    for idx, r in rows("glaiveai/glaive-function-calling-v2"):
        tools = glaive_tools(r["system"])
        m = re.match(r"USER: (.*?)\n\n\n(?:ASSISTANT|FUNCTION)", r["chat"], re.S)
        if not tools or not m or not ok_text(m[1].strip(), 20, 1500):
            continue
        key = (tuple(sorted(t["function"]["name"] for t in tools)), m[1].strip().lower())
        names = tuple(sorted(t["function"]["name"] for t in tools))
        if key in seen or sum(1 for o in out if o["tool_names"] == list(names)) >= 2:
            continue  # at most two prompts per tool set, for variety
        seen.add(key)
        out.append({"source": "glaiveai/glaive-function-calling-v2", "row": idx, "source_id": str(idx), "lang": "en",
                    "tool_names": list(names), "tools": tools, "messages": [{"role": "user", "content": m[1].strip()}]})
        if len(out) >= n:
            return out
    return out


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def shingles(t: str) -> set:
    t = norm(t)
    words = t.split()
    out = {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}
    cjk = re.sub(r"\s", "", t)
    if any("\u3040" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af" for c in cjk):
        out |= {cjk[i:i + 16] for i in range(len(cjk) - 15)}
    return out


def prompt_set_texts(path: Path) -> list:
    ps = json.loads(path.read_text())
    texts = []
    for it in ps["items"]:
        for m in it["messages"]:
            c = m["content"]
            if "document" in it:
                c = c.replace("{document}", (path.parent / it["document"]).read_text())
            texts.append((it["id"], c))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_set", type=Path)
    ap.add_argument("--per-domain", type=int, default=56)
    a = ap.parse_args()
    n = a.per_domain
    items = []
    for domain, got in (("chat", oasst(["en"], n)), ("multilingual", oasst(ML_LANGS, n)), ("code", magicoder(n)),
                        ("tool_calls", glaive(n))):
        assert len(got) == n, (domain, len(got))
        for i, it in enumerate(got):
            items.append({"id": f"cal-{domain}-{i:03d}", "domain": domain, "max_new_tokens": 1024, **it})

    ref = prompt_set_texts(a.prompt_set)
    ref_norm = {norm(t) for _, t in ref}
    ref_sh = set().union(*(shingles(t) for _, t in ref))
    hits = []
    for it in items:
        text = it["messages"][0]["content"]
        sh = shingles(text)
        if norm(text) in ref_norm or sh & ref_sh:
            hits.append({"id": it["id"], "identical": norm(text) in ref_norm, "shared": sorted(sh & ref_sh)[:5]})
    report = {"prompt_set": json.loads(a.prompt_set.read_text())["prompt_set"], "calibration_prompts": len(items),
              "rule": "no identical prompt (NFKC, case and punctuation folded) and no shared 8-word run (16 characters for CJK)",
              "prompt_set_texts": len(ref), "overlaps": hits, "sources": SOURCES}
    (HERE / "overlap.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    if hits:
        raise SystemExit(f"{len(hits)} calibration prompts overlap the Prompt set: see overlap.json")
    with open(HERE / "prompts.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(items), "overlaps": 0, "by_domain": {d: sum(i["domain"] == d for i in items) for d in
                                                                         ("chat", "multilingual", "code", "tool_calls")}}))


if __name__ == "__main__":
    main()
