#!/usr/bin/env python3
"""Build a reduced draft vocabulary for --spec-draft-vocab (Qwen3.8 MTP).

Counts token frequencies in your own text and writes token ids, most frequent first, with the chat-template control
tokens always at the top. llama-server loads the first --spec-draft-vocab-n ids (16384 in the recipe) as the MTP draft
head's rows. Tokenization uses a running llama-server with the same model (its /tokenize endpoint).

  python3 build-draft-vocab.py --url http://127.0.0.1:8003 --pi-sessions ~/.pi/agent/sessions --out draft-vocab.txt
  python3 build-draft-vocab.py --url http://127.0.0.1:8003 --text wiki.train.raw src/**/*.cpp --out draft-vocab.txt

Pi session logs (*.jsonl) contribute only assistant output (reasoning, text and tool calls rendered like the Qwen3.8
template), which is what the draft model has to predict.
"""
import argparse, collections, glob, json, os, urllib.request

CONTROL = range(248044, 248077)  # Qwen3.8 special tokens (<|im_end|>, <think>, </think>, <tool_call>, ...)
CHUNK = 1 << 20                   # characters per /tokenize request


def render_assistant(m):
    think, text, calls = [], [], []
    for c in m.get("content") if isinstance(m.get("content"), list) else []:
        t = c.get("type")
        if t == "thinking" and isinstance(c.get("thinking"), str):
            think.append(c["thinking"])
        elif t == "text" and isinstance(c.get("text"), str):
            text.append(c["text"])
        elif t == "toolCall":
            calls.append(c)
    s = "<think>\n" + "\n".join(think).strip() + "\n</think>\n\n" + "\n".join(text).strip()
    for i, c in enumerate(calls):
        s += ("\n\n" if i == 0 and s.strip() else "\n" if i else "") + "<tool_call>\n<function=" + str(c.get("name")) + ">\n"
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            for k, v in args.items():
                s += "<parameter=" + k + ">\n" + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) + "\n</parameter>\n"
        s += "</function>\n</tool_call>"
    return s + "<|im_end|>\n"


def pi_session_texts(root):
    for fn in sorted(glob.glob(os.path.join(os.path.expanduser(root), "**", "*.jsonl"), recursive=True)):
        for line in open(fn, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message") or {}
            if d.get("type") == "message" and m.get("role") == "assistant":
                yield render_assistant(m)


def file_texts(paths):
    for p in paths:
        for fn in glob.glob(os.path.expanduser(p), recursive=True):
            if os.path.isfile(fn):
                yield open(fn, errors="replace").read()


def tokenize(url, api_key, text):
    req = urllib.request.Request(url + "/tokenize", data=json.dumps({"content": text, "add_special": False, "parse_special": True}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["tokens"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8003")
    ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", "llamacpp"))
    ap.add_argument("--pi-sessions", action="append", default=[], help="directory with Pi session *.jsonl logs (repeatable)")
    ap.add_argument("--text", nargs="*", default=[], help="plain text files or globs")
    ap.add_argument("--min-ids", type=int, default=65536, help="write at least this many ids (unseen ids by id order)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    counts, buf, total = collections.Counter(), [], 0
    def flush():
        nonlocal total
        if buf:
            ids = tokenize(a.url, a.api_key, "".join(buf))
            counts.update(ids); total += len(ids); buf.clear()
    sources = [t for d in a.pi_sessions for t in pi_session_texts(d)] if a.pi_sessions else []
    for text in [*sources, *file_texts(a.text)]:
        buf.append(text)
        if sum(map(len, buf)) >= CHUNK:
            flush()
    flush()
    if not total:
        raise SystemExit("no text found")

    control = list(CONTROL)
    ranked = [t for t, _ in counts.most_common() if t not in CONTROL]
    seen = set(control) | set(ranked)
    tail = [t for t in range(248320) if t not in seen][: max(0, a.min_ids - len(control) - len(ranked))]
    order = control + ranked + tail
    with open(a.out, "w") as f:
        f.write(f"# Qwen3.8 MTP reduced draft vocabulary: token ids, most frequent first (control tokens first). {total} tokens counted.\n")
        for i in range(0, len(order), 16):
            f.write(" ".join(map(str, order[i:i + 16])) + "\n")
    print(f"{a.out}: {len(order)} ids ({len(ranked)} seen) from {total} tokens")


if __name__ == "__main__":
    main()
