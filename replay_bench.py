#!/usr/bin/env python3
"""Agentic replay benchmark: regenerate real assistant turns from local Pi session logs.

For each selected session, walk its assistant turns in order and ask the server to generate that turn from the real
history (system prompt, user turns, earlier assistant turns with reasoning and tool calls, tool results), with the
server's production sampler and prompt caching on, like a live Pi session. Reports decode speed, draft acceptance and
prefill over all turns. Local only: session content never leaves the machine.

usage: replay_bench.py --url URL --out FILE [--sessions N] [--turns-per-session M] [--max-tokens T] [--max-prompt-chars C]
"""
import argparse, glob, json, os, time, urllib.request

SESSIONS = os.environ.get("REPLAY_SESSIONS", os.path.expanduser("~/.pi2/agent/sessions"))


def load(fn):
    msgs = []
    for line in open(fn, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        role = m.get("role")
        parts = m.get("content") if isinstance(m.get("content"), list) else []
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if role == "system":
            sec = m.get("sections") or {}
            msgs.append({"role": "system", "content": m.get("content") or "\n\n".join(str(v) for v in sec.values() if v)})
        elif role == "user":
            msgs.append({"role": "user", "content": text})
        elif role == "assistant":
            calls = [{"id": p.get("id") or f"call{i}", "type": "function",
                      "function": {"name": p.get("name"), "arguments": json.dumps(p.get("arguments") or {}, ensure_ascii=False)}}
                     for i, p in enumerate(parts) if p.get("type") == "toolCall"]
            a = {"role": "assistant", "content": text,
                 "reasoning_content": "\n".join(p.get("thinking", "") for p in parts if p.get("type") == "thinking")}
            if calls:
                a["tool_calls"] = calls
            msgs.append(a)
        elif role == "toolResult":
            msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId"), "content": text})
    return msgs


def tools_from(msgs):
    specs = {}
    for m in msgs:
        for c in m.get("tool_calls", []):
            f = c["function"]
            args = json.loads(f["arguments"]) if f["arguments"] else {}
            p = specs.setdefault(f["name"], {})
            for k, v in args.items():
                p[k] = {"type": "string" if isinstance(v, str) else "number" if isinstance(v, (int, float)) else "object"}
    return [{"type": "function", "function": {"name": n, "description": n,
             "parameters": {"type": "object", "properties": p}}} for n, p in specs.items()]


def post(url, payload):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer llamacpp"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18003")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns-per-session", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-prompt-chars", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    picked = []
    for fn in reversed(files):  # newest first; sessions with enough assistant turns
        msgs = load(fn)
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
        if len(turns) >= 4:
            picked.append((fn, msgs, turns))
        if len(picked) == a.sessions:
            break
    rows = []
    for fn, msgs, turns in picked:
        tools = tools_from(msgs)
        for i in turns[: a.turns_per_session]:
            hist = msgs[:i]
            if sum(len(json.dumps(m)) for m in hist) > a.max_prompt_chars:
                break
            payload = {"messages": hist, "max_tokens": a.max_tokens, "seed": a.seed + i, "cache_prompt": True,
                       "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh", "preserve_thinking": True}}
            if tools:
                payload["tools"] = tools
            t0 = time.monotonic()
            r = post(a.url, payload)
            t = r.get("timings", {})
            rows.append({"session": os.path.basename(fn)[:24], "turn": i, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                         "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                         "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"), "wall_s": time.monotonic() - t0})
    pn = sum(r["predicted_n"] or 0 for r in rows); pms = sum(r["predicted_ms"] or 0 for r in rows)
    dn = sum(r["draft_n"] or 0 for r in rows); da = sum(r["draft_n_accepted"] or 0 for r in rows)
    ppn = sum(r["prompt_n"] or 0 for r in rows); ppms = sum(r["prompt_ms"] or 0 for r in rows)
    summary = {"turns": len(rows), "decode_tok_s": 1000 * pn / pms if pms else None, "draft_acceptance": da / dn if dn else None,
               "tokens_generated": pn, "prefill_tok_s": 1000 * ppn / ppms if ppms else None, "prompt_tokens_evaluated": ppn}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
