#!/usr/bin/env python3
import argparse
import json
import subprocess
import threading
import time
import urllib.request

CORPUS = (
    "Local inference benchmark corpus. The engineer measures deterministic decode speed, "
    "prompt processing, memory use, and stable output while preserving model identity. " * 280
)
PROMPT = CORPUS + (
    "\n\nWrite a detailed technical analysis in continuous prose of at least 900 words. "
    "Discuss GPU memory bandwidth, quantization, KV cache precision, fused attention, "
    "and speculative decoding. Do not conclude early."
)


def request_json(url, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer llamacpp"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def gpu_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def rss_kib(pid):
    try:
        with open(f"/proc/{pid}/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenized = request_json(args.url + "/tokenize", {"content": PROMPT, "add_special": True})
    prompt_tokens = len(tokenized.get("tokens", []))
    results = []

    for run in range(args.runs):
        try:
            request_json(args.url + "/slots/0?action=erase", {}, "POST")
        except Exception:
            pass

        stop = threading.Event()
        samples = []

        def sample_memory():
            while not stop.is_set():
                samples.append(
                    {
                        "t": time.perf_counter(),
                        "vram_mib": gpu_used(),
                        "rss_kib": rss_kib(args.pid),
                    }
                )
                stop.wait(0.1)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()

        payload = {
            "prompt": PROMPT,
            "n_predict": args.tokens,
            "temperature": 0.0,
            "seed": 3407,
            "stream": True,
            "cache_prompt": False,
            "ignore_eos": True,
        }
        req = urllib.request.Request(
            args.url + "/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer llamacpp"},
        )

        started = time.perf_counter()
        first = None
        final = None
        text = []

        with urllib.request.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                obj = json.loads(body)
                piece = obj.get("content", "")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
                if obj.get("stop") or obj.get("timings"):
                    final = obj

        finished = time.perf_counter()
        stop.set()
        sampler.join(timeout=2)
        timings = (final or {}).get("timings", {})

        result = {
            "run": run + 1,
            "prompt_tokens": prompt_tokens,
            "wall_s": finished - started,
            "ttft_s": None if first is None else first - started,
            "predicted_tokens": timings.get("predicted_n"),
            "predicted_tps": timings.get("predicted_per_second"),
            "prompt_tps": timings.get("prompt_per_second"),
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "peak_vram_mib": max((x["vram_mib"] for x in samples), default=0),
            "peak_rss_kib": max((x["rss_kib"] for x in samples), default=0),
            "output_prefix": "".join(text)[:240],
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    doc = {
        "label": args.label,
        "server_context": 65536,
        "generation": {
            "n_predict": args.tokens,
            "temperature": 0.0,
            "seed": 3407,
            "ignore_eos": True,
        },
        "prompt_tokens": prompt_tokens,
        "results": results,
    }
    with open(args.output, "w") as output:
        json.dump(doc, output, indent=2)


if __name__ == "__main__":
    main()
