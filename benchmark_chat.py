#!/usr/bin/env python3
import argparse, json, subprocess, threading, time, urllib.request

CORPUS = "Local inference benchmark corpus. The engineer measures deterministic decode speed, prompt processing, memory use, and stable output while preserving model identity. "
INSTRUCTION = "\n\nWrite a detailed technical analysis in continuous prose of at least 900 words. Discuss GPU memory bandwidth, quantization, KV cache precision, fused attention, and speculative decoding. Do not conclude early."

def request_json(url, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"), headers={"Content-Type":"application/json", "Authorization":"Bearer llamacpp"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def gpu_used():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=2)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0

def rss_kib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8003")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=280)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    prompt = CORPUS * args.repeats + INSTRUCTION
    props = request_json(args.url + "/props")
    context = props["default_generation_settings"]["n_ctx"]
    model = request_json(args.url + "/v1/models")["data"][0]["id"]
    tok = request_json(args.url + "/tokenize", {"content": prompt, "add_special": True})
    n_prompt = len(tok.get("tokens", []))
    results = []
    for run in range(args.runs):
        try: request_json(args.url + "/slots/0?action=erase", {}, "POST")
        except Exception: pass
        stop = threading.Event(); samples=[]
        def sample():
            while not stop.is_set():
                samples.append({"t":time.perf_counter(), "vram_mib":gpu_used(), "rss_kib":rss_kib(args.pid)})
                stop.wait(0.1)
        th=threading.Thread(target=sample, daemon=True); th.start()
        payload={"model":model,"messages":[{"role":"user","content":prompt}],
                 "max_tokens":args.tokens,"temperature":0.0,"seed":3407,"stream":True,
                 "stream_options":{"include_usage":True},"cache_prompt":False,"ignore_eos":True,
                 "chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"xhigh","preserve_thinking":True}}
        req=urllib.request.Request(args.url+"/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json","Authorization":"Bearer llamacpp"})
        t0=time.perf_counter(); first=None; first_answer=None; final=None; text=[]
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line=raw.decode(errors="replace").strip()
                if not line.startswith("data:"): continue
                body=line[5:].strip()
                if not body or body=="[DONE]": continue
                obj=json.loads(body)
                delta=(obj.get("choices") or [{}])[0].get("delta",{})
                piece=delta.get("reasoning_content","") or delta.get("content","")
                if piece:
                    if first is None: first=time.perf_counter()
                    if delta.get("content") and first_answer is None: first_answer=time.perf_counter()
                    text.append(piece)
                if obj.get("stop") or obj.get("timings"): final=obj
        t1=time.perf_counter(); stop.set(); th.join(timeout=2)
        timings=(final or {}).get("timings",{})
        assert timings.get("predicted_n") == args.tokens, "incomplete generation"
        assert first is not None and timings.get("predicted_per_second", 0) > 0
        r={
          "run":run+1,"prompt_tokens":timings.get("prompt_n",n_prompt),"wall_s":t1-t0,
          "ttft_s":None if first is None else first-t0,
          "first_answer_s":None if first_answer is None else first_answer-t0,
          "predicted_tokens":timings.get("predicted_n"),
          "predicted_tps":timings.get("predicted_per_second"),
          "prompt_tps":timings.get("prompt_per_second"),
          "prompt_ms":timings.get("prompt_ms"),
          "predicted_ms":timings.get("predicted_ms"),
          "peak_vram_mib":max((x["vram_mib"] for x in samples),default=0),
          "peak_rss_kib":max((x["rss_kib"] for x in samples),default=0),
          "output_prefix":"".join(text)[:240]
        }
        results.append(r); print(json.dumps(r), flush=True)
    doc={"label":args.label,"server_context":context,"server_props":props,"generation":{"max_tokens":args.tokens,"temperature":0.0,"seed":3407,"ignore_eos":True,"reasoning_effort":"xhigh","endpoint":"/v1/chat/completions"},"prompt_tokens":n_prompt,"results":results}
    with open(args.output,"w") as f: json.dump(doc,f,indent=2)
if __name__=="__main__": main()
