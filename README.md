# Qwen3.8 27B on a 5070 Ti: the setup that actually worked

This is the local Qwen setup I ended up with after benchmarking the obvious stuff and a few less-obvious things on my desktop.

The short version: **llama.cpp auto-fit was quietly costing me a ton of speed.** Forcing the target fully onto the GPU took decode from ~29 to ~44 tok/s. Then the Qwen GGUF's own embedded MTP head took it to ~69 tok/s.

No model swap, no tiny context, no "technically faster" config that makes the model noticeably worse. Same Q3_K abliterated target model, 65K context, single RTX 5070 Ti.

## My box

This recipe is specifically from this machine, so don't blindly assume the VRAM thresholds will be right for yours.

- **GPU:** MSI / NVIDIA GeForce RTX 5070 Ti, GB203, 16,303 MiB VRAM, compute capability 12.0
- **CPU:** Ryzen 7 9800X3D, 8C/16T, 96 MiB L3
- **RAM:** 32 GB ADATA DDR5-6000, ~30.23 GiB usable, 8 GiB swap
- **OS:** Ubuntu 26.04 LTS, kernel 7.0.0-30-generic
- **NVIDIA driver:** 580.173.02
- **CUDA:** driver compatibility 13.0, toolkit/runtime 13.1.115
- **Pi:** 0.84.2
- **llama.cpp:** build 10505, commit `ee4c505a4fb37be8ea37a78af272e74dad2835c1`, Release, native `sm_120a`, CUDA graphs on, CUDA FlashAttention on
- **Model:** `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`
- **File:** `Huihui-Qwen3.8-27B-abliterated-Q3_K.gguf`
- **SHA-256:** `a82fe31ff8716377d1dd2f47e1dc1847b96566ddcfa9a3c394a1b6ecf7adc75c`
- **Context:** 65,536
- **Max output in Pi:** 16,384
- **KV cache:** Q4_0 K/V
- **Batch / ubatch:** 512 / 128
- **Parallel slots:** 1

The GGUF is a dense 27B model with 64 target layers plus one embedded MTP layer. Native context is 262,144, but I run 65K because this whole thing is being squeezed onto one 16 GB card while still leaving enough room for the desktop and speculation.

## What moved the needle

I benchmarked every serious candidate with the same 7,322-token prompt, 65,536 server context, 512 generated tokens, temperature 0, seed 3407, erased slot cache, three runs each.

| setup | decode | TTFT | peak VRAM | wall time |
|---|---:|---:|---:|---:|
| original Q3_K + auto-fit | 29.05 tok/s | 7.95 s | 14,710 MiB | 25.54 s |
| Q3_K, all target layers on GPU | 44.38 tok/s | 5.43 s | 14,605 MiB | 16.94 s |
| **Q3_K + embedded MTP, n=2** | **69.05 tok/s** | **5.58 s** | **15,315 MiB** | **12.98 s** |
| Q3_K + MTP, n=3 | 63.49 tok/s | 5.62 s | 15,497 MiB | 13.67 s |
| Q2_K + MTP, n=2 | 65.98 tok/s | 6.47 s | 12,921 MiB | 14.21 s |
| Q2_K + DFlash2 Q4 | 47.77 tok/s | 6.54 s | 14,533 MiB | 17.24 s |

Compared with the service I started with, the final config was:

- **+137.7% decode throughput**
- **-29.8% TTFT**
- **-49.2% total wall time**

The fun part is that Q2_K is faster if you run plain autoregressive decode, but once MTP is enabled the Q3_K target wins anyway. So dropping another quant level would've bought me worse output *and* less speed. Nice deal.

## The actual problem

The original service used:

```text
--n-gpu-layers auto
--fit on
--fit-target 1024
```

That sounds sensible on a tight card, except desktop VRAM usage moves around. llama.cpp would react to that pressure by changing placement and pushing work off the GPU. On this machine that meant roughly **29 tok/s instead of 44 tok/s**.

The fastest stable path is:

```text
--n-gpu-layers all
--fit off
--spec-type draft-mtp
--spec-draft-n-max 2
--spec-draft-p-min 0
--spec-draft-type-k q4_0
--spec-draft-type-v q4_0
```

At the benchmark prompt the embedded MTP path accepted about **60.4% of drafts**, with a mean verified span around 2.21 tokens. That's enough to make a very real difference.

## Don't hardcode the psycho-fast mode

MTP at 65K is tight enough that I don't just force it every boot and pray Chrome behaves.

The launcher in [`pi2-llama-server`](pi2-llama-server) checks free VRAM and picks one of three modes:

1. **`mtp`** at >= 15,150 MiB free: all target layers on GPU + embedded MTP n=2
2. **`base`** at >= 14,500 MiB free: all target layers on GPU, no draft cache
3. **`fit`** below that: safe auto placement with a 512 MiB target margin

So under normal conditions I get the ~69 tok/s path. If a browser or some other GPU app has eaten too much VRAM, it backs off instead of OOMing.

You can force a mode while testing:

```bash
PI2_LLAMA_MODE=mtp  /path/to/pi2-llama-server
PI2_LLAMA_MODE=base /path/to/pi2-llama-server
PI2_LLAMA_MODE=fit  /path/to/pi2-llama-server
```

The launcher defaults match my filesystem. Change `PI2_LLAMA_SERVER_BIN` and `PI2_LLAMA_MODEL` or edit the defaults if your paths differ.

## llama.cpp build

My working binary is a CUDA build compiled natively for the 5070 Ti / SM120 path. The important runtime facts from the tested build:

- CUDA graphs enabled
- llama.cpp CUDA fused attention enabled
- Q4_0 target KV
- native Blackwell target
- single CUDA GPU, so no tensor-parallel nonsense to consider here

I wasn't using a PyTorch stack, so installing FlashAttention 2/3 as a Python package would not magically accelerate this GGUF path. llama.cpp's own CUDA attention was already on.

## systemd

I run the server as a user service. [`pi2-llama.service.example`](pi2-llama.service.example) is the stripped-down version of my unit.

The only slightly weird part is the optional runtime env file:

```text
EnvironmentFile=-%t/pi2-llama.env
```

My Pi wrapper writes mode overrides there when I want to force `mtp`, `base`, or `fit`. You can skip that entirely if you just want the launcher's automatic VRAM selection.

## things I tried that lost

### Q8 KV

Basically the same speed as Q4 KV and about **900 MiB more VRAM**. On a 16 GB card that extra memory is way more valuable to MTP than it is to cache precision.

### bigger batches

`2048 / 512` improved TTFT by around 0.4 seconds, then made decode roughly 10% slower. I care more about single-user generation speed, so `512 / 128` stayed.

### MTP n=3

Slower than n=2 and used more VRAM. Easy loss.

### Q2_K

Plain Q2_K hit 52.07 tok/s vs 44.38 for Q3_K, but Q2_K + MTP only reached 65.98 tok/s. Q3_K + MTP hit 69.05 and keeps the better quant. No reason to switch.

### Q4_K

The model file alone is about 15.66 GiB before KV cache, CUDA workspace, and the desktop get a vote. It doesn't fit fully resident, so it loses the thing that matters most here: keeping decode entirely on-card.

### NVFP4

The 5070 Ti supports Blackwell FP4 and current llama.cpp has native NVFP4 kernels. The problem isn't support, it's capacity. The exact Huihui-derived NVFP4 GGUF I checked is about **18.3 GiB**, so it isn't a one-card 16 GB solution.

### DFlash2

I actually built the then-unmerged llama.cpp PR #27342 and tested the Q4 DFlash2 drafter. Against Q2_K it managed 47.77 tok/s with only ~24% draft acceptance on my benchmark. The built-in MTP head was much better here.

That doesn't mean DFlash2 is bad in general. It means I wasn't going to keep a custom llama.cpp branch around to run slower on this specific box.

## quick reproduce

Start the server, get its PID, then run [`benchmark.py`](benchmark.py):

```bash
PID=$(pgrep -n llama-server)
python3 benchmark.py \
  --pid "$PID" \
  --runs 3 \
  --tokens 512 \
  --label q3k-mtp \
  --output /tmp/q3k-mtp.json
```

The benchmark intentionally uses a fixed long prompt, fixed seed and temperature 0. It's there to compare configs, not to pretend one synthetic prompt tells you everything about agent workloads.

I also tested the final setup through Pi cold-start, normal chat formatting, code, arithmetic, and a longer creative probe. The target GGUF and checksum never changed. MTP only drafts; target verification stays authoritative.

## if you're copying this to another GPU

The flags are the easy part. **Re-measure the VRAM thresholds.** `15150` and `14500` are from my 16,303 MiB 5070 Ti with my desktop environment, 65K context, this exact Q3_K file and Q4 KV cache.

If you copy those numbers onto a different card/context/model and it explodes, that one is between you and `nvidia-smi`.

## references

- Model: <https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF>
- llama.cpp build docs: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- llama.cpp speculative decoding: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- DFlash2 llama.cpp PR tested during this run: <https://github.com/ggml-org/llama.cpp/pull/27342>
