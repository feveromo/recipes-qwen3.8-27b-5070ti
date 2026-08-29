# Qwen3.8 27B UD-Q3_K_XL on a 5070 Ti

Verified on 2026-08-29. This is the current one-card, 65K-context recipe from the same workstation as the original August benchmark. The old Q3_K + MTP n=2 recipe reached 69.05 tok/s. The current recipe uses Huihui's newer Dynamic 3.0 UD-Q3_K_XL quant, current llama.cpp, and adaptive MTP n=3/n=2 tiers; the final build-10711 systemd service reached 82.05 tok/s on the same benchmark shape.

The important caveat: this is a 16 GB configuration with little spare VRAM. Copy the flags, not the thresholds. Re-measure the thresholds with your desktop workload.

## Verified hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti, GB203, 16,303 MiB VRAM, compute capability 12.0
- CPU: Ryzen 7 9800X3D, 8C/16T
- RAM: 32 GB DDR5-6000
- OS: Ubuntu 26.04 LTS, kernel 7.0.0-30-generic
- NVIDIA driver: 595.84
- CUDA toolkit/runtime used for the build: 13.1.115
- llama.cpp: build 10711, commit `601ec3fd77d9575e40aa99f7d0ebacef1ab0d323`
- Model repository: `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`
- Model file: `Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf`
- Model size: 13,326,125,984 bytes
- Model SHA-256: `c6e144d78595c4656946187c2cee42f9062b3b4fa9874bc1be3e9263396253da`
- Context: 65,536
- Target and draft KV: Q4_0 K/V
- Batch / ubatch: 512 / 128
- Threads: 8
- Parallel slots: 1

The model has 64 target layers plus the embedded MTP layer. Native context is 262,144; 65,536 is the practical target here because the model, target KV, MTP context, CUDA workspace, and desktop all share one 16 GB card. This profile is text-only and does not load the optional multimodal projector.

## Download and verify the model

```bash
MODEL_DIR=/home/fever/Models/huihui-qwen3.8-27b-abliterated-ud-q3kxl
MODEL="$MODEL_DIR/Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf"
mkdir -p "$MODEL_DIR"
curl --location --fail --retry 5 --continue-at - \
  --output "$MODEL" \
  'https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf?download=true'
stat -c '%s bytes' "$MODEL"
sha256sum "$MODEL"
```

Expected output:

```text
13326125984 bytes
c6e144d78595c4656946187c2cee42f9062b3b4fa9874bc1be3e9263396253da  .../Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf
```

## Reproduce the exact llama.cpp build

```bash
git clone https://github.com/ggml-org/llama.cpp.git /home/fever/Dev/llama.cpp
cd /home/fever/Dev/llama.cpp
git checkout 601ec3fd77d9575e40aa99f7d0ebacef1ab0d323

export PATH=/usr/local/cuda-13.1/bin:$PATH
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc

cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build-cuda --config Release -j 8 \
  --target llama-server llama-cli llama-bench

LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 \
  ./build-cuda/bin/llama-server --version
LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 \
  ./build-cuda/bin/llama-cli --list-devices
```

The tested build uses native Blackwell `120a-real`, CUDA graphs, llama.cpp CUDA FlashAttention, and the current backend draft sampler. Python FlashAttention packages do not affect this GGUF runtime.

## Runtime configuration

[`pi2-llama-server`](pi2-llama-server) contains the exact launch flags. Its defaults match the paths above and can be overridden with `PI2_LLAMA_SERVER_BIN` and `PI2_LLAMA_MODEL`.

The fixed flags are:

```text
--ctx-size 65536
--flash-attn on
--cache-type-k q4_0
--cache-type-v q4_0
--batch-size 512
--ubatch-size 128
--threads 8
--threads-batch 8
--parallel 1
--jinja
--reasoning on
--reasoning-budget -1
--reasoning-preserve
--no-warmup
```

The wrapper samples free VRAM before loading and selects:

1. `mtp` at 15,250 MiB free or above: all target layers on GPU, fit off, embedded MTP depth 3.
2. `mtp2` at 15,000 MiB free or above: all target layers on GPU, fit off, embedded MTP depth 2.
3. `base` at 14,500 MiB free or above: all target layers on GPU, fit off, no speculation.
4. `fit` below 14,500 MiB: automatic layer placement with a 512 MiB target margin, no speculation.

Measured idle server residency was 13,694 MiB for base, 14,700 MiB for MTP n=2, and 14,850 MiB for MTP n=3. The final automatic launch saw 15,264 MiB free—14 MiB above the n=3 gate—and completed all three benchmark runs at a 15,438 MiB total-use peak, leaving about 382 MiB unreserved free at peak. This validates the boundary on this desktop but is intentionally narrow. The extra n=2 tier avoids jumping directly from the fastest mode to 47 tok/s base decode when desktop VRAM is only moderately busy.

Force a mode only when enough VRAM is actually free:

```bash
PI2_LLAMA_MODE=mtp  ./pi2-llama-server
PI2_LLAMA_MODE=mtp2 ./pi2-llama-server
PI2_LLAMA_MODE=base ./pi2-llama-server
PI2_LLAMA_MODE=fit  ./pi2-llama-server
```

## systemd user service

Install the launcher and example unit, then adjust paths if your layout differs:

```bash
install -m 0755 pi2-llama-server /home/fever/.pi2/agent/bin/pi2-llama-server
install -m 0644 pi2-llama.service.example \
  /home/fever/.config/systemd/user/pi2-llama.service
systemctl --user daemon-reload
systemctl --user start pi2-llama.service
curl --fail --silent \
  -H 'Authorization: Bearer llamacpp' \
  http://127.0.0.1:8003/health
curl --fail --silent \
  -H 'Authorization: Bearer llamacpp' \
  http://127.0.0.1:8003/v1/models
```

The unit uses `KillSignal=SIGINT`; the tested service exits cleanly on SIGINT. A SIGTERM-driven test exposed an abort in `stream_session_manager` teardown, so preserving SIGINT in the unit matters for this build line.

## Controlled benchmark

Every row below uses the same 7,322-token prompt, 65,536 server context, 512 generated tokens, temperature 0, seed 3407, erased slot cache, and three runs.

| setup | decode mean | TTFT mean | prompt mean | peak VRAM | wall mean |
|---|---:|---:|---:|---:|---:|
| pre-audit UD-IQ3_S, build 10639, MTP n=2 | 81.57 tok/s | 5.14 s | 1,428 tok/s | 14,033 MiB | 11.40 s |
| UD-Q3_K_XL base, build 10709 | 47.46 tok/s | 4.88 s | 1,504 tok/s | 14,466 MiB | 15.65 s |
| UD-Q3_K_XL MTP n=1 | 70.48 tok/s | 5.03 s | 1,458 tok/s | 15,028 MiB | 12.28 s |
| UD-Q3_K_XL MTP n=2 | 81.23 tok/s | 5.04 s | 1,456 tok/s | 15,178 MiB | 11.33 s |
| **UD-Q3_K_XL MTP n=3, final build 10711 service** | **82.05 tok/s** | **5.14 s** | **1,427 tok/s** | **15,438 MiB** | **11.37 s** |
| UD-Q3_K_XL MTP n=4 | 78.86 tok/s | 5.08 s | 1,445 tok/s | 15,371 MiB | 11.56 s |
| UD-IQ3_S MTP n=3 | 82.17 tok/s | 5.18 s | 1,415 tok/s | 13,837 MiB | 11.40 s |
| UD-IQ3_S + current DFlash2 Q4, n=7 | 79.31 tok/s | 5.26 s | 1,396 tok/s | 15,157 MiB | 11.70 s |

The installed profile improved decode by 0.6% and wall time by 0.3% over the active profile at the start of this audit, with TTFT unchanged within 0.1%, while moving from UD-IQ3_S to the higher-bitrate UD-Q3_K_XL. A quieter build-10709 run of the same CUDA path reached 83.85 tok/s; the final build-10711 run had 296 MiB more desktop VRAM in use. Against the original Q3_K auto-fit result from the first recipe, decode improved from 29.05 to 82.05 tok/s, a 182.4% increase. Against the old recipe's 69.05 tok/s Q3_K + MTP n=2 winner, decode improved by 18.8%.

Run [`benchmark.py`](benchmark.py) against the loaded service:

```bash
PID="$(systemctl --user show -p MainPID --value pi2-llama.service)"
python3 benchmark.py \
  --pid "$PID" \
  --runs 3 \
  --tokens 512 \
  --label q3kxl-mtp3-build-10711 \
  --output /tmp/q3kxl-mtp3-build-10711.json
```

This benchmark compares runtime configurations; one deterministic prompt is not a general model-quality benchmark.

## Candidates that lost

- **UD-IQ3_S:** more VRAM headroom and essentially the same decode speed in the controlled runs. UD-Q3_K_XL remains the default because its higher-bitrate target preserved roughly 82 tok/s with MTP n=3; the benchmark does not establish a general quality delta.
- **MTP n=4:** acceptance fell enough that extra verification overhead cut decode to 78.86 tok/s and increased memory use.
- **Larger microbatch:** 1024/256 reduced TTFT to 4.51 seconds and wall time to 10.79 seconds for this 7.3K/512 benchmark, but decode fell to 81.33 tok/s and server residency rose to 14,922 MiB. The 512/128 setting is safer and faster for longer coding-agent generations.
- **16 CPU threads and 100% polling:** they slightly improved the greedy synthetic run but lost about 1% on sampled tool/prose requests. Eight physical cores and the default poll setting stayed.
- **Current DFlash2:** the corrected Q4 draft could not load beside Q3_K_XL at 65K because its 1,079.61 MiB CUDA allocation OOMed. It fit beside IQ3_S, but 31.5% draft acceptance at n=7 produced 79.31 tok/s, slower than embedded MTP and with the lower target quant.
- **UD-DW Q6/Q8 and Q4-class targets:** they do not fit fully resident with 65K KV and the desktop on this 16 GB card.
- **Qwen3.8 Flash Next:** its practical quantized footprint is far beyond this 32 GB RAM / 16 GB VRAM workstation, so it is not a one-card replacement.

## MTP correctness caveat

llama.cpp issue [#27296](https://github.com/ggml-org/llama.cpp/issues/27296) tracks intermittent MTP state surviving across requests and tool-call truncation. The selected n=3 path passed three repeated tool-prompt to prose-prompt pairs with no leaked phrase, plus five three-turn OpenAI tool conversations: all 15 calls had complete, correct arguments and no HTTP 500. That does not close an intermittent upstream issue. If a coding session shows cross-request text, malformed tool arguments, or a server 500, restart in `PI2_LLAMA_MODE=base` and preserve the failing prompts.

The proposed upstream state fix in [PR #27173](https://github.com/ggml-org/llama.cpp/pull/27173) was not applied because it remains unmerged and has a reported large throughput regression on another CUDA system. The verified setup stays on upstream master and retains a no-speculation fallback.

## References

- Huihui model repository: <https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF>
- llama.cpp build documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- llama.cpp speculative decoding: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- MTP state issue: <https://github.com/ggml-org/llama.cpp/issues/27296>
- DFlash2 support: <https://github.com/ggml-org/llama.cpp/pull/27342>
- Tool-set decode performance fix included by the tested build: <https://github.com/ggml-org/llama.cpp/pull/27679>
