# Qwen3.8 27B abliterated on a 5070 Ti: 128K context, MTP-3, custom CUDA kernels v2

Verified on 2026-09-27. This is the current one-card recipe from the same workstation as the earlier versions. It runs Huihui's GSQ-RCO IQ3_S quant with its embedded MTP head at **131,072 tokens of context** on a 16 GB RTX 5070 Ti, with a local llama.cpp patch ([`cuda-kernels.patch`](cuda-kernels.patch), v2) that rewrites most of the decode and prefill hot path for this model and GPU.

Same weights and launcher, previous recipe's patched build (v1) vs this one (v2), 96K context, run back to back:

| | v1 (2026-09-26) | **v2** | change |
|---|---:|---:|---:|
| decode, short prompt | 99.4 tok/s | **131** | +32% |
| decode, 15.7K prompt | 83.2 | **104.7** | +26% |
| decode, 62.5K prompt | 74.9 | **98.0** | +31% |
| decode, 92.9K prompt | 68.4 | **94.8** | +39% |
| decode, real sampling (T 1.0), 15.7K, 3 seeds | 75.6 | **111.7** | +48% |
| prefill, 62.5K / 92.9K prompt | 1,276 / 1,125 | **1,814 / 1,696** | +42% / +51% |
| time to first token, 92.9K prompt | 82.7 s | **54.9 s** | −34% |
| llama-server peak VRAM at 96K | 14,618 MiB | **14,282 MiB** | −336 MiB |
| largest practical context | 96K | **128K** | |

At 128K a 128,794-token prompt prefills at 1,576 tok/s and then decodes at 91 tok/s. Output quality is unchanged within noise (KL divergence vs v1 below v1's own batch-size noise, see [Numerical checks](#numerical-checks)).

Temperature-0 decode rates move with draft acceptance, which changes whenever the greedy text changes. The steadier number is time per verify cycle: **21.2 / 22.0 / 23.6 / 24.8 ms** at short / 15.7K / 62.5K / 92.9K, down from 27.3 / 28.2 / 30.8 / 32.3 ms (−22% to −23%).

The caveats from the previous recipes still hold: this is a 16 GB configuration with little spare VRAM, and the patch is not upstream. It was written and tested only for this model on this GPU (`sm_120`). Every new kernel path checks the architecture, types and shapes it was tested for, and falls back to the v1/stock code otherwise.

## Verified hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti, GB203, 16,303 MiB VRAM, compute capability 12.0
- CPU: Ryzen 7 9800X3D, 8C/16T; power profile `performance`
- RAM: 32 GB DDR5-6000
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic
- NVIDIA driver: 610.57.04
- CUDA toolkit: 13.1.115, host compiler g++-13
- llama.cpp: build 11191, commit `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0`, plus [`cuda-kernels.patch`](cuda-kernels.patch) (SHA-256 `a4eaa130344bded44b220a5fb80fa39a1a4436546ed61048be660c4d754bb29c`; includes the v1 changes)
- Model repository: `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`, revision `3f101cd22b7999228bbd5d79a33975414eb9758b`
- Model file: `Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf`, 12,120,016,416 bytes, SHA-256 `eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19`
- Context: 131,072. Target KV: Q4_0 K/V. MTP draft KV: Q4_0 K/V
- Speculation: embedded MTP, up to 3 draft tokens, fused reduced-vocabulary drafting, Gumbel-coupled sampling
- Batch / ubatch: 512 / 256. Threads: 8. Parallel slots: 1

The model has 64 target layers (48 Gated DeltaNet linear-attention layers and 16 full-attention layers with 24 query heads, 4 KV heads, head size 256) plus the embedded MTP (NextN) layer. Native context is 262,144. 128K is the practical limit here with a desktop running on the same card (see [128K context](#128k-context)).

## Download and verify the model

```bash
MODEL_DIR=/home/fever/Models/huihui-qwen3.8-27b-abliterated-gsq-rco-iq3s
MODEL="$MODEL_DIR/Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf"
mkdir -p "$MODEL_DIR"
curl --location --fail --retry 5 --continue-at - \
  --output "$MODEL" \
  'https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/3f101cd22b7999228bbd5d79a33975414eb9758b/Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf?download=true'
stat -c '%s bytes' "$MODEL"
sha256sum "$MODEL"
```

Expected output:

```text
12120016416 bytes
eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19  .../Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf
```

## Build the patched llama.cpp

`RECIPE` is your checkout of this repository.

```bash
RECIPE=/home/fever/Dev/recipes-qwen3.8-27b-5070ti
LLAMA=/home/fever/Dev/llama.cpp-pi2-cuda-20260927

git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA"
cd "$LLAMA"
git checkout 4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0
sha256sum "$RECIPE/cuda-kernels.patch"   # a4eaa130...bb29c
git apply --check "$RECIPE/cuda-kernels.patch"
git apply "$RECIPE/cuda-kernels.patch"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DGGML_CCACHE=ON \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' \
  -DGGML_CUDA_COMPRESSION_MODE=size \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_CUDA_FORCE_MMQ=OFF \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_CUDA_NO_PEER_COPY=OFF \
  -DGGML_CUDA_NO_VMM=OFF \
  -DGGML_CPU=ON \
  -DGGML_CPU_REPACK=ON \
  -DGGML_OPENMP=ON \
  -DGGML_LLAMAFILE=ON \
  -DGGML_BLAS=OFF \
  -DGGML_LTO=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_BUILD_UI=ON \
  -DLLAMA_USE_PREBUILT_UI=ON

cmake --build build --target llama-server llama-bench --parallel 12

LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 ./build/bin/llama-server --version
```

The build takes about three minutes and `--version` reports `build 11191, commit 4b1a27fa0`. The configure flags are unchanged from the previous recipe; many restate defaults so the build is exactly the one measured. The patch is made against `4b1a27fa0`; on newer upstream commits expect manual merges (it touches the CUDA backend, the MTP code in `common/` and `src/`, and the server).

To run the tests, including the cases the patch adds:

```bash
cmake --build build --target test-backend-ops test-sampling-coupled --parallel 12
LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 ./build/bin/test-backend-ops test -b CUDA0
./build/bin/test-sampling-coupled
```

## Draft vocabulary

The fused drafting mode (`--spec-draft-vocab`) gives the MTP head a reduced lm-head: the first 16,384 token ids of a ranked vocabulary file, plus up to 8,192 rows filled with tokens seen in the current context. Two options:

- **Generic:** [`draft-vocab-generic.txt`](draft-vocab-generic.txt), ranked from public text only (the WikiText-2 training split and the llama.cpp sources, 10.7M tokens).
- **Your own (recommended):** rank tokens from what the model writes for you. [`build-draft-vocab.py`](build-draft-vocab.py) reads Pi session logs (assistant output only, rendered like the chat template) and/or plain text files, and tokenizes through a running llama-server:

```bash
python3 build-draft-vocab.py --url http://127.0.0.1:8003 \
  --pi-sessions ~/.pi/agent/sessions --out ~/.pi2/agent/draft-vocab.txt
```

The measurements in this README used a vocabulary ranked from my own coding-agent sessions (not published). On held-out sessions it covers 98.4% of generated tokens including the context rows. On this README's synthetic benchmark the generic file performs the same within noise ([`results/2026-09-27/vocab-generic-v2-*.json`](results/2026-09-27/)); on your own coding work a personal file should accept more drafts.

## What the patch changes

An nsys profile of the v1 build showed each verify cycle at short context as 21.5 ms of target verification (4-token batch), three 1.4 ms MTP draft graphs (each reading the full 248,320-row lm-head), a 0.5 ms MTP catch-up graph and about 2 ms of host gaps, with ~2,420 kernel launches per cycle. Prefill at ~40K depth spent 44% in quantized matmuls (MMQ) and 37% in flash attention.

**From v1** (unchanged, see the previous recipe): flash attention reads Q4_0 KV directly on the MMA path; MMVQ with one warp per 4 rows for 2–4 columns; tiny BF16 matrices on the mat-vec kernel.

**New in v2:**

1. **Multi-column quantized mat-vec** (`mmvq.cu`). A new kernel for 2–4 columns decodes each weight block once (grid lookups, signs, scales) and reuses it for every column, loads weights with an evict-first hint, and fuses FFN gate+up(+GLU) for up to 4 columns, including gate/up pairs of different quant types. 4-column DRAM bandwidth (64 distinct weight copies, so nothing stays in L2): IQ3_S 17408×5120 675 → 749 GB/s, IQ3_XXS 634 → 714, Q4_K 5120×17408 634 → 785, Q2_K 466 → 704, fused IQ3_S gate/up 676 → 770; the card reads about 845 GB/s at best.
2. **Cheaper MTP drafting** (`common/speculative.cpp`, `src/models/qwen35.cpp`, `src/llama-context.cpp`).
   - The per-cycle MTP catch-up only needs the draft layer's K/V, so it now computes just the K/V projection, rope and cache write instead of the whole block (always on).
   - `--spec-draft-vocab FILE` runs all draft steps in one graph (MTP block → reduced lm-head → argmax → token-id map → embedding row → next step): no per-step CPU round trip, no 248K-row lm-head, no top-k sort. The head is an exact copy of the selected `output.weight` and embedding rows (+106 MiB).
   - `--spec-draft-fuse-catchup` folds the verified rows into the next draft graph, so each cycle runs one MTP graph.
   - The target model's output is bit-identical to v1 (KLD 0).
3. **Gumbel-coupled sampling** (`--spec-coupled-sampling`; `common/sampling.cpp`, new `GGML_OP_ADD_GUMBEL`). With temperature > 0 the target draws a random token, so an argmax draft only matches when that draw happens to be the argmax. The target's final draw is now a Gumbel-max over the candidates the sampler chain keeps, with noise hashed from (seed, token index, token id), and the draft adds the same noise to its logits before its argmax. That is still an exact sample from the model's distribution, but draft and target now usually pick the same token. At T 1.0 draft acceptance rose from 0.52 to 0.59 (short) and 0.36 to 0.41 (15.7K) for +6% throughput on the same build. A fixed seed gives a different (equally distributed) text than without the flag.
4. **Decode attention** (`fattn-mma-f16.cuh`, `fattn.cu`). The Q4_0 MMA path is rewritten for ≤ 32 query columns: a cp.async ring prefetches mask/K/V stages and each warp builds its tensor-core fragments straight from the raw Q4_0 bytes. 1–2-token queries on Q4_0 with grouped heads now use it instead of the vector kernel, which read the cache once per query head: at 64K KV, 499 → 102 µs. The 4-token verify is ~15% faster (137 → 116 µs at 64K). This makes a Q4_0 MTP draft KV cache fast, which saves 276 MiB at 96K compared with F16.
5. **INT8 prefill attention** (new `fattn-mma-q4i8.cu`). This card does 403 INT8 TOPS versus 104 TFLOPS for FP16 with FP32 accumulation. For prompt batches of ≥ 64 tokens, Q is quantized to int8 once and K tiles are re-quantized to int8 per row in shared memory, so the whole Q·Kᵀ runs on `mma.m16n8k32.s8`; softmax stays FP32 and P·V stays FP16 MMA. Q4_0 KV at 64K: 6.3 → 2.5 ms per 256-query call; also used for the MTP layer's KV. `GGML_CUDA_FA_NO_Q4I8=1` disables it.
6. **Kernel fusion** (`ggml-cuda.cu`, norm, conv, gated-delta-net kernels). Mat-vecs that share an input reuse one quantized copy, which the producing norm/activation kernel writes directly (`GGML_CUDA_DISABLE_Q8_CACHE=1` disables it); residual add + RMS norm (+quantize), activation × input (+quantize), the conv-state gather/concat/snapshot/conv/silu chain, the GDN q/k L2 norms and the BF16 alpha/beta gates are fused. Launches per decode cycle: ~2,420 → ~1,090; verify graph 21.3 → 19.5 ms on its own.
7. **Prefill matmuls and GDN** (`mmq*.cuh`, `gated_delta_net.cu`). The next weight/activation block is prefetched into shared memory while the tensor cores work (Blackwell only, bit-identical), IQ2/IQ3 sign decoding is cheaper, and new GDN kernels for prompt batches are about 3.5× faster (372 → ~105 µs per layer per 256 tokens). IQ1_M (one tensor in this quant) now runs through MMQ instead of dequantizing to F16 for cuBLAS, which removes a 178 MB temporary buffer (−170 MiB peak).

After v2, a short-context decode cycle is about 21 ms, of which about 16 ms is the new 4-column mat-vec kernel running close to DRAM bandwidth.

## Runtime configuration

[`pi2-llama-server`](pi2-llama-server) contains the exact launch flags. Its defaults match the paths above and can be overridden with `PI2_LLAMA_SERVER_BIN`, `PI2_LLAMA_MODEL`, `PI2_DRAFT_VOCAB` and `PI2_CTX`.

```text
--ctx-size 131072
--n-gpu-layers all
--fit off
--no-context-shift
--flash-attn on
--cache-type-k q4_0
--cache-type-v q4_0
--batch-size 512
--ubatch-size 256
--threads 8
--threads-batch 8
--parallel 1
--cache-ram 0
--ctx-checkpoints 0
--jinja
--reasoning on
--reasoning-effort xhigh
--reasoning-budget 16384
--reasoning-preserve
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp
--spec-draft-n-max 3
--spec-draft-type-k q4_0
--spec-draft-type-v q4_0
--spec-draft-vocab /home/fever/.pi2/agent/draft-vocab.txt
--spec-draft-vocab-n 16384
--spec-draft-fuse-catchup
--spec-coupled-sampling
```

Why these values:

- **128K context:** llama-server peaks at 14,926 MiB with a 128.8K-token prompt, which leaves about 1.2 GB of the 16,303 MiB card for the desktop. With heavier desktop GPU use (a game launcher measured up to 1.9 GB) use 96K: `systemctl --user set-environment PI2_CTX=98304` before the service starts, or `Environment=PI2_CTX=98304` in the unit.
- **Q4_0 draft KV:** with v2's attention path it is as fast as F16 or faster at long context, and 276 MiB smaller at 96K.
- **16,384 draft rows:** 8K rows cover fewer of the tokens generated in real sessions (97.2% vs 98.4% with context rows); 32K rows (99.3%) gave no measurable speedup for 2× the head memory.
- **MTP n=3:** with fused drafting, n=4 was slower than n=3 at 15.7K and 62.5K.
- **ubatch 256:** unchanged from v1; the VRAM goes to context instead.

## systemd user service

```bash
install -m 0755 pi2-llama-server /home/fever/.pi2/agent/bin/pi2-llama-server
install -m 0644 pi2-llama.service.example \
  /home/fever/.config/systemd/user/pi2-llama.service
install -m 0644 draft-vocab-generic.txt /home/fever/.pi2/agent/draft-vocab.txt   # or build your own, see above
systemctl --user daemon-reload
systemctl --user start pi2-llama.service
curl --fail --silent -H 'Authorization: Bearer llamacpp' http://127.0.0.1:8003/health
```

The unit keeps `KillSignal=SIGINT`; SIGINT shutdown is clean.

## Results

Raw JSON and logs are in [`results/2026-09-27/`](results/2026-09-27/). "v1" is the previous recipe's build (commit `4b1a27fa0` + the 2026-09-26 patch, its flags, 96K) and "v2" this recipe's build and flags at 96K unless noted; same model and desktop, run back to back. Decode rates include thinking tokens.

### Long-context benchmark

[`benchmark_chat.py`](benchmark_chat.py) streams from `/v1/chat/completions` with reasoning effort xhigh, prompt caching off and `ignore_eos`, on a repetitive synthetic prompt (real draft acceptance differs). Temperature 0, seed 3407; short 2 runs, others 1.

| workload | build | decode tok/s | ms per cycle | prefill tok/s | TTFT s | draft acceptance |
|---|---|---:|---:|---:|---:|---:|
| short: 120 prompt / 256 out | v1 | 99.4 | 27.3 | — | — | 0.58 |
| | **v2** | **130.0 / 131.6** | **21.2** | — | — | 0.59 / 0.61 |
| mid: 15,694 prompt / 512 out | v1 | 83.2 | 28.2 | 1,591 | 9.9 | 0.45 |
| | **v2** | **104.7** | **22.0** | **2,030** | **7.7** | 0.43 |
| long: 62,494 prompt / 512 out | v1 | 74.9 | 30.8 | 1,276 | 49.0 | 0.44 |
| | **v2** | **98.0** | **23.6** | **1,814** | **34.5** | 0.44 |
| vlong: 92,914 prompt / 512 out | v1 | 68.4 | 32.3 | 1,125 | 82.7 | 0.40 |
| | **v2** | **94.8** | **24.8** | **1,696** | **54.9** | 0.45 |

A verify cycle is one 4-token target pass plus drafting; ms per cycle = mean accepted length × 1000 / decode tok/s.

```bash
PID="$(systemctl --user show -p MainPID --value pi2-llama.service)"
python3 benchmark_chat.py --pid "$PID" --runs 2 --tokens 256 --repeats 1    --label short --output /tmp/short.json
python3 benchmark_chat.py --pid "$PID" --runs 1 --tokens 512 --repeats 600  --label mid   --output /tmp/mid.json
python3 benchmark_chat.py --pid "$PID" --runs 1 --tokens 512 --repeats 2400 --label long  --output /tmp/long.json
python3 benchmark_chat.py --pid "$PID" --runs 1 --tokens 512 --repeats 3570 --label vlong --output /tmp/vlong.json
```

### With the production sampler

Same prompts with the server's own sampler (temperature 1.0, top-k 20, top-p 0.95) and a different seed per run (`--server-sampling --vary-seed`), which is how the model is actually used:

| workload | runs | v1 decode tok/s | v2 decode tok/s | v1 acceptance | v2 acceptance |
|---|---:|---:|---:|---:|---:|
| short | 6 | 98.6 | **129.5** | 0.57 | 0.59 |
| mid (15.7K) | 3 | 75.6 | **111.7** | 0.38 | 0.49 |
| long (62.5K) | 1 | 65.7 | **96.0** | 0.34 | 0.43 |

The single long run is noisy; its cycle time (30.7 → 23.8 ms) matches the temperature-0 table.

### 128K context

With `--ctx-size 131072`:

- A 128,794-token prompt prefilled at 1,576 tok/s (81.7 s) and decoded at 91.3 tok/s. llama-server loaded at 14,804 MiB and peaked at 14,926 MiB; the card as a whole peaked at 15,074 MiB with a light desktop.
- **Ledger recall:** four values planted across a 119,457-token prompt were all returned exactly (cold prefill 1,606 tok/s). An append-only follow-up (the previous answer plus a new question, as an agent session sends it) reused 119,645 cached tokens and answered in 0.7 s ([`recall-120k-v2.json`](results/2026-09-27/recall-120k-v2.json)).
- 160K (163,840) ran out of memory during prefill in a run made before the IQ1_M change freed 170 MiB. It was not retried: even if it fits now, it would leave well under 1 GB for the desktop.

## Numerical checks

**Output probabilities vs v1.** `llama-perplexity --kl-divergence` on WikiText-2 test, 8 chunks of 2,048 tokens, Q4_0 KV, flash attention. Reference: the v1 build at `-ub 4`, which runs the same small-batch kernels as MTP verification.

| run vs v1 `-ub 4` | mean KLD | same top token | PPL ratio |
|---|---:|---:|---:|
| v1 `-ub 256` (v1's own batch-size noise) | 0.0074 | 97.91% | 0.9986 ± 0.0012 |
| v2 `-ub 4` (verify path) | 0.0031 | 98.16% | 1.0003 ± 0.0009 |
| v2 `-ub 256` (prefill path) | 0.0054 | 97.78% | 1.0004 ± 0.0010 |

Both v2 paths differ from v1 by less than v1 differs from itself across batch sizes. The drafting changes (items 2 and 3) do not touch the target's logits.

**Sampling exactness** (`tests/test-sampling-coupled.cpp`). One million coupled draws over 1,000 seeds × 1,000 token positions match the truncated softmax (chi-square p 0.30–0.97; the stock sampler scores 0.06–0.51 on the same test). A simulated speculative decoder with five different drafters (none, coupled, random, pure noise and an oracle that picks cycle lengths from future noise) emits exactly the non-speculative token sequence over 12,000 seeds, and planting a noise-reuse bug fails the test with chi² 5,418. On the server, 16 of 16 seeded requests were byte-identical with the fused and the classic drafter.

**Draft acceptance** depends on the text generated, so at temperature 0 the two builds' acceptance is measured on different outputs; all throughput numbers already include it.

**Kernel tests and sanitizers:**

- The full `test-backend-ops` suite on CUDA0 passes (17,436 cases, including new ones for this patch: masked attention tails, GQA 2–12, 1–1,030-token batches, row and K tails, strided/permuted mat-vec operands, fusion patterns with shared intermediates and in-place rewrites).
- `compute-sanitizer` memcheck, racecheck and synccheck report no errors or hazards on the new attention, mat-vec, fusion and GDN paths.
- Independent reviews of each change found and fixed 8 bugs before release, none of them reachable with this recipe's flags: NaN output when a stream-K block started past a fully masked KV tail (multi-sequence or ≥1,024-token batches), FP16 precision loss for very small V scales, a multi-architecture build break, a server abort when a draft decode triggered a memory update (`n_cmpl` > 1, KV shift, cache reuse), a crash after a prompt-cache restore with deferred catch-up rows, a read-before-wait race in two fused kernels, fused mat-vec + add ignoring a non-unit addend stride, and a missing eligibility check on the quantized-input cache.

**Service checks with the installed service:** the Pi2 API smoke test passes (quality probes, tool-call round trips, cross-request state isolation, long-to-short isolation), and a real Pi coding session (read, edit, bash, write) completes with the prompt cache reused on every turn.

One behaviour to know about, unchanged from v1: re-sending an *identical* long prompt after a reply does not hit the cache. The model's recurrent layers cannot be rolled back past the generated reply without `--ctx-checkpoints`. Append-only follow-ups, which is what chat clients send, reuse it.

## Candidates that lost

- **MTP n=4:** slower than n=3 at 15.7K and 62.5K.
- **Q8_0 draft KV:** ran out of memory at 96K.
- **Re-quantizing the MTP block to Q4_K:** no speed change.
- **32K draft rows:** no measurable gain over 16K; 8K rows cover fewer generated tokens.
- **MMQ with 128-row tiles for IQ3/IQ4 (+6–9% prefill):** failed the quality bar (same-top 97.42%, KLD 0.0078).
- **L2 prefetch and shared-memory grid tables in the mat-vec kernel:** 2–12% slower.
- **Prefetching GDN state into L2, a separate gated-norm kernel:** no measurable gain.
- **Backend (GPU) sampling:** llama.cpp disables it when a reasoning budget is set; CPU sampling costs < 1% of a cycle here anyway.
- **160K context:** out of memory during prefill (before the IQ1_M change), and too little headroom for a desktop in any case.

Ideas not pursued:

- The GDN kernel writes four full recurrent-state snapshots per layer per verify for rollback (~600 MB per cycle); a cheaper rollback scheme could save up to ~0.5 ms per cycle.
- Staging mat-vec weights through shared memory with `cp.async`; IQ3_S/IQ3_XXS/IQ2 at 4 columns still have 5–10% to the bandwidth limit.
- An MMQ rewrite that overlaps IQ dequantization with the tensor-core work (dequant is ~20% of an IQ3_S prefill matmul).

## MTP correctness caveat

llama.cpp issue [#27296](https://github.com/ggml-org/llama.cpp/issues/27296) tracks intermittent MTP state surviving across requests and tool-call truncation; it is still open upstream. The installed MTP-3 service passed the tool round-trip and state-isolation tests, and the fused drafting paths were tested across long→short requests, prompt-cache restores and multiple completions per request. If a coding session shows cross-request text, malformed tool arguments or a server 500, restart with `--spec-type none` and keep the failing prompts. The fused drafting modes can also be dropped individually by removing `--spec-draft-vocab`, `--spec-draft-fuse-catchup` or `--spec-coupled-sampling`.

## Previous recipes

- **2026-09-26:** GSQ-RCO IQ3_S at 96K with MTP-3 and the v1 CUDA patch; 100.4 / 72.0 / 68.4 tok/s at short / 62K / 93K on the 1,024-token benchmark (commit `e6e9f1b`).
- **2026-08-29:** Huihui UD-Q3_K_XL at 65K with adaptive MTP n=3/n=2 tiers, build 10711; 82.05 tok/s on the 7.3K `benchmark.py` shape (commit `e9a3cb7`).
- **2026-08-19:** the original Q3_K + MTP n=2 recipe; 69.05 tok/s (commit `259ce0f`).

## Files

- [`pi2-llama-server`](pi2-llama-server): launcher with the exact flags.
- [`pi2-llama.service.example`](pi2-llama.service.example): systemd user unit.
- [`cuda-kernels.patch`](cuda-kernels.patch): the llama.cpp patch (v1 + v2), including its tests.
- [`build-draft-vocab.py`](build-draft-vocab.py): builds a draft vocabulary from your own text or Pi sessions.
- [`draft-vocab-generic.txt`](draft-vocab-generic.txt): draft vocabulary ranked from public text.
- [`benchmark_chat.py`](benchmark_chat.py): long-context chat-endpoint benchmark (`--server-sampling`, `--vary-seed` added).
- [`benchmark.py`](benchmark.py): the earlier recipes' 7.3K `/completion` benchmark.
- [`results/2026-09-27/`](results/2026-09-27/): v1/v2 benchmark JSON (`long-context-*`, `sampling-*`), `context-128k-v2.json`, `recall-120k-v2.json`, `vocab-generic-v2-*`, per-run draft acceptance, and the KLD logs. The 2026-09-26 results are still in [`results/2026-09-26/`](results/2026-09-26/).

## References

- Huihui model repository: <https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF>
- llama.cpp base commit: <https://github.com/ggml-org/llama.cpp/commit/4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0>
- llama.cpp build documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- llama.cpp speculative decoding: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- MTP state issue: <https://github.com/ggml-org/llama.cpp/issues/27296>
- FR-Spec (frequency-ranked draft vocabularies): <https://arxiv.org/abs/2502.14856>
- SageAttention (INT8 attention): <https://arxiv.org/abs/2410.02367>
