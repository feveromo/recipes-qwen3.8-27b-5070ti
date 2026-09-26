# Qwen3.8 27B abliterated on a 5070 Ti: 96K context, MTP-3, custom CUDA kernels

Verified on 2026-09-26. This is the current one-card recipe from the same workstation as the earlier versions. It runs Huihui's GSQ-RCO IQ3_S quant with its embedded MTP head at **98,304 tokens of context**. It also applies a small local CUDA patch to llama.cpp ([`cuda-kernels.patch`](cuda-kernels.patch)). Everything is fully resident on a 16 GB RTX 5070 Ti.

Same weights, flags and context; stock llama.cpp vs the patched build:

| decode tok/s | short prompt | 62K prompt | 93K prompt |
|---|---:|---:|---:|
| stock llama.cpp `4b1a27fa0` | 84.5 | 52.6 | 43.1 |
| **+ `cuda-kernels.patch`** | **100.4** | **72.0** | **68.4** |
| change | +19% | +37% | +59% |

Long-prompt prefill is 16–21% faster. The server process uses 360–460 MiB less VRAM. On the older 7.3K-token benchmark shape from the previous recipes, decode went from 73.0 (stock) to **92.7 tok/s** (patched). The August recipe reached 82.05 tok/s on that benchmark with a different quant and only 65K context.

The important caveat is unchanged: this is a 16 GB configuration with little spare VRAM. At a 93K prompt the card's total use peaked at 15,522 of 16,303 MiB with a normal desktop running. The patch is also not upstream. It was written and tested only for this model on this GPU (`sm_120`).

## Verified hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti, GB203, 16,303 MiB VRAM, compute capability 12.0
- CPU: Ryzen 7 9800X3D, 8C/16T; power profile `performance`
- RAM: 32 GB DDR5-6000
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic
- NVIDIA driver: 610.57.04
- CUDA toolkit: 13.1.115, host compiler g++-13
- llama.cpp: build 11191, commit `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0`, plus [`cuda-kernels.patch`](cuda-kernels.patch) (SHA-256 `61b8b3d211af5d79beb95e9e4cfe29088fb22f324c396af0d6397de34bf8ca46`)
- Model repository: `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`, revision `3f101cd22b7999228bbd5d79a33975414eb9758b`
- Model file: `Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf`
- Model size: 12,120,016,416 bytes
- Model SHA-256: `eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19`
- Context: 98,304
- Target KV: Q4_0 K/V. MTP draft KV: F16 K/V
- Speculation: embedded MTP, up to 3 draft tokens
- Batch / ubatch: 512 / 256
- Threads: 8
- Parallel slots: 1

The model has 64 target layers plus the embedded MTP (NextN) layer. The `-mtp` file's 851 trunk tensors are byte-identical to the publisher's non-MTP `GSQ-RCO-IQ3_S.gguf`; it only adds the 15 `blk.64.*` MTP tensors. Native context is 262,144. 98,304 is the practical limit here because the model, target KV, MTP KV, CUDA workspace and desktop all share one 16 GB card; 128K ran out of memory. This profile is text-only and does not load the multimodal projector.

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
LLAMA=/home/fever/Dev/llama.cpp-pi2-cuda-20260926

git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA"
cd "$LLAMA"
git checkout 4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0
sha256sum "$RECIPE/cuda-kernels.patch"   # 61b8b3d2...8ca46
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

The build took about three minutes. `--version` should report `build 11191, commit 4b1a27fa0`. Many of the `-D` flags restate llama.cpp defaults; they are listed so the build is exactly the one measured.

The patch is made against `4b1a27fa0`. As of 2026-09-26, upstream master has not touched the four CUDA files it changes, but it has changed `tests/test-backend-ops.cpp`. The test hunk may need a manual merge on newer commits. For an unpatched comparison build, skip the two `git apply` lines.

To run the kernel tests, which include the cases the patch adds:

```bash
cmake --build build --target test-backend-ops --parallel 12
LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 ./build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 ./build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT
```

## What the patch changes

An `nsys` profile of stock MTP-3 decode at 62K showed where each ~48 ms verify step went:

- about 26 ms in quantized mat-vec (MMVQ);
- about 7 ms converting the Q4_0 KV cache to F16;
- about 5.4 ms in flash attention;
- about 2.9 ms in cuBLAS BF16 GEMMs for tiny matrices.

The patch has three parts:

1. **Flash attention reads Q4_0 KV directly** (`fattn-mma-f16.cuh`, `fattn.cu`).
   - **The problem:** stock llama.cpp converts the entire K and V cache to F16 before every attention call on the tensor-core (MMA) path. That happens twice per full-attention layer on every MTP verify step.
   - **The fix:** the MMA kernel now copies raw Q4_0 tiles into shared memory with `cp.async`. It dequantizes them there into the existing F16 tile layout, with the same rounding as the stock converter.
   - **Speed:**
     - Verify step (4 queries × 6 heads): 1,188 → 135 µs at 62K KV and 1,509 → 207 µs at 93K, conversion included.
     - Prefill (256 queries): 11.6 → 6.2 ms at 62K and 16.3 → 9.6 ms at 93K.
   - **Memory:** the ~400 MB F16 conversion buffer is no longer allocated.
   - **When it applies:** only for Q4_0 K and V at head size 256 with GQA ≥ 2. It also needs a mask, no ALiBi, softcap or sparse attention, padded KV, 16-byte alignment and a GPU with `cp.async`. Any other case uses stock code.
2. **Quantized mat-vec for 2–4 columns** (`mmvq.cu`).
   - **The problem:** MTP verification multiplies each weight matrix by 2–4 token vectors.
   - **The fix:** one warp computes four rows, instead of four warps splitting the K dimension. Split-K is kept only for small or odd row counts.
   - **Speed:** the IQ3_S 69632×5120 verify shape went from 307 to 222 µs. Most shapes take 72–90% of the stock time; Q6_K with two columns is about 6% slower.
3. **Tiny BF16 matrices use the mat-vec kernel** (`mmvf.cu`) instead of cuBLAS, for ≤ 256 rows and ≤ 8 columns: 21.3 → 3.6 µs at 48×4.

1–2-token queries on Q4_0 KV still use stock's vector attention kernel. They are about 0.5% of decode time here.

After the patch, total GPU busy time in the 62K decode trace fell from 4,660 to 3,262 ms (−30%). The Q4_0→F16 conversion (693 ms) and cuBLAS BF16 work (277 ms) are gone, and attention fell from 613 to 308 ms. About 70% of what remains is quantized mat-vec, which already runs at 540–830 GB/s of the card's 896 GB/s.

## Runtime configuration

[`pi2-llama-server`](pi2-llama-server) contains the exact launch flags. Its defaults match the paths above and can be overridden with `PI2_LLAMA_SERVER_BIN` and `PI2_LLAMA_MODEL`.

```text
--ctx-size 98304
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
--spec-draft-type-k f16
--spec-draft-type-v f16
```

Why these values:

- **F16 draft KV:** the MTP layer has GQA 6 at head size 256. With F16 K/V, llama.cpp sends its attention to the MMA kernel, which reads the cache once per KV head. With Q4_0, the 1-token draft steps use the vector kernel, which rereads the cache for every query head. The F16 draft cache costs about 276 MiB at 96K for one layer. It was worth +4 tok/s at 62K and +6 at 93K on the stock build.
- **ubatch 256:** saves about 88 MiB per compute buffer (three buffers) compared with 512, for about 1% prefill.
- **MTP n=3:** n=4 added only about 2% more decode for another 110–150 MiB.

The launcher always uses full GPU offload. Unified memory, automatic fitting and context shifting are disabled, so a VRAM shortage fails loudly instead of silently spilling layers to the CPU.

The previous recipe's free-VRAM tiers (`mtp` / `mtp2` / `base` / `fit`) are gone. If you need a smaller footprint, edit the `speculation` line:

- `--spec-draft-n-max 2` saves about 150 MiB;
- `--spec-type none` disables MTP entirely.

Lowering `--ctx-size` is the other lever.

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

The unit keeps `KillSignal=SIGINT`. An earlier build aborted in `stream_session_manager` teardown on SIGTERM. SIGINT shutdown is clean on this build.

## Results

Raw JSON and logs are in [`results/2026-09-26/`](results/2026-09-26/). "Stock" and "patched" are both commit `4b1a27fa0` with the same configure flags, model and launch flags; the only difference is the patch. Decode rates include thinking tokens.

### Long-context benchmark

[`benchmark_chat.py`](benchmark_chat.py) sends requests to the OpenAI-compatible streaming chat endpoint with:

- reasoning effort xhigh, temperature 0, seed 3407 and `ignore_eos`;
- prompt caching off, on a 98,304-token server;
- a repetitive synthetic prompt, so draft acceptance on real work will differ.

Short and 62K were measured 3 times each, 93K twice.

| workload | build | decode tok/s | prefill tok/s | TTFT s | wall s | peak total VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
| short: 120 prompt / 256 out | stock | 84.55 | 1,298 | 0.10 | 3.11 | 15,302 |
| | **patched** | **100.38** | 1,272 | 0.10 | **2.64** | 15,401 |
| 62K: 62,494 prompt / 1,024 out | stock | 52.58 | 1,091 | 57.33 | 76.79 | 15,336 |
| | **patched** | **71.97** | **1,269** | **49.33** | **63.54** | 15,409 |
| 93K: 92,914 prompt / 1,024 out | stock | 43.14 | 920 | 101.13 | 124.84 | 15,332 |
| | **patched** | **68.36** | **1,113** | **83.58** | **98.54** | 15,522 |

Peak *total* VRAM includes the desktop. The desktop was using about 460 MiB more during the patched runs than during the stock runs. Measured per process, llama-server's own footprint went down:

- loaded: 14,786 → 14,324 MiB;
- peak: 14,982 → 14,618 MiB.

The back-to-back runs below show the same thing.

Prompt processing on short prompts is about 2% slower with the patch (120 and 7.3K tokens). Long prompts are 16–21% faster. The short-prompt loss has not been investigated.

```bash
PID="$(systemctl --user show -p MainPID --value pi2-llama.service)"
python3 benchmark_chat.py --pid "$PID" --runs 3 --tokens 256  --repeats 1    --label short --output /tmp/short.json
python3 benchmark_chat.py --pid "$PID" --runs 3 --tokens 1024 --repeats 2400 --label 62k   --output /tmp/62k.json
python3 benchmark_chat.py --pid "$PID" --runs 2 --tokens 1024 --repeats 3570 --label 93k   --output /tmp/93k.json
```

### Previous-recipe benchmark shape

[`benchmark.py`](benchmark.py) is the benchmark from the earlier recipes:

- the raw `/completion` endpoint, with no chat template and no thinking;
- a 7,322-token prompt and 512 generated tokens;
- temperature 0, seed 3407, three runs.

The stock and patched rows were run back to back under the same desktop load.

| setup | decode mean | TTFT mean | prompt mean | peak VRAM | wall mean |
|---|---:|---:|---:|---:|---:|
| Aug 29 recipe: UD-Q3_K_XL, 65K, MTP n=3, build 10711 | 82.05 tok/s | 5.14 s | 1,427 tok/s | 15,438 MiB | 11.37 s |
| this recipe, stock build | 73.01 tok/s | 4.39 s | 1,670 tok/s | 15,276 MiB | 11.39 s |
| **this recipe, patched build** | **92.69 tok/s** | **4.46 s** | **1,644 tok/s** | **14,914 MiB** | **9.97 s** |

- **Patched vs stock:** +27% decode and −12% wall time. VRAM was 462 MiB lower after loading and 362 MiB lower at peak.
- **Patched vs the August recipe:** +13% decode with 50% more context.
- **Stock vs the August recipe:** the stock build of this recipe is slower on this shape. The weights, context, draft-KV type and llama.cpp version all changed, and those factors were not separated.

```bash
PID="$(systemctl --user show -p MainPID --value pi2-llama.service)"
python3 benchmark.py --pid "$PID" --runs 3 --tokens 512 \
  --label patched --output /tmp/patched.json
```

Neither benchmark measures general model quality.

### Numerical checks

**Output probabilities vs stock.** I compared logits from the two builds using `llama-perplexity --kl-divergence`:

- **Setup:** WikiText-2 test, 8 chunks of 2,048 tokens, Q4_0 KV, flash attention.
- **Reference:** a stock build at `-ub 4`, which exercises the same small-batch kernels as MTP verification.

| run vs stock `-ub 4` | mean KLD | same top token | PPL ratio |
|---|---:|---:|---:|
| stock `-ub 256` (stock's own batch-size noise) | 0.00397 | 98.06% | 0.9989 ± 0.0010 |
| patched `-ub 4` (verify path) | 0.00323 | 98.07% | 1.0002 ± 0.0011 |
| patched `-ub 256` (prefill path) | 0.00397 | 98.06% | 0.9989 ± 0.0010 |

The patched verify path differs from stock by less than stock differs from itself across batch sizes. The patched prefill path is bit-identical to stock prefill.

**Draft acceptance.** At temperature 0, any rounding difference changes the greedy text, so the two builds' acceptance rates are measured on different outputs.

| run | stock | patched |
|---|---:|---:|
| chat benchmark, short | 0.61 | 0.58 |
| chat benchmark, 62K | 0.49 | 0.41 |
| chat benchmark, 93K | 0.44 | 0.42 |
| 7.3K `/completion` benchmark | 0.49 | 0.53 |

The throughput numbers above already include this effect.

**Kernel tests and sanitizers:**

- The full `FLASH_ATTN_EXT` (4,021 cases) and `MUL_MAT` (1,332 cases) `test-backend-ops` suites pass. They include the new cases for Q4_0 head-256 attention with GQA 2/4/6/8 and batches of 1–64, plus the fallbacks.
- `compute-sanitizer` memcheck and synccheck report 0 errors on the Q4_0 attention cases.
- Racecheck reports no hazards in the new kernel. Its two warnings are in the unchanged vector kernel.

**Service checks with the installed service:**

- An API smoke test passed. It covers quality probes, tool-call round trips, cross-request state isolation and long-to-short isolation.
- A 93K ledger-recall test returned all four planted values exactly, cold and on a cached follow-up. Cold prefill was 1,120 tok/s (stock 924) and decode 97.2 tok/s (stock 63.6).

## Candidates that lost

- **128K context:** ran out of CUDA memory on a 125K prompt without MTP; a 128-token microbatch failed during warmup.
- **MTP n=4:** about +2% decode over n=3 for another 110–150 MiB (stock build).
- **Q4_0 draft KV:** slower than F16 draft KV, because of the vector-kernel rereads explained above.
- **Vector attention for 3–4-query verify batches:** raising llama.cpp's `≤ 2` vector-kernel cutoff to `≤ 4` passed all tests but was slightly slower (80.7 / 46.3 tok/s vs 81.1 / 47.3 with MTP n=2). The Q4_0-direct MMA path above replaced this idea.
- **`GGML_CUDA_GRAPH_OPT=1`:** no gain (53.48 vs 53.95 tok/s without MTP).
- **ubatch 512:** about 1% more prefill for about 260 MiB more VRAM.
- **ubatch 2048:** +11% prefill at 32K without MTP, but its compute buffer does not fit beside MTP at 96K.
- **UD-Q3_K_XL at 65K (the previous recipe):** replaced by GSQ-RCO IQ3_S. The IQ3_S file is 1.2 GB smaller, which leaves room for 96K context plus MTP. In a 2026-09-25 trial at 64K without MTP, RCO IQ3_S peaked 1.7 GiB lower than the IQ4 quant then in use, and decoded 9% faster at short context. It was not benchmarked directly against UD-Q3_K_XL.

Ideas not pursued:

- A reduced-vocabulary lm-head for the MTP draft. The full 248K-vocab Q4_K draft head is about 8% of decode time.
- Tuning the IQ2/IQ3 mat-vec kernels, which reach 540–690 GB/s against about 820 achievable.
- A Q4_0 draft KV cache that uses the new attention path.

## MTP correctness caveat

llama.cpp issue [#27296](https://github.com/ggml-org/llama.cpp/issues/27296) tracks intermittent MTP state surviving across requests and tool-call truncation; it is still open. The installed MTP-3 service passed the tool round-trip and state-isolation smoke tests, but that does not close an intermittent upstream issue. If a coding session shows cross-request text, malformed tool arguments or a server 500, do two things:

1. Restart with `--spec-type none`.
2. Keep the failing prompts.

The proposed upstream fix in [PR #27173](https://github.com/ggml-org/llama.cpp/pull/27173) is still unmerged and was not applied.

## Previous recipes

- **2026-08-29:** Huihui UD-Q3_K_XL at 65K with adaptive MTP n=3/n=2 tiers, build 10711; 82.05 tok/s on the `benchmark.py` shape (commit `e9a3cb7`).
- **2026-08-19:** the original Q3_K + MTP n=2 recipe; 69.05 tok/s (commit `259ce0f`).

## Files

- [`pi2-llama-server`](pi2-llama-server): launcher with the exact flags.
- [`pi2-llama.service.example`](pi2-llama.service.example): systemd user unit.
- [`cuda-kernels.patch`](cuda-kernels.patch): the llama.cpp patch, including its `test-backend-ops` cases.
- [`benchmark_chat.py`](benchmark_chat.py): long-context chat-endpoint benchmark.
- [`benchmark.py`](benchmark.py): the earlier recipes' 7.3K `/completion` benchmark.
- [`results/2026-09-26/`](results/2026-09-26/): contains:
  - `long-context-{stock,patched}-{short,62k,93k}.json`;
  - `august-shape-{stock,patched}.json`;
  - `kld-*.log` from `llama-perplexity`.

## References

- Huihui model repository: <https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF>
- llama.cpp base commit: <https://github.com/ggml-org/llama.cpp/commit/4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0>
- llama.cpp build documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- llama.cpp speculative decoding: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- MTP state issue: <https://github.com/ggml-org/llama.cpp/issues/27296>
- Proposed MTP state fix: <https://github.com/ggml-org/llama.cpp/pull/27173>
