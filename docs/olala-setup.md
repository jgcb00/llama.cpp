# Olala on llama.cpp — setup procedure for every hardware

Step-by-step: get the code, make the GGUF files once, then build and run on
**CPU (x86 / ARM)**, **Mac (Apple Silicon, Metal)** or **NVIDIA (CUDA)**.
Background, measurements and debug switches: [dragon.md](dragon.md).

| hardware | weights | status |
|---|---|---|
| CPU x86-64 (AVX-512 / AVX2) | **q5_k_m** (q6_k for best quality) | validated |
| CPU, ≈ 1.2–2 GiB of free RAM (laptops) | q5_k_m or the 4.0 GiB small file, `--lazy-mode on` | measured on Linux x86; paging test passes on Windows/macOS CI ([low-RAM mode](#low-ram-mode-experts-read-from-disk)) |
| CPU ARM64 (Linux, macOS without Metal) | q5_k_m | compiles; not benchmarked |
| Mac, Metal | **bf16** | validated (M5 Pro 24 GB, macOS 26.4): pp512 ≈ 2070 t/s, tg128 ≈ 70 t/s |
| NVIDIA, CUDA | **bf16** | validated (H100) |

Quantized weights on GPUs (CUDA, Metal) can produce NaN/garbage: Dragon's outlier
activations overflow the fp16 activation scales those backends use. Use bf16 there.

---

## 0. Get the code

```sh
git clone -b olala-master https://github.com/jgcb00/llama.cpp.git llama.cpp-olala
cd llama.cpp-olala
```

## 1. Make the GGUF files (once, on any Linux/Mac machine)

Requirements: Python ≥ 3.10, ~30 GB free disk, a CPU build of this repo for the
quantizer (step 2 "CPU" below).

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# HF checkpoint dir: config.json, model.safetensors, tokenizer.json,
# tokenizer_config.json, chat_template.jinja
CKPT=/path/to/dpo-checkpoint-99k-newRLHF
python3 convert_hf_to_gguf.py $CKPT --outtype bf16 --outfile olala-bf16.gguf
```

Converting straight from a private HF repo (`convert_hf_to_gguf.py OVHaiLLM/olala-7a1b-dpo99k --remote …`)
uses the token saved by `hf auth login` (or `HF_TOKEN`). Two caveats: `--remote` does not fetch
`chat_template.jinja`, so download the checkpoint instead if you want the template in the GGUF
(see the troubleshooting table); and if the Xet downloads keep failing, set `HF_HUB_DISABLE_XET=1`.

`olala-bf16.gguf` (12.7 GiB) is the GPU file. For CPU, quantize it — **always with the
safe overrides** (plain llama-quantize recipes silently produce NaN-prone files on this
model, see [dragon.md §3](dragon.md)):

```sh
SAFE="--tensor-type ffn_latent_up=bf16 --tensor-type ffn_up_exps=q8_0"
# importance matrix (~15 min on 16 threads); any text corpus works, wikitext-2 used here
./build/bin/llama-imatrix -m olala-bf16.gguf -f wiki.test.raw -o olala.imatrix --chunks 120 -t 16
./build/bin/llama-quantize --imatrix olala.imatrix $SAFE olala-bf16.gguf olala-q5_k_m.gguf q5_k_m 16
./build/bin/llama-quantize --imatrix olala.imatrix $SAFE olala-bf16.gguf olala-q6_k.gguf   q6_k   16
```

Small-RAM variant (4.0 GiB, for 8 GB machines, CPU only):

```sh
./build/bin/llama-quantize --imatrix olala.imatrix --tensor-type ffn_latent_up=bf16 \
    --tensor-type ffn_up_exps=iq4_nl --output-tensor-type q4_k \
    olala-bf16.gguf olala-q4_k_m-small.gguf q4_k_m 16
```

Sanity check of any file (must print a finite perplexity; bf16 ≈ 13 on 8 wikitext-2 chunks):

```sh
./build/bin/llama-perplexity -m olala-q5_k_m.gguf -f wiki.test.raw --chunks 8 -t 16 -fa 1
```

---

## 2a. CPU — Linux / Windows / macOS without Metal

Build:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

The fast Dragon kernels are ggml-cpu ops with AVX-512, AVX2 and NEON paths. For a binary
that runs on several CPU generations, add `-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`
(runtime ISA dispatch).

Check it works:

```sh
./build/bin/llama-bench -m olala-q5_k_m.gguf -t 16 -fa 1 --repack 0 -p 512 -n 128
# reference, EPYC 9334 @ 16 threads: pp512 ≈ 600 t/s, tg128 ≈ 109 t/s (q4_k_m: 615 / 116)
```

Serve:

```sh
./build/bin/llama-server -m olala-q5_k_m.gguf -t 16 -fa 1 -nr \
    -ctk q8_0 -ctv q8_0 -c 32768 -np 4 --host 0.0.0.0 --port 8080
```

- `-t`: number of threads. On a shared machine use half the physical cores; more
  than the physical core count only slows it down.
- `-fa 1 -ctk q8_0 -ctv q8_0`: always on CPU (faster at long context, half the KV memory).
- `-nr` (`--no-repack`): the x86 "repack" GEMM kernels are slower on this model and
  block the fused MoE op; always pass it on CPU (`--repack 0` for llama-bench).
- `-np 4`: parallel slots (4 users ≈ 173 t/s total on 16 threads, 8 users ≈ 184).

### Low-RAM mode: experts read from disk

The routed experts are 82% of the file, but a token only uses 6 of the 256 per layer.
With `--lazy-mode on` they stay in the GGUF on disk: after routing, each layer reads
the selected experts (≈ 114 MiB per token for q5_k_m, 84 MiB for the small variant)
and hands them back to the kernel once used. The rest of the model (≈ 0.85 GiB for
q5_k_m, 0.7 GiB small) plus ≈ 0.3 GiB of buffers stays in RAM.

```sh
./build/bin/llama-server -m olala-q5_k_m.gguf -t 16 -fa 1 -nr --lazy-mode on \
    -ctk q8_0 -ctv q8_0 -c 8192 -np 1 --port 8080
```

- Put the GGUF on an **SSD/NVMe**: every token is a burst of random reads. From a
  hard-disk RAID decode drops to ≈ 1 t/s.
- Keep `-c` modest: the KV cache is regular RAM (64k context = 2.7 GiB in f16).
  Do not lower `-ub`: prefill re-reads the experts once per micro-batch.
- Output is identical to the normal fused CPU path (greedy, token for token).
- Experts that were released stay in the OS page cache, which the kernel gives back
  under memory pressure, so the mode adapts to the RAM that is free. To keep the page
  cache out of it too (strict footprint, slower), set `DRAGON_EXPERT_DROP_CACHE=1`.
  `DRAGON_EXPERT_CACHE_MB=N` keeps the N MiB of most recently used experts mapped.
- Linux, Windows and macOS all release the experts after use (Linux `madvise`,
  Windows `VirtualUnlock` = out of the working set, macOS `madvise`), and all read
  them ahead in large requests (`madvise` / `PrefetchVirtualMemory`). Linux also maps
  them in one call per thread; on Windows and macOS this is page faults, a bit slower.
  The CI test `test-dragon-moe-paging` checks the release on Linux and Windows
  (macOS does not count clean file-backed pages in a process's memory anyway).
- Windows laptop: in a terminal from the "x64 Native Tools" prompt (or any shell
  with Visual Studio 2022 + CMake),

  ```bat
  cmake -B build -DLLAMA_CURL=OFF
  cmake --build build --config Release -j
  build\bin\Release\llama-server.exe -m olala-q4_k_m-small.gguf -t 6 -fa 1 -nr ^
      --lazy-mode on -c 8192 --port 8080
  ```

  `-t` = number of performance cores (not threads). Close the browser tabs you do
  not need: the model's working set is ≈ 1.1–1.3 GiB, the rest of the RAM is cache.

Measured on EPYC 9334, 16 threads, Micron 7450 NVMe, cold page cache, RAM capped with a
cgroup (`systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`, page cache
included in the cap):

| file | mode | RAM cap | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| q5_k_m (5.6 GiB) | normal | none (5.9 GiB RSS) | 592 | 109 |
| q5_k_m | normal | 2 GiB | 104 | 10.6 |
| q5_k_m | low-RAM | 2 GiB | 174 | 27.5 |
| q5_k_m | low-RAM | 1.5 GiB | 128 | 21.1 |
| q5_k_m | low-RAM | 1.2 GiB | 143 | 11.7 |
| small (4.0 GiB) | normal | 2 GiB | 124 | 18.5 |
| small | low-RAM | 2 GiB | 298 | 41.2 |
| small | low-RAM | 1.5 GiB | 296 | 30.6 |
| small | low-RAM | 1.2 GiB | 199 | 26.3 |
| small | low-RAM | 1 GiB | 198 | 15.9 |

Below ≈ 1.2 GiB (q5_k_m) / 1 GiB (small) the always-used weights no longer fit and
decode collapses (≈ 2.7 t/s). Without a cap the process stays at ≈ 1.2–1.3 GiB RSS and
runs at 65–70 t/s once the file sits in the page cache.

## 2b. Mac — Apple Silicon (Metal)

Requirements: Xcode command-line tools (`xcode-select --install`), CMake, macOS 14+
(bf16 needs Metal 3.1). Unified memory ≥ 16 GB for bf16.

```sh
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Validated on an Apple M5 Pro (24 GB, macOS 26.4.1, PR #1): op tests 18/18, CPU-vs-Metal
greedy output identical on 15- and 481-token prompts, pp512 ≈ 2070 t/s, pp2048 ≈ 2250 t/s,
tg128 ≈ 70 t/s (49 t/s at 32k context), 8 users × 128 tokens ≈ 290 t/s total. On a new
machine or macOS version it is still worth running the checklist in
[dragon-metal-testing.md](dragon-metal-testing.md) (`test-backend-ops -b MTL0`,
CPU-vs-Metal greedy diff) before relying on the output.

Then:

```sh
./build/bin/llama-bench  -m olala-bf16.gguf -ngl 99 -fa 1 -p 512 -n 128
./build/bin/llama-server -m olala-bf16.gguf -ngl 99 -fa 1 -c 32768 -np 4 --port 8080
```

If Metal fails or the machine has < 16 GB: run on the Mac CPU instead with a quantized file
— same build, add `-ngl 0` (or `--device none`) and use `olala-q5_k_m.gguf`
(or the 4.0 GiB small variant), `-t` = number of performance cores.

## 2c. NVIDIA — CUDA

Requirements: CUDA toolkit ≥ 12, a GPU with ≥ 16 GB (bf16 weights 12.7 GiB + KV).

```sh
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 16
# several toolkits installed? pin them:
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.x/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
```

Check the Dragon ops against the CPU reference, then benchmark:

```sh
./build-cuda/bin/test-backend-ops -o MAMBA3_MIMO
./build-cuda/bin/test-backend-ops -o GEODESIC
./build-cuda/bin/llama-bench -m olala-bf16.gguf -ngl 99 -fa 1 -p 8192 -n 128
# reference, H100 PCIe: pp8192 ≈ 5300 t/s, tg128 ≈ 163 t/s
```

Serve:

```sh
CUDA_VISIBLE_DEVICES=0 ./build-cuda/bin/llama-server -m olala-bf16.gguf -ngl 99 -fa 1 \
    -c 65536 -np 8 --host 0.0.0.0 --port 8080
```

No environment variable is needed: layers on the GPU automatically use the fused
`MAMBA3_MIMO` / `GEODESIC` kernels. Partial offload (`-ngl N < 37`) works; the CPU layers
use the CPU kernels.

---

## 3. Using the server (all hardware)

OpenAI-compatible API on `http://<host>:8080/v1`. The chat template is embedded in the
GGUF and a dedicated Olala parser returns `reasoning_content` (analysis channel),
`content` (final channel) and `tool_calls`, streaming included.

```sh
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {
     "name": "get_weather",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "reasoning_effort": "low",
  "temperature": 0.6
}' | python3 -m json.tool
```

- `reasoning_effort`: `none`, `low`, `medium`, `high` (passed to the template).
- The recurrent (Mamba) state is kept in f32 — do not set `DRAGON_BF16_STATE`
  (long generations degrade into repetition loops with a bf16 state).
- Prompts longer than `-c` fail (recurrent models cannot context-shift): size `-c` for
  your longest conversation × `-np`.

## 4. Troubleshooting

| symptom | cause / fix |
|---|---|
| NaN / `?????` / empty output with a quantized GGUF on GPU | fp16 activation-scale overflow; use bf16 on GPUs |
| NaN with a quantized GGUF on CPU | file made without the `$SAFE` overrides; re-quantize |
| `unknown model architecture: 'dragon'` | binary not built from `olala-master` |
| no `reasoning_content` / raw `<\|channel_start\|>` in output | GGUF converted without `chat_template.jinja`; reconvert |
| CPU much slower than the table | too many threads (`-t` > physical cores, or machine busy), `-fa` off, or repack on (add `-nr` / `--repack 0`) |
| Mac: `failed to build 'mamba3' library` | Metal shader compile error — send the log (see Metal checklist) |
