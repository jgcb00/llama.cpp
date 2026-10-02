# Olala on llama.cpp — setup procedure for every hardware

Step-by-step: get the code, make the GGUF files once, then build and run on
**CPU (x86 / ARM)**, **Mac (Apple Silicon, Metal)** or **NVIDIA (CUDA)**.
Background, measurements and debug switches: [dragon.md](dragon.md).

| hardware | weights | status |
|---|---|---|
| CPU x86-64 (AVX-512 / AVX2) | **q5_k_m** (q6_k for best quality) | validated |
| CPU ARM64 (Linux, macOS without Metal) | q5_k_m | compiles; not benchmarked |
| Mac, Metal | **bf16** | written, **not yet run on a Mac** — follow [dragon-metal-testing.md](dragon-metal-testing.md) first |
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

## 2a. CPU — Linux / Windows (WSL) / macOS without Metal

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
./build/bin/llama-bench -m olala-q5_k_m.gguf -t 16 -fa 1 -p 512 -n 128
# reference, EPYC 9334 @ 16 threads: pp512 ≈ 530 t/s, tg128 ≈ 95 t/s
```

Serve:

```sh
./build/bin/llama-server -m olala-q5_k_m.gguf -t 16 -fa 1 \
    -ctk q8_0 -ctv q8_0 -c 32768 -np 4 --host 0.0.0.0 --port 8080
```

- `-t`: number of threads. On a shared machine use half the physical cores; more
  than the physical core count only slows it down.
- `-fa 1 -ctk q8_0 -ctv q8_0`: always on CPU (faster at long context, half the KV memory).
- `-np 4`: parallel slots (4 users ≈ 155 t/s total on 16 threads).

## 2b. Mac — Apple Silicon (Metal)

Requirements: Xcode command-line tools (`xcode-select --install`), CMake, macOS 14+
(bf16 needs Metal 3.1). Unified memory ≥ 16 GB for bf16.

```sh
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**First run the validation checklist** in [dragon-metal-testing.md](dragon-metal-testing.md)
(shader compile, `test-backend-ops -b MTL0`, CPU-vs-Metal greedy diff). The Metal kernels
have not been run on Apple hardware yet; send back the outputs listed in its section 6.

Once validated:

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
| CPU much slower than the table | too many threads (`-t` > physical cores, or machine busy), or `-fa` off |
| Mac: `failed to build 'mamba3' library` | Metal shader compile error — send the log (see Metal checklist) |
