# Dragon 7A1B in llama.cpp — conversion, quantization & usage guide

Dragon 7A1B is a hybrid LLM: 36 blocks in an `MMMMV` pattern — 29 **Mamba3-MIMO**
recurrent mixers (rank-4 MIMO, trapezoid state update, halved rotary with cumulative
angles), 7 **Differential-TPA-V2** attention mixers (GQA-4, 48 q-heads, head_dim 128,
logit softcap, token shift, scalable softmax), a **256-expert / 6-active MoE** with a
dense shared expert, and geodesic-rotation residuals. 6.8 B params, ~1 B active,
vocab 151936, 64k train context.

Branch to use: **`dragon-gpu-master`** (tracks upstream master; carries the arch, the
`GGML_OP_MAMBA3_MIMO` op with CPU + CUDA backends, and all CPU/GPU performance work).

---

## 1. Building

CPU (AVX-512 machine recommended — the fast kernels are AVX-512F/DQ):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j
```

CUDA (tested on H100 / CUDA 13.2 — pin the compiler explicitly, mixed system
toolchains break the arch detection):

```sh
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
      -DGGML_CUDA_COMPRESSION_MODE=""
cmake --build build-cuda -j
```

---

## 2. Converting the HF checkpoint to GGUF

The converter (`conversion/dragon.py`) registers `DragonForCausalLM` with the
standard converter, so conversion is the stock command:

```sh
python3 convert_hf_to_gguf.py /path/to/dragon_checkpoint \
        --outfile dragon-7a1b-bf16.gguf --outtype bf16
```

The HF directory must contain `config.json` (with `layers_config` — the per-block
M/V letter string), `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
and `chat_template.jinja` (embedded into the GGUF automatically).

Notes:
- **Always convert to `bf16`** as the base. The checkpoint is bf16; f16 would clip
  nothing today but bf16 keeps the exponent range that this model actually uses
  (see the outlier discussion below).
- Small mixers' tensors (MIMO projections, biases, geodesic scalars, MoE router
  `ffn_gate_inp`, norms) are stored **F32** and are never quantized by any later
  step — this is intentional and required.
- Sanity check after conversion:
  `./build/bin/dragon-generate -m dragon-7a1b-bf16.gguf -t 16 -n 32 --incremental -p "Once upon a time"`
  should produce fluent English.

---

## 3. Quantization — read this before running llama-quantize

### 3.1 The one thing you must know

Dragon has **massive outlier activations** (up to ~1e14 at the M-mixer output of two
layers; ~1e9 after the MoE ReLU²). Quantized matmuls quantize *activations* on the
fly, and the storage format of those activation scales decides everything:

| weight type | activation format | scale type | outcome on Dragon |
|---|---|---|---|
| K-quants (q3_K…q6_K) | q8_K | **fp32** | ✅ safe |
| type-0 (q4_0/q5_0/q8_0) | q8_0 | fp16 (max 65504) | ❌ overflows → NaN |
| any quant on **CUDA** | q8_1 | fp16 | ❌ NaN (see §5) |

Two of Dragon's tensor classes (`ffn_up_exps`, `ffn_latent_up`) have **384 columns**,
which K-quants cannot encode (they need multiples of 256). `llama-quantize` then
**silently falls back to type-0** — which poisons every default recipe. All standard
mixes (plain `q4_k_m`, `q6_k`, `q8_0`, …) produce a model that generates plausibly on
short prompts and then collapses to NaN/garbage on real prompts.

### 3.2 The safe recipe

Always pass these overrides:

```sh
SAFE="--tensor-type ffn_latent_up=bf16 --tensor-type ffn_up_exps=q8_0"

# recommended: importance matrix from ~64 chunks of general text
./build/bin/llama-imatrix -m dragon-7a1b-bf16.gguf -f wiki.train.raw -o dragon.imatrix --chunks 64

# the ladder
./build/bin/llama-quantize --imatrix dragon.imatrix $SAFE dragon-7a1b-bf16.gguf dragon-q4km-safe.gguf q4_k_m
./build/bin/llama-quantize --imatrix dragon.imatrix $SAFE dragon-7a1b-bf16.gguf dragon-q5km-safe.gguf q5_k_m
./build/bin/llama-quantize                          $SAFE dragon-7a1b-bf16.gguf dragon-q6k-safe.gguf  q6_k
# 8-bit tier ("fp8-like"): q8_0 base needs three extra protections
./build/bin/llama-quantize --tensor-type attn_output=bf16 --tensor-type ffn_down_exps=q6_k \
        --tensor-type ffn_down_shexp=bf16 $SAFE dragon-7a1b-bf16.gguf dragon-q8-safe2.gguf q8_0
```

Why each override: `ffn_latent_up` (384-col, its *input* carries the 1e8+ MoE
outliers — bf16, it is tiny: 21 M params) · `ffn_up_exps` (384-col; its input is
bounded, so q8_0 weights are fine *on CPU*) · in the q8_0 base additionally
`attn_output` (input ~1e14), `ffn_down_exps`/`ffn_down_shexp` (inputs are ReLU²-squared,
~5e8) must not be type-0.

### 3.3 Measured quality & speed (EPYC 9334, 32 threads)

| variant | size | wiki PPL (32 chk) | gsm8k | MMLU-gen | humaneval+ | decode t/s | RAM @32k ctx |
|---|---|---|---|---|---|---|---|
| bf16 | 13.6 GB | 8.211 | 64.0 | 45.1 | 39.6 | 49 | 14.7 GB |
| **q8-safe2** | 7.3 GB | **8.212 (lossless)** | **65.2** | 45.3 | 39.0 | 62 | 8.9 GB |
| q6k-safe | 6.3 GB | 8.252 (+0.5%) | — | — | — | 74 | 7.9 GB |
| q5km-safe | 6.0 GB | 8.295 (+1.0%) | — | — | — | 77 | 7.6 GB |
| **q4km-safe** | 5.7 GB | 8.403 (+2.3%) | 61.2 | 44.8 | 39.0 | **79** | 8.3 GB* |
| q3km-safe | 5.2 GB | 9.146 (+11%) | — | — | — | 79 | — |

\* includes the 1.1 GB repack buffer; `--no-repack` trades ~20% prefill for −1.1 GB.

**Recommendations**: `q4km-safe` is the sweet spot (max speed, ~3 gsm8k points inside
the error bar); `q8-safe2` when you want provably-lossless; below q4 the quality cliff
is steep. Per-tensor sensitivity if you build custom mixes (ΔPPL when demoted to
q3_K): shared expert +0.39 ≫ attn_output +0.17 > ssm_in +0.13 > routed experts +0.08
> lm_head +0.03 ≈ embeddings 0 — i.e. keep the *shared* expert high, squeeze the
routed experts and embeddings freely. The MoE router is F32 and untouchable.

### 3.4 KV cache quantization (free on CPU)

`-ctk q8_0 -ctv q8_0` halves the attention KV cache (42 → ~22 KB/token) with **no
measurable quality loss** and, with the blocked FA kernel on this branch, **no speed
loss** (decode at depth 8k: 57.4 vs 57.4–58.9 t/s f16). Use it whenever RAM matters.

---

## 4. Running on CPU

```sh
./build/bin/llama-completion -m dragon-q4km-safe.gguf -t <physical cores> -fa 1 -fit off -p "..."
```

- **`-fa 1` always** — the blocked/tiled FA kernels are a large win (up to +84%
  multi-user long-context). `-fit off` is currently required (memory-fitting probe
  segfaults on hybrid models — known issue).
- Do not exceed physical core count for `-t` (SMT collapses throughput).
- **llama-server**: run with `-np 1` (one slot). An upstream bug corrupts recurrent
  state with concurrent slots (affects all recurrent/hybrid models, not just Dragon).
  For parallel batch throughput use `llama-parallel` / `llama-batched-bench`, which
  are safe.
- Reference numbers (32 threads, q4km-safe): decode 79 t/s short-ctx / 37.6 t/s at
  24k depth; prefill ~540 t/s short, 369 t/s at 24k (bf16). Memory: ~6 GB @4k,
  ~8.3 GB @32k, ~12 GB @128k (see the KV-q8 and no-repack levers above).
- Recurrent-state note: the M-layer state cache is stored bf16-packed by default
  (validated bit-equivalent generation). `DRAGON_F32_STATE=1` reverts.

## 5. Running on GPU (CUDA)

**Use bf16 on GPU. Do not use quantized GGUFs with CUDA visible.** CUDA quantized
matmuls use fp16 activation scales (q8_1) with no fp32-scale option, so Dragon's
outliers NaN the recurrent state — q4 fails at the first token, q8 within ~30 tokens.
This *also* applies at `-ngl 0` (large-batch matmuls auto-offload): for CPU runs of
quantized models on a GPU machine, hide the devices (`CUDA_VISIBLE_DEVICES=`).

Full-offload environment (enables the GGML op for the M-recurrence and the
GPU-capable primitive forms of the fused CPU ops):

```sh
export DRAGON_GGML_OP=1 DRAGON_NO_FUSED_MOE=1 DRAGON_NO_FUSED_SHIFT=1 DRAGON_NO_FUSED_GEO=1
./build-cuda/bin/llama-completion -m dragon-7a1b-bf16.gguf -ngl 99 -fa 1 -fit off -p "..."
```

Measured (H100 PCIe, bf16, 12.5 GB VRAM + ~41 MB/seq context): prefill 8k **937 t/s**,
decode 60 t/s single-stream, **262 t/s total at 8 concurrent users** (multi-sequence
is fully supported and token-exact vs single-stream). Generation is token-identical
to the CPU path. The M-op uses a chunk-parallel prefill kernel (exact decomposition,
chunk 32) and a shared-memory serial kernel for decode.

## 6. Environment-variable reference

| variable | effect |
|---|---|
| `DRAGON_GGML_OP=1` | M-recurrence via `GGML_OP_MAMBA3_MIMO` (required for GPU; CPU: same results, slightly slower than the fused custom op) |
| `DRAGON_NO_FUSED_MOE/SHIFT/GEO=1` | replace fused CPU custom ops with ggml primitives (required for GPU offload) |
| `DRAGON_F32_STATE=1` | store recurrent state f32 instead of bf16-packed |
| `DRAGON_STATE_Q8=1` | experimental int8 state (quality OK, ~7% slower — not recommended) |
| `DRAGON_M_PRIM=1` / `DRAGON_M_CHUNK_SIZE=N` / `DRAGON_M_DECODE_PRIM=1` | debug/reference primitive paths (single-sequence only) |
| `GGML_OP_PROFILE=1` | per-op CPU wall-time tables at exit |
| `GGML_FA_PROFILE=1` | rdtsc section profile of the FA prefill kernel |

## 7. Known issues

- `-fit on` (default memory fitting) segfaults → always `-fit off`.
- llama-server concurrent slots corrupt recurrent state (upstream bug) → `-np 1`.
- Quantized weights + CUDA = NaN (fp16 activation scales; see §5).
- Recurrent models cannot context-shift; requests beyond the context fail rather
  than slide. Prompt-prefix reuse across requests is not available for the M-state.
- lm-eval via llama-server: use generative tasks (`mmlu_generative`, `gsm8k`,
  `humaneval_plus`); the completions API does not expose loglikelihood logprobs.
