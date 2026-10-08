# Olala / Dragon 7A1B in llama.cpp — conversion, quantization & usage

Olala (architecture name `dragon` in GGUF; HF classes `OlalaForCausalLM` /
`DragonForCausalLM`) is a hybrid LLM: 36 blocks in an `MMMMV` pattern — 29
**Mamba3-MIMO** recurrent mixers (rank-4 MIMO, trapezoid state update, rotary phase
state), 7 **Differential-TPA** attention mixers (48 q-heads / 12 kv-heads, head_dim
128, logit softcap 150, token shift, scalable softmax), a **256-expert / 6-active
MoE** with a dense shared expert, and geodesic-rotation residuals. 6.8 B params,
~1 B active, vocab 151936.

Branch: **`olala-master`** (rebased on upstream master of 2026-10-02).

**Setup procedure per hardware (CPU / Mac / NVIDIA): [olala-setup.md](olala-setup.md).**

---

## 1. Building

| target | command | notes |
|---|---|---|
| CPU (x86 / ARM) | `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` | fast kernels are ggml-cpu ops: AVX-512, AVX2 and NEON paths, runtime ISA dispatch in multi-variant builds |
| Apple (Metal) | `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` | validated on an M5 Pro (see [olala-setup.md](olala-setup.md) §2b); checklist in [dragon-metal-testing.md](dragon-metal-testing.md) |
| NVIDIA (CUDA) | `cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cuda -j` | on boxes with several CUDA toolkits pin `-DCMAKE_CUDA_COMPILER=… -DCMAKE_CUDA_HOST_COMPILER=…` |

No environment variables are needed: layers whose weights sit on a GPU automatically
use the portable graph (`GGML_OP_MAMBA3_MIMO` + `GGML_OP_GEODESIC` + standard MoE),
CPU layers use the CPU fast path.

## 2. Converting a checkpoint

```sh
python3 convert_hf_to_gguf.py /path/to/checkpoint --outtype bf16 --outfile olala-bf16.gguf
```

The directory needs `config.json` (with `layers_config`), `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` (embedded in the
GGUF). The converter also writes packed copies of the M-layer constants
(`blk.N.ssm_m3_*`, ~10 MB) used by the GPU kernels; GGUFs without them still work.

Always convert to **bf16**. Biases, MIMO projections, norms, geodesic scalars and
the MoE router stay F32 and are never quantized.

## 3. Quantization — read before running llama-quantize

Dragon has **massive outlier activations** (up to ~1e14 at two M-mixer outputs,
~1e9 after the MoE ReLU²). Quantized matmuls quantize activations on the fly:

| weight type | activation format | scale type | on Dragon |
|---|---|---|---|
| K-quants (q3_K…q6_K), iq4_xs | q8_K | fp32 | safe |
| type-0 (q4_0/q5_0/q8_0, iq4_nl) | q8_0 | fp16 | overflows → NaN where the input has outliers |
| any quant on CUDA / Metal (large batches) | fp16-scaled | fp16 | risky — use bf16 on GPUs |

`ffn_up_exps` and `ffn_latent_up` have 384 columns, which K-quants cannot encode, and
llama-quantize **silently falls back to type-0** for them — so plain recipes are
poisoned. Always pass:

```sh
SAFE="--tensor-type ffn_latent_up=bf16 --tensor-type ffn_up_exps=q8_0"
./build/bin/llama-imatrix -m olala-bf16.gguf -f wiki.test.raw -o olala.imatrix --chunks 120
./build/bin/llama-quantize --imatrix olala.imatrix $SAFE olala-bf16.gguf olala-q5_k_m.gguf q5_k_m
./build/bin/llama-quantize --imatrix olala.imatrix $SAFE olala-bf16.gguf olala-q4_k_m.gguf q4_k_m
./build/bin/llama-quantize --imatrix olala.imatrix $SAFE olala-bf16.gguf olala-q6_k.gguf   q6_k
# 8-bit tier: the q8_0 base needs three more protections
./build/bin/llama-quantize --tensor-type attn_output=bf16 --tensor-type ffn_down_exps=q6_k \
    --tensor-type ffn_down_shexp=bf16 $SAFE olala-bf16.gguf olala-q8_0-safe2.gguf q8_0
```

`ffn_up_exps`'s input is the bounded MoE latent, so a 4-bit 32-block type is safe
there too (`--tensor-type ffn_up_exps=iq4_nl`, plus `--output-tensor-type q4_k`):
-25% file size, same speed — useful for 8 GB machines.

### Measured (DPO-99k checkpoint; EPYC 9334, **16 threads**, `-fa 1 --no-repack`)

| variant | size | KLD vs bf16 | same top-1 | pp512 t/s | tg128 t/s |
|---|---|---|---|---|---|
| bf16 | 12.7 GiB | — | — | 530 | 56.8 |
| q8_0-safe2 | 6.4 GiB | 0.0040 | 96.9% | 501 | 84.8 |
| q6_k | 5.9 GiB | 0.0106 | 94.6% | 579 | 101.8 |
| **q5_k_m** | 5.6 GiB | 0.0194 | 92.8% | **598** | **108.9** |
| q4_k_m | 5.3 GiB | 0.0487 | 88.5% | 614 | 115.8 |
| q4_k_m + up iq4_nl + out q4_k | 4.0 GiB | 0.0610 | 87.2% | 596 | 113.1 |

KLD over 16 wikitext-2 chunks. **Recommendation: q5_k_m** (q4_k_m is 7% faster at
2.5× the divergence); q6_k when quality matters most. The box delivers ~190 GB/s and
a token reads ~1.0 GB of weights with q5_k_m, so ~190 t/s is the decode roofline;
the rest is per-node dispatch (~1100 graph nodes per token).

**`--no-repack` on x86** (`-nr` for llama-server/llama-cli, `--repack 0` for
llama-bench): ggml's interleaved "repack" GEMM kernels are slower than the plain
q4_K path on this model (q4_k_m: pp512 483 → 526, tg 94 → 99 before the kernels
below), and the fused MoE op cannot read repacked weights. q5_K/q6_K/q8_0 files are
not repacked on x86 anyway.

**CPU kernels** (2026-10-06): the whole MoE block of a layer runs as one ggml-cpu op
(`GGML_OP_DRAGON_MOE`: latent down, f32 router, top-k, experts, latent up, shared
expert; 3 thread barriers instead of ~14 nodes) for ubatches of ≤ 8 tokens, i.e.
decode and up to 8 concurrent slots (tg 92.6 → 109 t/s). The Mamba3-MIMO prefill sweep
is d-major with the trapezoid term folded into the state (shifted-γ identity of the
chunked reference); its inner loop runs at the AVX-512 FMA bound, 7.6 → 3.7 ms per
layer at pp512 (pp512 531 → 598 t/s).

**KV cache: use `-ctk q8_0 -ctv q8_0` on CPU** — half the attention KV memory and
faster at depth (q5_k_m decode: 52 t/s at 24k context vs 47 with f16; prefill equal).

Long context / multi-user (q5_k_m, 16 threads): pp8192 405 t/s; decode 86 t/s at 1k,
67 at 8k, 52 at 24k (q8_0 KV; before the 2026-10-06 kernels); 1/2/4/8 concurrent users
103/135/173/184 t/s total.

## 4. Running

```sh
./build/bin/llama-server -m olala-q5_k_m.gguf -t <threads> -fa 1 -c 32768 -np 4
```

- Chat: the embedded template and a dedicated Olala parser give OpenAI-style
  `reasoning_content` / `content` / `tool_calls` (streaming included).
  `reasoning_effort` (`none` … `high`) is passed through to the template.
- `-fa 1` always on CPU. Memory fitting (`-fit`, default) works.
- Several slots (`-np N`) are fine. Concurrent requests can differ slightly from
  single-request output (batch-shape rounding in the CPU GEMMs), never corrupted.
- Recurrent state: kept in **f32** (a reduced-precision state degrades long
  generations into repetition loops); the rotary phase is kept wrapped to [-π, π].
  `DRAGON_BF16_STATE=1` stores the K/V sections bf16 (not recommended).
- Low RAM: `--lazy-mode on` keeps the routed experts and the token embeddings
  on disk and pages the selected experts in per layer (CPU fused MoE op: read-ahead
  after routing, release after use). ≈ 1.2 GiB RSS for q5_k_m; 27 t/s decode under a
  2 GiB cap from NVMe. Details and measurements: [olala-setup.md](olala-setup.md)
  (low-RAM mode).
- GPU: `-ngl 99`, **bf16 weights** (see §3). H100 PCIe bf16: pp8192 5313 t/s,
  tg128 163 t/s.

## 5. Environment variables (debug / A-B only)

| variable | effect |
|---|---|
| `DRAGON_PORTABLE=1` | force the GPU graph on CPU (reference for backend checks) |
| `DRAGON_CPU_CUSTOM_M=1`, `DRAGON_CPU_CUSTOM_GEO=1` | legacy libllama custom-op CPU kernels |
| `DRAGON_NO_INPLACE_STATE=1` | gather + copy the recurrent state instead of updating it in place |
| `DRAGON_NO_FUSED_MOE/SHIFT/GEO=1` | ggml-primitive MoE / token shift / geodesic on CPU |
| `DRAGON_FUSED_MOE_MAX_T=N` | largest ubatch handled by the fused MoE op (default 8, 0 disables) |
| `GGML_M3_LEGACY=1` | previous (blocked, p-major) Mamba3 CPU kernel for every ubatch size |
| `DRAGON_EXPERT_CACHE_MB=N` | low-RAM mode: keep the N MiB of most recently used experts mapped (default 0) |
| `DRAGON_EXPERT_DROP_CACHE=1` | low-RAM mode: also drop released experts from the OS page cache (strict, slower) |
| `DRAGON_BF16_STATE=1` | bf16 K/V state sections |
| `DRAGON_M_PRIM=1` / `DRAGON_M_CHUNK_SIZE=N` / `DRAGON_M_DECODE_PRIM=1` | closed-form primitive reference paths (single sequence) |
| `GGML_OP_PROFILE=1` (`GGML_OP_PROFILE_TOP=N`) | per-op CPU time tables at exit |

## 6. Known limitations

- Low-RAM mode (`--lazy-mode on`) is CPU-only and untested with GPU offload
  (`-ngl`): lazy experts always live in host memory and only the CPU MoE op
  releases them.
- Quantized weights on GPUs: fp16 activation scales overflow on Dragon's outliers.
- Recurrent models cannot context-shift; prompts beyond the context fail.
- Truncated tool calls at `max_tokens` are returned partially (shared llama.cpp
  behaviour; vLLM drops them).
