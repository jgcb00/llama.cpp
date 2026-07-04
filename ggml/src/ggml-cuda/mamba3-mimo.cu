#include "mamba3-mimo.cuh"

// Dragon Mamba3-MIMO trapezoid recurrence (see ggml_mamba3_mimo in ggml.h):
//   kv[t]  = sum_r k[:,r,t] v[:,r,t]^T
//   S[t]   = alpha_t * S[t-1] + beta_t * kv[t-1] + gamma_t * kv[t]   (kv[-1] = 0)
//   y_r[t] = q[:,r,t]^T S[t]
//
// semantic reference: ggml_compute_forward_mamba3_mimo (ggml/src/ggml-cpu/ops.cpp).
//
// one block per (head, seq); the t-loop is serial inside the block so the state
// is loaded/stored once per ubatch (T == 1 is the decode step, larger T is the
// prefill v1 that amortizes the state load).
//
// shared memory holds the current/previous token q/k/v head slices (the beta
// term needs kv[t-1], recomputed from the double-buffered k/v) and, when it
// fits (D_qk*D_v*4 + staging <= 48 KB), the running state; otherwise the state
// lives directly in its final-state slot of dst, like the CPU forward.
template <bool state_in_smem>
static __global__ void mamba3_mimo_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ coefs,
        const float * __restrict__ s_in,
        float       *              dst,
        const int64_t              D_qk,
        const int64_t              R,
        const int64_t              D_v,
        const int64_t              H,
        const int64_t              n_tokens,
        const int64_t              n_seqs) {
    const int64_t h   = blockIdx.x; // head
    const int64_t seq = blockIdx.y; // sequence

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    // per-token strides in floats
    const int64_t qk_tok = D_qk * R * H;
    const int64_t v_tok  = D_v  * R * H;
    const int64_t y_row  = D_v  * R * H;

    const int64_t qk_head = D_qk * R;   // per-head slice inside one token
    const int64_t v_head  = D_v  * R;
    const int64_t SD      = D_qk * D_v; // state elements per (seq, head)

    // output layout: [y rows | final state rows], all rows D_v*R*H wide
    float * y_base     = dst;
    float * state_base = dst + y_row * n_tokens * n_seqs;

    // final-state slot for this (seq, head); layout per seq: (D_qk, D_v, H)
    float       * S_out  = state_base + (seq * H + h) * SD;
    const float * S_init = s_in       + (seq * H + h) * SD;

    // shared memory: [ qbuf | kbuf x2 | vbuf x2 | (state) ]
    extern __shared__ float smem[];
    float * qbuf = smem;
    float * kbuf = qbuf + qk_head;
    float * vbuf = kbuf + 2 * qk_head;
    float * S    = state_in_smem ? vbuf + 2 * v_head : S_out;

    for (int64_t i = tid; i < SD; i += nth) {
        S[i] = S_init[i];
    }
    __syncthreads();

    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t tok = seq * n_tokens + t;
        const int64_t cur =  t      & 1;
        const int64_t prv = (t + 1) & 1;

        const float * q_t = q + tok * qk_tok + h * qk_head;
        const float * k_t = k + tok * qk_tok + h * qk_head;
        const float * v_t = v + tok * v_tok  + h * v_head;

        // stage this token's q/k/v head slices; the previous token's k/v stay
        // in the other half of the double buffer (trapezoid beta term)
        for (int64_t i = tid; i < qk_head; i += nth) {
            qbuf[i]                 = q_t[i];
            kbuf[cur * qk_head + i] = k_t[i];
        }
        for (int64_t i = tid; i < v_head; i += nth) {
            vbuf[cur * v_head + i] = v_t[i];
        }
        __syncthreads();

        const float * cf    = coefs + 3 * (h + H * (t + n_tokens * seq));
        const float   alpha = cf[0];
        const float   beta  = cf[1];
        const float   gamma = cf[2];

        const float * k_c = kbuf + cur * qk_head;
        const float * v_c = vbuf + cur * v_head;
        const float * k_p = kbuf + prv * qk_head;
        const float * v_p = vbuf + prv * v_head;

        // S[p*D_qk + d] = alpha*S + beta*kv_prev + gamma*kv_cur
        for (int64_t i = tid; i < SD; i += nth) {
            const int64_t d = i % D_qk;
            const int64_t p = i / D_qk;

            float c_acc = 0.0f;
            for (int64_t r = 0; r < R; ++r) {
                c_acc += v_c[r * D_v + p] * k_c[r * D_qk + d];
            }
            float p_acc = 0.0f;
            if (t > 0) {
                // kv[-1] is zero: the cross-batch carry is folded into s0 by the caller
                for (int64_t r = 0; r < R; ++r) {
                    p_acc += v_p[r * D_v + p] * k_p[r * D_qk + d];
                }
            }
            S[i] = alpha * S[i] + beta * p_acc + gamma * c_acc;
        }
        __syncthreads();

        // y_t[r*D_v + p] = q[:,r,t] . S[p,:]
        float * y_t = y_base + tok * y_row + h * v_head;
        for (int64_t j = tid; j < v_head; j += nth) {
            const int64_t r = j / D_v;
            const int64_t p = j % D_v;

            const float * S_row = S    + p * D_qk;
            const float * q_r   = qbuf + r * D_qk;

            float acc = 0.0f;
            for (int64_t d = 0; d < D_qk; ++d) {
                acc += S_row[d] * q_r[d];
            }
            y_t[j] = acc;
        }
        // protects qbuf and the prv halves of kbuf/vbuf, overwritten next iteration
        __syncthreads();
    }

    if constexpr (state_in_smem) {
        for (int64_t i = tid; i < SD; i += nth) {
            S_out[i] = S[i];
        }
    }
}

// chunk-parallel prefill: the recurrence is linear (affine) in S, so with
// chunks of MAMBA3_CHUNK tokens:
//
//   S_end(n) = P_n * S_end(n-1) + L_n
//
// where P_n = prod(alpha over chunk n) and L_n is the chunk-local state
// obtained by running the recurrence from S = 0. the trapezoid beta term at a
// chunk start uses the REAL previous token (global t0-1, one-token lookback
// into the previous chunk); kv[-1] = 0 only at global t = 0.
//
//   phase 1: per (chunk, h, seq) block: compute L_n and P_n         (parallel)
//   phase 2: per (h, seq) block: scan S_end(n) = P_n*S_end(n-1)+L_n; leaves
//            the carry-in state of each chunk in the workspace and writes the
//            final state to dst                                     (serial over chunks, cheap)
//   phase 3: per (chunk, h, seq) block: serial-kernel body seeded with the
//            carry-in state, computing y[t]                         (parallel)
//
// phases 1 and 3 redo the state updates (2x the serial flops) but expose
// n_chunks-fold more parallelism, which is what the serial kernel lacks.

#define MAMBA3_CHUNK 32

// shared body for phases 1 (compute_y = false: local state + decay product)
// and 3 (compute_y = true: outputs from the carry-in state)
template <bool state_in_smem, bool compute_y>
static __global__ void mamba3_mimo_chunk_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ coefs,
        float       * __restrict__ wstate, // (n_chunks, h, seq) x D_qk*D_v: phase-1 out / phase-3 in
        float       * __restrict__ wdecay, // (n_chunks, h, seq): per-chunk alpha product (phase-1 out)
        float       *              dst,
        const int64_t              D_qk,
        const int64_t              R,
        const int64_t              D_v,
        const int64_t              H,
        const int64_t              n_tokens,
        const int64_t              n_seqs,
        const int64_t              n_chunks) {
    const int64_t nc  = blockIdx.x; // chunk
    const int64_t h   = blockIdx.y; // head
    const int64_t seq = blockIdx.z; // sequence

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    const int64_t t0 = nc * MAMBA3_CHUNK;                                              // first token of the chunk
    const int64_t tc = n_tokens - t0 < MAMBA3_CHUNK ? n_tokens - t0 : MAMBA3_CHUNK;    // tokens in the chunk

    // per-token strides in floats
    const int64_t qk_tok = D_qk * R * H;
    const int64_t v_tok  = D_v  * R * H;
    const int64_t y_row  = D_v  * R * H;

    const int64_t qk_head = D_qk * R;
    const int64_t v_head  = D_v  * R;
    const int64_t SD      = D_qk * D_v;

    float * y_base = dst;

    // workspace slot for this (seq, head, chunk); chunks contiguous per (seq, head)
    float * W = wstate + ((seq * H + h) * n_chunks + nc) * SD;

    // shared memory: [ (qbuf) | kbuf x2 | vbuf x2 | (state) ]
    extern __shared__ float smem[];
    float * qbuf = smem; // phase 3 only
    float * kbuf = smem + (compute_y ? qk_head : 0);
    float * vbuf = kbuf + 2 * qk_head;
    float * S    = state_in_smem ? vbuf + 2 * v_head : W;

    if constexpr (compute_y) {
        // seed with the carry-in state (already in place when W is the state)
        if constexpr (state_in_smem) {
            for (int64_t i = tid; i < SD; i += nth) {
                S[i] = W[i];
            }
        }
    } else {
        for (int64_t i = tid; i < SD; i += nth) {
            S[i] = 0.0f;
        }
    }

    // one-token lookback: the beta term at chunk-local t = 0 uses the previous
    // chunk's last token; stage it in the prv half of the double buffer
    if (t0 > 0) {
        const int64_t ptok = seq * n_tokens + t0 - 1;
        const float * k_pt = k + ptok * qk_tok + h * qk_head;
        const float * v_pt = v + ptok * v_tok  + h * v_head;
        for (int64_t i = tid; i < qk_head; i += nth) {
            kbuf[1 * qk_head + i] = k_pt[i];
        }
        for (int64_t i = tid; i < v_head; i += nth) {
            vbuf[1 * v_head + i] = v_pt[i];
        }
    }
    __syncthreads();

    float pa = 1.0f; // running alpha product (phase 1; redundant across threads)

    for (int64_t t = 0; t < tc; ++t) {
        const int64_t gt  = t0 + t;                // global token position in the sequence
        const int64_t tok = seq * n_tokens + gt;
        const int64_t cur =  t      & 1;
        const int64_t prv = (t + 1) & 1;

        const float * k_t = k + tok * qk_tok + h * qk_head;
        const float * v_t = v + tok * v_tok  + h * v_head;

        for (int64_t i = tid; i < qk_head; i += nth) {
            kbuf[cur * qk_head + i] = k_t[i];
        }
        for (int64_t i = tid; i < v_head; i += nth) {
            vbuf[cur * v_head + i] = v_t[i];
        }
        if constexpr (compute_y) {
            const float * q_t = q + tok * qk_tok + h * qk_head;
            for (int64_t i = tid; i < qk_head; i += nth) {
                qbuf[i] = q_t[i];
            }
        }
        __syncthreads();

        const float * cf    = coefs + 3 * (h + H * (gt + n_tokens * seq));
        const float   alpha = cf[0];
        const float   beta  = cf[1];
        const float   gamma = cf[2];

        if constexpr (!compute_y) {
            pa *= alpha;
        }

        const float * k_c = kbuf + cur * qk_head;
        const float * v_c = vbuf + cur * v_head;
        const float * k_p = kbuf + prv * qk_head;
        const float * v_p = vbuf + prv * v_head;

        const bool has_prev = gt > 0; // kv[-1] is zero only at the true sequence start

        for (int64_t i = tid; i < SD; i += nth) {
            const int64_t d = i % D_qk;
            const int64_t p = i / D_qk;

            float c_acc = 0.0f;
            for (int64_t r = 0; r < R; ++r) {
                c_acc += v_c[r * D_v + p] * k_c[r * D_qk + d];
            }
            float p_acc = 0.0f;
            if (has_prev) {
                for (int64_t r = 0; r < R; ++r) {
                    p_acc += v_p[r * D_v + p] * k_p[r * D_qk + d];
                }
            }
            S[i] = alpha * S[i] + beta * p_acc + gamma * c_acc;
        }
        __syncthreads();

        if constexpr (compute_y) {
            float * y_t = y_base + tok * y_row + h * v_head;
            for (int64_t j = tid; j < v_head; j += nth) {
                const int64_t r = j / D_v;
                const int64_t p = j % D_v;

                const float * S_row = S    + p * D_qk;
                const float * q_r   = qbuf + r * D_qk;

                float acc = 0.0f;
                for (int64_t d = 0; d < D_qk; ++d) {
                    acc += S_row[d] * q_r[d];
                }
                y_t[j] = acc;
            }
            __syncthreads();
        }
    }

    if constexpr (!compute_y) {
        if constexpr (state_in_smem) {
            for (int64_t i = tid; i < SD; i += nth) {
                W[i] = S[i];
            }
        }
        if (tid == 0) {
            wdecay[(seq * H + h) * n_chunks + nc] = pa;
        }
    }
}

// phase 2: serial scan over chunks; replaces each chunk's local state in the
// workspace with its carry-in state and writes the final state to dst
template <bool state_in_smem>
static __global__ void mamba3_mimo_chunk_scan_cuda(
        const float * __restrict__ s_in,
        float       * __restrict__ wstate,
        const float * __restrict__ wdecay,
        float       *              dst,
        const int64_t              D_qk,
        const int64_t              R,
        const int64_t              D_v,
        const int64_t              H,
        const int64_t              n_tokens,
        const int64_t              n_seqs,
        const int64_t              n_chunks) {
    const int64_t h   = blockIdx.x;
    const int64_t seq = blockIdx.y;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    const int64_t SD    = D_qk * D_v;
    const int64_t y_row = D_v * R * H;

    float * state_base = dst + y_row * n_tokens * n_seqs;

    float       * S_out  = state_base + (seq * H + h) * SD;
    const float * S_init = s_in       + (seq * H + h) * SD;

    extern __shared__ float smem[];
    float * S = state_in_smem ? smem : S_out;

    // each thread owns a fixed subset of state elements: no syncs needed
    for (int64_t i = tid; i < SD; i += nth) {
        S[i] = S_init[i];
    }

    for (int64_t n = 0; n < n_chunks; ++n) {
        float     * W  = wstate + ((seq * H + h) * n_chunks + n) * SD;
        const float pa = wdecay[(seq * H + h) * n_chunks + n];

        for (int64_t i = tid; i < SD; i += nth) {
            const float local = W[i];
            W[i] = S[i];                // carry-in for phase 3
            S[i] = pa * S[i] + local;
        }
    }

    if constexpr (state_in_smem) {
        for (int64_t i = tid; i < SD; i += nth) {
            S_out[i] = S[i];
        }
    }
}

void ggml_cuda_op_mamba3_mimo(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_coefs = dst->src[3];
    const ggml_tensor * src_state = dst->src[4];

    const int64_t D_qk     = src_q->ne[0];
    const int64_t R        = src_q->ne[1];
    const int64_t H        = src_q->ne[2];
    const int64_t D_v      = src_v->ne[0];
    const int64_t n_tokens = src_coefs->ne[2];
    const int64_t n_seqs   = src_coefs->ne[3];

    GGML_ASSERT(src_q->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src_q));
    GGML_ASSERT(ggml_is_contiguous(src_k));
    GGML_ASSERT(ggml_is_contiguous(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_coefs));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    const float * q_d  = (const float *) src_q->data;
    const float * k_d  = (const float *) src_k->data;
    const float * v_d  = (const float *) src_v->data;
    const float * c_d  = (const float *) src_coefs->data;
    const float * s_d  = (const float *) src_state->data;
    float       * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    // q/k/v staging (double-buffered k/v for the trapezoid beta term)
    const size_t staging_bytes = (3 * D_qk * R + 2 * D_v * R) * sizeof(float);
    const size_t state_bytes   = D_qk * D_v * sizeof(float);

    // stay within the default per-block shared memory limit; larger states
    // work directly in a global-memory slot (dst final state / workspace)
    constexpr size_t SMEM_MAX = 48 * 1024;
    GGML_ASSERT(staging_bytes <= SMEM_MAX);

    const int  n_threads     = 256;
    const bool state_in_smem = staging_bytes + state_bytes <= SMEM_MAX;

    if (n_tokens >= 2 * MAMBA3_CHUNK) {
        // chunk-parallel prefill
        const int64_t n_chunks = (n_tokens + MAMBA3_CHUNK - 1) / MAMBA3_CHUNK;
        const int64_t SD       = D_qk * D_v;

        ggml_cuda_pool_alloc<float> wstate(ctx.pool(), (size_t) (n_chunks * n_seqs * H) * SD);
        ggml_cuda_pool_alloc<float> wdecay(ctx.pool(), (size_t) (n_chunks * n_seqs * H));

        const dim3 grid_chunks((unsigned) n_chunks, (unsigned) H, (unsigned) n_seqs);
        const dim3 grid_scan  ((unsigned) H, (unsigned) n_seqs, 1);

        // phase 1 stages k/v only, phase 3 also stages q
        const size_t smem1 = (2 * D_qk * R + 2 * D_v * R) * sizeof(float) + (state_in_smem ? state_bytes : 0);
        const size_t smem3 = staging_bytes                               + (state_in_smem ? state_bytes : 0);
        const size_t smem2 = state_in_smem ? state_bytes : 0;

        if (state_in_smem) {
            mamba3_mimo_chunk_cuda<true, false><<<grid_chunks, n_threads, smem1, stream>>>(
                q_d, k_d, v_d, c_d, wstate.get(), wdecay.get(), dst_d, D_qk, R, D_v, H, n_tokens, n_seqs, n_chunks);
            mamba3_mimo_chunk_scan_cuda<true><<<grid_scan, n_threads, smem2, stream>>>(
                s_d, wstate.get(), wdecay.get(), dst_d, D_qk, R, D_v, H, n_tokens, n_seqs, n_chunks);
            mamba3_mimo_chunk_cuda<true, true><<<grid_chunks, n_threads, smem3, stream>>>(
                q_d, k_d, v_d, c_d, wstate.get(), wdecay.get(), dst_d, D_qk, R, D_v, H, n_tokens, n_seqs, n_chunks);
        } else {
            mamba3_mimo_chunk_cuda<false, false><<<grid_chunks, n_threads, smem1, stream>>>(
                q_d, k_d, v_d, c_d, wstate.get(), wdecay.get(), dst_d, D_qk, R, D_v, H, n_tokens, n_seqs, n_chunks);
            mamba3_mimo_chunk_scan_cuda<false><<<grid_scan, n_threads, smem2, stream>>>(
                s_d, wstate.get(), wdecay.get(), dst_d, D_qk, R, D_v, H, n_tokens, n_seqs, n_chunks);
            mamba3_mimo_chunk_cuda<false, true><<<grid_chunks, n_threads, smem3, stream>>>(
                q_d, k_d, v_d, c_d, wstate.get(), wdecay.get(), dst_d, D_qk, R, D_v, H, n_tokens, n_seqs, n_chunks);
        }
        return;
    }

    const dim3 grid_dims((unsigned) H, (unsigned) n_seqs, 1);

    if (state_in_smem) {
        mamba3_mimo_cuda<true><<<grid_dims, n_threads, staging_bytes + state_bytes, stream>>>(
            q_d, k_d, v_d, c_d, s_d, dst_d, D_qk, R, D_v, H, n_tokens, n_seqs);
    } else {
        mamba3_mimo_cuda<false><<<grid_dims, n_threads, staging_bytes, stream>>>(
            q_d, k_d, v_d, c_d, s_d, dst_d, D_qk, R, D_v, H, n_tokens, n_seqs);
    }
}
