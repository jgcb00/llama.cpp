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
    // work directly in their final-state slot of dst (global memory)
    constexpr size_t SMEM_MAX = 48 * 1024;
    GGML_ASSERT(staging_bytes <= SMEM_MAX);

    const int n_threads = 256;
    const dim3 grid_dims((unsigned) H, (unsigned) n_seqs, 1);

    if (staging_bytes + state_bytes <= SMEM_MAX) {
        mamba3_mimo_cuda<true><<<grid_dims, n_threads, staging_bytes + state_bytes, stream>>>(
            q_d, k_d, v_d, c_d, s_d, dst_d, D_qk, R, D_v, H, n_tokens, n_seqs);
    } else {
        mamba3_mimo_cuda<false><<<grid_dims, n_threads, staging_bytes, stream>>>(
            q_d, k_d, v_d, c_d, s_d, dst_d, D_qk, R, D_v, H, n_tokens, n_seqs);
    }
}
