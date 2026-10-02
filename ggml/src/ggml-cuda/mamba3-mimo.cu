#include "mamba3-mimo.cuh"

// Fused Dragon Mamba3-MIMO mixer core (see ggml_mamba3_mimo in ggml.h).
// semantic reference: ggml_compute_forward_mamba3_mimo (ggml/src/ggml-cpu/ops.cpp).
//
// One block per (head, seq[, chunk]). Every token step builds its own operands
// from the raw projections in shared memory (B/C rms-norm + weight + bias +
// rotary, x/z MIMO scaling, softplus/exp/sigmoid discretization), then updates
// the state and, when producing outputs, applies the D skip, the SiLU(z) gate
// and the MIMO output combine. Warps own state rows p, lanes stride d, so the
// state update and the q.S contraction share registers and need no sync.
//
// short ubatches (decode): one kernel, serial over the tokens of the sequence.
// long ubatches (prefill): the recurrence is affine in S, so with chunks of
// M3_CHUNK tokens S_end(n) = P_n S_end(n-1) + L_n, where P_n is the product of
// the chunk's alphas and L_n the chunk-local state run from S = 0:
//   angles:  per (head, seq): serial phase cumsum -> per-token angle buffer
//   phase 1: per (chunk, head, seq): L_n and P_n                (parallel)
//   phase 2: per (head, seq): S_end scan, leaves chunk carry-ins (serial over chunks)
//   phase 3: per (chunk, head, seq): outputs from the carry-in  (parallel)

#define M3_CHUNK    32
#define M3_MAX_R    8
#define M3_NTHREADS 256

struct m3_params {
    const float * pdyn;   int64_t pdyn_s;   // token stride (floats)
    const float * pstat;  int64_t pstat_s;
    const float * bias;                     // (D_qk, 2R, H)
    const float * mxz;                      // (D_v, 2R, H)
    const float * mimo_o;                   // (D_v, R, H)
    const float * norms;                    // (D_qk, 2)
    const float * misc;                     // (H, 2)
    const float * s_in;   int64_t s_in_s;   // seq stride (floats)
    float       * y;                        // (D_v, H, n_tok)
    float       * s_out;                    // (n_embd_s, n_seqs), contiguous
    float       * ang;                      // (n_ang, H, n_tok) per-token angles (chunked path)
    float       * wstate;                   // (n_chunks, H, n_seqs) x SD (chunked path)
    float       * wdecay;                   // (n_chunks, H, n_seqs)
    int D_qk, R, D_v, H, n_ang;
    int T, n_seqs, n_chunks;
    int64_t n_embd_s, off_K, off_V, off_A;
    float eps, a_floor;
};

enum m3_mode { M3_SERIAL, M3_PHASE1, M3_PHASE3 };

static __device__ __forceinline__ float m3_softplus(float x) {
    return fmaxf(x, 0.0f) + logf(expf(-fabsf(x)) + 1.0f);
}

struct m3_coefs { float dt, alpha, beta, gamma; };

static __device__ __forceinline__ m3_coefs m3_token_coefs(const m3_params & P, int h, int64_t tok) {
    const float * ps = P.pstat + tok*P.pstat_s + (int64_t) h*(2*P.D_v + 3);
    m3_coefs c;
    c.dt    = m3_softplus(ps[2*P.D_v + 0] + P.misc[h]);
    const float a    = fminf(-m3_softplus(ps[2*P.D_v + 1]), -P.a_floor);
    c.alpha = expf(a*c.dt);
    const float trap = 1.0f/(1.0f + expf(-ps[2*P.D_v + 2]));
    c.beta  = (1.0f - trap)*c.dt*c.alpha;
    c.gamma = trap*c.dt;
    return c;
}

static __device__ __forceinline__ float m3_wrap(float a) {
    return a - 6.283185307179586f*rintf(a*0.15915494309189535f);
}

// builds k (and q, z if wanted) for token `tok` into shared memory from the raw
// projections; cs/sn must hold the token's rotary cos/sin, rms the 2R scales.
// v always. Caller syncs before and after.
static __device__ __forceinline__ void m3_build_operands(
        const m3_params & P, int h, int64_t tok, const float * cs, const float * sn, const float * rms,
        float * k, float * q, float * v, float * z, bool want_qz) {
    const int tid = threadIdx.x, nth = blockDim.x;
    const int D_qk = P.D_qk, R = P.R, D_v = P.D_v;
    const float * pd     = P.pdyn + tok*P.pdyn_s;
    const float * bias_h = P.bias + (int64_t) h*2*R*D_qk;
    const int     half   = D_qk/2, quarter = D_qk/4;

    const int n_kq = (want_qz ? 2 : 1)*R*D_qk;
    for (int j = tid; j < n_kq; j += nth) {
        const int g = j / D_qk;
        const int d = j % D_qk;
        const float * src = pd + g*D_qk;
        const float * w   = P.norms + (g < R ? 0 : D_qk);
        const float * b   = bias_h + g*D_qk;
        const float   sc  = rms[g];
        float out;
        if (d < quarter) {
            const float v0 = src[d]*sc*w[d] + b[d];
            const float v2 = src[d + half]*sc*w[d + half] + b[d + half];
            out = v0*cs[d] - v2*sn[d];
        } else if (d >= half && d < half + quarter) {
            const int i = d - half;
            const float v0 = src[i]*sc*w[i] + b[i];
            const float v2 = src[d]*sc*w[d] + b[d];
            out = v0*sn[i] + v2*cs[i];
        } else {
            out = src[d]*sc*w[d] + b[d];
        }
        if (g < R) {
            k[j] = out;
        } else {
            q[j - R*D_qk] = out;
        }
    }

    const float * ps   = P.pstat + tok*P.pstat_s + (int64_t) h*(2*D_v + 3);
    const float * mx_h = P.mxz + (int64_t) h*2*R*D_v;
    for (int j = tid; j < R*D_v; j += nth) {
        const int p = j % D_v;
        v[j] = ps[D_v + p]*mx_h[j];
        if (want_qz) {
            z[j] = ps[p]*mx_h[R*D_v + j];
        }
    }
}

// rms scales of the 2R groups of B/C of token `tok` (one warp per group)
static __device__ __forceinline__ void m3_rms(const m3_params & P, int64_t tok, float * rms, int n_groups) {
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE, nw = blockDim.x / WARP_SIZE;
    const float * pd = P.pdyn + tok*P.pdyn_s;
    for (int g = warp; g < n_groups; g += nw) {
        float s = 0.0f;
        for (int d = lane; d < P.D_qk; d += WARP_SIZE) {
            const float x = pd[g*P.D_qk + d];
            s += x*x;
        }
        s = warp_reduce_sum(s);
        if (lane == 0) {
            rms[g] = rsqrtf(s/P.D_qk + P.eps);
        }
    }
}

template <m3_mode mode, bool state_in_smem, int RT>
static __global__ void __launch_bounds__(M3_NTHREADS) mamba3_mimo_scan_cuda(const m3_params P) {
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int nc  = blockIdx.x;

    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid % WARP_SIZE, warp = tid / WARP_SIZE, nw = nth / WARP_SIZE;

    constexpr int R = RT;
    const int D_qk = P.D_qk, D_v = P.D_v, H = P.H, n_ang = P.n_ang;
    const int64_t SD = (int64_t) D_qk*D_v;

    const int t0 = mode == M3_SERIAL ? 0   : nc*M3_CHUNK;
    const int tc = mode == M3_SERIAL ? P.T : min(M3_CHUNK, P.T - t0);
    const int64_t tok0 = (int64_t) seq*P.T;

    const float * s_in  = P.s_in  + seq*P.s_in_s;
    float       * s_out = P.s_out + seq*P.n_embd_s;
    float       * W     = mode == M3_SERIAL ? nullptr : P.wstate + (((int64_t) seq*H + h)*P.n_chunks + nc)*SD;

    // shared: [ k x2 | q | v x2 | z | cs | sn | ang | rms | (S) ]
    extern __shared__ float smem[];
    float * kb  = smem;
    float * qb  = kb  + 2*R*D_qk;
    float * vb  = qb  + R*D_qk;
    float * zb  = vb  + 2*R*D_v;
    float * cs  = zb  + R*D_v;
    float * sn  = cs  + n_ang;
    float * ang = sn  + n_ang;
    float * rms = ang + n_ang;
    float * S   = state_in_smem ? rms + 2*R : (mode == M3_SERIAL ? s_out + h*SD : W);

    // initial state
    for (int64_t i = tid; i < SD; i += nth) {
        float s0;
        if constexpr (mode == M3_SERIAL) {
            s0 = s_in[h*SD + i];
        } else if constexpr (mode == M3_PHASE1) {
            s0 = 0.0f;
        } else {
            s0 = W[i]; // carry-in left by phase 2
        }
        if (state_in_smem || mode != M3_PHASE3) {
            S[i] = s0;
        }
    }
    if constexpr (mode == M3_SERIAL) {
        for (int i = tid; i < n_ang; i += nth) {
            ang[i] = s_in[P.off_A + h*n_ang + i];
        }
    }

    // previous token's k/v: from the cache at the sequence start, else rebuilt
    int prv = 1;
    if (t0 == 0) {
        for (int j = tid; j < R*D_qk; j += nth) {
            kb[prv*R*D_qk + j] = s_in[P.off_K + (int64_t) h*R*D_qk + j];
        }
        for (int j = tid; j < R*D_v; j += nth) {
            vb[prv*R*D_v + j] = s_in[P.off_V + (int64_t) h*R*D_v + j];
        }
    } else {
        const int64_t ptok = tok0 + t0 - 1;
        for (int i = tid; i < n_ang; i += nth) {
            float s, c;
            sincosf(P.ang[(ptok*H + h)*n_ang + i], &s, &c);
            cs[i] = c; sn[i] = s;
        }
        m3_rms(P, ptok, rms, R);
        __syncthreads();
        m3_build_operands(P, h, ptok, cs, sn, rms, kb + prv*R*D_qk, qb, vb + prv*R*D_v, zb, false);
    }
    __syncthreads();

    float pa = 1.0f; // running alpha product (phase 1)
    const float D_h = P.misc[H + h];
    const float * mo_h = P.mimo_o + (int64_t) h*R*D_v;

    for (int t = 0; t < tc; ++t) {
        const int64_t tok = tok0 + t0 + t;
        const int cur = prv ^ 1;
        constexpr bool want_y = mode != M3_PHASE1;

        const m3_coefs cf = m3_token_coefs(P, h, tok);
        if constexpr (mode == M3_PHASE1) {
            pa *= cf.alpha;
        }

        for (int i = tid; i < n_ang; i += nth) {
            float a;
            if constexpr (mode == M3_SERIAL) {
                a = m3_wrap(ang[i] + tanhf(P.pdyn[tok*P.pdyn_s + 2*R*D_qk + i])*3.14159265358979f*cf.dt);
                ang[i] = a;
            } else {
                a = P.ang[(tok*H + h)*n_ang + i];
            }
            float s, c;
            sincosf(a, &s, &c);
            cs[i] = c; sn[i] = s;
        }
        m3_rms(P, tok, rms, want_y ? 2*R : R);
        __syncthreads();

        float * k_c = kb + cur*R*D_qk;
        float * v_c = vb + cur*R*D_v;
        const float * k_p = kb + prv*R*D_qk;
        const float * v_p = vb + prv*R*D_v;
        m3_build_operands(P, h, tok, cs, sn, rms, k_c, qb, v_c, zb, want_y);
        __syncthreads();

        for (int p = warp; p < D_v; p += nw) {
            float vc[R], vp[R], acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                vc[r]  = v_c[r*D_v + p];
                vp[r]  = v_p[r*D_v + p];
                acc[r] = 0.0f;
            }
            float * Sp = S + (int64_t) p*D_qk;
            for (int d = lane; d < D_qk; d += WARP_SIZE) {
                float c_acc = 0.0f, p_acc = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    c_acc += vc[r]*k_c[r*D_qk + d];
                    p_acc += vp[r]*k_p[r*D_qk + d];
                }
                const float s = cf.alpha*Sp[d] + cf.beta*p_acc + cf.gamma*c_acc;
                Sp[d] = s;
                if constexpr (want_y) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acc[r] += s*qb[r*D_qk + d];
                    }
                }
            }
            if constexpr (want_y) {
                float yp = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float o  = warp_reduce_sum(acc[r]) + D_h*vc[r];
                    const float zv = zb[r*D_v + p];
                    yp += mo_h[r*D_v + p]*o*(zv/(1.0f + expf(-zv)));
                }
                if (lane == 0) {
                    P.y[(tok*H + h)*D_v + p] = yp;
                }
            }
        }
        __syncthreads();
        prv = cur;
    }

    const bool last = mode == M3_SERIAL || (mode == M3_PHASE3 && nc == P.n_chunks - 1);
    if (last) {
        // k/v of the last token feed the next ubatch's trapezoid term
        for (int j = tid; j < R*D_qk; j += nth) {
            s_out[P.off_K + (int64_t) h*R*D_qk + j] = kb[prv*R*D_qk + j];
        }
        for (int j = tid; j < R*D_v; j += nth) {
            s_out[P.off_V + (int64_t) h*R*D_v + j] = vb[prv*R*D_v + j];
        }
    }
    if constexpr (mode == M3_SERIAL) {
        for (int i = tid; i < n_ang; i += nth) {
            s_out[P.off_A + h*n_ang + i] = ang[i];
        }
        if constexpr (state_in_smem) {
            for (int64_t i = tid; i < SD; i += nth) {
                s_out[h*SD + i] = S[i];
            }
        }
    }
    if constexpr (mode == M3_PHASE1) {
        if constexpr (state_in_smem) {
            for (int64_t i = tid; i < SD; i += nth) {
                W[i] = S[i];
            }
        }
        if (tid == 0) {
            P.wdecay[((int64_t) seq*H + h)*P.n_chunks + nc] = pa;
        }
    }
}

// per-token wrapped rotary phases for the chunked path; one block per (head, seq)
static __global__ void mamba3_mimo_angles_cuda(const m3_params P) {
    const int h = blockIdx.x, seq = blockIdx.y;
    const int R = P.R, D_qk = P.D_qk, H = P.H, n_ang = P.n_ang;
    for (int i = threadIdx.x; i < n_ang; i += blockDim.x) {
        float a = P.s_in[seq*P.s_in_s + P.off_A + h*n_ang + i];
        for (int t = 0; t < P.T; ++t) {
            const int64_t tok = (int64_t) seq*P.T + t;
            const m3_coefs cf = m3_token_coefs(P, h, tok);
            a = m3_wrap(a + tanhf(P.pdyn[tok*P.pdyn_s + 2*R*D_qk + i])*3.14159265358979f*cf.dt);
            P.ang[(tok*H + h)*n_ang + i] = a;
        }
        P.s_out[seq*P.n_embd_s + P.off_A + h*n_ang + i] = a;
    }
}

// phase 2: serial scan over chunks; leaves each chunk's carry-in in the
// workspace and writes the final state
template <bool state_in_smem>
static __global__ void mamba3_mimo_chunk_scan_cuda(const m3_params P) {
    const int h = blockIdx.x, seq = blockIdx.y;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int64_t SD = (int64_t) P.D_qk*P.D_v;

    const float * S_init = P.s_in  + seq*P.s_in_s   + h*SD;
    float       * S_out  = P.s_out + seq*P.n_embd_s + h*SD;

    extern __shared__ float smem[];
    float * S = state_in_smem ? smem : S_out;

    // each thread owns a fixed subset of state elements: no syncs needed
    for (int64_t i = tid; i < SD; i += nth) {
        S[i] = S_init[i];
    }
    for (int n = 0; n < P.n_chunks; ++n) {
        float     * W  = P.wstate + (((int64_t) seq*P.H + h)*P.n_chunks + n)*SD;
        const float pa = P.wdecay[((int64_t) seq*P.H + h)*P.n_chunks + n];
        for (int64_t i = tid; i < SD; i += nth) {
            const float local = W[i];
            W[i] = S[i];
            S[i] = pa*S[i] + local;
        }
    }
    if constexpr (state_in_smem) {
        for (int64_t i = tid; i < SD; i += nth) {
            S_out[i] = S[i];
        }
    }
}

template <m3_mode mode, int RT>
static void m3_launch_r(const m3_params & P, dim3 grid, size_t smem_ops, size_t smem_state, bool in_smem, cudaStream_t stream) {
    if (in_smem) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mamba3_mimo_scan_cuda<mode, true, RT>), smem_ops + smem_state);
        mamba3_mimo_scan_cuda<mode, true, RT><<<grid, M3_NTHREADS, smem_ops + smem_state, stream>>>(P);
    } else {
        mamba3_mimo_scan_cuda<mode, false, RT><<<grid, M3_NTHREADS, smem_ops, stream>>>(P);
    }
}

template <m3_mode mode>
static void m3_launch(const m3_params & P, dim3 grid, size_t smem_ops, size_t smem_state, bool in_smem, cudaStream_t stream) {
    switch (P.R) {
        case 1: m3_launch_r<mode, 1>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        case 2: m3_launch_r<mode, 2>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        case 4: m3_launch_r<mode, 4>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        case 8: m3_launch_r<mode, 8>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        default: GGML_ABORT("mamba3_mimo: unsupported MIMO rank %d", P.R);
    }
}

void ggml_cuda_op_mamba3_mimo(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * pdyn   = dst->src[0];
    const ggml_tensor * pstat  = dst->src[1];
    const ggml_tensor * state  = dst->src[7];

    m3_params P = {};
    P.D_qk  = (int) dst->src[2]->ne[0];
    P.R     = (int) dst->src[4]->ne[1];
    P.H     = (int) dst->src[4]->ne[2];
    P.D_v   = (int) dst->src[4]->ne[0];
    P.n_ang = P.D_qk/4;
    GGML_ASSERT(P.R <= M3_MAX_R);

    const int64_t n_tok = pdyn->ne[1];
    P.n_seqs = (int) state->ne[1];
    P.T      = (int) (n_tok / P.n_seqs);

    for (int i = 2; i < 7; ++i) {
        GGML_ASSERT(ggml_is_contiguous(dst->src[i]));
    }
    GGML_ASSERT(pdyn->nb[0] == sizeof(float) && pstat->nb[0] == sizeof(float) && state->nb[0] == sizeof(float));

    P.pdyn   = (const float *) pdyn->data;   P.pdyn_s  = pdyn->nb[1]/sizeof(float);
    P.pstat  = (const float *) pstat->data;  P.pstat_s = pstat->nb[1]/sizeof(float);
    P.bias   = (const float *) dst->src[2]->data;
    P.mxz    = (const float *) dst->src[3]->data;
    P.mimo_o = (const float *) dst->src[4]->data;
    P.norms  = (const float *) dst->src[5]->data;
    P.misc   = (const float *) dst->src[6]->data;
    P.s_in   = (const float *) state->data;  P.s_in_s  = state->nb[1]/sizeof(float);

    P.off_K    = (int64_t) P.H*P.D_v*P.D_qk;
    P.off_V    = P.off_K + (int64_t) P.H*P.R*P.D_qk;
    P.off_A    = P.off_V + (int64_t) P.H*P.R*P.D_v;
    P.n_embd_s = P.off_A + (int64_t) P.H*P.n_ang;

    P.y     = (float *) dst->data;
    P.s_out = P.y + (int64_t) P.D_v*P.H*n_tok;

    P.eps     = ggml_get_op_params_f32(dst, 0);
    P.a_floor = ggml_get_op_params_f32(dst, 1);

    cudaStream_t stream = ctx.stream();

    const size_t smem_ops   = (size_t) (3*P.R*P.D_qk + 3*P.R*P.D_v + 3*P.n_ang + 2*P.R)*sizeof(float);
    const size_t smem_state = (size_t) P.D_qk*P.D_v*sizeof(float);
    const size_t smem_max   = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    GGML_ASSERT(smem_ops <= 48*1024);
    const bool in_smem = smem_ops + smem_state <= smem_max;

    if (P.T < 2*M3_CHUNK) {
        P.n_chunks = 1;
        m3_launch<M3_SERIAL>(P, dim3(1, P.H, P.n_seqs), smem_ops, smem_state, in_smem, stream);
        return;
    }

    P.n_chunks = (P.T + M3_CHUNK - 1)/M3_CHUNK;
    const int64_t SD = (int64_t) P.D_qk*P.D_v;
    ggml_cuda_pool_alloc<float> angbuf(ctx.pool(), (size_t) n_tok*P.H*P.n_ang);
    ggml_cuda_pool_alloc<float> wstate(ctx.pool(), (size_t) P.n_chunks*P.n_seqs*P.H*SD);
    ggml_cuda_pool_alloc<float> wdecay(ctx.pool(), (size_t) P.n_chunks*P.n_seqs*P.H);
    P.ang    = angbuf.get();
    P.wstate = wstate.get();
    P.wdecay = wdecay.get();

    mamba3_mimo_angles_cuda<<<dim3(P.H, P.n_seqs), WARP_SIZE, 0, stream>>>(P);

    const dim3 grid_chunks(P.n_chunks, P.H, P.n_seqs);
    m3_launch<M3_PHASE1>(P, grid_chunks, smem_ops, smem_state, in_smem, stream);
    if (smem_state <= smem_max) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mamba3_mimo_chunk_scan_cuda<true>), smem_state);
        mamba3_mimo_chunk_scan_cuda<true><<<dim3(P.H, P.n_seqs), M3_NTHREADS, smem_state, stream>>>(P);
    } else {
        mamba3_mimo_chunk_scan_cuda<false><<<dim3(P.H, P.n_seqs), M3_NTHREADS, 0, stream>>>(P);
    }
    m3_launch<M3_PHASE3>(P, grid_chunks, smem_ops, smem_state, in_smem, stream);
}

// ---------------------------------------------------------------------------
// geodesic residual: one block per row

static __global__ void geodesic_cuda(
        const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        const float * __restrict__ scale, const float * __restrict__ bias, const float inv_depth,
        const int n, const int64_t ne1, const int64_t ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1, i2 = (ir/ne1) % ne2, i3 = ir/(ne1*ne2);
    const float * xr = x   + i1*sx1 + i2*sx2 + i3*sx3;
    const float * gr = g   + i1*sg1 + i2*sg2 + i3*sg3;
    float       * yr = dst + i1*sd1 + i2*sd2 + i3*sd3;

    __shared__ float red[32];
    __shared__ float bc[2];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE, nw = blockDim.x / WARP_SIZE;

    auto block_sum = [&](float v) -> float {
        v = warp_reduce_sum(v);
        if (lane == 0) red[warp] = v;
        __syncthreads();
        v = lane < nw ? red[lane] : 0.0f;
        v = warp_reduce_sum(v);
        __syncthreads();
        return v;
    };

    float xx = 0.0f, xg = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        xx += xr[i]*xr[i];
        xg += xr[i]*gr[i];
    }
    xx = block_sum(xx);
    xg = block_sum(xg);
    const float x_norm_sq = fmaxf(xx, 1e-12f);
    const float coef      = xg/x_norm_sq;

    float pp = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float gp = gr[i] - coef*xr[i];
        pp += gp*gp;
    }
    pp = block_sum(pp);

    if (tid == 0) {
        const float pi4      = 0.785398163f;
        const float tan_norm = fmaxf(sqrtf(pp), 1e-8f);
        const float xn       = sqrtf(x_norm_sq);
        float theta = fminf(tan_norm/fmaxf(xn, 1e-6f), pi4);
        theta = fminf((theta*scale[0] + bias[0])*inv_depth, pi4);
        bc[0] = cosf(theta);
        bc[1] = sinf(theta)*xn/tan_norm;
    }
    __syncthreads();
    const float c = bc[0], s = bc[1];
    for (int i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i]*c + (gr[i] - coef*xr[i])*s;
    }
}

void ggml_cuda_op_geodesic(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = dst->src[1];
    GGML_ASSERT(x->nb[0] == sizeof(float) && g->nb[0] == sizeof(float));
    const int64_t nr = ggml_nrows(x);
    const int n = (int) x->ne[0];
    const int nthreads = n >= 1024 ? 512 : (n >= 256 ? 256 : 128);
    geodesic_cuda<<<nr, nthreads, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) g->data, (float *) dst->data,
        (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, ggml_get_op_params_f32(dst, 0),
        n, x->ne[1], x->ne[2],
        x->nb[1]/4, x->nb[2]/4, x->nb[3]/4,
        g->nb[1]/4, g->nb[2]/4, g->nb[3]/4,
        dst->nb[1]/4, dst->nb[2]/4, dst->nb[3]/4);
}
