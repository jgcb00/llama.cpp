#include "models.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

// SIMD helpers for the Mamba3-MIMO kernel hot loops. All operate on fp32
// vectors of length n; n is typically a multiple of 8 (D_qk=128, D_v=64) so
// the tail loop usually runs zero iterations.

// Fused: out1[i] += a1*in1[i] ; out2[i] += a2*in2[i]
// Used to update curr_kv and prev_kv in one pass (shared p, d iteration).
static inline float dragon_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Storage precision of the recurrent state cache. Working precision in the
// kernel is always f32; this only affects the cache-resident bytes.
//   F32  — plain f32 everywhere (DRAGON_F32_STATE=1, or forced by any opt-in
//          path that reads the cache as raw f32: prim/chunked/mega/colsplit).
//   BF16 — default. K/V trapezoid sections packed to bf16 (low half of their
//          f32 sections); the big ssm S-matrix and angle scalars stay f32.
//   Q8   — DRAGON_STATE_Q8=1 (experiment). ssm S-matrix AND K/V trapezoids
//          stored as per-32-block int8 with an f32 scale ([scale][32xint8]
//          in each 32-float slot); angles stay f32. Recurrent feedback of the
//          quantization error accumulates over t — validate before adopting.
enum dragon_state_prec { DRAGON_STATE_F32, DRAGON_STATE_BF16, DRAGON_STATE_Q8 };
static dragon_state_prec dragon_state_mode() {
    static const dragon_state_prec v = [] {
        if (std::getenv("DRAGON_F32_STATE"))     return DRAGON_STATE_F32;
        if (std::getenv("DRAGON_M_PRIM"))        return DRAGON_STATE_F32;
        if (std::getenv("DRAGON_M_DECODE_PRIM")) return DRAGON_STATE_F32;
        if (std::getenv("DRAGON_M_CHUNK_SIZE"))  return DRAGON_STATE_F32;
        if (std::getenv("DRAGON_MEGA_DECODE"))   return DRAGON_STATE_F32;
        if (std::getenv("DRAGON_M_COLSPLIT"))    return DRAGON_STATE_F32; // q8/bf16 slots assume unsplit 32-aligned runs
        if (std::getenv("DRAGON_STATE_Q8"))      return DRAGON_STATE_Q8;
        return DRAGON_STATE_BF16;
    }();
    return v;
}
static bool dragon_state_bf16() { return dragon_state_mode() != DRAGON_STATE_F32; }

static inline void dragon_simd_fma_inplace_dual(
        float * __restrict__ out1, const float * __restrict__ in1, float a1,
        float * __restrict__ out2, const float * __restrict__ in2, float a2,
        int64_t n) {
    int64_t i = 0;
#if defined(__AVX512F__)
    const __m512 va1z = _mm512_set1_ps(a1);
    const __m512 va2z = _mm512_set1_ps(a2);
    for (; i + 16 <= n; i += 16) {
        __m512 o1 = _mm512_loadu_ps(out1 + i);
        __m512 o2 = _mm512_loadu_ps(out2 + i);
        __m512 v1 = _mm512_loadu_ps(in1  + i);
        __m512 v2 = _mm512_loadu_ps(in2  + i);
        o1 = _mm512_fmadd_ps(va1z, v1, o1);
        o2 = _mm512_fmadd_ps(va2z, v2, o2);
        _mm512_storeu_ps(out1 + i, o1);
        _mm512_storeu_ps(out2 + i, o2);
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 va1 = _mm256_set1_ps(a1);
    const __m256 va2 = _mm256_set1_ps(a2);
    for (; i + 8 <= n; i += 8) {
        __m256 o1 = _mm256_loadu_ps(out1 + i);
        __m256 o2 = _mm256_loadu_ps(out2 + i);
        __m256 v1 = _mm256_loadu_ps(in1  + i);
        __m256 v2 = _mm256_loadu_ps(in2  + i);
        o1 = _mm256_fmadd_ps(va1, v1, o1);
        o2 = _mm256_fmadd_ps(va2, v2, o2);
        _mm256_storeu_ps(out1 + i, o1);
        _mm256_storeu_ps(out2 + i, o2);
    }
#endif
    for (; i < n; ++i) {
        out1[i] += a1 * in1[i];
        out2[i] += a2 * in2[i];
    }
}

// state[i] = alpha * state[i] + beta * prev[i] + gamma * curr[i]
static inline void dragon_simd_state_update(float * __restrict__ state,
                                            const float * __restrict__ prev,
                                            const float * __restrict__ curr,
                                            float alpha, float beta, float gamma,
                                            int64_t n) {
    int64_t i = 0;
#if defined(__AVX512F__)
    const __m512 vaz = _mm512_set1_ps(alpha);
    const __m512 vbz = _mm512_set1_ps(beta);
    const __m512 vgz = _mm512_set1_ps(gamma);
    for (; i + 16 <= n; i += 16) {
        __m512 s = _mm512_loadu_ps(state + i);
        __m512 p = _mm512_loadu_ps(prev + i);
        __m512 c = _mm512_loadu_ps(curr + i);
        s = _mm512_mul_ps(vaz, s);
        s = _mm512_fmadd_ps(vbz, p, s);
        s = _mm512_fmadd_ps(vgz, c, s);
        _mm512_storeu_ps(state + i, s);
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 va = _mm256_set1_ps(alpha);
    const __m256 vb = _mm256_set1_ps(beta);
    const __m256 vg = _mm256_set1_ps(gamma);
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(state + i);
        __m256 p = _mm256_loadu_ps(prev + i);
        __m256 c = _mm256_loadu_ps(curr + i);
        s = _mm256_mul_ps(va, s);
        s = _mm256_fmadd_ps(vb, p, s);
        s = _mm256_fmadd_ps(vg, c, s);
        _mm256_storeu_ps(state + i, s);
    }
#endif
    for (; i < n; ++i) {
        state[i] = alpha * state[i] + beta * prev[i] + gamma * curr[i];
    }
}

// Fused per-(h, t) state step over one unit's row slice. For each state row p:
//   curr[d] = Σ_r v[p,r]·k_rot[d,r]      prev[d] = Σ_r V_st[p,r]·K_st[d,r]
//   st[p,d] = α·st[p,d] + β·prev[d] + γ·curr[d]
//   o[r]    = Σ_d st[p,d]·q_rot[d,r];  o[r] += D·v[p,r];  o[r] *= z·σ(z)
//   y[p]   += Σ_r o[r]·mimo_o[p,r]
// Single pass per row: curr/prev/o live in registers. The previous buffered
// version streamed two (Dv_sl, D_qk) scratch matrices (~3× the state traffic)
// per (h, t), which dominated kernel time. Accumulation order matches the
// buffered helpers exactly (per-element fma chains over r; α-mul then β-, γ-
// fma; chunk-sequential dot with one final reduce), so results are
// bit-identical on the AVX-512 path.
template <int RC>
static inline void dragon_simd_fused_step_r(
        float * __restrict__ st,           // (Dv_sl, D_qk)
        const float * __restrict__ K_st,   // (RC, D_qk)  previous-step k_rot
        const float * __restrict__ V_st,   // (RC, Dv_sl) previous-step v
        const float * __restrict__ k_rot,  // (RC, D_qk)
        const float * __restrict__ q_rot,  // (RC, D_qk)
        const float * __restrict__ v_loc,  // (RC, Dv_sl)
        const float * __restrict__ z_loc,  // (RC, Dv_sl)
        const float * __restrict__ mo,     // mimo_o at (p0, ·, h): stride m_s1 per r
        int64_t m_s1,
        float * __restrict__ y_row,        // (Dv_sl)
        float alpha, float beta, float gamma, float D_h,
        int64_t D_qk, int64_t Dv_sl) {
#if defined(__AVX512F__)
    if ((D_qk % 16) == 0) {
        const __m512 va = _mm512_set1_ps(alpha);
        const __m512 vb = _mm512_set1_ps(beta);
        const __m512 vg = _mm512_set1_ps(gamma);
        for (int64_t p = 0; p < Dv_sl; ++p) {
            float * st_row = st + p * D_qk;
            __m512 qacc[RC];
            __m512 vp[RC], Vp[RC];
            for (int r = 0; r < RC; ++r) {
                qacc[r] = _mm512_setzero_ps();
                vp[r]   = _mm512_set1_ps(v_loc[r * Dv_sl + p]);
                Vp[r]   = _mm512_set1_ps(V_st[r * Dv_sl + p]);
            }
            for (int64_t d = 0; d < D_qk; d += 16) {
                __m512 c_acc = _mm512_setzero_ps();
                __m512 p_acc = _mm512_setzero_ps();
                for (int r = 0; r < RC; ++r) {
                    c_acc = _mm512_fmadd_ps(vp[r], _mm512_loadu_ps(k_rot + r * D_qk + d), c_acc);
                    p_acc = _mm512_fmadd_ps(Vp[r], _mm512_loadu_ps(K_st  + r * D_qk + d), p_acc);
                }
                __m512 s = _mm512_loadu_ps(st_row + d);
                s = _mm512_mul_ps(va, s);
                s = _mm512_fmadd_ps(vb, p_acc, s);
                s = _mm512_fmadd_ps(vg, c_acc, s);
                _mm512_storeu_ps(st_row + d, s);
                for (int r = 0; r < RC; ++r) {
                    qacc[r] = _mm512_fmadd_ps(s, _mm512_loadu_ps(q_rot + r * D_qk + d), qacc[r]);
                }
            }
            float yp = y_row[p];
            for (int r = 0; r < RC; ++r) {
                float acc = _mm512_reduce_add_ps(qacc[r]);
                acc += D_h * v_loc[r * Dv_sl + p];
                const float zv = z_loc[r * Dv_sl + p];
                acc *= zv * dragon_sigmoidf(zv);
                yp += acc * mo[r * m_s1 + p];
            }
            y_row[p] = yp;
        }
        return;
    }
#endif
    // Portable fallback (same r/d accumulation order, scalar).
    for (int64_t p = 0; p < Dv_sl; ++p) {
        float * st_row = st + p * D_qk;
        for (int64_t d = 0; d < D_qk; ++d) {
            float c_acc = 0.0f, p_acc = 0.0f;
            for (int r = 0; r < RC; ++r) {
                c_acc += v_loc[r * Dv_sl + p] * k_rot[r * D_qk + d];
                p_acc += V_st[r * Dv_sl + p]  * K_st[r * D_qk + d];
            }
            st_row[d] = alpha * st_row[d] + beta * p_acc + gamma * c_acc;
        }
        float yp = y_row[p];
        for (int r = 0; r < RC; ++r) {
            float acc = 0.0f;
            for (int64_t d = 0; d < D_qk; ++d) {
                acc += st_row[d] * q_rot[r * D_qk + d];
            }
            acc += D_h * v_loc[r * Dv_sl + p];
            const float zv = z_loc[r * Dv_sl + p];
            acc *= zv * dragon_sigmoidf(zv);
            yp += acc * mo[r * m_s1 + p];
        }
        y_row[p] = yp;
    }
}

static inline void dragon_simd_fused_step(
        float * st, const float * K_st, const float * V_st,
        const float * k_rot, const float * q_rot,
        const float * v_loc, const float * z_loc,
        const float * mo, int64_t m_s1, float * y_row,
        float alpha, float beta, float gamma, float D_h,
        int64_t R, int64_t D_qk, int64_t Dv_sl) {
    switch (R) {
        case 4:  dragon_simd_fused_step_r<4>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_row, alpha, beta, gamma, D_h, D_qk, Dv_sl); break;
        case 2:  dragon_simd_fused_step_r<2>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_row, alpha, beta, gamma, D_h, D_qk, Dv_sl); break;
        case 1:  dragon_simd_fused_step_r<1>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_row, alpha, beta, gamma, D_h, D_qk, Dv_sl); break;
        case 8:  dragon_simd_fused_step_r<8>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_row, alpha, beta, gamma, D_h, D_qk, Dv_sl); break;
        default: GGML_ABORT("dragon: unsupported MIMO rank %d", (int) R);
    }
}

// Temporal-blocked variant: applies TB consecutive tokens' updates in a single
// pass over the state rows — the state (32 KB/head) is by far the dominant
// traffic of the scan, so TB tokens per sweep cuts it ~TB×. Per state element
// the op sequence is identical to TB sequential calls of the single-token
// version (the running value just stays in a register between tokens), so
// results are bit-identical on the AVX-512 path. prev_kv for token tt comes
// from K_st/V_st for tt==0 and from token tt-1's k/v buffers otherwise.
template <int RC, int TB>
static inline void dragon_simd_fused_step_tb(
        float * __restrict__ st,
        const float * K_st, const float * V_st,
        const float * const * k_rot,   // TB pointers, each (RC, D_qk)
        const float * const * q_rot,
        const float * const * v_loc,   // TB pointers, each (RC, Dv_sl)
        const float * const * z_loc,
        const float * mo, int64_t m_s1,
        float * const * y_rows,        // TB pointers, each (Dv_sl)
        const float * alpha, const float * beta, const float * gamma, float D_h,
        int64_t D_qk, int64_t Dv_sl) {
#if defined(__AVX512F__)
    if ((D_qk % 16) == 0) {
        for (int64_t p = 0; p < Dv_sl; ++p) {
            float * st_row = st + p * D_qk;
            __m512 qacc[TB][RC];
            for (int tt = 0; tt < TB; ++tt) {
                for (int r = 0; r < RC; ++r) {
                    qacc[tt][r] = _mm512_setzero_ps();
                }
            }
            for (int64_t d = 0; d < D_qk; d += 16) {
                __m512 s = _mm512_loadu_ps(st_row + d);
                for (int tt = 0; tt < TB; ++tt) {
                    const float * kp = (tt == 0) ? K_st : k_rot[tt - 1];
                    const float * vp = (tt == 0) ? V_st : v_loc[tt - 1];
                    __m512 c_acc = _mm512_setzero_ps();
                    __m512 p_acc = _mm512_setzero_ps();
                    for (int r = 0; r < RC; ++r) {
                        c_acc = _mm512_fmadd_ps(_mm512_set1_ps(v_loc[tt][r * Dv_sl + p]),
                                                _mm512_loadu_ps(k_rot[tt] + r * D_qk + d), c_acc);
                        p_acc = _mm512_fmadd_ps(_mm512_set1_ps(vp[r * Dv_sl + p]),
                                                _mm512_loadu_ps(kp + r * D_qk + d), p_acc);
                    }
                    s = _mm512_mul_ps(_mm512_set1_ps(alpha[tt]), s);
                    s = _mm512_fmadd_ps(_mm512_set1_ps(beta[tt]),  p_acc, s);
                    s = _mm512_fmadd_ps(_mm512_set1_ps(gamma[tt]), c_acc, s);
                    for (int r = 0; r < RC; ++r) {
                        qacc[tt][r] = _mm512_fmadd_ps(s, _mm512_loadu_ps(q_rot[tt] + r * D_qk + d), qacc[tt][r]);
                    }
                }
                _mm512_storeu_ps(st_row + d, s);
            }
            for (int tt = 0; tt < TB; ++tt) {
                float yp = y_rows[tt][p];
                for (int r = 0; r < RC; ++r) {
                    float acc = _mm512_reduce_add_ps(qacc[tt][r]);
                    acc += D_h * v_loc[tt][r * Dv_sl + p];
                    const float zv = z_loc[tt][r * Dv_sl + p];
                    acc *= zv * dragon_sigmoidf(zv);
                    yp += acc * mo[r * m_s1 + p];
                }
                y_rows[tt][p] = yp;
            }
        }
        return;
    }
#endif
    // Portable fallback: TB sequential single-token steps (extra state sweeps,
    // same results).
    for (int tt = 0; tt < TB; ++tt) {
        const float * kp = (tt == 0) ? K_st : k_rot[tt - 1];
        const float * vp = (tt == 0) ? V_st : v_loc[tt - 1];
        dragon_simd_fused_step_r<RC>(st, kp, vp, k_rot[tt], q_rot[tt], v_loc[tt], z_loc[tt],
                                     mo, m_s1, y_rows[tt], alpha[tt], beta[tt], gamma[tt], D_h,
                                     D_qk, Dv_sl);
    }
}

// Runtime dispatch over TB (1/2/4) for rank 4 (Dragon); other ranks fall back
// to sequential single-token steps.
static inline void dragon_simd_fused_step_group(
        float * st, const float * K_st, const float * V_st,
        const float * const * k_rot, const float * const * q_rot,
        const float * const * v_loc, const float * const * z_loc,
        const float * mo, int64_t m_s1, float * const * y_rows,
        const float * alpha, const float * beta, const float * gamma, float D_h,
        int64_t R, int64_t TB, int64_t D_qk, int64_t Dv_sl) {
    if (R == 4) {
        switch (TB) {
            case 4: dragon_simd_fused_step_tb<4, 4>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_rows, alpha, beta, gamma, D_h, D_qk, Dv_sl); return;
            case 3: dragon_simd_fused_step_tb<4, 3>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_rows, alpha, beta, gamma, D_h, D_qk, Dv_sl); return;
            case 2: dragon_simd_fused_step_tb<4, 2>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_rows, alpha, beta, gamma, D_h, D_qk, Dv_sl); return;
            case 1: dragon_simd_fused_step_tb<4, 1>(st, K_st, V_st, k_rot, q_rot, v_loc, z_loc, mo, m_s1, y_rows, alpha, beta, gamma, D_h, D_qk, Dv_sl); return;
            default: break;
        }
    }
    for (int64_t tt = 0; tt < TB; ++tt) {
        const float * kp = (tt == 0) ? K_st : k_rot[tt - 1];
        const float * vp = (tt == 0) ? V_st : v_loc[tt - 1];
        dragon_simd_fused_step(st, kp, vp, k_rot[tt], q_rot[tt], v_loc[tt], z_loc[tt],
                               mo, m_s1, y_rows[tt], alpha[tt], beta[tt], gamma[tt], D_h,
                               R, D_qk, Dv_sl);
    }
}

// dot product
static inline float dragon_simd_dot(const float * __restrict__ a,
                                    const float * __restrict__ b,
                                    int64_t n) {
    int64_t i = 0;
    float acc = 0.0f;
#if defined(__AVX512F__)
    __m512 vaccz = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        vaccz = _mm512_fmadd_ps(va, vb, vaccz);
    }
    acc += _mm512_reduce_add_ps(vaccz);
#endif
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    acc += _mm_cvtss_f32(s);
#endif
    for (; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

// ---- per-layer hidden-state dump (DRAGON_DUMP_DIR env var to enable) -------
// When set, each block writes its INPUT hidden state and the final output
// hidden state as raw fp32 bytes to {dir}/block_{il}_in.bin and {dir}/final.bin.
// Header per file: int32 n_embd, int32 n_tokens, then n_embd*n_tokens fp32 values.
struct dragon_dump_userdata {
    char filename[1024];
};
static void dragon_dump_kernel(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) return;
    if (dst != dst->src[0] && dst->data && dst->src[0]->data) {
        std::memcpy(dst->data, dst->src[0]->data, ggml_nbytes(dst));
    }
    auto * ud = (dragon_dump_userdata *) userdata;
    FILE * f = std::fopen(ud->filename, "wb");
    if (!f) return;
    // dst here is fp32 contiguous (we cast/ensure before calling).
    int32_t hdr[4] = { (int32_t) dst->ne[0], (int32_t) dst->ne[1],
                       (int32_t) dst->ne[2], (int32_t) dst->ne[3] };
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(dst->data, ggml_nbytes(dst), 1, f);
    std::fclose(f);
}

static const char * dragon_dump_dir() {
    const char * d = std::getenv("DRAGON_DUMP_DIR");
    return (d && d[0]) ? d : nullptr;
}

// Insert a dump-pass on `t` into the graph; returns a tensor with the same
// values as `t` (an identity passthrough that side-effects to disk).
// Allocates a userdata struct heap-side so the filename outlives graph build.
static ggml_tensor * dragon_maybe_dump(ggml_context * ctx, ggml_tensor * t,
                                       const char * name_fmt, int il) {
    const char * dir = dragon_dump_dir();
    if (!dir) return t;
    auto * ud = (dragon_dump_userdata *) std::malloc(sizeof(dragon_dump_userdata));
    std::snprintf(ud->filename, sizeof(ud->filename), "%s/", dir);
    int n = (int) std::strlen(ud->filename);
    std::snprintf(ud->filename + n, sizeof(ud->filename) - n, name_fmt, il);
    // make fp32 + contiguous
    ggml_tensor * t32 = ggml_cont(ctx, ggml_cast(ctx, t, GGML_TYPE_F32));
    ggml_tensor * args[1] = { t32 };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
                          t32->ne[0], t32->ne[1], t32->ne[2], t32->ne[3],
                          args, 1, dragon_dump_kernel, /*n_tasks=*/1, ud);
}

// Dragon 7A1B — hybrid Mamba3-MIMO ("M") + Differential-TPA-V2 ("V") + MoE,
// with a geodesic-rotation residual on every block.

void llama_model_dragon::load_arch_hparams(llama_model_loader & ml) {
    const std::string arch = arch_name();

    // ---- standard hparams ----
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // SWA window is used by the scalable-softmax pre-multiply only (see the V
    // mixer). We don't forward it into hparams.n_swa: llama.cpp's iswa cache
    // requires at least one full-attention layer, which Dragon doesn't have.
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.dragon_slw_wsize, false);
    ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
    if (hparams.f_attn_logit_softcapping > 0.0f) {
        hparams.attn_soft_cap = true;
    }

    // ---- MoE ----
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                   hparams.moe_latent_size, false);

    // ---- Mamba3-MIMO ----
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);

    // ---- Dragon-specific ----
    ml.get_key(arch + ".mamba_mimo_dim",   hparams.dragon_mamba_mimo_dim);
    ml.get_key(arch + ".mamba_headdim",    hparams.dragon_mamba_headdim);
    ml.get_key(arch + ".num_signal_heads", hparams.dragon_n_signal_heads);
    ml.get_key(arch + ".num_noise_heads",  hparams.dragon_n_noise_heads);
    ml.get_key(arch + ".tpa_rank",         hparams.dragon_tpa_rank);
    ml.get_key(arch + ".num_rope_angles",  hparams.dragon_num_rope_angles);
    ml.get_key(arch + ".gate_bias",        hparams.dragon_gate_bias, false);

    // Layer-types string: 'M' for Mamba3-MIMO (recurrent), 'V' for Diff-TPA-V2 (attention).
    std::string layer_types;
    ml.get_key(arch + ".layer_types", layer_types);
    if (layer_types.size() != hparams.n_layer()) {
        throw std::runtime_error("dragon: layer_types length (" + std::to_string(layer_types.size()) +
                                 ") != n_layer (" + std::to_string(hparams.n_layer()) + ")");
    }
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        const bool is_M = (layer_types[i] == 'M');
        hparams.is_recr_impl[i] = is_M;
        hparams.n_head_arr[i]    = hparams.dragon_n_signal_heads + hparams.dragon_n_noise_heads;
        hparams.n_head_kv_arr[i] = is_M ? 0u : hparams.dragon_n_noise_heads;
        hparams.n_ff_arr[i]      = 0u; // no dense FFN — MoE per block
        hparams.swa_layers[i]    = false; // see SLW comment above
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dragon::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    auto is_M = [&](int il) { return hparams.is_recr(il); };

    // SSM (Mamba3-MIMO) constants
    const int64_t H_ssm    = (int64_t) hparams.ssm_d_inner / (int64_t) hparams.ssm_dt_rank; // 48
    const int64_t Hdim_ssm = (int64_t) hparams.ssm_dt_rank;                                 // 64
    const int64_t d_state  = (int64_t) hparams.ssm_d_state;                                 // 128
    const int64_t R        = (int64_t) hparams.dragon_mamba_mimo_dim;                       // 4
    const int64_t G        = (int64_t) hparams.ssm_n_group;                                 // 1
    const int64_t angles_d = (int64_t) hparams.dragon_num_rope_angles;                      // 32
    const int64_t d_in_static = H_ssm * (2 * Hdim_ssm + 3);                                 // 6288
    const int64_t d_in_dyn    = 2 * G * d_state * R + angles_d;                             // 1056

    // Attention (Diff-TPA-V2) constants. `n_head_attn` avoids LLAMA_LOAD_LOCALS' n_head.
    const int64_t n_head_attn = (int64_t) hparams.dragon_n_signal_heads + (int64_t) hparams.dragon_n_noise_heads; // 48
    const int64_t n_kv      = (int64_t) hparams.dragon_n_noise_heads;                                              // 12
    const int64_t n_signal  = (int64_t) hparams.dragon_n_signal_heads;                                             // 36
    const int64_t head_dim  = (int64_t) hparams.n_embd_head_k_full;                                                // 128
    const int64_t tpa_rank  = (int64_t) hparams.dragon_tpa_rank;                                                   // 4

    // MoE constants
    const int64_t moe_n_embd = hparams.moe_latent_size > 0 ? (int64_t) hparams.moe_latent_size : (int64_t) n_embd;
    const int64_t ff_exp     = (int64_t) hparams.n_ff_exp;
    const int64_t ff_shexp   = (int64_t) hparams.n_ff_shexp;

    // ---- embeddings + LM head (no final norm: dragon.final_norm == false) ----
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // geodesic-residual scalars (always 4 per block)
        layer.geo_mixer_scale = create_tensor(tn(LLM_TENSOR_GEODESIC_MIXER_SCALE, i), { 1 }, 0);
        layer.geo_mixer_bias  = create_tensor(tn(LLM_TENSOR_GEODESIC_MIXER_BIAS,  i), { 1 }, 0);
        layer.geo_mlp_scale   = create_tensor(tn(LLM_TENSOR_GEODESIC_MLP_SCALE,   i), { 1 }, 0);
        layer.geo_mlp_bias    = create_tensor(tn(LLM_TENSOR_GEODESIC_MLP_BIAS,    i), { 1 }, 0);

        const int64_t mixer_in = is_M(i) ? (H_ssm * Hdim_ssm) : (n_signal * head_dim);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { mixer_in, n_embd }, 0);

        if (is_M(i)) {
            layer.ssm_in     = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), { n_embd, d_in_static }, 0);
            layer.ssm_in_dyn = create_tensor(tn(LLM_TENSOR_SSM_IN_DYN, "weight", i), { n_embd, d_in_dyn   }, 0);

            layer.ssm_b_norm = create_tensor(tn(LLM_TENSOR_SSM_B_NORM, "weight", i), { d_state }, 0);
            layer.ssm_c_norm = create_tensor(tn(LLM_TENSOR_SSM_C_NORM, "weight", i), { d_state }, 0);

            layer.ssm_b_bias = create_tensor(tn(LLM_TENSOR_SSM_B_BIAS, i), { d_state, R, H_ssm }, 0);
            layer.ssm_c_bias = create_tensor(tn(LLM_TENSOR_SSM_C_BIAS, i), { d_state, R, H_ssm }, 0);

            layer.ssm_dt_bias = create_tensor(tn(LLM_TENSOR_SSM_DT_BIAS, i), { H_ssm }, 0);
            layer.ssm_d       = create_tensor(tn(LLM_TENSOR_SSM_D,       i), { H_ssm }, 0);

            layer.ssm_mimo_x = create_tensor(tn(LLM_TENSOR_SSM_MIMO_X, i), { Hdim_ssm, R, H_ssm }, 0);
            layer.ssm_mimo_z = create_tensor(tn(LLM_TENSOR_SSM_MIMO_Z, i), { Hdim_ssm, R, H_ssm }, 0);
            layer.ssm_mimo_o = create_tensor(tn(LLM_TENSOR_SSM_MIMO_O, i), { Hdim_ssm, R, H_ssm }, 0);
        } else {
            layer.wq                  = create_tensor(tn(LLM_TENSOR_ATTN_Q,              "weight", i), { n_embd, n_head_attn * head_dim }, 0);
            layer.attn_wa_k           = create_tensor(tn(LLM_TENSOR_ATTN_WA_K,           "weight", i), { n_embd, n_kv  * tpa_rank }, 0);
            layer.attn_wa_v           = create_tensor(tn(LLM_TENSOR_ATTN_WA_V,           "weight", i), { n_embd, n_kv  * tpa_rank }, 0);
            layer.attn_wb_k           = create_tensor(tn(LLM_TENSOR_ATTN_WB_K,           "weight", i), { n_embd, tpa_rank * head_dim }, 0);
            layer.attn_wb_v           = create_tensor(tn(LLM_TENSOR_ATTN_WB_V,           "weight", i), { n_embd, tpa_rank * head_dim }, 0);
            layer.attn_shift_k        = create_tensor(tn(LLM_TENSOR_ATTN_SHIFT_K,        "weight", i), { n_embd, n_kv }, 0);
            layer.attn_shift_v        = create_tensor(tn(LLM_TENSOR_ATTN_SHIFT_V,        "weight", i), { n_embd, n_kv }, 0);
            layer.attn_lambda         = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA,         "weight", i), { n_embd, n_kv }, 0);
            layer.attn_softmax_scaler = create_tensor(tn(LLM_TENSOR_ATTN_SOFTMAX_SCALER,           i), { n_head_attn }, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { head_dim }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { head_dim }, 0);

            layer.attn_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, n_signal * head_dim }, 0);
        }

        // MoE (every block)
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,           i), { n_expert }, 0);
        layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), { n_embd, moe_n_embd }, 0);
        layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), { moe_n_embd, n_embd }, 0);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), { moe_n_embd, ff_exp, n_expert }, 0);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), { ff_exp, moe_n_embd, n_expert }, 0);
        layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,    "weight", i), { n_embd, ff_shexp }, 0);
        layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,  "weight", i), { ff_shexp, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dragon::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Geodesic-residual update. Given x (residual) and g (mixer/MLP output),
// rotate x along the unit tangent direction by an angle θ ∈ [−∞, π/4]:
//     output = x · cos(θ) + (g_⊥ / ||g_⊥||) · ||x|| · sin(θ)
// where g_⊥ = g − ((x·g) / ||x||²) · x is the component of g orthogonal to x,
// and θ = clamp_max(((||g_⊥||/||x||) · scale + bias) / (layer_idx + 1), π/4).
// HF does not clamp θ from below — bias can drive θ negative, which flips the
// sign of the sin(θ) tangent term. Reductions are over the n_embd axis (ne[0]).
// Fused single-node version of the geodesic residual below. The graph version
// expands to ~22 ggml nodes per call (x2 per block x 36 blocks ~ 1.6k nodes per
// token); at decode each node is a 1536-float op whose cost is pure thread-
// barrier latency. The fused kernel does the same math in one node.
struct dragon_geodesic_userdata {
    int il;
};

static void dragon_geodesic_kernel(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * ud = (const dragon_geodesic_userdata *) userdata;
    const float kPi4 = 0.785398163f; // π/4

    const ggml_tensor * x_t = dst->src[0]; // (n_embd, T)
    const ggml_tensor * g_t = dst->src[1]; // (n_embd, T)

    GGML_ASSERT(x_t->type == GGML_TYPE_F32 && x_t->nb[0] == sizeof(float));
    GGML_ASSERT(g_t->type == GGML_TYPE_F32 && g_t->nb[0] == sizeof(float));
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[3]->type == GGML_TYPE_F32);

    const float scale = ((const float *) dst->src[2]->data)[0];
    const float bias  = ((const float *) dst->src[3]->data)[0];
    const float il_div = 1.0f / (float) (ud->il + 1);

    const int64_t n = x_t->ne[0];
    const int64_t T = x_t->ne[1];

    for (int64_t t = ith; t < T; t += nth) {
        const float * x = (const float *) ((const char *) x_t->data + t*x_t->nb[1]);
        const float * g = (const float *) ((const char *) g_t->data + t*g_t->nb[1]);
        float       * y = (float *)       ((char *)       dst->data + t*dst->nb[1]);

        float xx = 0.0f, xg = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            xx += x[i]*x[i];
            xg += x[i]*g[i];
        }
        const float x_norm_sq  = std::max(xx, 1e-12f);
        const float proj_coeff = xg / x_norm_sq;

        // g_⊥ = g − proj·x; stash it in dst while accumulating ||g_⊥||²
        float tn2 = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float d = g[i] - proj_coeff*x[i];
            y[i] = d;
            tn2 += d*d;
        }
        const float tan_norm = std::max(sqrtf(tn2), 1e-8f);
        const float R        = sqrtf(x_norm_sq);
        const float R_min    = std::max(R, 1e-6f);

        float theta = std::min(tan_norm / R_min, kPi4);
        theta = std::min((theta*scale + bias) * il_div, kPi4);

        const float c = cosf(theta);
        const float s = sinf(theta) * R / tan_norm;
        for (int64_t i = 0; i < n; ++i) {
            y[i] = x[i]*c + y[i]*s;
        }
    }
}

static ggml_tensor * build_dragon_geodesic_fused(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * g,
        ggml_tensor  * scale_scalar,
        ggml_tensor  * bias_scalar,
        int            il) {
    ggml_tensor * args[4] = { x, g, scale_scalar, bias_scalar };
    auto * ud = (dragon_geodesic_userdata *) std::malloc(sizeof(dragon_geodesic_userdata));
    ud->il = il;
    ggml_tensor * geo = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1,
                          args, 4, dragon_geodesic_kernel,
                          /*n_tasks=*/ GGML_N_TASKS_MAX, ud);
    ggml_set_name(geo, "geodesic");
    return geo;
}

static ggml_tensor * build_dragon_geodesic_ref(
        ggml_context * ctx,
        ggml_tensor  * x,           // (n_embd, n_tokens)
        ggml_tensor  * g,           // (n_embd, n_tokens)
        ggml_tensor  * scale_scalar,// (1)
        ggml_tensor  * bias_scalar, // (1)
        int            il) {
    const float kPi4 = 0.785398163f; // π/4

    // ||x||²    (per token; min-clamped)
    ggml_tensor * x_norm_sq = ggml_sum_rows(ctx, ggml_sqr(ctx, x));
    x_norm_sq = ggml_clamp(ctx, x_norm_sq, 1e-12f, FLT_MAX);

    // proj_coeff = (x · g) / ||x||²    (per token)
    ggml_tensor * xg_dot     = ggml_sum_rows(ctx, ggml_mul(ctx, x, g));
    ggml_tensor * proj_coeff = ggml_div(ctx, xg_dot, x_norm_sq);

    // g_⊥ = g − proj_coeff · x       (broadcasts proj_coeff over n_embd)
    ggml_tensor * grad = ggml_sub(ctx, g, ggml_mul(ctx, x, proj_coeff));

    // ||g_⊥||                          (min-clamped to avoid NaN)
    ggml_tensor * tan_norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, grad)));
    tan_norm = ggml_clamp(ctx, tan_norm, 1e-8f, FLT_MAX);

    // ||x||                             (from x_norm_sq, already clamped)
    ggml_tensor * R     = ggml_sqrt(ctx, x_norm_sq);
    ggml_tensor * R_min = ggml_clamp(ctx, R, 1e-6f, FLT_MAX);

    // θ = clamp_max((||g_⊥||/||x|| · scale + bias) / (layer_idx + 1), π/4).
    // -FLT_MAX as the min mirrors HF's `torch.clamp(x, max=π/4)` exactly.
    ggml_tensor * theta = ggml_clamp(ctx, ggml_div(ctx, tan_norm, R_min), -FLT_MAX, kPi4);
    theta = ggml_mul(ctx, theta, scale_scalar);
    theta = ggml_add(ctx, theta, bias_scalar);
    theta = ggml_scale(ctx, theta, 1.0f / (float)(il + 1));
    theta = ggml_clamp(ctx, theta, -FLT_MAX, kPi4);

    // output = x·cos(θ) + (g_⊥/||g_⊥||) · ||x|| · sin(θ)
    ggml_tensor * unit_tangent = ggml_div(ctx, grad, tan_norm);
    ggml_tensor * cos_t = ggml_cos(ctx, theta);
    ggml_tensor * sin_t = ggml_sin(ctx, theta);
    ggml_tensor * left  = ggml_mul(ctx, x, cos_t);
    ggml_tensor * right = ggml_mul(ctx, ggml_mul(ctx, unit_tangent, R), sin_t);
    return ggml_add(ctx, left, right);
}

static ggml_tensor * build_dragon_geodesic(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * g,
        ggml_tensor  * scale_scalar,
        ggml_tensor  * bias_scalar,
        int            il) {
    // DRAGON_NO_FUSED_GEO=1 selects the pure-ggml reference path (also the
    // path for non-CPU backends if this graph is ever offloaded).
    static const bool use_ref = std::getenv("DRAGON_NO_FUSED_GEO") != nullptr;
    if (use_ref) {
        return build_dragon_geodesic_ref(ctx, x, g, scale_scalar, bias_scalar, il);
    }
    return build_dragon_geodesic_fused(ctx, x, g, scale_scalar, bias_scalar, il);
}

// Mamba3-MIMO CPU kernel. O(L · H · D_qk · D_v · R) per layer; wired in via
// ggml_custom_4d. State is zero-initialised on every call, so this matches HF
// for single-pass prefill but not for incremental decode (state in/out are not
// yet plumbed through the recurrent cache).
//
// Per-step recurrence (per head h, per timestep t):
//   angle_state[h] += tanh(ang_raw[t]) · dt[h,t] · π
//   q_rot, k_rot   = halved-rotary(q, k, angle_state)      // rotary_dim_divisor=4
//   curr_kv[p,d]   = sum_r k_rot[d,r] · v[p,r]
//   prev_kv[p,d]   = sum_r K_state[h,r,d] · V_state[h,r,p]
//   state[h,p,d]   = α·state[h,p,d] + (1−trap)·dt·α·prev_kv[p,d] + trap·dt·curr_kv[p,d]
//   o[r,p]         = sum_d state[h,p,d] · q_rot[r,d] + D[h]·v[p,r]
//   o[r,p]        *= z[p,r,h,t] · sigmoid(z[p,r,h,t])
//   y[p,h,t]       = sum_r o[r,p] · MIMO_O[p,r,h]
//
// Inputs (8 or 9 src tensors, all fp32 contiguous):
//   src[0] q_biased (D_qk, R, H, L)   src[4] adts    (3, H, L)   // α, dt, trap
//   src[1] k_biased (D_qk, R, H, L)   src[5] D_skip  (H,)
//   src[2] v_proj   (D_v,  R, H, L)   src[6] mimo_o  (D_v, R, H)
//   src[3] z_proj   (D_v,  R, H, L)   src[7] ang_raw (num_angles, L)
//   src[8] state_in (n_embd_s, n_seqs) — OPTIONAL. If present, used as initial
//          ssm_state/K_state/V_state/angle_state (packed as defined in
//          llama_hparams::n_embd_s()); if absent, all are zero-initialised.
//
// Output: when src[8] is absent → y (D_v, H, L). When src[8] is present →
//   a flat tensor of length D_v·H·L + n_embd_s·n_seqs, containing y followed
//   by the new packed state (caller views/splits and ggml_cpy's the state
//   tail back to the cache).
// Per-layer constant weights of the M-mixer, converted to f32 and packed ONCE
// per process. Rebuilding these packs as graph ops (casts + concats of frozen
// tensors) cost ~16 barrier-floor nodes per layer per token. Keyed by the
// b_bias tensor's data pointer; entries live until process exit.
struct dragon_packed_weights {
    std::vector<float> buf;
    const float * bias;   // (D_qk, 2R, H)  rows [b_bias | c_bias]
    const float * mxz;    // (D_v, 2R, H)   rows [mimo_x | mimo_z]
    const float * mimo_o; // (D_v, R, H)
    const float * norms;  // (D_qk, 2)      cols [b_norm | c_norm]
    const float * misc;   // (H, 2)         cols [dt_bias | D_skip]
};

static float dragon_weight_f32(const ggml_tensor * t, int64_t i) {
    switch (t->type) {
        case GGML_TYPE_F32:  return ((const float *) t->data)[i];
        case GGML_TYPE_BF16: {
            // same bit-shift conversion ggml_cast does
            const uint32_t u = (uint32_t) (((const uint16_t *) t->data)[i]) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(((const ggml_fp16_t *) t->data)[i]);
        default: GGML_ABORT("dragon: unsupported weight type %s", ggml_type_name(t->type));
    }
}

static const dragon_packed_weights * dragon_get_packed_weights(
        const llama_layer & layer, int64_t D_qk, int64_t D_v, int64_t R, int64_t H) {
    static std::mutex mtx;
    static std::unordered_map<const void *, std::unique_ptr<dragon_packed_weights>> cache;

    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(layer.ssm_b_bias->data);
    if (it != cache.end()) {
        return it->second.get();
    }

    auto pw = std::make_unique<dragon_packed_weights>();
    const int64_t n_bias = D_qk*2*R*H, n_mxz = D_v*2*R*H, n_mo = D_v*R*H;
    pw->buf.resize((size_t) (n_bias + n_mxz + n_mo + 2*D_qk + 2*H));
    float * p = pw->buf.data();

    pw->bias = p;  // (D_qk, R, H) pairs interleaved to (D_qk, [b·R | c·R], H)
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t d = 0; d < D_qk; ++d) {
                const int64_t src = d + r*D_qk + h*R*D_qk;
                p[d + r*D_qk       + h*2*R*D_qk] = dragon_weight_f32(layer.ssm_b_bias, src);
                p[d + (R + r)*D_qk + h*2*R*D_qk] = dragon_weight_f32(layer.ssm_c_bias, src);
            }
        }
    }
    p += n_bias;
    pw->mxz = p;
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t d = 0; d < D_v; ++d) {
                const int64_t src = d + r*D_v + h*R*D_v;
                p[d + r*D_v       + h*2*R*D_v] = dragon_weight_f32(layer.ssm_mimo_x, src);
                p[d + (R + r)*D_v + h*2*R*D_v] = dragon_weight_f32(layer.ssm_mimo_z, src);
            }
        }
    }
    p += n_mxz;
    pw->mimo_o = p;
    for (int64_t i = 0; i < n_mo; ++i) {
        p[i] = dragon_weight_f32(layer.ssm_mimo_o, i);
    }
    p += n_mo;
    pw->norms = p;
    for (int64_t d = 0; d < D_qk; ++d) {
        p[d]        = dragon_weight_f32(layer.ssm_b_norm, d);
        p[D_qk + d] = dragon_weight_f32(layer.ssm_c_norm, d);
    }
    p += 2*D_qk;
    pw->misc = p;
    for (int64_t h = 0; h < H; ++h) {
        p[h]     = dragon_weight_f32(layer.ssm_dt_bias, h);
        p[H + h] = dragon_weight_f32(layer.ssm_d, h);
    }

    const dragon_packed_weights * ret = pw.get();
    cache.emplace(layer.ssm_b_bias->data, std::move(pw));
    return ret;
}

// Userdata struct passed via ggml_custom_4d. NULL → n_seqs = 1 (legacy).
struct dragon_m_kernel_userdata {
    int64_t n_seqs;
    float   rms_eps;
    int64_t D_qk, D_v, R, H;
    const dragon_packed_weights * w;
    // raw bf16 projection weights for the decode mega-kernel
    const ggml_tensor * w_in = nullptr;
    const ggml_tensor * w_wo = nullptr;
};

// ---- Decode MIMO mega-kernel ----
// Decode-only (one token per sequence): fuses the per-head in_proj GEMV, the
// Mamba3-MIMO state step and the per-head out_proj GEMV into one pass, so the
// big projection weights stream from DRAM *while* the state math runs, and
// each head's weight slice is streamed ONCE for all concurrent sequences
// (multi-user M-layer weight traffic becomes independent of user count).
// Each head accumulates into its own output slab (dst is (n_embd, L, H));
// a small follow-up custom op reduces the slabs.

#if defined(__AVX512F__)
static inline __m512 dragon_bf16_load16(const uint16_t * p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) p)), 16));
}
#endif

static void dragon_m_mega_decode_kernel(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * cur   = dst->src[0]; // (n_embd, L) f32
    const ggml_tensor * pdyn  = dst->src[1]; // (2·R·D_qk + num_angles, L) f32
    const ggml_tensor * state_in = dst->src[2]; // (n_embd_s, n_seqs) gathered
    ggml_tensor * state_dst      = dst->src[3]; // cache view, written in-place

    const auto * ud = (const dragon_m_kernel_userdata *) userdata;
    const int64_t D_qk = ud->D_qk, D_v = ud->D_v, R = ud->R, H = ud->H;
    const int64_t L = cur->ne[1];           // == n_seqs (one token per seq)
    const int64_t n_embd = cur->ne[0];
    const float rms_eps = ud->rms_eps;
    const int64_t num_angles = pdyn->ne[0] - 2 * R * D_qk;
    const int64_t quarter    = D_qk / 4;
    const bool    do_rotary  = (num_angles == quarter) && (D_qk >= 4);
    const int64_t per_head   = 2 * D_v + 3;

    const float * cur_d   = (const float *) cur->data;
    const int64_t cur_s1  = cur->nb[1] / sizeof(float);
    const float * pdyn_d  = (const float *) pdyn->data;
    const int64_t pdyn_s1 = pdyn->ne[0];
    const float * bias_d  = ud->w->bias;
    const float * mxz_d   = ud->w->mxz;
    const float * norms_d = ud->w->norms;
    const float * misc_d  = ud->w->misc;
    const float * mimo_d  = ud->w->mimo_o;
    const float * state_in_d = (const float *) state_in->data;
    float * state_out = (float *) state_dst->data;
    const int64_t n_state_per_seq = (int64_t) (state_dst->nb[1] / sizeof(float));
    float * y_d = (float *) dst->data;       // (n_embd, L, H) per-head slabs

    GGML_ASSERT(ud->w_in != nullptr && ud->w_wo != nullptr);
    GGML_ASSERT(ud->w_in->type == GGML_TYPE_BF16 && ud->w_wo->type == GGML_TYPE_BF16);
    const uint16_t * win_d = (const uint16_t *) ud->w_in->data;  // (n_embd, per_head·H)
    const uint16_t * wo_d  = (const uint16_t *) ud->w_wo->data;  // (D_v·H, n_embd)

    const int64_t off_K_glob   = H * D_v * D_qk;
    const int64_t off_V_glob   = off_K_glob + H * R * D_qk;
    const int64_t off_ang_glob = off_V_glob + H * R * D_v;

    // dragon_state_bf16() returns false whenever DRAGON_MEGA_DECODE is set,
    // so this kernel always sees an f32 state cache.
    GGML_ASSERT(!dragon_state_bf16() && "mega decode kernel does not support bf16 state");

    const int64_t h_per = (H + nth - 1) / nth;
    const int64_t h0 = std::min<int64_t>((int64_t) ith * h_per, H);
    const int64_t h1 = std::min<int64_t>(h0 + h_per, H);

    // zero this thread's head slabs (always, so the reducer can sum all H)
    for (int64_t h = h0; h < h1; ++h) {
        std::memset(y_d + h * n_embd * L, 0, (size_t) n_embd * L * sizeof(float));
    }
    if (h0 >= h1) {
        return;
    }

    auto sp = [](float x) {
        const float pos = x > 0.0f ? x : 0.0f;
        return pos + logf(expf(-fabsf(x)) + 1.0f);
    };

    std::vector<float> ph_buf((size_t) L * per_head);
    std::vector<float> bcn((size_t) 2 * R * D_qk);
    std::vector<float> k_rot((size_t) R * D_qk), q_rot((size_t) R * D_qk);
    std::vector<float> v_loc((size_t) R * D_v),  z_loc((size_t) R * D_v);
    std::vector<float> sstate((size_t) D_v * D_qk);
    std::vector<float> Kst((size_t) R * D_qk), Vst((size_t) R * D_v);
    std::vector<float> angst((size_t) num_angles);
    std::vector<float> cs_buf((size_t) num_angles), ss_buf((size_t) num_angles);
    std::vector<float> y_h((size_t) D_v);

    for (int64_t h = h0; h < h1; ++h) {
        // --- in_proj GEMV for this head: rows [h·per_head, (h+1)·per_head) ---
        for (int64_t j = 0; j < per_head; ++j) {
            const uint16_t * wrow = win_d + (h * per_head + j) * n_embd;
            for (int64_t seq = 0; seq < L; ++seq) {
                const float * x = cur_d + seq * cur_s1;
#if defined(__AVX512F__)
                __m512 acc = _mm512_setzero_ps();
                for (int64_t d = 0; d < n_embd; d += 16) {
                    acc = _mm512_fmadd_ps(dragon_bf16_load16(wrow + d), _mm512_loadu_ps(x + d), acc);
                }
                ph_buf[seq * per_head + j] = _mm512_reduce_add_ps(acc);
#else
                float acc = 0.0f;
                for (int64_t d = 0; d < n_embd; ++d) {
                    uint32_t u = (uint32_t) wrow[d] << 16; float wv; memcpy(&wv, &u, 4);
                    acc += wv * x[d];
                }
                ph_buf[seq * per_head + j] = acc;
#endif
            }
        }

        const float * bias_h = bias_d + h * 2 * R * D_qk;
        const float * mx_h   = mxz_d + h * 2 * R * D_v;
        const float * mz_h   = mx_h + R * D_v;
        const float * mo     = mimo_d + h * R * D_v;
        const float   D_h    = misc_d[H + h];

        for (int64_t seq = 0; seq < L; ++seq) {
            // B/C RMS-norm for this token (same math as the main kernel)
            const float * bc_t = pdyn_d + seq * pdyn_s1;
            for (int64_t g = 0; g < 2 * R; ++g) {
                const float * src = bc_t + g * D_qk;
                const float * w   = norms_d + (g < R ? 0 : D_qk);
                float       * out = bcn.data() + g * D_qk;
                double sum = 0.0;
                for (int64_t d = 0; d < D_qk; ++d) sum += (double) src[d] * (double) src[d];
                const float scale = 1.0f / sqrtf((float) (sum / D_qk) + rms_eps);
                for (int64_t d = 0; d < D_qk; ++d) out[d] = (src[d] * scale) * w[d];
            }

            const float * ph = ph_buf.data() + seq * per_head;
            const float * z_t = ph;
            const float * x_t = ph + D_v;
            const float dt_sp  = sp(ph[2*D_v + 0] + misc_d[h]);
            float a_neg        = -sp(ph[2*D_v + 1]);
            if (a_neg > -1e-4f) a_neg = -1e-4f;
            const float a_h    = expf(a_neg * dt_sp);
            const float trap_h = dragon_sigmoidf(ph[2*D_v + 2]);
            const float beta_h  = (1.0f - trap_h) * dt_sp * a_h;
            const float gamma_h = trap_h * dt_sp;

            // state in
            const float * sis = state_in_d + seq * state_in->nb[1] / sizeof(float);
            std::memcpy(sstate.data(), sis + h * D_v * D_qk, (size_t) D_v * D_qk * sizeof(float));
            std::memcpy(Kst.data(), sis + off_K_glob + h * R * D_qk, (size_t) R * D_qk * sizeof(float));
            std::memcpy(Vst.data(), sis + off_V_glob + h * R * D_v, (size_t) R * D_v * sizeof(float));
            std::memcpy(angst.data(), sis + off_ang_glob + h * num_angles, (size_t) num_angles * sizeof(float));

            // angle update + k/q materialize + rotary
            const float * ang_t = pdyn_d + seq * pdyn_s1 + 2 * R * D_qk;
            for (int64_t i = 0; i < num_angles; ++i) {
                angst[i] += tanhf(ang_t[i]) * (float) M_PI * dt_sp;
            }
            const int64_t half = R * D_qk;
            for (int64_t j = 0; j < half; ++j) k_rot[j] = bcn[j] + bias_h[j];
            for (int64_t j = 0; j < half; ++j) q_rot[j] = bcn[half + j] + bias_h[half + j];
            for (int64_t r2 = 0; r2 < R; ++r2) {
                float * vr = v_loc.data() + r2 * D_v;
                float * zr = z_loc.data() + r2 * D_v;
                const float * mxr = mx_h + r2 * D_v;
                const float * mzr = mz_h + r2 * D_v;
                for (int64_t p = 0; p < D_v; ++p) { vr[p] = x_t[p] * mxr[p]; zr[p] = z_t[p] * mzr[p]; }
            }
            if (do_rotary) {
                for (int64_t i = 0; i < num_angles; ++i) { cs_buf[i] = cosf(angst[i]); ss_buf[i] = sinf(angst[i]); }
                const int64_t i2_off = 2 * quarter;
                for (int64_t r2 = 0; r2 < R; ++r2) {
                    float * qr = q_rot.data() + r2 * D_qk;
                    float * kr = k_rot.data() + r2 * D_qk;
                    for (int64_t i = 0; i < quarter; ++i) {
                        const float c = cs_buf[i], s2 = ss_buf[i];
                        const float q0 = qr[i], q2 = qr[i + i2_off];
                        qr[i] = q0 * c - q2 * s2; qr[i + i2_off] = q0 * s2 + q2 * c;
                        const float kk0 = kr[i], kk2 = kr[i + i2_off];
                        kr[i] = kk0 * c - kk2 * s2; kr[i + i2_off] = kk0 * s2 + kk2 * c;
                    }
                }
            }

            std::fill(y_h.begin(), y_h.end(), 0.0f);
            dragon_simd_fused_step(sstate.data(), Kst.data(), Vst.data(),
                                   k_rot.data(), q_rot.data(), v_loc.data(), z_loc.data(),
                                   mo, D_v, y_h.data(),
                                   a_h, beta_h, gamma_h, D_h, R, D_qk, D_v);

            // state out (this head's sections only — no cross-thread races)
            float * sos = state_out + seq * n_state_per_seq;
            std::memcpy(sos + h * D_v * D_qk, sstate.data(), (size_t) D_v * D_qk * sizeof(float));
            std::memcpy(sos + off_K_glob + h * R * D_qk, k_rot.data(), (size_t) R * D_qk * sizeof(float));
            std::memcpy(sos + off_V_glob + h * R * D_v, v_loc.data(), (size_t) R * D_v * sizeof(float));
            std::memcpy(sos + off_ang_glob + h * num_angles, angst.data(), (size_t) num_angles * sizeof(float));

            // --- out_proj for this head: out[o] += Σ_p wo[h·D_v + p, o] · y_h[p] ---
            float * slab = y_d + h * n_embd * L + seq * n_embd;
            const uint16_t * wo_h = wo_d + h * D_v; // element (j=h·D_v+p, o) at j + o·(D_v·H)
            const int64_t wo_s1 = D_v * H;
#if defined(__AVX512F__)
            if (D_v == 64) {
                const __m512 y0 = _mm512_loadu_ps(y_h.data());
                const __m512 y1 = _mm512_loadu_ps(y_h.data() + 16);
                const __m512 y2 = _mm512_loadu_ps(y_h.data() + 32);
                const __m512 y3 = _mm512_loadu_ps(y_h.data() + 48);
                for (int64_t o = 0; o < n_embd; ++o) {
                    const uint16_t * w = wo_h + o * wo_s1;
                    __m512 acc = _mm512_mul_ps(dragon_bf16_load16(w), y0);
                    acc = _mm512_fmadd_ps(dragon_bf16_load16(w + 16), y1, acc);
                    acc = _mm512_fmadd_ps(dragon_bf16_load16(w + 32), y2, acc);
                    acc = _mm512_fmadd_ps(dragon_bf16_load16(w + 48), y3, acc);
                    slab[o] = _mm512_reduce_add_ps(acc);
                }
            } else
#endif
            {
                for (int64_t o = 0; o < n_embd; ++o) {
                    const uint16_t * w = wo_h + o * wo_s1;
                    float acc = 0.0f;
                    for (int64_t p = 0; p < D_v; ++p) {
                        uint32_t u = (uint32_t) w[p] << 16; float wv; memcpy(&wv, &u, 4);
                        acc += wv * y_h[p];
                    }
                    slab[o] = acc;
                }
            }
        }
    }
}

// Sum the per-head output slabs: dst (n_embd, L) = Σ_h src0[:, :, h].
static void dragon_m_mega_reduce_kernel(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * slabs = dst->src[0]; // (n_embd, L, H)
    const int64_t n  = slabs->ne[0] * slabs->ne[1];
    const int64_t H  = slabs->ne[2];
    const float * s  = (const float *) slabs->data;
    float       * o  = (float *) dst->data;
    const int64_t per = (n + nth - 1) / nth;
    const int64_t i0 = std::min<int64_t>((int64_t) ith * per, n);
    const int64_t i1 = std::min<int64_t>(i0 + per, n);
    for (int64_t i = i0; i < i1; ++i) {
        float acc = s[i];
        for (int64_t h = 1; h < H; ++h) {
            acc += s[i + h * n];
        }
        o[i] = acc;
    }
}

// ---- Fused MoE selection / weighting / reduction (dragon-local) ----
// Mirrors build_moe_ffn with sigmoid gating + DeepSeek-V3 selection bias +
// norm_w + w_scale for Dragon's config. The ggml chain is 8 selection nodes +
// mul + 5 adds per layer; fused it is 3 custom nodes around the mul_mat_ids.

struct dragon_moe_userdata {
    float w_scale;
};

// scores = sigmoid(logits) + bias; writes ids of the k largest (desc, ties →
// lower index — same as a stable descending argsort).
static void dragon_moe_select(const float * lt, const float * bd, int64_t n_expert,
                              int64_t k, int32_t * ids) {
    for (int64_t i = 0; i < k; ++i) {
        int64_t best = -1;
        float   bv   = -FLT_MAX;
        for (int64_t e = 0; e < n_expert; ++e) {
            bool taken = false;
            for (int64_t j = 0; j < i; ++j) {
                if (ids[j] == (int32_t) e) { taken = true; break; }
            }
            if (taken) {
                continue;
            }
            const float s = dragon_sigmoidf(lt[e]) + bd[e];
            if (s > bv) { bv = s; best = e; }
        }
        if (best < 0) {
            // all remaining scores are NaN (activation blow-up at this token,
            // seen with quantized weights). Match upstream argsort semantics:
            // pick an in-bounds index so mul_mat_id stays safe; the token's
            // output is garbage either way.
            static std::atomic<int> warned{0};
            if (warned.fetch_add(1) == 0) {
                fprintf(stderr, "dragon: WARNING: all-NaN MoE router scores for a token (quantization-induced activation overflow?); selecting expert %d\n", (int) i);
            }
            best = i;
        }
        ids[i] = (int32_t) best;
    }
}

static void dragon_moe_topk_kernel(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * logits = dst->src[0]; // (n_expert, L) f32
    const ggml_tensor * bias   = dst->src[1]; // (n_expert)   f32
    const int64_t n_expert = logits->ne[0];
    const int64_t L        = logits->ne[1];
    const int64_t k        = dst->ne[0];
    const float * ld = (const float *) logits->data;
    const float * bd = (const float *) bias->data;
    int32_t     * od = (int32_t *) dst->data;
    const int64_t per = (L + nth - 1) / nth;
    const int64_t t0 = std::min<int64_t>((int64_t) ith * per, L);
    const int64_t t1 = std::min<int64_t>(t0 + per, L);
    for (int64_t t = t0; t < t1; ++t) {
        dragon_moe_select(ld + t * n_expert, bd, n_expert, k, od + t * k);
    }
}

// weights = sigmoid(logits)[sel] (un-biased), normalized (sum → clamp → div),
// then × w_scale — same op order as build_moe_ffn's norm_w path.
static void dragon_moe_weights_kernel(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * logits = dst->src[0];
    const ggml_tensor * bias   = dst->src[1];
    const float w_scale = ((const dragon_moe_userdata *) userdata)->w_scale;
    const int64_t n_expert = logits->ne[0];
    const int64_t L        = logits->ne[1];
    const int64_t k        = dst->ne[0];
    const float * ld = (const float *) logits->data;
    const float * bd = (const float *) bias->data;
    float       * od = (float *) dst->data;
    const int64_t per = (L + nth - 1) / nth;
    const int64_t t0 = std::min<int64_t>((int64_t) ith * per, L);
    const int64_t t1 = std::min<int64_t>(t0 + per, L);
    // NOTE: llama.cpp warmup graphs run with n_expert_used == n_expert (256)
    // to touch all expert weights — size for that, not just the normal top-6.
    int32_t ids[256];
    GGML_ASSERT(k <= 256);
    for (int64_t t = t0; t < t1; ++t) {
        const float * lt = ld + t * n_expert;
        dragon_moe_select(lt, bd, n_expert, k, ids);
        float * w = od + t * k;
        float sum = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            w[i] = dragon_sigmoidf(lt[ids[i]]);
            sum += w[i];
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f; // same clamp as build_moe_ffn
        for (int64_t i = 0; i < k; ++i) {
            w[i] = (w[i] / sum) * w_scale;
        }
    }
}

// out[d, t] = Σ_i experts[d, i, t] · w[i, t], accumulated in expert order —
// identical per-element fp order to build_moe_ffn's mul + chained adds.
static void dragon_moe_reduce_kernel(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * experts = dst->src[0]; // (ne, k, L) f32
    const ggml_tensor * w       = dst->src[1]; // (k, L)     f32
    const int64_t ne = experts->ne[0];
    const int64_t k  = experts->ne[1];
    const int64_t L  = experts->ne[2];
    const size_t  e_s1 = experts->nb[1] / sizeof(float);
    const size_t  e_s2 = experts->nb[2] / sizeof(float);
    const float * ed = (const float *) experts->data;
    const float * wd = (const float *) w->data;
    float       * od = (float *) dst->data;
    const int64_t per = (L + nth - 1) / nth;
    const int64_t t0 = std::min<int64_t>((int64_t) ith * per, L);
    const int64_t t1 = std::min<int64_t>(t0 + per, L);
    for (int64_t t = t0; t < t1; ++t) {
        const float * et = ed + t * e_s2;
        const float * wt = wd + t * k;
        float * ot = od + t * ne;
        for (int64_t d = 0; d < ne; ++d) {
            ot[d] = et[d] * wt[0];
        }
        for (int64_t i = 1; i < k; ++i) {
            const float * ei = et + i * e_s1;
            const float  wi = wt[i];
            for (int64_t d = 0; d < ne; ++d) {
                ot[d] += ei[d] * wi;
            }
        }
    }
}

// ---- Fused V-layer token shift ----
// Computes K̃/Ṽ = α·X_prev + (1−α)·X with α = sigmoid(gate)·[pos>0], where
// X_prev is the previous token (cache for each seq's first in-batch token),
// and writes each seq's pre-shift last K/V back into the recurrent cache.
// Replaces ~19 small graph nodes (conts/concats/sigmoid/muls/cpy) per V-layer.
struct dragon_vshift_userdata {
    int64_t n_seqs;
};

static void dragon_v_shift_kernel(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * K    = dst->src[0]; // (head_dim, n_kv, L) f32, pre-shift
    const ggml_tensor * V    = dst->src[1];
    const ggml_tensor * ak   = dst->src[2]; // (n_kv, L) raw gate logits
    const ggml_tensor * av   = dst->src[3];
    const ggml_tensor * pos  = dst->src[4]; // (L) i32
    const ggml_tensor * prev = dst->src[5]; // (n_embd_r, n_seqs) gathered [K_prev | V_prev]
    ggml_tensor * cache      = dst->src[6]; // (n_embd_r, n_seqs) cache view, written in-place

    const int64_t hd    = K->ne[0];
    const int64_t n_kv  = K->ne[1];
    const int64_t L     = K->ne[2];
    const int64_t n_kp  = hd * n_kv;
    const int64_t n_seqs   = ((const dragon_vshift_userdata *) userdata)->n_seqs;
    const int64_t n_seq_t  = L / n_seqs;

    const float   * K_d   = (const float *) K->data;
    const float   * V_d   = (const float *) V->data;
    const float   * ak_d  = (const float *) ak->data;
    const float   * av_d  = (const float *) av->data;
    const int32_t * pos_d = (const int32_t *) pos->data;
    const float   * pv_d  = (const float *) prev->data;
    const int64_t   pv_s1 = prev->nb[1] / sizeof(float);
    float * cache_d       = (float *) cache->data;
    const int64_t cache_s1 = cache->nb[1] / sizeof(float);
    float * y_d           = (float *) dst->data;       // (hd, n_kv, L, 2): [K̃ | Ṽ]
    const int64_t y_s2 = hd * n_kv, y_s3 = hd * n_kv * L;

    const int64_t per = (L + nth - 1) / nth;
    const int64_t t0  = std::min<int64_t>((int64_t) ith * per, L);
    const int64_t t1  = std::min<int64_t>(t0 + per, L);

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t seq = t / n_seq_t;
        const int64_t tt  = t % n_seq_t;   // index within this sequence's block
        // α-zero mask at sequence position 0 — same cast+clamp as the graph did
        float pmask = (float) pos_d[t];
        pmask = pmask < 0.0f ? 0.0f : (pmask > 1.0f ? 1.0f : pmask);
        for (int64_t j = 0; j < n_kv; ++j) {
            const float a_k = dragon_sigmoidf(ak_d[j + t * n_kv]) * pmask;
            const float a_v = dragon_sigmoidf(av_d[j + t * n_kv]) * pmask;
            const float b_k = a_k * -1.0f + 1.0f;  // same scale_bias form
            const float b_v = a_v * -1.0f + 1.0f;
            const float * Kt = K_d + j * hd + t * y_s2;
            const float * Vt = V_d + j * hd + t * y_s2;
            const float * Kp = (tt == 0) ? pv_d + seq * pv_s1        + j * hd
                                         : K_d  + j * hd + (t - 1) * y_s2;
            const float * Vp = (tt == 0) ? pv_d + seq * pv_s1 + n_kp + j * hd
                                         : V_d  + j * hd + (t - 1) * y_s2;
            float * Ko = y_d + j * hd + t * y_s2;
            float * Vo = y_d + j * hd + t * y_s2 + y_s3;
            for (int64_t d = 0; d < hd; ++d) {
                Ko[d] = Kp[d] * a_k + Kt[d] * b_k;
                Vo[d] = Vp[d] * a_v + Vt[d] * b_v;
            }
        }
        // pre-shift writeback: last token of each seq → cache column
        if (tt == n_seq_t - 1) {
            float * col = cache_d + seq * cache_s1;
            std::memcpy(col,        K_d + t * y_s2, (size_t) n_kp * sizeof(float));
            std::memcpy(col + n_kp, V_d + t * y_s2, (size_t) n_kp * sizeof(float));
        }
    }
}



static void dragon_mamba3_mimo_kernel(ggml_tensor * dst, int ith, int nth, void * userdata) {
    // Raw-input kernel (CuteDSL-step style): consumes the two projection GEMM
    // outputs directly and folds B/C RMS-norm (+weight), per-head bias, x/z
    // MIMO scaling and the softplus/exp/sigmoid discretisation into the
    // per-(h, t) loop — no packing/cont/norm glue nodes in the graph.
    const ggml_tensor * pdyn  = dst->src[0]; // (2·R·D_qk + num_angles, L)  [B | C | ang]
    const ggml_tensor * pstat = dst->src[1]; // ((2·D_v + 3)·H, L)  per head [z | x | dt | A | trap]
    // src[2] state_in is OPTIONAL. (n_embd_s, n_seqs) packed as
    // ssm_state / K_state / V_state / angle_state. NULL → zero-init.
    const ggml_tensor * state_in = dst->src[2];
    // src[3] state_dst is OPTIONAL: a view of the recurrent cache the kernel
    // writes the new packed state into directly (column-contiguous).
    ggml_tensor * state_dst = dst->src[3];

    // Per-layer constants come pre-packed (once per process) via userdata —
    // see dragon_get_packed_weights.
    const auto * ud = (const dragon_m_kernel_userdata *) userdata;
    GGML_ASSERT(ud != nullptr && ud->w != nullptr);
    const int64_t n_seqs_kernel = ud->n_seqs;
    const float   rms_eps       = ud->rms_eps;

    const int64_t D_qk = ud->D_qk;
    const int64_t R    = ud->R;
    const int64_t H    = ud->H;
    const int64_t D_v  = ud->D_v;
    const int64_t L    = pdyn->ne[1];

    const int64_t num_angles = pdyn->ne[0] - 2 * R * D_qk; // num_rope_angles (= 32 in Dragon 7A1B)
    const int64_t per_head   = 2 * D_v + 3;
    GGML_ASSERT(pstat->ne[0] == per_head * H);
    GGML_ASSERT(pstat->ne[1] == L);

    // Activations are fp32; weights come from the f32 packed cache.
    const float * pdyn_d  = (const float *) pdyn->data;
    const float * pstat_d = (const float *) pstat->data;
    const float * bias_d  = ud->w->bias;
    const float * mxz_d   = ud->w->mxz;
    const float * norms_d = ud->w->norms;
    const float * misc_d  = ud->w->misc;
    const float * mimo_d  = ud->w->mimo_o;
    float       * y_d     = (float       *) dst->data;
    const float * state_in_d = state_in ? (const float *) state_in->data : nullptr;

    const int64_t quarter    = D_qk / 4;           // rotary_dim_divisor = 4 → pair (i, i+D_qk/2)
    const bool    do_rotary  = (num_angles == quarter) && (D_qk >= 4);

    // Strides in elements (row-major; ne[0] innermost).
    const int64_t pdyn_s1  = pdyn->ne[0];                      // t-stride of proj_dyn
    const int64_t pstat_s1 = pstat->ne[0];                     // t-stride of proj_static
    const int64_t kq_s2    = D_qk * 2 * R;                     // bias h-stride; bc_norm size
    const int64_t mxz_s2   = D_v * 2 * R;                      // mxz: (D_v, 2R, H) h-stride
    const int64_t m_s1 = D_v, m_s2 = D_v * R;                  // mimo_o: (D_v, R, H)
    const int64_t y_s1 = D_v, y_s2 = D_v * H;                  // y:     (D_v, H, L)

    // softplus(x) = max(x, 0) + log(1 + exp(-|x|)) — numerically stable form
    auto sp = [](float x) {
        const float pos = x > 0.0f ? x : 0.0f;
        return pos + logf(expf(-fabsf(x)) + 1.0f);
    };

    // State-cache storage codec (see dragon_state_mode()). Working precision
    // in-kernel remains f32. vLLM runs the state in bf16 as well, so the
    // default also matches the reference serving stack.
    // section_base points at the start of a section in the packed cache row;
    // elt_off is the LOGICAL element offset within that section. `is_ssm`
    // marks the big S-matrix section: kept f32 in BF16 mode (validated
    // config), packed like the rest in the Q8 experiment.
    const dragon_state_prec st_mode = dragon_state_mode();
    auto load_kv = [&](float * dstp, const float * section_base, int64_t elt_off, int64_t n, bool is_ssm = false) {
        const dragon_state_prec m = (is_ssm && st_mode == DRAGON_STATE_BF16) ? DRAGON_STATE_F32 : st_mode;
        switch (m) {
        case DRAGON_STATE_F32:
            std::memcpy(dstp, section_base + elt_off, (size_t) n * sizeof(float));
            return;
        case DRAGON_STATE_BF16: {
            const uint16_t * s16 = (const uint16_t *) section_base + elt_off;
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t) s16[i] << 16;
                std::memcpy(&dstp[i], &u, 4);
            }
            return;
        }
        case DRAGON_STATE_Q8: {
            GGML_ASSERT(elt_off % 32 == 0 && n % 32 == 0);
            for (int64_t b = 0; b < n / 32; ++b) {
                const char  * slot  = (const char *) (section_base + elt_off + b * 32);
                float         scale;
                std::memcpy(&scale, slot, 4);
                const int8_t * qi = (const int8_t *) (slot + 4);
                for (int64_t i = 0; i < 32; ++i) {
                    dstp[b * 32 + i] = scale * (float) qi[i];
                }
            }
            return;
        }
        }
    };
    auto store_kv = [&](float * section_base, int64_t elt_off, const float * srcp, int64_t n, bool is_ssm = false) {
        const dragon_state_prec m = (is_ssm && st_mode == DRAGON_STATE_BF16) ? DRAGON_STATE_F32 : st_mode;
        switch (m) {
        case DRAGON_STATE_F32:
            std::memcpy(section_base + elt_off, srcp, (size_t) n * sizeof(float));
            return;
        case DRAGON_STATE_BF16: {
            uint16_t * d16 = (uint16_t *) section_base + elt_off;
            for (int64_t i = 0; i < n; ++i) {
                uint32_t u;
                std::memcpy(&u, &srcp[i], 4);
                u += 0x7fff + ((u >> 16) & 1);  // round-to-nearest-even
                d16[i] = (uint16_t) (u >> 16);
            }
            return;
        }
        case DRAGON_STATE_Q8: {
            GGML_ASSERT(elt_off % 32 == 0 && n % 32 == 0);
            for (int64_t b = 0; b < n / 32; ++b) {
                const float * s = srcp + b * 32;
                float amax = 0.0f;
                for (int64_t i = 0; i < 32; ++i) {
                    amax = std::max(amax, fabsf(s[i]));
                }
                const float scale = amax / 127.0f;
                const float inv   = scale != 0.0f ? 1.0f / scale : 0.0f;
                char * slot = (char *) (section_base + elt_off + b * 32);
                std::memcpy(slot, &scale, 4);
                int8_t * qi = (int8_t *) (slot + 4);
                for (int64_t i = 0; i < 32; ++i) {
                    qi[i] = (int8_t) lrintf(s[i] * inv);
                }
            }
            return;
        }
        }
    };

    // Parallelize over (head × state-column-slice) units. With H=48 heads and
    // nth=32 threads, plain head-parallelism gives 24 threads 2 heads and 8
    // threads none (wall = 2·L head-scans vs the ideal 1.5·L). Splitting each
    // head's D_v state rows into S slices yields H·S balanced units with NO
    // synchronisation or recompute: every state element (h, p, d) still sees
    // exactly the serial op sequence, so results are bit-identical. Only the
    // small per-(h, t) prep (k/q bias add, rotary, angles) is duplicated per
    // slice. (An L-chunked scan à la the tilelang mamba3 kernel was evaluated
    // but its cross-chunk carry recompute costs more than the imbalance it
    // removes at CPU-scale thread counts.)
    // Opt-in (DRAGON_M_COLSPLIT=1): measured neutral-to-slightly-negative at
    // nth=32 on EPYC 9334 — the duplicated per-(h, t) prep eats the balance
    // gain — but kept for experimentation at other H/nth ratios.
    int64_t S = 1;
    if (const char * cs_env = std::getenv("DRAGON_M_COLSPLIT"); cs_env && cs_env[0] && cs_env[0] != '0') {
        auto imbalance = [&](int64_t s) {
            const double u = (double) (H * s);
            return std::ceil(u / nth) * nth / u;
        };
        for (int64_t cand : {2, 4}) {
            if (D_v % cand == 0 && imbalance(cand) + 1e-9 < imbalance(S)) {
                S = cand;
            }
        }
    }
    const int64_t Dv_sl   = D_v / S;
    const int64_t n_units = H * S;          // unit u = h·S + s → rows [s·Dv_sl, (s+1)·Dv_sl)
    const int64_t u_per_thread = (n_units + nth - 1) / nth;
    const int64_t u_begin = std::min<int64_t>((int64_t) ith * u_per_thread, n_units);
    const int64_t u_end   = std::min<int64_t>(u_begin + u_per_thread, n_units);
    const int64_t my_U    = u_end - u_begin;
    if (my_U <= 0) {
        return;
    }

    // Local SSM state for this thread's units: (my_U, Dv_sl, D_qk) fp32.
    std::vector<float> ssm_state((size_t) my_U * Dv_sl * D_qk, 0.0f);
    // Per-unit previous-step K and V (for trapezoid β term). K is duplicated
    // across the S slices of a head (identical values); V is row-sliced.
    std::vector<float> K_state((size_t) my_U * R * D_qk, 0.0f);
    std::vector<float> V_state((size_t) my_U * R * Dv_sl, 0.0f);
    // Cumulative rotary angle per unit (duplicated across slices of a head).
    std::vector<float> angle_state((size_t) my_U * num_angles, 0.0f);

    // Offsets into the packed (n_embd_s,) state blob — must match the layout
    // computed by llama_hparams::n_embd_s() for Dragon.
    const int64_t off_K_glob   = H * D_v * D_qk;
    const int64_t off_V_glob   = off_K_glob + H * R * D_qk;
    const int64_t off_ang_glob = off_V_glob + H * R * D_v;
    const int64_t n_state_per_seq = off_ang_glob + H * num_angles;

    // Per-step rotated q, k scratch: (D_qk, R) × token-group (TB_MAX = 4).
    std::vector<float> q_rot((size_t) 4 * D_qk * R);
    std::vector<float> k_rot((size_t) 4 * D_qk * R);
    // Per-(unit, t) materialized v/z rows: (Dv_sl, R) × token-group.
    std::vector<float> v_loc((size_t) 4 * Dv_sl * R);
    std::vector<float> z_loc((size_t) 4 * Dv_sl * R);

    // L is laid out as n_seqs blocks of n_seq_tokens consecutive tokens.
    const int64_t n_seqs       = std::max<int64_t>(1, n_seqs_kernel);
    const int64_t n_seq_tokens = L / n_seqs;
    GGML_ASSERT(n_seq_tokens * n_seqs == L && "L must be divisible by n_seqs");
    const int64_t n_y          = D_v * H * L;
    float * state_out = nullptr;
    if (state_dst != nullptr) {
        // direct write into the recurrent cache; columns must be contiguous so
        // the seq * n_state_per_seq addressing below holds
        GGML_ASSERT(state_dst->nb[1] == (size_t) n_state_per_seq * sizeof(float));
        state_out = (float *) state_dst->data;
    } else if (state_in_d) {
        state_out = y_d + n_y;  // legacy: packed state appended to y in dst
    }

    // Zero output rows owned by this thread (we accumulate y over R).
    for (int64_t u = u_begin; u < u_end; ++u) {
        const int64_t h  = u / S;
        const int64_t p0 = (u % S) * Dv_sl;
        for (int64_t t = 0; t < L; ++t) {
            std::memset(y_d + h * y_s1 + p0 + t * y_s2, 0,
                        (size_t) Dv_sl * sizeof(float));
        }
    }

    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        // Reset / seed state for this sequence.
        if (state_in_d) {
            const float * state_in_seq = state_in_d + seq * n_state_per_seq;
            for (int64_t u = u_begin; u < u_end; ++u) {
                const int64_t ul = u - u_begin;
                const int64_t h  = u / S;
                const int64_t p0 = (u % S) * Dv_sl;
                load_kv(ssm_state.data() + ul * Dv_sl * D_qk,
                        state_in_seq, h * D_v * D_qk + p0 * D_qk,
                        Dv_sl * D_qk, /*is_ssm=*/true);
                load_kv(K_state.data() + ul * R * D_qk,
                        state_in_seq + off_K_glob, h * R * D_qk,
                        R * D_qk);
                for (int64_t r = 0; r < R; ++r) {
                    load_kv(V_state.data() + (ul * R + r) * Dv_sl,
                            state_in_seq + off_V_glob, h * R * D_v + r * D_v + p0,
                            Dv_sl);
                }
                std::memcpy(angle_state.data() + ul * num_angles,
                            state_in_seq + off_ang_glob + h * num_angles,
                            (size_t) num_angles * sizeof(float));
            }
        } else {
            std::fill(ssm_state.begin(),  ssm_state.end(),  0.0f);
            std::fill(K_state.begin(),    K_state.end(),    0.0f);
            std::fill(V_state.begin(),    V_state.end(),    0.0f);
            std::fill(angle_state.begin(),angle_state.end(),0.0f);
        }

    // Per-token scratch reused across heads in this thread's slice. Tokens are
    // processed in groups of up to TB_MAX so the state sweep (the dominant
    // memory traffic) runs once per group — see dragon_simd_fused_step_tb.
    constexpr int64_t TB_MAX = 4;
    std::vector<float> tanh_ang_pi((size_t) TB_MAX * num_angles);
    std::vector<float> cs_buf((size_t) num_angles);
    std::vector<float> ss_buf((size_t) num_angles);
    std::vector<float> bc_norm((size_t) TB_MAX * 2 * R * D_qk); // per-t RMS-normed [B | C]
    const int64_t seq_t0  = seq * n_seq_tokens;
    const int64_t seq_t1  = (seq + 1) * n_seq_tokens;
    for (int64_t tg = seq_t0; tg < seq_t1; tg += TB_MAX) {
        const int64_t TB = std::min<int64_t>(TB_MAX, seq_t1 - tg);
        // --- head-independent per-token prep for the group ---
        for (int64_t tt = 0; tt < TB; ++tt) {
            const int64_t t = tg + tt;
            // tanh(ang[i, t]) · π — depends only on t, not h.
            const float * ang_t = pdyn_d + t * pdyn_s1 + 2 * R * D_qk;
            float * tan_tt = tanh_ang_pi.data() + tt * num_angles;
            for (int64_t i = 0; i < num_angles; ++i) {
                tan_tt[i] = tanhf(ang_t[i]) * (float) M_PI;
            }
            // RMS-norm B and C rows of proj_dyn (per (r, t) group of D_qk), then
            // apply the per-channel norm weights — same math/accumulator as
            // ggml_rms_norm (double sum) followed by the broadcast ggml_mul.
            const float * bc_t = pdyn_d + t * pdyn_s1;     // (D_qk, 2R): [B | C]
            float * bcn_tt = bc_norm.data() + tt * 2 * R * D_qk;
            for (int64_t g = 0; g < 2 * R; ++g) {
                const float * src = bc_t + g * D_qk;
                const float * w   = norms_d + (g < R ? 0 : D_qk); // b_norm | c_norm
                float       * out = bcn_tt + g * D_qk;
                double sum = 0.0;
                for (int64_t d = 0; d < D_qk; ++d) {
                    sum += (double) src[d] * (double) src[d];
                }
                const float scale = 1.0f / sqrtf((float) (sum / D_qk) + rms_eps);
                for (int64_t d = 0; d < D_qk; ++d) {
                    out[d] = (src[d] * scale) * w[d];
                }
            }
        }
        for (int64_t u = u_begin; u < u_end; ++u) {
            const int64_t ul = u - u_begin;
            const int64_t h  = u / S;
            const int64_t p0 = (u % S) * Dv_sl;
            const float * bias_h = bias_d + h * kq_s2;
            const float * mx_h   = mxz_d + h * mxz_s2 + p0;          // (D_v, 2R): [mimo_x | mimo_z]
            const float * mz_h   = mx_h + R * D_v;
            const float   D_h    = misc_d[H + h];
            float * ang_st = angle_state.data() + ul * num_angles;

            float alpha_g[TB_MAX], beta_g[TB_MAX], gamma_g[TB_MAX];
            const float * kpv[TB_MAX], * qpv[TB_MAX], * vpv[TB_MAX], * zpv[TB_MAX];
            float * ypv[TB_MAX];

            for (int64_t tt = 0; tt < TB; ++tt) {
                const int64_t t = tg + tt;
                // Per-(h, t) raw scalars from proj_static: [z | x | dt | A | trap]
                const float * ph    = pstat_d + t * pstat_s1 + h * per_head;
                const float * z_t   = ph;                 // length D_v
                const float * x_t   = ph + D_v;           // length D_v
                const float dt_sp   = sp(ph[2*D_v + 0] + misc_d[h]);
                float a_neg         = -sp(ph[2*D_v + 1]);
                if (a_neg > -1e-4f) a_neg = -1e-4f;       // clamp(max = -A_floor)
                alpha_g[tt]         = expf(a_neg * dt_sp);
                const float dt_h    = dt_sp;
                const float trap_h  = dragon_sigmoidf(ph[2*D_v + 2]);
                beta_g[tt]          = (1.0f - trap_h) * dt_h * alpha_g[tt];
                gamma_g[tt]         = trap_h * dt_h;

                // --- Rotary update ---
                // angle_state[h, i] += (precomputed tanh(ang_t[i]) · π) · dt_h
                const float * tan_tt = tanh_ang_pi.data() + tt * num_angles;
                for (int64_t i = 0; i < num_angles; ++i) {
                    ang_st[i] += tan_tt[i] * dt_h;
                }

                // Materialize raw q, k for this (h, t) into the group scratch:
                // shared normed B/C row plus the per-head bias.
                float * krt = k_rot.data() + tt * R * D_qk;
                float * qrt = q_rot.data() + tt * R * D_qk;
                {
                    const float * bcn_tt = bc_norm.data() + tt * 2 * R * D_qk;
                    const int64_t half = R * D_qk;
                    for (int64_t j = 0; j < half; ++j) {
                        krt[j] = bcn_tt[j] + bias_h[j];
                    }
                    for (int64_t j = 0; j < half; ++j) {
                        qrt[j] = bcn_tt[half + j] + bias_h[half + j];
                    }
                }
                // Materialize this unit's v, z rows for (h, t).
                float * vlt = v_loc.data() + tt * R * Dv_sl;
                float * zlt = z_loc.data() + tt * R * Dv_sl;
                {
                    const float * xs = x_t + p0;                          // length Dv_sl
                    const float * zs = z_t + p0;
                    for (int64_t r = 0; r < R; ++r) {
                        float * vr = vlt + r * Dv_sl;
                        float * zr = zlt + r * Dv_sl;
                        const float * mxr = mx_h + r * D_v;
                        const float * mzr = mz_h + r * D_v;
                        for (int64_t p = 0; p < Dv_sl; ++p) {
                            vr[p] = xs[p] * mxr[p];
                            zr[p] = zs[p] * mzr[p];
                        }
                    }
                }
                if (do_rotary) {
                    // cos / sin per angle once for this (h, t); reused across R.
                    for (int64_t i = 0; i < num_angles; ++i) {
                        cs_buf[i] = cosf(ang_st[i]);
                        ss_buf[i] = sinf(ang_st[i]);
                    }
                    const int64_t i2_off = 2 * quarter;
                    for (int64_t r = 0; r < R; ++r) {
                        float * qr = qrt + r * D_qk;
                        float * kr = krt + r * D_qk;
                        int64_t i = 0;
#if defined(__AVX512F__)
                        for (; i + 16 <= quarter; i += 16) {
                            __m512 c = _mm512_loadu_ps(cs_buf.data() + i);
                            __m512 s = _mm512_loadu_ps(ss_buf.data() + i);
                            __m512 q0 = _mm512_loadu_ps(qr + i);
                            __m512 q2 = _mm512_loadu_ps(qr + i + i2_off);
                            __m512 k0 = _mm512_loadu_ps(kr + i);
                            __m512 k2 = _mm512_loadu_ps(kr + i + i2_off);
                            // q0_new = q0*c - q2*s ; q2_new = q0*s + q2*c
                            _mm512_storeu_ps(qr + i,           _mm512_fnmadd_ps(q2, s, _mm512_mul_ps(q0, c)));
                            _mm512_storeu_ps(qr + i + i2_off,  _mm512_fmadd_ps (q0, s, _mm512_mul_ps(q2, c)));
                            _mm512_storeu_ps(kr + i,           _mm512_fnmadd_ps(k2, s, _mm512_mul_ps(k0, c)));
                            _mm512_storeu_ps(kr + i + i2_off,  _mm512_fmadd_ps (k0, s, _mm512_mul_ps(k2, c)));
                        }
#endif
                        for (; i < quarter; ++i) {
                            const float c = cs_buf[i];
                            const float s = ss_buf[i];
                            const int64_t i0 = i;
                            const int64_t i2 = i + i2_off;
                            const float q0 = qr[i0], q2 = qr[i2];
                            qr[i0] = q0 * c - q2 * s;
                            qr[i2] = q0 * s + q2 * c;
                            const float kk0 = kr[i0], kk2 = kr[i2];
                            kr[i0] = kk0 * c - kk2 * s;
                            kr[i2] = kk0 * s + kk2 * c;
                        }
                    }
                }
                kpv[tt] = krt;
                qpv[tt] = qrt;
                vpv[tt] = vlt;
                zpv[tt] = zlt;
                ypv[tt] = y_d + h * y_s1 + p0 + t * y_s2;
            }

            // --- Fused group: TB tokens' updates in one pass over the state ---
            float * K_st = K_state.data() + ul * R * D_qk;
            float * V_st = V_state.data() + ul * R * Dv_sl;
            float * st   = ssm_state.data() + ul * Dv_sl * D_qk;
            const float * mo = mimo_d + h * m_s2 + p0;
            dragon_simd_fused_step_group(st, K_st, V_st, kpv, qpv, vpv, zpv,
                                         mo, m_s1, ypv, alpha_g, beta_g, gamma_g, D_h,
                                         R, TB, D_qk, Dv_sl);

            // Save K_state / V_state = last token of the group.
            std::memcpy(K_st, kpv[TB - 1], (size_t) R * D_qk * sizeof(float));
            std::memcpy(V_st, vpv[TB - 1], (size_t) R * Dv_sl * sizeof(float));
        }
    }

        // Per-seq state writeback. Each unit writes only its own row slice;
        // the head-wide K/angle sections (duplicated across slices) are
        // written by slice 0 only.
        if (state_out) {
            float * state_out_seq = state_out + seq * n_state_per_seq;
            for (int64_t u = u_begin; u < u_end; ++u) {
                const int64_t ul = u - u_begin;
                const int64_t h  = u / S;
                const int64_t s  = u % S;
                const int64_t p0 = s * Dv_sl;
                store_kv(state_out_seq, h * D_v * D_qk + p0 * D_qk,
                         ssm_state.data() + ul * Dv_sl * D_qk,
                         Dv_sl * D_qk, /*is_ssm=*/true);
                for (int64_t r = 0; r < R; ++r) {
                    store_kv(state_out_seq + off_V_glob, h * R * D_v + r * D_v + p0,
                             V_state.data() + (ul * R + r) * Dv_sl,
                             Dv_sl);
                }
                if (s == 0) {
                    store_kv(state_out_seq + off_K_glob, h * R * D_qk,
                             K_state.data() + ul * R * D_qk,
                             R * D_qk);
                    std::memcpy(state_out_seq + off_ang_glob + h * num_angles,
                                angle_state.data() + ul * num_angles,
                                (size_t) num_angles * sizeof(float));
                }
            }
        }
    }  // end per-seq loop
}

// Primitive re-expression of the Mamba3-MIMO recurrence. Uses only standard
// ggml ops (mul, add, cumsum, exp, tri, mul_mat), so it runs on any backend
// — gated on the DRAGON_M_PRIM env var; default uses the CPU custom op which
// is faster on CPU but only runs on CPU.
//
// Algorithm follows mamba3_MIMO_chunk_ref from the production tilelang tests
// (rotary on Q/K + factor-scaled K + decay matrix + diagonal correction).
// Non-chunked: materialises a (L, L, H) decay matrix and a (D_qk, D_v, H, L)
// kv tensor. Memory at L=128 / H=48 / D_qk=128 / D_v=64 is ~200 MB for kv;
// fine for short prefill, needs chunking for long contexts.
//
// Output: y (D_v, H, L) fp32, math equivalent to dragon_mamba3_mimo_kernel.
// Validated: 125/127 argmax match vs CPU custom op on a 127-token prompt, cos
// ≈ 0.9997 (limit set by fp32 op-order rounding).
static ggml_tensor * build_dragon_m_recurrence_prim(
        ggml_context * ctx,
        ggml_tensor * q_biased,    // (D_qk, R, H, L)
        ggml_tensor * k_biased,    // (D_qk, R, H, L)
        ggml_tensor * v_proj,      // (D_v,  R, H, L)
        ggml_tensor * z_proj,      // (D_v,  R, H, L)
        ggml_tensor * alpha,       // (H, L)  = exp(ADT)
        ggml_tensor * dt,          // (H, L)  post-softplus
        ggml_tensor * trap_post,   // (H, L)  post-sigmoid
        ggml_tensor * ang_raw,     // (num_angles, L)  raw angle projection
        ggml_tensor * D_skip,      // (H,)
        ggml_tensor * mimo_o,      // (D_v, R, H)
        int64_t D_qk, int64_t D_v, int64_t R, int64_t H, int64_t L,
        int64_t num_angles,
        int il,
        // Optional state in/out. If state_in_packed is non-NULL, it's a flat
        // (n_embd_s, 1) tensor containing prior batch's ssm_state/K_state/V_state/
        // angle_state; state_out_packed is filled with the new packed state
        // (state[L-1], k_rot[L-1], v[L-1], cum_angle[L-1]). Pass NULL to disable.
        ggml_tensor * state_in_packed = nullptr,
        ggml_tensor ** state_out_packed = nullptr) {
    (void) il;  // currently unused (reserved for future per-layer dump hooks)
    const int64_t quarter = D_qk / 4;
    GGML_ASSERT(num_angles == quarter && "halved rotary expects num_angles = D_qk/4");

    // Packed-state offsets (must match llama_hparams::n_embd_s() for Dragon).
    const int64_t off_K_glob   = H * D_v * D_qk;
    const int64_t off_V_glob   = off_K_glob + H * R * D_qk;
    const int64_t off_ang_glob = off_V_glob + H * R * D_v;

    // Unpack state_in into the four components if provided.
    ggml_tensor * state_in_4d  = nullptr;  // (D_qk, D_v, H, 1)
    ggml_tensor * K_in_4d      = nullptr;  // (D_qk, R,   H, 1)
    ggml_tensor * V_in_4d      = nullptr;  // (D_v,  R,   H, 1)
    ggml_tensor * angle_in_2d  = nullptr;  // (na, H)
    if (state_in_packed != nullptr) {
        const size_t fs = ggml_element_size(state_in_packed);
        ggml_tensor * sp = ggml_reshape_1d(ctx, state_in_packed,
                                            off_ang_glob + H * num_angles);
        auto slice = [&](int64_t off, int64_t n) {
            return ggml_view_1d(ctx, sp, n, off * fs);
        };
        state_in_4d = ggml_reshape_4d(ctx, slice(0,                H*D_v*D_qk),    D_qk, D_v, H, 1);
        K_in_4d     = ggml_reshape_4d(ctx, slice(off_K_glob,       H*R  *D_qk),    D_qk, R,   H, 1);
        V_in_4d     = ggml_reshape_4d(ctx, slice(off_V_glob,       H*R  *D_v ),    D_v,  R,   H, 1);
        angle_in_2d = ggml_reshape_2d(ctx, slice(off_ang_glob,     H*num_angles),  num_angles, H);
    }

    // 1) Cumulative rotary angle per (i, h, t).
    //    contrib[i, h, t] = π · tanh(ang_raw[i, t]) · dt[h, t]
    //    cum_angle = cumsum over t.
    //    (The custom kernel applies tanhf internally; the input is the raw
    //    projection, not tanh'd despite some older comments suggesting otherwise.)
    ggml_tensor * ang_tanh = ggml_tanh(ctx, ang_raw);                                     // (na, L)
    ggml_tensor * ang_pi   = ggml_scale(ctx, ang_tanh, (float) M_PI);                     // (na, L)
    ggml_tensor * ang_3d  = ggml_reshape_3d(ctx, ang_pi, num_angles, 1, L);                // (na, 1, L)
    ggml_tensor * ang_b   = ggml_repeat_4d(ctx, ang_3d, num_angles, H, L, 1);              // (na, H, L)
    ggml_tensor * dt_3d   = ggml_reshape_3d(ctx, dt, 1, H, L);                              // (1, H, L)
    ggml_tensor * contrib = ggml_mul(ctx, ang_b, dt_3d);                                    // (na, H, L)
    // cumsum runs along ne[0]; permute so L is innermost.
    // ggml_permute(t, p0,p1,p2,p3) uses "perm[k] = destination of old axis k"
    // (unlike numpy.transpose). For non-self-inverse permutations the two
    // conventions give opposite results.
    ggml_tensor * contrib_lt = ggml_cont(ctx, ggml_permute(ctx, contrib, 1, 2, 0, 3));      // (L, na, H)
    ggml_tensor * cum_lt     = ggml_cumsum(ctx, contrib_lt);                                // (L, na, H)
    // If state_in is provided, the cumulative angle starts from angle_in (the
    // running angle saved at the end of the prior batch) rather than zero.
    if (angle_in_2d != nullptr) {
        // angle_in (na, H) → (1, na, H), broadcasts over L into cum_lt (L, na, H).
        ggml_tensor * angle_in_1lh = ggml_reshape_3d(ctx, angle_in_2d, 1, num_angles, H);
        ggml_tensor * angle_in_full = ggml_repeat_4d(ctx, angle_in_1lh, L, num_angles, H, 1);
        cum_lt = ggml_add(ctx, cum_lt, angle_in_full);
    }
    ggml_tensor * cos_lt = ggml_cos(ctx, cum_lt);                                            // (L, na, H)
    ggml_tensor * sin_lt = ggml_sin(ctx, cum_lt);
    // Back to (na, H, L) for the rotary application.
    ggml_tensor * cos_a = ggml_cont(ctx, ggml_permute(ctx, cos_lt, 2, 0, 1, 3));            // (na, H, L)
    ggml_tensor * sin_a = ggml_cont(ctx, ggml_permute(ctx, sin_lt, 2, 0, 1, 3));

    // 2) Halved rotary on q, k. Group 0 (d∈[0, quarter)) pairs with group 2
    //    (d∈[D_qk/2, D_qk/2+quarter)); groups 1 and 3 pass through.
    auto apply_rotary = [&](ggml_tensor * x) {
        const size_t es = ggml_element_size(x);
        auto group = [&](int gi) {
            return ggml_cont(ctx,
                ggml_view_4d(ctx, x, quarter, R, H, L,
                             x->nb[1], x->nb[2], x->nb[3],
                             gi * quarter * es));
        };
        ggml_tensor * g0 = group(0);
        ggml_tensor * g1 = group(1);
        ggml_tensor * g2 = group(2);
        ggml_tensor * g3 = group(3);
        // c, s broadcast over R: reshape (quarter, H, L) → (quarter, 1, H, L).
        ggml_tensor * c4 = ggml_reshape_4d(ctx, cos_a, quarter, 1, H, L);
        ggml_tensor * s4 = ggml_reshape_4d(ctx, sin_a, quarter, 1, H, L);
        ggml_tensor * g0_new = ggml_sub(ctx, ggml_mul(ctx, g0, c4), ggml_mul(ctx, g2, s4));
        ggml_tensor * g2_new = ggml_add(ctx, ggml_mul(ctx, g0, s4), ggml_mul(ctx, g2, c4));
        ggml_tensor * lo = ggml_concat(ctx, g0_new, g1, /*dim=*/ 0);    // (2·quarter, R, H, L)
        ggml_tensor * hi = ggml_concat(ctx, g2_new, g3, /*dim=*/ 0);    // (2·quarter, R, H, L)
        return ggml_concat(ctx, lo, hi, /*dim=*/ 0);                    // (D_qk, R, H, L)
    };
    ggml_tensor * q_rot = apply_rotary(q_biased);
    ggml_tensor * k_rot = apply_rotary(k_biased);

    // 3) Cumulative log α along the L axis. log(α) = ADT but we only have α here.
    ggml_tensor * log_a   = ggml_log(ctx, alpha);                                            // (H, L)
    ggml_tensor * log_a_T = ggml_cont(ctx, ggml_transpose(ctx, log_a));                      // (L, H)
    ggml_tensor * cum_a   = ggml_cumsum(ctx, log_a_T);                                       // (L, H)

    // Outer diff: diff[u, t, h] = cum_a[t, h] - cum_a[u, h].
    // ggml_sub broadcasts the second operand into the first (one-way), so the
    // (1, L, H) side must be materialised to (L, L, H) first.
    ggml_tensor * ca_t = ggml_reshape_3d(ctx, cum_a, 1, L, H);                                // (1, L, H)
    ggml_tensor * ca_u = ggml_reshape_3d(ctx, cum_a, L, 1, H);                                // (L, 1, H)
    ggml_tensor * ca_t_full = ggml_repeat_4d(ctx, ca_t, L, L, H, 1);                          // (L, L, H)
    ggml_tensor * diff = ggml_sub(ctx, ca_t_full, ca_u);                                      // (L, L, H)
    ggml_tensor * W    = ggml_exp(ctx, diff);                                                  // (L, L, H)
    // Keep only u ≤ t (W[u, t, h] for u > t becomes 0).
    W = ggml_tri(ctx, W, GGML_TRI_TYPE_LOWER_DIAG);

    // 4) factor[h, u] = γ[h, u] + γ_shifted[h, u]
    //    γ[h, t]         = trap_post[h, t] · dt[h, t]
    //    γ_shifted[h, t] = (1 − trap_post[h, t+1]) · dt[h, t+1], 0 at t = L−1
    ggml_tensor * gamma          = ggml_mul(ctx, trap_post, dt);                              // (H, L)
    ggml_tensor * one_minus_trap = ggml_scale_bias(ctx, trap_post, -1.0f, 1.0f);              // (H, L)
    ggml_tensor * coeff_prev     = ggml_mul(ctx, one_minus_trap, dt);                         // (H, L)
    ggml_tensor * gamma_shifted;
    if (L > 1) {
        // Shift coeff_prev one column left along ne[1]; pad zero at the end.
        ggml_tensor * tail = ggml_view_2d(ctx, coeff_prev, H, L - 1,
                                          coeff_prev->nb[1], coeff_prev->nb[1]);
        gamma_shifted = ggml_pad(ctx, ggml_cont(ctx, tail), 0, 1, 0, 0);                      // (H, L)
    } else {
        gamma_shifted = ggml_scale(ctx, coeff_prev, 0.0f);
    }
    ggml_tensor * factor = ggml_add(ctx, gamma, gamma_shifted);                                // (H, L)

    // 5) Wf[u, t, h] = W[u, t, h] · factor[h, u].
    //    factor (H, L) → transpose to (L, H) → reshape (L, 1, H) for broadcast over t.
    ggml_tensor * factor_LH    = ggml_cont(ctx, ggml_transpose(ctx, factor));                  // (L, H)
    ggml_tensor * factor_bcast = ggml_reshape_3d(ctx, factor_LH, L, 1, H);                     // (L, 1, H)
    ggml_tensor * Wf = ggml_mul(ctx, W, factor_bcast);                                          // (L, L, H)

    // 6) kv[d, p, h, u] = sum_r k_rot[d, r, h, u] · v[p, r, h, u].
    //    Permute k, v so R is at ne[0] for the matmul reduction.
    ggml_tensor * k_for_mm = ggml_cont(ctx, ggml_permute(ctx, k_rot,  1, 0, 2, 3));            // (R, D_qk, H, L)
    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_proj, 1, 0, 2, 3));            // (R, D_v,  H, L)
    ggml_tensor * kv = ggml_mul_mat(ctx, k_for_mm, v_for_mm);                                  // (D_qk, D_v, H, L)

    // 7) state[d, p, t, h] = sum_u Wf[u, t, h] · kv[d, p, h, u].
    //    Permute kv (D_qk, D_v, H, L=u) → (L=u, D_qk, D_v, H); reshape to (L, D_qk·D_v, H).
    ggml_tensor * kv_perm = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 2, 3, 0));                 // (L, D_qk, D_v, H)
    ggml_tensor * kv_flat = ggml_reshape_3d(ctx, kv_perm, L, D_qk * D_v, H);                   // (L, D_qk·D_v, H)
    ggml_tensor * state_flat = ggml_mul_mat(ctx, kv_flat, Wf);                                  // (D_qk·D_v, L=t, H)
    ggml_tensor * state = ggml_reshape_4d(ctx, state_flat, D_qk, D_v, L, H);                   // (D_qk, D_v, L, H)

    // 7b) Carry from prior batch: state[t] += exp(cum_a[t,h]) · state_in_eff[d,p,h]
    //     where state_in_eff = state_in + γ_shifted[-1]·prev_kv_in.
    //     γ_shifted[-1] = (1-trap[0])·dt[0]; prev_kv_in = K_in @ V_in.
    //     decay[t,h] = exp(cum_a[t,h]) is just exp(cum_a) reshaped.
    if (state_in_4d != nullptr) {
        // prev_kv_in (D_qk, D_v, H, 1) = sum_r K_in[d,r,h] · V_in[p,r,h]
        ggml_tensor * K_R0_in = ggml_cont(ctx, ggml_permute(ctx, K_in_4d, 1, 0, 2, 3));   // (R, D_qk, H, 1)
        ggml_tensor * V_R0_in = ggml_cont(ctx, ggml_permute(ctx, V_in_4d, 1, 0, 2, 3));   // (R, D_v,  H, 1)
        ggml_tensor * prev_kv_in = ggml_mul_mat(ctx, K_R0_in, V_R0_in);                   // (D_qk, D_v, H, 1)

        // γ_shifted[-1] per head = (1-trap_post[0])·dt[0]. Take from first column.
        const size_t fs = ggml_element_size(coeff_prev);
        ggml_tensor * gshift_neg1 = ggml_view_1d(ctx, coeff_prev, H, 0 * fs);             // (H,) at t=0 column
        ggml_tensor * gshift_neg1_4d = ggml_reshape_4d(ctx,
            ggml_cont(ctx, gshift_neg1), 1, 1, H, 1);
        ggml_tensor * pk_scaled = ggml_mul(ctx, prev_kv_in, gshift_neg1_4d);              // (D_qk, D_v, H, 1)
        ggml_tensor * state_in_eff = ggml_add(ctx, state_in_4d, pk_scaled);               // (D_qk, D_v, H, 1)

        // decay (L, H) = exp(cum_a). Permute to (1, 1, L, H) for broadcast.
        ggml_tensor * decay_LH = ggml_exp(ctx, cum_a);                                    // (L, H)
        ggml_tensor * decay_4d = ggml_reshape_4d(ctx, decay_LH, 1, 1, L, H);               // (1, 1, L, H)

        // state_in_eff is (D_qk, D_v, 1, H). Repeat to (D_qk, D_v, L, H) so mul broadcasts.
        ggml_tensor * sie_perm = ggml_cont(ctx, ggml_permute(ctx, state_in_eff, 0, 1, 3, 2));  // (D_qk, D_v, 1, H)
        ggml_tensor * sie_rep  = ggml_repeat_4d(ctx, sie_perm, D_qk, D_v, L, H);              // (D_qk, D_v, L, H)
        ggml_tensor * carry    = ggml_mul(ctx, sie_rep, decay_4d);                            // (D_qk, D_v, L, H)
        state = ggml_add(ctx, state, carry);
    }

    // 8) qstate[p, r, t, h] = sum_d state[d, p, t, h] · q_rot[d, r, h, t].
    //    Permute q_rot (D_qk, R, H, L) → (D_qk, R, L, H) so its batch dims line up with state.
    ggml_tensor * q_for_mm = ggml_cont(ctx, ggml_permute(ctx, q_rot, 0, 1, 3, 2));             // (D_qk, R, L, H)
    ggml_tensor * qstate   = ggml_mul_mat(ctx, state, q_for_mm);                                // (D_v, R, L, H)

    // 8b) Diagonal correction. For u = t the recurrence's true coefficient is γ[t],
    //     but Wf[t, t, h] = factor[t] = γ[t] + γ_shifted[t]. So qstate over-counts
    //     the diagonal by γ_shifted[t] · sum_d kv[d, p, h, t] · q_rot[d, r, h, t].
    //     Subtract it. (γ_shifted[L-1] = 0 by construction, so the last position is
    //     unaffected.)
    ggml_tensor * qk_diag    = ggml_mul_mat(ctx, kv, q_rot);                                    // (D_v, R, H, L)
    ggml_tensor * gshift_4d  = ggml_reshape_4d(ctx, gamma_shifted, 1, 1, H, L);
    ggml_tensor * over_count = ggml_mul(ctx, qk_diag, gshift_4d);                                // (D_v, R, H, L)
    ggml_tensor * over_LH    = ggml_cont(ctx, ggml_permute(ctx, over_count, 0, 1, 3, 2));        // (D_v, R, L, H)
    qstate = ggml_sub(ctx, qstate, over_LH);

    // 9) o = qstate + D[h] · v_proj. Permute v to (D_v, R, L, H); broadcast D over (D_v, R, L).
    ggml_tensor * v_LH = ggml_cont(ctx, ggml_permute(ctx, v_proj, 0, 1, 3, 2));                // (D_v, R, L, H)
    ggml_tensor * D4   = ggml_reshape_4d(ctx, D_skip, 1, 1, 1, H);
    ggml_tensor * Dv   = ggml_mul(ctx, v_LH, D4);
    ggml_tensor * o    = ggml_add(ctx, qstate, Dv);                                              // (D_v, R, L, H)

    // 10) Apply silu(z) gate.
    ggml_tensor * z_LH = ggml_cont(ctx, ggml_permute(ctx, z_proj, 0, 1, 3, 2));                // (D_v, R, L, H)
    o = ggml_mul(ctx, o, ggml_silu(ctx, z_LH));                                                 // (D_v, R, L, H)

    // 11) y[p, t, h] = sum_r o[p, r, t, h] · MIMO_O[p, r, h]. Broadcast MIMO_O over L,
    //     then sum over r via permute + sum_rows.
    ggml_tensor * mimo_4d    = ggml_reshape_4d(ctx, mimo_o, D_v, R, 1, H);
    ggml_tensor * o_weighted = ggml_mul(ctx, o, mimo_4d);                                       // (D_v, R, L, H)
    ggml_tensor * o_rsum_in  = ggml_cont(ctx, ggml_permute(ctx, o_weighted, 1, 0, 2, 3));       // (R, D_v, L, H)
    ggml_tensor * y_rsum     = ggml_sum_rows(ctx, o_rsum_in);                                   // (1, D_v, L, H)
    ggml_tensor * y_DvLH     = ggml_reshape_3d(ctx, y_rsum, D_v, L, H);                          // (D_v, L, H)

    // Pack the closing state for the next batch if requested.
    if (state_out_packed != nullptr) {
        // state[L-1]: view of state (D_qk, D_v, L, H) at the L-1 slot.
        ggml_tensor * state_last = ggml_view_4d(ctx, state, D_qk, D_v, 1, H,
                                                 state->nb[1], state->nb[2], state->nb[3],
                                                 (L - 1) * state->nb[2]);
        // K_state = k_rot at L-1 (D_qk, R, H, 1)
        ggml_tensor * k_last = ggml_view_4d(ctx, k_rot, D_qk, R, H, 1,
                                             k_rot->nb[1], k_rot->nb[2], k_rot->nb[3],
                                             (L - 1) * k_rot->nb[3]);
        // V_state = v_proj at L-1
        ggml_tensor * v_last = ggml_view_4d(ctx, v_proj, D_v, R, H, 1,
                                             v_proj->nb[1], v_proj->nb[2], v_proj->nb[3],
                                             (L - 1) * v_proj->nb[3]);
        // angle = cum_lt at L-1. cum_lt is (L, na, H); take L-1 slot → (1, na, H) → (na, H).
        ggml_tensor * ang_last = ggml_view_3d(ctx, cum_lt, 1, num_angles, H,
                                               cum_lt->nb[1], cum_lt->nb[2],
                                               (L - 1) * cum_lt->nb[0]);

        auto flat = [&](ggml_tensor * t, int64_t n) {
            return ggml_reshape_2d(ctx, ggml_cont(ctx, t), n, 1);
        };
        *state_out_packed = ggml_concat(ctx,
            ggml_concat(ctx,
                ggml_concat(ctx,
                    flat(state_last, H * D_v * D_qk),
                    flat(k_last,     H * R   * D_qk), /*dim=*/ 0),
                flat(v_last,         H * R   * D_v ), /*dim=*/ 0),
            flat(ang_last,           H * num_angles), /*dim=*/ 0);
    }

    // Match the kernel output layout: (D_v, H, L).
    return ggml_cont(ctx, ggml_permute(ctx, y_DvLH, 0, 2, 1, 3));
}


// Chunked variant of build_dragon_m_recurrence_prim. Splits L into chunks of
// size CS; computes within-chunk decay matrix + state with the closed form,
// then carries a (D_qk, D_v, H) state across chunks via the exact
// linear-recurrence update. Peak compute-buffer is O(CS·D_qk·D_v·H) instead
// of O(L·D_qk·D_v·H), letting prefill scale to slw_wsize=2400 on tight GPUs.
//
// Recommended CS ∈ [64, 256]. Below ~32, ggml's per-context tensor-object pool
// runs out (the graph adds ~40 nodes per chunk per layer).
//
// Math: with cum_local[t_local,h] = cumsum_over_chunk(log α)[t_local,h],
//   decay_carry_to_t[t_local,h] = exp(cum_local[t_local,h])
//   state_carry_update          = exp(cum_local[CS-1,h]) · state_carry_old
//                                 + state_within_chunk[:,:,CS-1,:]
// The diagonal correction (subtract γ_shifted · kv·q at u=t) is purely local
// to the chunk, so it's applied unchanged.
static ggml_tensor * build_dragon_m_recurrence_prim_chunked(
        ggml_context * ctx,
        ggml_tensor * q_biased,    // (D_qk, R, H, L)
        ggml_tensor * k_biased,    // (D_qk, R, H, L)
        ggml_tensor * v_proj,      // (D_v,  R, H, L)
        ggml_tensor * z_proj,      // (D_v,  R, H, L)
        ggml_tensor * alpha,       // (H, L)
        ggml_tensor * dt,          // (H, L)
        ggml_tensor * trap_post,   // (H, L)
        ggml_tensor * ang_raw,     // (num_angles, L)
        ggml_tensor * D_skip,      // (H,)
        ggml_tensor * mimo_o,      // (D_v, R, H)
        int64_t D_qk, int64_t D_v, int64_t R, int64_t H, int64_t L,
        int64_t num_angles,
        int64_t CS,
        int il,
        ggml_tensor * state_in_packed = nullptr,
        ggml_tensor ** state_out_packed = nullptr) {
    (void) il;
    const int64_t quarter = D_qk / 4;
    GGML_ASSERT(num_angles == quarter && "halved rotary expects num_angles = D_qk/4");
    GGML_ASSERT(CS > 0 && CS <= L);

    // Packed-state offsets (must match llama_hparams::n_embd_s() for Dragon).
    const int64_t off_K_glob   = H * D_v * D_qk;
    const int64_t off_V_glob   = off_K_glob + H * R * D_qk;
    const int64_t off_ang_glob = off_V_glob + H * R * D_v;

    // Unpack state_in if provided.
    ggml_tensor * state_in_4d  = nullptr;
    ggml_tensor * K_in_4d      = nullptr;
    ggml_tensor * V_in_4d      = nullptr;
    ggml_tensor * angle_in_2d  = nullptr;
    if (state_in_packed != nullptr) {
        const size_t fs = ggml_element_size(state_in_packed);
        ggml_tensor * sp = ggml_reshape_1d(ctx, state_in_packed,
                                            off_ang_glob + H * num_angles);
        auto slice = [&](int64_t off, int64_t n) {
            return ggml_view_1d(ctx, sp, n, off * fs);
        };
        state_in_4d = ggml_reshape_4d(ctx, slice(0,                H*D_v*D_qk),    D_qk, D_v, H, 1);
        K_in_4d     = ggml_reshape_4d(ctx, slice(off_K_glob,       H*R  *D_qk),    D_qk, R,   H, 1);
        V_in_4d     = ggml_reshape_4d(ctx, slice(off_V_glob,       H*R  *D_v ),    D_v,  R,   H, 1);
        angle_in_2d = ggml_reshape_2d(ctx, slice(off_ang_glob,     H*num_angles),  num_angles, H);
    }

    // --- Phase A: whole-L cheap stuff (same as non-chunked) ---

    ggml_tensor * ang_tanh = ggml_tanh(ctx, ang_raw);
    ggml_tensor * ang_pi   = ggml_scale(ctx, ang_tanh, (float) M_PI);
    ggml_tensor * ang_3d   = ggml_reshape_3d(ctx, ang_pi, num_angles, 1, L);
    ggml_tensor * ang_b    = ggml_repeat_4d(ctx, ang_3d, num_angles, H, L, 1);
    ggml_tensor * dt_3d    = ggml_reshape_3d(ctx, dt, 1, H, L);
    ggml_tensor * contrib  = ggml_mul(ctx, ang_b, dt_3d);
    ggml_tensor * contrib_lt = ggml_cont(ctx, ggml_permute(ctx, contrib, 1, 2, 0, 3));
    ggml_tensor * cum_lt   = ggml_cumsum(ctx, contrib_lt);
    // Shift by prior batch's angle if provided.
    if (angle_in_2d != nullptr) {
        ggml_tensor * angle_in_1lh = ggml_reshape_3d(ctx, angle_in_2d, 1, num_angles, H);
        ggml_tensor * angle_in_full = ggml_repeat_4d(ctx, angle_in_1lh, L, num_angles, H, 1);
        cum_lt = ggml_add(ctx, cum_lt, angle_in_full);
    }
    ggml_tensor * cos_lt   = ggml_cos(ctx, cum_lt);
    ggml_tensor * sin_lt   = ggml_sin(ctx, cum_lt);
    ggml_tensor * cos_a    = ggml_cont(ctx, ggml_permute(ctx, cos_lt, 2, 0, 1, 3));
    ggml_tensor * sin_a    = ggml_cont(ctx, ggml_permute(ctx, sin_lt, 2, 0, 1, 3));

    auto apply_rotary = [&](ggml_tensor * x) {
        const size_t es = ggml_element_size(x);
        auto group = [&](int gi) {
            return ggml_cont(ctx,
                ggml_view_4d(ctx, x, quarter, R, H, L,
                             x->nb[1], x->nb[2], x->nb[3],
                             gi * quarter * es));
        };
        ggml_tensor * g0 = group(0);
        ggml_tensor * g1 = group(1);
        ggml_tensor * g2 = group(2);
        ggml_tensor * g3 = group(3);
        ggml_tensor * c4 = ggml_reshape_4d(ctx, cos_a, quarter, 1, H, L);
        ggml_tensor * s4 = ggml_reshape_4d(ctx, sin_a, quarter, 1, H, L);
        ggml_tensor * g0_new = ggml_sub(ctx, ggml_mul(ctx, g0, c4), ggml_mul(ctx, g2, s4));
        ggml_tensor * g2_new = ggml_add(ctx, ggml_mul(ctx, g0, s4), ggml_mul(ctx, g2, c4));
        ggml_tensor * lo = ggml_concat(ctx, g0_new, g1, /*dim=*/ 0);
        ggml_tensor * hi = ggml_concat(ctx, g2_new, g3, /*dim=*/ 0);
        return ggml_concat(ctx, lo, hi, /*dim=*/ 0);
    };
    ggml_tensor * q_rot = apply_rotary(q_biased);  // (D_qk, R, H, L)
    ggml_tensor * k_rot = apply_rotary(k_biased);

    ggml_tensor * log_a    = ggml_log(ctx, alpha);
    ggml_tensor * log_a_LH = ggml_cont(ctx, ggml_transpose(ctx, log_a));  // (L, H)

    ggml_tensor * gamma          = ggml_mul(ctx, trap_post, dt);
    ggml_tensor * one_minus_trap = ggml_scale_bias(ctx, trap_post, -1.0f, 1.0f);
    ggml_tensor * coeff_prev     = ggml_mul(ctx, one_minus_trap, dt);
    ggml_tensor * gamma_shifted;
    if (L > 1) {
        ggml_tensor * tail = ggml_view_2d(ctx, coeff_prev, H, L - 1,
                                          coeff_prev->nb[1], coeff_prev->nb[1]);
        gamma_shifted = ggml_pad(ctx, ggml_cont(ctx, tail), 0, 1, 0, 0);
    } else {
        gamma_shifted = ggml_scale(ctx, coeff_prev, 0.0f);
    }
    ggml_tensor * factor = ggml_add(ctx, gamma, gamma_shifted);                                   // (H, L)
    ggml_tensor * factor_LH = ggml_cont(ctx, ggml_transpose(ctx, factor));                        // (L, H)

    // Permute q,k,v,z to chunk-friendly layouts ONCE; we then slice these.
    ggml_tensor * q_LH = ggml_cont(ctx, ggml_permute(ctx, q_rot,  0, 1, 3, 2));   // (D_qk, R, L, H)
    ggml_tensor * v_LH = ggml_cont(ctx, ggml_permute(ctx, v_proj, 0, 1, 3, 2));   // (D_v,  R, L, H)
    ggml_tensor * z_LH = ggml_cont(ctx, ggml_permute(ctx, z_proj, 0, 1, 3, 2));   // (D_v,  R, L, H)
    ggml_tensor * k_R0 = ggml_cont(ctx, ggml_permute(ctx, k_rot,  1, 0, 2, 3));   // (R, D_qk, H, L)
    ggml_tensor * v_R0 = ggml_cont(ctx, ggml_permute(ctx, v_proj, 1, 0, 2, 3));   // (R, D_v,  H, L)

    // --- Phase B: chunk loop ---

    const int64_t n_chunks = (L + CS - 1) / CS;
    ggml_tensor * state_carry = nullptr;
    std::vector<ggml_tensor *> y_chunks;
    y_chunks.reserve((size_t) n_chunks);

    ggml_tensor * mimo_4d = ggml_reshape_4d(ctx, mimo_o, D_v, R, 1, H);
    ggml_tensor * D4      = ggml_reshape_4d(ctx, D_skip, 1, 1, 1, H);

    // If state_in is provided, seed state_carry with state_in_eff =
    //   state_in + γ_shifted[-1] · prev_kv_in
    // where γ_shifted[-1] = (1 - trap_post[0]) · dt[0] and prev_kv_in = K_in @ V_in.
    if (state_in_4d != nullptr) {
        ggml_tensor * K_R0_in = ggml_cont(ctx, ggml_permute(ctx, K_in_4d, 1, 0, 2, 3));
        ggml_tensor * V_R0_in = ggml_cont(ctx, ggml_permute(ctx, V_in_4d, 1, 0, 2, 3));
        ggml_tensor * prev_kv_in = ggml_mul_mat(ctx, K_R0_in, V_R0_in);                       // (D_qk, D_v, H, 1)
        // (1-trap_post[0])·dt[0] per head — take first column of coeff_prev (H,L) at t=0.
        const size_t fs = ggml_element_size(coeff_prev);
        ggml_tensor * gshift_neg1 = ggml_view_1d(ctx, coeff_prev, H, 0 * fs);                 // (H,)
        ggml_tensor * gshift_neg1_4d = ggml_reshape_4d(ctx,
            ggml_cont(ctx, gshift_neg1), 1, 1, H, 1);
        ggml_tensor * pk_scaled = ggml_mul(ctx, prev_kv_in, gshift_neg1_4d);                  // (D_qk, D_v, H, 1)
        // Reshape state_in to (D_qk, D_v, 1, H) so the chunk loop's broadcasts work.
        ggml_tensor * state_in_perm = ggml_cont(ctx, ggml_permute(ctx, state_in_4d, 0, 1, 3, 2));  // (D_qk, D_v, 1, H)
        ggml_tensor * pk_scaled_p   = ggml_cont(ctx, ggml_permute(ctx, pk_scaled,    0, 1, 3, 2)); // (D_qk, D_v, 1, H)
        state_carry = ggml_add(ctx, state_in_perm, pk_scaled_p);                              // (D_qk, D_v, 1, H)
    }

    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t t_start = c * CS;
        const int64_t cs_act  = std::min(CS, L - t_start);

        // Slices along the L axis (all are contiguous slabs).
        ggml_tensor * q_rot_c = ggml_view_4d(ctx, q_rot, D_qk, R, H, cs_act,
                                              q_rot->nb[1], q_rot->nb[2], q_rot->nb[3],
                                              t_start * q_rot->nb[3]);
        ggml_tensor * k_R0_c  = ggml_view_4d(ctx, k_R0, R, D_qk, H, cs_act,
                                              k_R0->nb[1], k_R0->nb[2], k_R0->nb[3],
                                              t_start * k_R0->nb[3]);
        ggml_tensor * v_R0_c  = ggml_view_4d(ctx, v_R0, R, D_v, H, cs_act,
                                              v_R0->nb[1], v_R0->nb[2], v_R0->nb[3],
                                              t_start * v_R0->nb[3]);
        ggml_tensor * q_LH_c  = ggml_view_4d(ctx, q_LH, D_qk, R, cs_act, H,
                                              q_LH->nb[1], q_LH->nb[2], q_LH->nb[3],
                                              t_start * q_LH->nb[2]);
        ggml_tensor * v_LH_c  = ggml_view_4d(ctx, v_LH, D_v, R, cs_act, H,
                                              v_LH->nb[1], v_LH->nb[2], v_LH->nb[3],
                                              t_start * v_LH->nb[2]);
        ggml_tensor * z_LH_c  = ggml_view_4d(ctx, z_LH, D_v, R, cs_act, H,
                                              z_LH->nb[1], z_LH->nb[2], z_LH->nb[3],
                                              t_start * z_LH->nb[2]);
        ggml_tensor * log_a_c  = ggml_view_2d(ctx, log_a_LH, cs_act, H,
                                               log_a_LH->nb[1], t_start * log_a_LH->nb[0]);
        ggml_tensor * factor_c = ggml_view_2d(ctx, factor_LH, cs_act, H,
                                               factor_LH->nb[1], t_start * factor_LH->nb[0]);
        // gamma_shifted is (H, L) — slice ne[1] for L.
        ggml_tensor * gshift_c = ggml_view_2d(ctx, gamma_shifted, H, cs_act,
                                               gamma_shifted->nb[1],
                                               t_start * gamma_shifted->nb[1]);

        // Within-chunk cumsum of log α (needed for both the (cs, cs) decay matrix
        // and the carry alpha_decay vector).
        ggml_tensor * cum_c = ggml_cumsum(ctx, ggml_cont(ctx, log_a_c));               // (cs, H)
        ggml_tensor * cum_c_t = ggml_reshape_3d(ctx, cum_c, 1, cs_act, H);
        ggml_tensor * cum_c_u = ggml_reshape_3d(ctx, cum_c, cs_act, 1, H);
        ggml_tensor * cum_c_full = ggml_repeat_4d(ctx, cum_c_t, cs_act, cs_act, H, 1);
        ggml_tensor * diff_c = ggml_sub(ctx, cum_c_full, cum_c_u);                     // (cs, cs, H)
        ggml_tensor * W_c    = ggml_tri(ctx, ggml_exp(ctx, diff_c), GGML_TRI_TYPE_LOWER_DIAG);

        ggml_tensor * factor_c_cont = ggml_cont(ctx, factor_c);
        ggml_tensor * factor_c_bc   = ggml_reshape_3d(ctx, factor_c_cont, cs_act, 1, H);
        ggml_tensor * Wf_c          = ggml_mul(ctx, W_c, factor_c_bc);                 // (cs, cs, H)

        // kv_c (D_qk, D_v, H, cs).
        ggml_tensor * kv_c = ggml_mul_mat(ctx, ggml_cont(ctx, k_R0_c), ggml_cont(ctx, v_R0_c));

        // state_within_c (D_qk, D_v, cs=t, H).
        ggml_tensor * kv_perm_c     = ggml_cont(ctx, ggml_permute(ctx, kv_c, 1, 2, 3, 0));
        ggml_tensor * kv_flat_c     = ggml_reshape_3d(ctx, kv_perm_c, cs_act, D_qk * D_v, H);
        ggml_tensor * state_flat_c  = ggml_mul_mat(ctx, kv_flat_c, Wf_c);
        ggml_tensor * state_within_c = ggml_reshape_4d(ctx, state_flat_c, D_qk, D_v, cs_act, H);

        // Add carry contribution: decay_carry[t_local, h] · state_carry
        // where decay_carry[t_local, h] = exp(cum_c[t_local, h]).
        ggml_tensor * state_total_c;
        if (state_carry != nullptr) {
            ggml_tensor * alpha_decay = ggml_exp(ctx, cum_c);                          // (cs, H)
            ggml_tensor * ad_4d       = ggml_reshape_4d(ctx, alpha_decay, 1, 1, cs_act, H);
            ggml_tensor * sc_rep      = ggml_repeat_4d(ctx, state_carry, D_qk, D_v, cs_act, H);
            ggml_tensor * carry       = ggml_mul(ctx, sc_rep, ad_4d);
            state_total_c             = ggml_add(ctx, state_within_c, carry);
        } else {
            state_total_c = state_within_c;
        }

        // qstate_c (D_v, R, cs, H).
        ggml_tensor * qstate_c = ggml_mul_mat(ctx, state_total_c, ggml_cont(ctx, q_LH_c));

        // Diagonal correction.
        ggml_tensor * qk_diag_c = ggml_mul_mat(ctx, kv_c, ggml_cont(ctx, q_rot_c));    // (D_v, R, H, cs)
        ggml_tensor * gshift_c_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, gshift_c), 1, 1, H, cs_act);
        ggml_tensor * over_c    = ggml_mul(ctx, qk_diag_c, gshift_c_4d);
        ggml_tensor * over_LH_c = ggml_cont(ctx, ggml_permute(ctx, over_c, 0, 1, 3, 2));   // (D_v, R, cs, H)
        qstate_c = ggml_sub(ctx, qstate_c, over_LH_c);

        // D-skip + silu(z) + MIMO_O + sum_r.
        ggml_tensor * Dv_c       = ggml_mul(ctx, ggml_cont(ctx, v_LH_c), D4);
        ggml_tensor * o_c        = ggml_add(ctx, qstate_c, Dv_c);
        o_c                      = ggml_mul(ctx, o_c, ggml_silu(ctx, ggml_cont(ctx, z_LH_c)));
        ggml_tensor * o_w_c      = ggml_mul(ctx, o_c, mimo_4d);
        ggml_tensor * o_rsum_in  = ggml_cont(ctx, ggml_permute(ctx, o_w_c, 1, 0, 2, 3));
        ggml_tensor * y_rsum     = ggml_sum_rows(ctx, o_rsum_in);                     // (1, D_v, cs, H)
        ggml_tensor * y_Dv_cs_H  = ggml_reshape_3d(ctx, y_rsum, D_v, cs_act, H);
        ggml_tensor * y_c_DvHcs  = ggml_cont(ctx, ggml_permute(ctx, y_Dv_cs_H, 0, 2, 1, 3));  // (D_v, H, cs)
        y_chunks.push_back(y_c_DvHcs);

        // Update state_carry for next chunk (and for state_out at the end):
        //   new = exp(cum_c[cs-1, h]) · old + state_within_c[:, :, cs-1, :]
        // Always run, including the last chunk, so state_carry holds state[L-1] after the loop.
        {
            ggml_tensor * cum_c_last = ggml_view_2d(ctx, cum_c, 1, H,
                                                     cum_c->nb[1],
                                                     (cs_act - 1) * cum_c->nb[0]);     // (1, H)
            ggml_tensor * decay_full = ggml_exp(ctx, ggml_cont(ctx, cum_c_last));
            ggml_tensor * decay_4d   = ggml_reshape_4d(ctx, decay_full, 1, 1, 1, H);

            ggml_tensor * sw_last = ggml_view_4d(ctx, state_within_c, D_qk, D_v, 1, H,
                                                  state_within_c->nb[1],
                                                  state_within_c->nb[2],
                                                  state_within_c->nb[3],
                                                  (cs_act - 1) * state_within_c->nb[2]);
            ggml_tensor * sw_last_cont = ggml_cont(ctx, sw_last);

            if (state_carry != nullptr) {
                ggml_tensor * sc_decayed = ggml_mul(ctx, state_carry, decay_4d);
                state_carry = ggml_add(ctx, sc_decayed, sw_last_cont);
            } else {
                state_carry = sw_last_cont;
            }
        }
    }

    // --- Phase C: concat chunks along ne[2] (the L axis of y_c_DvHcs) ---
    ggml_tensor * y = y_chunks[0];
    for (size_t i = 1; i < y_chunks.size(); ++i) {
        y = ggml_concat(ctx, y, y_chunks[i], /*dim=*/ 2);
    }

    // Pack state_out_packed if requested. state_carry is state at L-1.
    if (state_out_packed != nullptr) {
        ggml_tensor * k_last = ggml_view_4d(ctx, k_rot, D_qk, R, H, 1,
                                             k_rot->nb[1], k_rot->nb[2], k_rot->nb[3],
                                             (L - 1) * k_rot->nb[3]);
        ggml_tensor * v_last = ggml_view_4d(ctx, v_proj, D_v, R, H, 1,
                                             v_proj->nb[1], v_proj->nb[2], v_proj->nb[3],
                                             (L - 1) * v_proj->nb[3]);
        ggml_tensor * ang_last = ggml_view_3d(ctx, cum_lt, 1, num_angles, H,
                                               cum_lt->nb[1], cum_lt->nb[2],
                                               (L - 1) * cum_lt->nb[0]);
        auto flat = [&](ggml_tensor * t, int64_t n) {
            return ggml_reshape_2d(ctx, ggml_cont(ctx, t), n, 1);
        };
        *state_out_packed = ggml_concat(ctx,
            ggml_concat(ctx,
                ggml_concat(ctx,
                    flat(state_carry, H * D_v * D_qk),
                    flat(k_last,      H * R   * D_qk), /*dim=*/ 0),
                flat(v_last,          H * R   * D_v ), /*dim=*/ 0),
            flat(ang_last,            H * num_angles), /*dim=*/ 0);
    }

    return y;
}


// One-step decode for the Mamba3-MIMO recurrence, in pure ggml ops. Reads the
// per-sequence packed state from the recurrent cache (state + K + V + angle),
// runs one recurrence step on the L=1 input, and returns both the output y
// (D_v, H, 1) and a packed new-state blob (n_embd_s, 1) the caller is expected
// to ggml_cpy back into the cache.
//
// Layout of the packed state blob (offsets in floats from start of one seq):
//   [0]                         (D_qk, D_v, H)  ssm_state
//   [H*D_v*D_qk]                (D_qk, R,   H)  K_state    (last step's k_rot)
//   [H*D_v*D_qk + H*R*D_qk]     (D_v,  R,   H)  V_state    (last step's v)
//   [... + H*R*D_v]             (na,        H)  angle_state
// — matches llama_hparams::n_embd_s() for Dragon.
static ggml_tensor * build_dragon_m_decode_step(
        ggml_context * ctx,
        ggml_tensor * q_biased,    // (D_qk, R, H, 1)
        ggml_tensor * k_biased,    // (D_qk, R, H, 1)
        ggml_tensor * v_proj,      // (D_v,  R, H, 1)
        ggml_tensor * z_proj,      // (D_v,  R, H, 1)
        ggml_tensor * alpha,       // (H, 1)
        ggml_tensor * dt,          // (H, 1)
        ggml_tensor * trap_post,   // (H, 1)
        ggml_tensor * ang_raw,     // (num_angles, 1)
        ggml_tensor * D_skip,      // (H,)
        ggml_tensor * mimo_o,      // (D_v, R, H)
        ggml_tensor * state_packed_in,  // (n_embd_s, 1) read from the cache
        int64_t D_qk, int64_t D_v, int64_t R, int64_t H,
        int64_t num_angles,
        ggml_tensor ** state_packed_out /* (n_embd_s, 1) — caller ggml_cpy's this back */) {
    const int64_t quarter = D_qk / 4;
    GGML_ASSERT(num_angles == quarter && "halved rotary expects num_angles = D_qk/4");

    const size_t fs = sizeof(float);
    const int64_t off_state = 0;
    const int64_t off_K     = off_state + H * D_v  * D_qk;
    const int64_t off_V     = off_K     + H * R    * D_qk;
    const int64_t off_ang   = off_V     + H * R    * D_v;
    const int64_t n_total   = off_ang   + H * num_angles;

    // --- Split state_packed_in into its four components (as contiguous views) ---
    ggml_tensor * sp = ggml_reshape_1d(ctx, state_packed_in, n_total);
    auto slice = [&](int64_t off, int64_t n) {
        return ggml_view_1d(ctx, sp, n, off * fs);
    };
    ggml_tensor * state_in_4d = ggml_reshape_4d(ctx, slice(off_state, H*D_v*D_qk), D_qk, D_v, H, 1);     // (D_qk, D_v, H, 1)
    ggml_tensor * K_in_4d     = ggml_reshape_4d(ctx, slice(off_K,     H*R  *D_qk), D_qk, R,   H, 1);     // (D_qk, R,   H, 1)
    ggml_tensor * V_in_4d     = ggml_reshape_4d(ctx, slice(off_V,     H*R  *D_v ), D_v,  R,   H, 1);     // (D_v,  R,   H, 1)
    ggml_tensor * angle_in_2d = ggml_reshape_2d(ctx, slice(off_ang,   H*num_angles), num_angles, H);     // (na, H)

    // --- 1. Update angle: angle_new[i, h] = angle_in[i, h] + π·tanh(ang_raw[i])·dt[h] ---
    ggml_tensor * ang_pi = ggml_scale(ctx, ggml_tanh(ctx, ang_raw), (float) M_PI);     // (na, 1)
    // outer product (na, 1) × (H, 1) → (na, H): use mul_mat(ang^T_2d (1, na), dt^T (1, H)) → result (na, H).
    ggml_tensor * ang_1na = ggml_reshape_2d(ctx, ang_pi, 1, num_angles);                // (1, na)
    ggml_tensor * dt_1H   = ggml_reshape_2d(ctx, dt,     1, H);                          // (1, H)
    ggml_tensor * delta_angle = ggml_mul_mat(ctx, ang_1na, dt_1H);                       // (na, H)
    ggml_tensor * angle_new   = ggml_add(ctx, angle_in_2d, delta_angle);                 // (na, H)

    // --- 2. Halved-rotary on q, k using angle_new. Same partitioning as the prefill primitive. ---
    ggml_tensor * cos_a = ggml_cos(ctx, angle_new);                                      // (na, H)
    ggml_tensor * sin_a = ggml_sin(ctx, angle_new);
    // Broadcast over R and the singleton L axis. cos_a is (na, H) — reshape (na, 1, H, 1).
    ggml_tensor * c4 = ggml_reshape_4d(ctx, cos_a, num_angles, 1, H, 1);
    ggml_tensor * s4 = ggml_reshape_4d(ctx, sin_a, num_angles, 1, H, 1);
    auto apply_rotary = [&](ggml_tensor * x) {
        const size_t es = ggml_element_size(x);
        auto group = [&](int gi) {
            return ggml_cont(ctx,
                ggml_view_4d(ctx, x, quarter, R, H, 1,
                             x->nb[1], x->nb[2], x->nb[3],
                             gi * quarter * es));
        };
        ggml_tensor * g0 = group(0);
        ggml_tensor * g1 = group(1);
        ggml_tensor * g2 = group(2);
        ggml_tensor * g3 = group(3);
        ggml_tensor * g0_new = ggml_sub(ctx, ggml_mul(ctx, g0, c4), ggml_mul(ctx, g2, s4));
        ggml_tensor * g2_new = ggml_add(ctx, ggml_mul(ctx, g0, s4), ggml_mul(ctx, g2, c4));
        ggml_tensor * lo = ggml_concat(ctx, g0_new, g1, 0);
        ggml_tensor * hi = ggml_concat(ctx, g2_new, g3, 0);
        return ggml_concat(ctx, lo, hi, 0);
    };
    ggml_tensor * q_rot = apply_rotary(q_biased);   // (D_qk, R, H, 1)
    ggml_tensor * k_rot = apply_rotary(k_biased);

    // --- 3. curr_kv[d, p, h] = sum_r k_rot[d, r, h] · v[p, r, h] ---
    // Permute k, v to put R at ne[0] for the contraction. (q-flavoured matmul:
    // ggml_mul_mat(a, b) computes (a^T · b), contracting over ne[0].)
    ggml_tensor * k_R0 = ggml_cont(ctx, ggml_permute(ctx, k_rot,  1, 0, 2, 3));   // (R, D_qk, H, 1)
    ggml_tensor * v_R0 = ggml_cont(ctx, ggml_permute(ctx, v_proj, 1, 0, 2, 3));   // (R, D_v,  H, 1)
    ggml_tensor * curr_kv = ggml_mul_mat(ctx, k_R0, v_R0);                        // (D_qk, D_v, H, 1)

    // --- 4. prev_kv[d, p, h] = sum_r K_in[d, r, h] · V_in[p, r, h] ---
    ggml_tensor * K_R0 = ggml_cont(ctx, ggml_permute(ctx, K_in_4d, 1, 0, 2, 3));  // (R, D_qk, H, 1)
    ggml_tensor * V_R0 = ggml_cont(ctx, ggml_permute(ctx, V_in_4d, 1, 0, 2, 3));  // (R, D_v,  H, 1)
    ggml_tensor * prev_kv = ggml_mul_mat(ctx, K_R0, V_R0);                        // (D_qk, D_v, H, 1)

    // --- 5. β = (1-trap)·dt·α, γ = trap·dt — all (H, 1) ---
    ggml_tensor * one_minus_trap = ggml_scale_bias(ctx, trap_post, -1.0f, 1.0f);   // (H, 1)
    ggml_tensor * gamma = ggml_mul(ctx, trap_post, dt);                              // (H, 1)
    ggml_tensor * beta  = ggml_mul(ctx, ggml_mul(ctx, one_minus_trap, dt), alpha);  // (H, 1)

    // Broadcast α, β, γ to (1, 1, H, 1) so they multiply (D_qk, D_v, H, 1) tensors.
    auto bcast_H = [&](ggml_tensor * t) {
        return ggml_reshape_4d(ctx, t, 1, 1, H, 1);
    };
    ggml_tensor * a_bc = bcast_H(alpha);
    ggml_tensor * b_bc = bcast_H(beta);
    ggml_tensor * g_bc = bcast_H(gamma);

    // --- 6. state_new = α·state_in + β·prev_kv + γ·curr_kv ---
    ggml_tensor * state_new_4d = ggml_add(ctx,
        ggml_add(ctx,
            ggml_mul(ctx, state_in_4d, a_bc),
            ggml_mul(ctx, prev_kv,     b_bc)),
        ggml_mul(ctx, curr_kv,     g_bc));                                          // (D_qk, D_v, H, 1)

    // --- 7. qstate[p, r, h] = sum_d state_new[d, p, h] · q_rot[d, r, h] ---
    // state_new is (D_qk=row, D_v=col, H, 1). Treat (D_qk, D_v) as the "weight matrix"
    // and q_rot as the activations: ggml_mul_mat(state_new, q_rot) gives (D_v, R, H, 1).
    ggml_tensor * qstate = ggml_mul_mat(ctx, state_new_4d, q_rot);                 // (D_v, R, H, 1)

    // --- 8. o = qstate + D[h]·v, then o *= silu(z), then sum_r (o · MIMO_O) ---
    ggml_tensor * D4 = ggml_reshape_4d(ctx, D_skip, 1, 1, H, 1);
    ggml_tensor * o  = ggml_add(ctx, qstate, ggml_mul(ctx, v_proj, D4));            // (D_v, R, H, 1)
    o = ggml_mul(ctx, o, ggml_silu(ctx, z_proj));
    ggml_tensor * mimo_4d = ggml_reshape_4d(ctx, mimo_o, D_v, R, H, 1);
    ggml_tensor * o_w     = ggml_mul(ctx, o, mimo_4d);                              // (D_v, R, H, 1)
    // sum over R via permute + sum_rows along ne[0]
    ggml_tensor * o_rsum_in = ggml_cont(ctx, ggml_permute(ctx, o_w, 1, 0, 2, 3));   // (R, D_v, H, 1)
    ggml_tensor * y_rsum    = ggml_sum_rows(ctx, o_rsum_in);                        // (1, D_v, H, 1)
    ggml_tensor * y         = ggml_reshape_3d(ctx, y_rsum, D_v, H, 1);              // (D_v, H, 1)

    // --- 9. Pack new state. Layout matches state_packed_in. ---
    // state_new (D_qk, D_v, H, 1) → flat H*D_v*D_qk
    // K_out = k_rot   (D_qk, R, H, 1) → flat H*R*D_qk
    // V_out = v_proj  (D_v,  R, H, 1) → flat H*R*D_v
    // angle_out = angle_new (na, H) → flat H*na
    auto flat = [&](ggml_tensor * t, int64_t n) {
        return ggml_reshape_2d(ctx, ggml_cont(ctx, t), n, 1);
    };
    ggml_tensor * new_packed = ggml_concat(ctx,
        ggml_concat(ctx,
            ggml_concat(ctx,
                flat(state_new_4d, H*D_v*D_qk),
                flat(k_rot,        H*R  *D_qk), /*dim=*/ 0),
            flat(v_proj,           H*R  *D_v ), /*dim=*/ 0),
        flat(angle_new,            H*num_angles), /*dim=*/ 0);                       // (n_embd_s, 1)
    *state_packed_out = new_packed;

    return y;
}


// Graph builder for the M-layer (Mamba3-MIMO) mixer. Wires the static and
// dynamic projections, normalises B/C, computes α/dt/trap, and dispatches the
// recurrence — either to dragon_mamba3_mimo_kernel via ggml_custom_4d (default)
// or to build_dragon_m_recurrence_prim when DRAGON_M_PRIM is set in the env.
static ggml_tensor * build_dragon_m_mixer_real(
        llm_graph_context & gctx,
        const llama_model & model,
        llm_graph_input_rs * inp_rs,
        const llama_memory_recurrent_context * mctx_recr,
        ggml_tensor * cur,
        const llama_hparams & hparams,
        int64_t n_tokens,
        const llama_ubatch & ubatch,
        int il) {
    auto * ctx = gctx.ctx0;

    const int64_t H_ssm   = (int64_t) hparams.ssm_d_inner / (int64_t) hparams.ssm_dt_rank; // 48
    const int64_t D_v     = (int64_t) hparams.ssm_dt_rank;                                 // 64 (headdim)
    const int64_t D_qk    = (int64_t) hparams.ssm_d_state;                                 // 128 (d_state)
    const int64_t R       = (int64_t) hparams.dragon_mamba_mimo_dim;                       // 4
    const int64_t G       = (int64_t) hparams.ssm_n_group;                                 // 1
    const int64_t angles_d = (int64_t) hparams.dragon_num_rope_angles;                     // 32
    const auto  & layer   = model.layers[il];

    // Bind the recurrent state buffer so the cache is wired in even though we
    // do not yet read/write through it (single-pass prefill only).
    ggml_tensor * ssm_states_all = mctx_recr->get_s_l(il);
    ggml_tensor * rs_view = gctx.build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), gctx.ubatch.n_seqs);

    // Per-stage dump scaffolding — only dumps for the first M layer, gated on
    // the DRAGON_DUMP_DIR env var; no-op otherwise.
    auto dump_m_stage = [&](ggml_tensor * t, const char * tag) -> ggml_tensor * {
        if (il != 0) return t;
        char namebuf[256];
        std::snprintf(namebuf, sizeof(namebuf), "m_%%02d_%s.bin", tag);
        return dragon_maybe_dump(ctx, t, namebuf, il);
    };
    cur = dump_m_stage(cur, "00_in");

    // Force fp32 for the downstream RMS-norm + bias-add path; with bf16 input
    // the intermediate tensors carry bf16 precision and lose ~0.5 dB on q/k.
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cont(ctx, ggml_cast(ctx, cur, GGML_TYPE_F32));
    }

    // Decide the recurrence path up-front (also gates whether the static
    // projection is built as a graph GEMM at all — the decode mega-kernel
    // computes it internally while streaming the weights).
    bool expanded_path = false;
    if (n_tokens == 1 && std::getenv("DRAGON_M_DECODE_PRIM")) {
        expanded_path = true;
    }
    if (const char * ec = std::getenv("DRAGON_M_CHUNK_SIZE"); ec && ec[0] && atoll(ec) > 0) {
        expanded_path = true;
    }
    if (const char * ep = std::getenv("DRAGON_M_PRIM"); ep && ep[0] && ep[0] != '0') {
        expanded_path = true;
    }
    // Opt-in (DRAGON_MEGA_DECODE=1): measured 2-4% SLOWER than the default
    // path — ggml's batched GEMM already amortizes projection weights across
    // concurrent sequences, and its native AVX512-BF16 dots outrun the
    // hand-rolled bf16 loads here. Kept for experimentation (the overlap idea
    // pays on GPU, not on this CPU).
    static const bool want_mega = []() {
        const char * e = std::getenv("DRAGON_MEGA_DECODE");
        return e && e[0] && e[0] != '0';
    }();
    const bool mega_decode = !expanded_path && want_mega && R == 4 &&
                             n_tokens == (int64_t) gctx.ubatch.n_seqs &&
                             layer.ssm_in->type == GGML_TYPE_BF16 &&
                             layer.wo->type == GGML_TYPE_BF16;

    // 1) Static projection: ssm_in @ cur → (H · (2·D_v + 3), L).
    ggml_tensor * proj_static = nullptr;
    if (!mega_decode) {
        proj_static = ggml_mul_mat(ctx, layer.ssm_in, cur);
        ggml_set_name(proj_static, "m_in_proj");
    }
    const int64_t per_head_size = 2 * D_v + 3;
    ggml_tensor * per = nullptr, * z_in = nullptr, * x_in = nullptr;
    if (!mega_decode) {
        per = ggml_reshape_3d(ctx, proj_static, per_head_size, H_ssm, n_tokens);
        // Split z, x, dt_raw, A_raw, trap_raw via views.
        z_in = ggml_view_3d(ctx, per, D_v, H_ssm, n_tokens,
                            per->nb[1], per->nb[2], /*off*/ 0);
        x_in = ggml_view_3d(ctx, per, D_v, H_ssm, n_tokens,
                            per->nb[1], per->nb[2], /*off*/ D_v * per->nb[0]);
    }
    // dt_raw / A_raw / trap_raw: extract ONE scalar per (head, token).
    // ggml_view_2d uses nb[0]=element_size, which would stride along the
    // per_head_size axis. We need the H_ssm axis to stride by per->nb[1].
    // Use view_3d with a leading singleton, then cont + reshape to drop it.
    auto pick_scalar = [&](int64_t off_elt) -> ggml_tensor * {
        ggml_tensor * v3 = ggml_view_3d(ctx, per, /*ne0*/ 1, /*ne1*/ H_ssm, /*ne2*/ n_tokens,
                                        per->nb[1], per->nb[2], off_elt * per->nb[0]);
        return ggml_reshape_2d(ctx, ggml_cont(ctx, v3), H_ssm, n_tokens);
    };
    // (dt/A/trap extraction happens only on the expanded paths — the default
    // kernel reads them from proj_static directly)

    // 2) Dynamic projection: ssm_in_dyn @ cur → (2·R·G·D_qk + num_angles, L),
    //    laid out as [B(R·G·D_qk) | C(R·G·D_qk) | angle(num_angles)] along ne[0].
    ggml_tensor * proj_dyn = ggml_mul_mat(ctx, layer.ssm_in_dyn, cur);
    ggml_set_name(proj_dyn, "m_proj_dyn");
    const int64_t BC_size = R * G * D_qk;
    ggml_tensor * B_flat = ggml_view_2d(ctx, proj_dyn, BC_size, n_tokens,
                                        proj_dyn->nb[1], /*off*/ 0);
    ggml_tensor * C_flat = ggml_view_2d(ctx, proj_dyn, BC_size, n_tokens,
                                        proj_dyn->nb[1], /*off*/ BC_size * proj_dyn->nb[0]);
    // angle_raw: trailing `angles_d` columns of proj_dyn, shape (angles_d, L).
    ggml_tensor * ang_raw_view = ggml_view_2d(ctx, proj_dyn, angles_d, n_tokens,
                                              proj_dyn->nb[1], /*off*/ 2 * BC_size * proj_dyn->nb[0]);

    // Ensure all inputs to the custom kernel are fp32 and contiguous.
    auto to_f32 = [&](ggml_tensor * t_in) {
        return ggml_cont(ctx, ggml_cast(ctx, t_in, GGML_TYPE_F32));
    };

    ggml_tensor * q_f = nullptr, * k_f = nullptr, * v_f = nullptr, * z_f = nullptr;
    ggml_tensor * dt_raw = nullptr, * A_raw = nullptr, * trap_raw = nullptr;
    ggml_tensor * D_f = nullptr, * ang_f = nullptr;
    ggml_tensor * M_f = nullptr;          // mimo_o f32 — expanded paths only

    if (expanded_path) {
        dt_raw   = pick_scalar(2 * D_v + 0);
        A_raw    = pick_scalar(2 * D_v + 1);
        trap_raw = pick_scalar(2 * D_v + 2);
        D_f      = to_f32(layer.ssm_d);
        M_f      = to_f32(layer.ssm_mimo_o);
        ang_f    = to_f32(ang_raw_view);
        ang_f    = dump_m_stage(ang_f, "05c_ang");

        // Reshape B, C to (D_qk, R, L). G=1 collapses out.
        ggml_tensor * B = ggml_reshape_3d(ctx, ggml_cont(ctx, B_flat), D_qk, R, n_tokens);
        ggml_tensor * C = ggml_reshape_3d(ctx, ggml_cont(ctx, C_flat), D_qk, R, n_tokens);

        // Apply B_norm, C_norm — broadcast over R, L.
        B = ggml_mul(ctx, ggml_rms_norm(ctx, B, hparams.f_norm_rms_eps), layer.ssm_b_norm);
        C = ggml_mul(ctx, ggml_rms_norm(ctx, C, hparams.f_norm_rms_eps), layer.ssm_c_norm);

        // Broadcast B, C from (D_qk, R, L) → (D_qk, R, H, L) so we can add per-head bias.
        ggml_tensor * B_target = ggml_new_tensor_4d(ctx, B->type, D_qk, R, H_ssm, n_tokens);
        ggml_tensor * C_target = ggml_new_tensor_4d(ctx, C->type, D_qk, R, H_ssm, n_tokens);
        ggml_tensor * B_with_H = ggml_reshape_4d(ctx, B, D_qk, R, 1, n_tokens);
        ggml_tensor * C_with_H = ggml_reshape_4d(ctx, C, D_qk, R, 1, n_tokens);
        ggml_tensor * B_bh = ggml_repeat(ctx, B_with_H, B_target);
        ggml_tensor * C_bh = ggml_repeat(ctx, C_with_H, C_target);

        // Add per-head bias (D_qk, R, H) — broadcast over L. The biases are fp32 in
        // HF but get quantized to bf16 by the MOSTLY_BF16 converter; cast back to
        // fp32 here so we don't lose ~0.8% precision per element.
        ggml_tensor * b_bias_f = ggml_cast(ctx, layer.ssm_b_bias, GGML_TYPE_F32);
        ggml_tensor * c_bias_f = ggml_cast(ctx, layer.ssm_c_bias, GGML_TYPE_F32);
        ggml_tensor * k_biased = ggml_add(ctx, B_bh, b_bias_f);
        ggml_tensor * q_biased = ggml_add(ctx, C_bh, c_bias_f);

        // v_proj, z_proj: tile (D_v, H, L) to (D_v, R, H, L) then multiply by
        // MIMO_x / MIMO_z (D_v, R, H), broadcast over L.
        ggml_tensor * x_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, x_in), D_v, 1, H_ssm, n_tokens);
        ggml_tensor * z_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, z_in), D_v, 1, H_ssm, n_tokens);
        ggml_tensor * x_target = ggml_new_tensor_4d(ctx, x_in->type, D_v, R, H_ssm, n_tokens);
        ggml_tensor * z_target = ggml_new_tensor_4d(ctx, z_in->type, D_v, R, H_ssm, n_tokens);
        ggml_tensor * x_bh = ggml_repeat(ctx, x_4d, x_target);
        ggml_tensor * z_bh = ggml_repeat(ctx, z_4d, z_target);
        ggml_tensor * v_proj = ggml_mul(ctx, x_bh, layer.ssm_mimo_x);
        ggml_tensor * z_proj = ggml_mul(ctx, z_bh, layer.ssm_mimo_z);

        q_f = to_f32(q_biased);
        k_f = to_f32(k_biased);
        v_f = to_f32(v_proj);
        z_f = to_f32(z_proj);
        // Optional input dumps for kernel-vs-primitive validation (DRAGON_DUMP_DIR gated).
        q_f = dump_m_stage(q_f, "01_qf");
        k_f = dump_m_stage(k_f, "02_kf");
        v_f = dump_m_stage(v_f, "03_vf");
        z_f = dump_m_stage(z_f, "04_zf");
    }
    // (default path: per-layer constants come from the process-wide packed
    // weight cache — no graph nodes at all; see dragon_get_packed_weights)

    // 4) Discretisation: α = exp(-softplus(A_raw) · softplus(dt_raw + dt_bias)).
    //    Only built as graph ops on the expanded paths (the primitives consume
    //    α / dt / trap as separate tensors); the default kernel computes it
    //    inline from proj_static.
    ggml_tensor * a_f_ = nullptr, * d_f_ = nullptr;
    ggml_tensor * adts_packed = nullptr;
    if (expanded_path) {
        ggml_tensor * dt_pos = ggml_add(ctx, dt_raw, layer.ssm_dt_bias);
        // Use the numerically-stable form softplus(x) = max(x, 0) + log(1 + exp(-|x|))
        // to avoid overflow for x ≫ 0 (where naive log(1+exp(x)) saturates to inf).
        auto softplus = [&](ggml_tensor * t_in) {
            ggml_tensor * pos     = ggml_relu(ctx, t_in);
            ggml_tensor * neg_abs = ggml_scale(ctx, ggml_abs(ctx, t_in), -1.0f);
            ggml_tensor * inner   = ggml_scale_bias(ctx, ggml_exp(ctx, neg_abs), 1.0f, 1.0f);
            return ggml_add(ctx, pos, ggml_log(ctx, inner));
        };
        ggml_tensor * dt_sp = softplus(dt_pos);
        // -A clamped above by -A_floor (= -1e-4 in Dragon's config.time_step_floor).
        ggml_tensor * a_neg = ggml_clamp(ctx, ggml_scale(ctx, softplus(A_raw), -1.0f),
                                         /*min=*/ -FLT_MAX, /*max=*/ -1e-4f);
        ggml_tensor * ADT     = ggml_mul(ctx, a_neg, dt_sp);
        ggml_tensor * alpha_t = ggml_exp(ctx, ADT);                             // (H, L)

        a_f_ = to_f32(alpha_t);   // (H, L) — fed to the primitive paths
        d_f_ = to_f32(dt_sp);     // (H, L)

        // Pack (α, dt, trap) into one (3, H, L) tensor for the kernel. ggml_concat
        // along dim=0 stacks them in ne[0], so reshape each to (1, H, L) first.
        auto pack_scalar = [&](ggml_tensor * scalar_HL) {
            return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_cast(ctx, scalar_HL, GGML_TYPE_F32)),
                                   1, H_ssm, n_tokens);
        };
        ggml_tensor * a_in    = pack_scalar(alpha_t);
        ggml_tensor * d_in    = pack_scalar(dt_sp);
        ggml_tensor * trap_in = pack_scalar(ggml_sigmoid(ctx, trap_raw));
        adts_packed = ggml_cont(ctx, ggml_concat(ctx,
                                    ggml_concat(ctx, a_in, d_in, /*dim=*/ 0),
                                    trap_in, /*dim=*/ 0));
        adts_packed = dump_m_stage(adts_packed, "05d_adts");
    }

    // Dispatch the recurrence. Four paths:
    //   * n_tokens == 1                — decode primitive (single step, reads/writes
    //                                    state from the recurrent cache). Limitation:
    //                                    correctness depends on the prefill having
    //                                    written its final state; the prefill paths
    //                                    don't do that yet, so the first decode step
    //                                    after a prefill starts from a zero state.
    //                                    See `examples/dragon-generate` for a fully
    //                                    correct repeated-prefill workaround.
    //   * DRAGON_M_CHUNK_SIZE=<N>      — chunked primitive (peak buffer O(CS),
    //                                    needed for long contexts on tight GPUs).
    //                                    Recommended N ∈ [64, 256].
    //   * DRAGON_M_PRIM=1              — non-chunked primitive (any backend, but
    //                                    allocates (D_qk, D_v, H, L) kv and
    //                                    (L, L, H) decay tensors; fine to L≈1000).
    //   * Default                      — CPU-only custom op (fastest on CPU).
    ggml_tensor * y = nullptr;
    ggml_tensor * out_pre = nullptr;
    if (n_tokens == 1 && std::getenv("DRAGON_M_DECODE_PRIM")) {
        // (Debug) route decode through the pure-ggml-ops decode-step primitive.
        ggml_tensor * trap_post = ggml_cont(ctx, ggml_sigmoid(ctx, trap_raw));
        ggml_tensor * state_packed_out = nullptr;
        y = build_dragon_m_decode_step(ctx,
                q_f, k_f, v_f, z_f,
                a_f_, d_f_, trap_post, ang_f,
                D_f, M_f,
                rs_view,
                D_qk, D_v, R, H_ssm,
                /*num_angles=*/ angles_d,
                &state_packed_out);
        const auto kv_head = mctx_recr->get_head();
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);
        ggml_build_forward_expand(gctx.gf,
            ggml_cpy(ctx, state_packed_out,
                ggml_view_2d(ctx, ssm_states_all,
                    hparams.n_embd_s(), gctx.ubatch.n_seqs,
                    ssm_states_all->nb[1],
                    kv_head * row_size)));
        goto post_recurrence;
    }
    {
    int64_t chunk_size = 0;
    if (const char * ec = std::getenv("DRAGON_M_CHUNK_SIZE"); ec && ec[0]) {
        chunk_size = atoll(ec);
    }
    if (chunk_size > 0) {
        ggml_tensor * trap_post = ggml_cont(ctx, ggml_sigmoid(ctx, trap_raw));
        ggml_tensor * state_out = nullptr;
        y = build_dragon_m_recurrence_prim_chunked(ctx,
                q_f, k_f, v_f, z_f,
                a_f_, d_f_, trap_post, ang_f,
                D_f, M_f,
                D_qk, D_v, R, H_ssm, n_tokens,
                /*num_angles=*/ angles_d,
                std::min<int64_t>(chunk_size, n_tokens),
                il,
                /*state_in_packed=*/  rs_view,
                /*state_out_packed=*/ &state_out);
        const auto kv_head = mctx_recr->get_head();
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);
        ggml_build_forward_expand(gctx.gf,
            ggml_cpy(ctx, state_out,
                ggml_view_2d(ctx, ssm_states_all,
                    hparams.n_embd_s(), ubatch.n_seqs,
                    ssm_states_all->nb[1],
                    kv_head * row_size)));
    } else if (const char * e = std::getenv("DRAGON_M_PRIM"); e && e[0] && e[0] != '0') {
        ggml_tensor * trap_post = ggml_cont(ctx, ggml_sigmoid(ctx, trap_raw));
        ggml_tensor * state_out = nullptr;
        y = build_dragon_m_recurrence_prim(ctx,
                q_f, k_f, v_f, z_f,
                a_f_, d_f_, trap_post, ang_f,
                D_f, M_f,
                D_qk, D_v, R, H_ssm, n_tokens,
                /*num_angles=*/ angles_d,
                il,
                /*state_in_packed=*/  rs_view,
                /*state_out_packed=*/ &state_out);
        const auto kv_head = mctx_recr->get_head();
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);
        ggml_build_forward_expand(gctx.gf,
            ggml_cpy(ctx, state_out,
                ggml_view_2d(ctx, ssm_states_all,
                    hparams.n_embd_s(), ubatch.n_seqs,
                    ssm_states_all->nb[1],
                    kv_head * row_size)));
    } else if (mega_decode) {
        // Decode mega-kernel: fused per-head in_proj GEMV + state step +
        // out_proj GEMV — weights stream once per head for ALL sequences.
        const auto kv_head = mctx_recr->get_head();
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);
        ggml_tensor * state_dst = ggml_view_2d(ctx, ssm_states_all,
                hparams.n_embd_s(), gctx.ubatch.n_seqs,
                ssm_states_all->nb[1], kv_head * row_size);
        ggml_tensor * args[4] = { cur, proj_dyn, rs_view, state_dst };
        auto * ud = (dragon_m_kernel_userdata *) std::malloc(sizeof(dragon_m_kernel_userdata));
        ud->n_seqs  = ubatch.n_seqs;
        ud->rms_eps = hparams.f_norm_rms_eps;
        ud->D_qk = D_qk; ud->D_v = D_v; ud->R = R; ud->H = H_ssm;
        ud->w    = dragon_get_packed_weights(layer, D_qk, D_v, R, H_ssm);
        ud->w_in = layer.ssm_in;
        ud->w_wo = layer.wo;
        const int64_t n_embd_m = cur->ne[0];
        ggml_tensor * slabs = ggml_custom_4d(ctx, GGML_TYPE_F32,
                n_embd_m, n_tokens, H_ssm, 1,
                args, 4, dragon_m_mega_decode_kernel, GGML_N_TASKS_MAX, ud);
        ggml_set_name(slabs, "m_mega");
        ggml_tensor * rargs[1] = { slabs };
        out_pre = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd_m, n_tokens, 1, 1,
                rargs, 1, dragon_m_mega_reduce_kernel, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(out_pre, "m_mega_out");
    } else {
        // State-aware custom op: rs_view (src[8]) is the gathered state input;
        // src[9] is a view of the recurrent cache that the kernel writes the
        // new packed state into directly (saves a 1.6 MB/layer ggml_cpy and a
        // y cont per token). dst is just y, in its final 3-D shape.
        const auto kv_head = mctx_recr->get_head();
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);
        ggml_tensor * state_dst = ggml_view_2d(ctx, ssm_states_all,
                hparams.n_embd_s(), gctx.ubatch.n_seqs,
                ssm_states_all->nb[1],
                kv_head * row_size);
        ggml_tensor * args[4] = { proj_dyn, proj_static, rs_view, state_dst };
        // Heap-allocated so the userdata pointer outlives graph build. ggml runs
        // the kernel during graph execution, by which time the local stack would
        // be gone.
        auto * ud = (dragon_m_kernel_userdata *) std::malloc(sizeof(dragon_m_kernel_userdata));
        ud->n_seqs  = ubatch.n_seqs;
        ud->rms_eps = hparams.f_norm_rms_eps;
        ud->D_qk = D_qk; ud->D_v = D_v; ud->R = R; ud->H = H_ssm;
        ud->w = dragon_get_packed_weights(layer, D_qk, D_v, R, H_ssm);
        ud->w_in = nullptr; ud->w_wo = nullptr;
        y = ggml_custom_4d(ctx, GGML_TYPE_F32,
                           D_v, H_ssm, n_tokens, 1,
                           args, 4,
                           dragon_mamba3_mimo_kernel,
                           /*n_tasks=*/ GGML_N_TASKS_MAX,
                           /*userdata=*/ ud);
        ggml_set_name(y, "m_kernel");
    }
    }

post_recurrence:
    ggml_tensor * out = nullptr;
    if (out_pre != nullptr) {
        out = out_pre;  // mega path: out_proj already applied in-kernel
    } else {
        y = dump_m_stage(y, "06_kernel_out");
        // mixer_proj: (D_v · H, L) → (n_embd, L)
        ggml_tensor * y_flat = ggml_reshape_2d(ctx, y, D_v * H_ssm, n_tokens);
        out = ggml_mul_mat(ctx, layer.wo, y_flat);
        ggml_set_name(out, "m_out_proj");
    }

    // Anchor rs_view in the graph so the recurrent cache buffer stays bound
    // even on paths that don't consume it (set_input asserts otherwise).
    // A 1-element view is enough — the old full-row ggml_sum_rows anchor cost
    // ~340 µs/layer/token single-threaded.
    if (rs_view != nullptr) {
        ggml_tensor * anchor = ggml_scale(ctx, ggml_view_1d(ctx, rs_view, 1, 0), 0.0f);
        out = ggml_add(ctx, out, anchor);
    }
    return out;
}


// V-layer mixer: Differential Tensor-Product Attention V2.
//   Q from `wq` (n_head full heads). K and V via TPA factorization:
//        A_k:(n_kv·rank, L)  B_k:(rank·head_dim, L)
//        K[L, h, d] = (1/rank) · sum_r A_k[L, h, r] · B_k[L, r, d]
//   followed by token-shift on K/V (config.token_shift_attn), per-head QK-norm,
//   scalable-softmax pre-multiply of Q, sliding-window attention with softcap,
//   the differential combine:
//        output[B,L,n_noise,snr,D] = signal − sigmoid(λ_proj(x)) · noise
//   flattened to (B,L,n_signal,D), and a block-level SiLU gate.
static ggml_tensor * build_dragon_v_mixer(
        llm_graph_context & gctx,
        const llama_model & model,
        llm_graph_input_attn_kv * inp_attn,
        llm_graph_input_rs * inp_rs,
        const llama_memory_recurrent_context * mctx_recr,
        ggml_tensor * cur,        // (n_embd, n_tokens)
        const llama_hparams & hparams,
        int64_t n_tokens,
        int il) {
    auto * ctx = gctx.ctx0;

    const int64_t head_dim = (int64_t) hparams.n_embd_head_k_full;
    const int64_t n_head   = (int64_t) hparams.dragon_n_signal_heads + (int64_t) hparams.dragon_n_noise_heads;
    const int64_t n_kv     = (int64_t) hparams.dragon_n_noise_heads;
    const int64_t n_signal = (int64_t) hparams.dragon_n_signal_heads;
    const int64_t n_noise  = n_kv;
    const int64_t snr      = n_signal / n_noise;
    const int64_t rank     = (int64_t) hparams.dragon_tpa_rank;
    const float   gate_b   = hparams.dragon_gate_bias;
    const auto  & layer    = model.layers[il];

    // Q projection: (n_head · head_dim, L) → (head_dim, n_head, L).
    ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.wq, cur);
    ggml_set_name(Qcur, "v_q_proj");
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head, n_tokens);

    // TPA-K, TPA-V. ggml_mul_mat reduces along ne[0] of both operands, so pack
    // the rank axis there: A has shape (rank, n_kv, L), B has (rank, head_dim, L),
    // output is (head_dim, n_kv, L). Scale by 1/rank to match HF's `.div_(rank)`.
    auto build_tpa = [&](ggml_tensor * Wa, ggml_tensor * Wb) {
        ggml_tensor * Aflat = ggml_mul_mat(ctx, Wa, cur);                          // (n_kv·rank, L)
        ggml_set_name(Aflat, "v_tpa_proj");
        ggml_tensor * Bflat = ggml_mul_mat(ctx, Wb, cur);                          // (rank·head_dim, L)
        ggml_set_name(Bflat, "v_tpa_proj");
        ggml_tensor * A     = ggml_reshape_3d(ctx, Aflat, rank, n_kv, n_tokens);
        ggml_tensor * Btmp  = ggml_reshape_3d(ctx, Bflat, head_dim, rank, n_tokens);
        ggml_tensor * B     = ggml_cont(ctx, ggml_permute(ctx, Btmp, 1, 0, 2, 3));
        ggml_tensor * kv_exp = ggml_scale(ctx, ggml_mul_mat(ctx, B, A), 1.0f / (float) rank);
        ggml_set_name(kv_exp, "v_tpa_expand");
        return kv_exp;
    };
    ggml_tensor * Kcur = build_tpa(layer.attn_wa_k, layer.attn_wb_k);              // (head_dim, n_kv, L)
    ggml_tensor * Vcur = build_tpa(layer.attn_wa_v, layer.attn_wb_v);              // (head_dim, n_kv, L)

    // Token-shift on K, V (config.token_shift_attn = true).
    //   K_new[t] = α_k[t] · K_prev[t] + (1 − α_k[t]) · K[t]
    // where K_prev[0] is read from the recurrent cache (= last K of the prior
    // batch, or zero on a fresh sequence) and K_prev[t>0] = K[t-1] within this
    // batch. α is zeroed at sequence position 0 (HF doc-start mask).
    if (layer.attn_shift_k != nullptr) {
        // Read K_prev / V_prev for this V-layer from the recurrent cache.
        const int64_t n_seqs   = gctx.ubatch.n_seqs;
        const int64_t n_seq_t  = n_tokens / n_seqs;
        GGML_ASSERT(n_seq_t * n_seqs == n_tokens && "n_tokens must be divisible by n_seqs");
        ggml_tensor * states_all = mctx_recr->get_r_l(il);                       // (n_embd_r, n_rs)
        ggml_tensor * shift_view = gctx.build_rs(inp_rs, states_all, hparams.n_embd_r(), n_seqs);
                                                                                 // (n_embd_r, n_seqs)
        const int64_t n_kp     = head_dim * n_kv;                                // K-side floats

        static const bool no_fused_shift = std::getenv("DRAGON_NO_FUSED_SHIFT") != nullptr;
        if (!no_fused_shift) {
            // Fused token shift: one custom op computes K̃/Ṽ for both tensors
            // and writes the pre-shift last K/V of each seq into the cache.
            ggml_tensor * ak = ggml_mul_mat(ctx, layer.attn_shift_k, cur);       // (n_kv, L)
            ggml_set_name(ak, "v_shift_gate");
            ggml_tensor * av = ggml_mul_mat(ctx, layer.attn_shift_v, cur);
            ggml_set_name(av, "v_shift_gate");
            const auto kv_head = mctx_recr->get_head();
            const size_t row_size = hparams.n_embd_r() * ggml_element_size(states_all);
            ggml_tensor * cache_view = ggml_view_2d(ctx, states_all,
                    hparams.n_embd_r(), n_seqs, states_all->nb[1], kv_head * row_size);
            ggml_tensor * args[7] = { Kcur, Vcur, ak, av, gctx.build_inp_pos(), shift_view, cache_view };
            auto * sud = (dragon_vshift_userdata *) std::malloc(sizeof(dragon_vshift_userdata));
            sud->n_seqs = n_seqs;
            ggml_tensor * kv_new = ggml_custom_4d(ctx, GGML_TYPE_F32,
                    head_dim, n_kv, n_tokens, 2,
                    args, 7, dragon_v_shift_kernel, GGML_N_TASKS_MAX, sud);
            ggml_set_name(kv_new, "v_shift_fused");
            Kcur = ggml_view_3d(ctx, kv_new, head_dim, n_kv, n_tokens,
                                kv_new->nb[1], kv_new->nb[2], 0);
            Vcur = ggml_view_3d(ctx, kv_new, head_dim, n_kv, n_tokens,
                                kv_new->nb[1], kv_new->nb[2], kv_new->nb[3]);
            goto post_shift;
        }
        {
        // Per-seq K_prev / V_prev: view of the first n_kp / second n_kp rows of each column.
        // shift_view shape (n_embd_r, n_seqs) — split along ne[0].
        ggml_tensor * K_prev_in = ggml_reshape_4d(ctx,
            ggml_cont(ctx, ggml_view_2d(ctx, shift_view, n_kp, n_seqs,
                                         shift_view->nb[1], 0)),
            head_dim, n_kv, 1, n_seqs);                                          // (head_dim, n_kv, 1, n_seqs)
        ggml_tensor * V_prev_in = ggml_reshape_4d(ctx,
            ggml_cont(ctx, ggml_view_2d(ctx, shift_view, n_kp, n_seqs,
                                         shift_view->nb[1],
                                         n_kp * ggml_element_size(shift_view))),
            head_dim, n_kv, 1, n_seqs);

        // α-zero mask: 0 at sequence position 0, 1 elsewhere. Computed from
        // build_inp_pos so it works for any batch start position AND multi-seq
        // (each seq's first token has pos=0).
        ggml_tensor * pos_f32   = ggml_cast(ctx, gctx.build_inp_pos(), GGML_TYPE_F32);
        ggml_tensor * pos_mask  = ggml_clamp(ctx, pos_f32, 0.0f, 1.0f);          // (n_tokens,)
        ggml_tensor * pos_mask_kv = ggml_reshape_3d(ctx, pos_mask, 1, 1, n_tokens);

        // Reshape X into per-seq blocks. X: (head_dim, n_kv, n_seq_t, n_seqs).
        auto reshape_per_seq = [&](ggml_tensor * X) {
            return ggml_reshape_4d(ctx, X, head_dim, n_kv, n_seq_t, n_seqs);
        };
        auto shift_kv = [&](ggml_tensor * X, ggml_tensor * shift_proj, ggml_tensor * X_prev_first) {
            // Build X_prev (head_dim, n_kv, n_seq_t, n_seqs) per seq: prepend
            // X_prev_first to X[:, :, 0..n_seq_t-2, :].
            ggml_tensor * X_4d = reshape_per_seq(X);                              // (head_dim, n_kv, n_seq_t, n_seqs)
            ggml_tensor * X_prev_4d;
            if (n_seq_t == 1) {
                X_prev_4d = X_prev_first;                                         // (head_dim, n_kv, 1, n_seqs)
            } else {
                ggml_tensor * X_head = ggml_view_4d(ctx, X_4d,
                    head_dim, n_kv, n_seq_t - 1, n_seqs,
                    X_4d->nb[1], X_4d->nb[2], X_4d->nb[3], 0);
                X_prev_4d = ggml_concat(ctx, X_prev_first, X_head, /*dim=*/ 2);   // (head_dim, n_kv, n_seq_t, n_seqs)
            }
            ggml_tensor * X_prev = ggml_reshape_3d(ctx, ggml_cont(ctx, X_prev_4d),
                                                    head_dim, n_kv, n_tokens);
            ggml_tensor * a = ggml_sigmoid(ctx, ggml_mul_mat(ctx, shift_proj, cur));  // (n_kv, n_tokens)
            a = ggml_reshape_3d(ctx, a, 1, n_kv, n_tokens);
            a = ggml_mul(ctx, a, pos_mask_kv);
            ggml_tensor * one_minus_a = ggml_scale_bias(ctx, a, -1.0f, 1.0f);
            return ggml_add(ctx,
                ggml_mul(ctx, X_prev, a),
                ggml_mul(ctx, X,      one_minus_a));
        };
        ggml_tensor * Knew = shift_kv(Kcur, layer.attn_shift_k, K_prev_in);
        ggml_tensor * Vnew = shift_kv(Vcur, layer.attn_shift_v, V_prev_in);

        // Per-seq writeback: take the (pre-shift) K/V at each seq's last position
        // (= position n_seq_t - 1 within that seq) and pack into (n_embd_r, n_seqs)
        // for the cache slot.
        ggml_tensor * K_4d = reshape_per_seq(Kcur);                               // (head_dim, n_kv, n_seq_t, n_seqs)
        ggml_tensor * V_4d = reshape_per_seq(Vcur);
        ggml_tensor * K_last = ggml_view_4d(ctx, K_4d,
            head_dim, n_kv, 1, n_seqs,
            K_4d->nb[1], K_4d->nb[2], K_4d->nb[3],
            (n_seq_t - 1) * K_4d->nb[2]);
        ggml_tensor * V_last = ggml_view_4d(ctx, V_4d,
            head_dim, n_kv, 1, n_seqs,
            V_4d->nb[1], V_4d->nb[2], V_4d->nb[3],
            (n_seq_t - 1) * V_4d->nb[2]);
        // Pack K and V per seq into (n_kp + n_kp = n_embd_r, n_seqs).
        ggml_tensor * K_last_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, K_last), n_kp, n_seqs);
        ggml_tensor * V_last_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, V_last), n_kp, n_seqs);
        ggml_tensor * KV_packed = ggml_concat(ctx, K_last_2d, V_last_2d, /*dim=*/ 0);
        const auto kv_head = mctx_recr->get_head();
        const size_t row_size = hparams.n_embd_r() * ggml_element_size(states_all);
        ggml_build_forward_expand(gctx.gf,
            ggml_cpy(ctx,
                KV_packed,
                ggml_view_2d(ctx, states_all,
                    hparams.n_embd_r(), gctx.ubatch.n_seqs,
                    states_all->nb[1],
                    kv_head * row_size)));

        Kcur = Knew;
        Vcur = Vnew;
        }
post_shift:;
    }

    // QK-norm (over head_dim).
    Qcur = ggml_mul(ctx, ggml_rms_norm(ctx, Qcur, hparams.f_norm_rms_eps), layer.attn_q_norm);
    Kcur = ggml_mul(ctx, ggml_rms_norm(ctx, Kcur, hparams.f_norm_rms_eps), layer.attn_k_norm);

    // Scalable softmax: Q *= softmax_scaler[h] · log(min(pos+1, slw_wsize)).
    // https://arxiv.org/abs/2501.19399
    {
        const float slw_wsize_f = (float) std::max<uint32_t>(hparams.dragon_slw_wsize, 1u);
        ggml_tensor * pos_f32      = ggml_cast(ctx, gctx.build_inp_pos(), GGML_TYPE_F32);
        ggml_tensor * pos_plus_one = ggml_scale_bias(ctx, pos_f32, 1.0f, 1.0f);
        ggml_tensor * pos_clamped  = ggml_clamp(ctx, pos_plus_one, 1.0f, slw_wsize_f);
        ggml_tensor * log_pos      = ggml_reshape_3d(ctx, ggml_log(ctx, pos_clamped), 1, 1, n_tokens);
        ggml_tensor * scaler       = ggml_reshape_3d(ctx, layer.attn_softmax_scaler, 1, n_head, 1);
        Qcur = ggml_mul(ctx, Qcur, log_pos);
        Qcur = ggml_mul(ctx, Qcur, scaler);
    }

    // Sliding-window attention with softcap. Dragon uses softmax_scale = 1/head_dim
    // (not 1/√head_dim) when use_completed_p = true; combined with the scalable
    // softmax pre-multiply this gives effective scale softmax_scaler[h] ·
    // log(min(pos+1, slw_wsize)) / head_dim on Q·Kᵀ.
    const float kq_scale = 1.0f / (float) head_dim;
    ggml_tensor * attn_out = gctx.build_attn(inp_attn,
            /* wo */ nullptr, /* wo_b */ nullptr, /* wo_s */ nullptr,
            Qcur, Kcur, Vcur, /*sinks*/ nullptr, /*kq_b*/ nullptr, /*v_mla*/ nullptr,
            kq_scale, il);
    attn_out = ggml_reshape_3d(ctx, attn_out, head_dim, n_head, n_tokens);

    // Differential combine. attn_out (head_dim, n_head, L) splits as
    // (head_dim, snr+1, n_noise, L) with the first `snr` slabs as signal and
    // the last slab as noise. Output = signal − sigmoid(λ_proj(x)) · noise.
    ggml_tensor * reshaped = ggml_reshape_4d(ctx, attn_out, head_dim, snr + 1, n_noise, n_tokens);
    ggml_tensor * sig = ggml_view_4d(ctx, reshaped, head_dim, snr, n_noise, n_tokens,
                                     reshaped->nb[1], reshaped->nb[2], reshaped->nb[3], 0);
    ggml_tensor * noi = ggml_view_4d(ctx, reshaped, head_dim, 1, n_noise, n_tokens,
                                     reshaped->nb[1], reshaped->nb[2], reshaped->nb[3],
                                     (size_t) snr * reshaped->nb[1]);
    ggml_tensor * lam = ggml_sigmoid(ctx, ggml_mul_mat(ctx, layer.attn_lambda, cur)); // (n_noise, L)
    lam = ggml_reshape_4d(ctx, lam, 1, 1, n_noise, n_tokens);
    ggml_tensor * combined = ggml_sub(ctx, ggml_cont(ctx, sig), ggml_mul(ctx, noi, lam));
    ggml_tensor * y_attn   = ggml_reshape_2d(ctx, ggml_cont(ctx, combined),
                                             head_dim * n_signal, n_tokens);

    // Block-level gate: silu(gate_proj(x) + gate_bias).
    ggml_tensor * g = ggml_silu(ctx, ggml_scale_bias(ctx, ggml_mul_mat(ctx, layer.attn_gate, cur),
                                                    1.0f, gate_b));
    y_attn = ggml_mul(ctx, y_attn, g);

    // mixer_proj: (n_signal · head_dim, L) → (n_embd, L)
    ggml_tensor * v_out = ggml_mul_mat(ctx, layer.wo, y_attn);
    ggml_set_name(v_out, "v_out_proj");
    return v_out;
}


llama_model_dragon::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    // normalize_embeddings_ngpt = true: each token's embedding row is L2-normalised
    // before entering the first block. normalize_embeddings = false ⇒ no √n_embd
    // scaling, just F.normalize().
    inpL = ggml_l2_norm(ctx0, inpL, 1e-12f);
    ggml_build_forward_expand(gf, inpL);

    auto * inp_hybrid = build_inp_mem_hybrid();
    auto * inp_attn   = inp_hybrid->get_attn();
    auto * inp_rs     = inp_hybrid->get_recr();
    const auto * mctx_recr = inp_rs->mctx;
    const llama_ubatch & ubatch_ref = this->ubatch;

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const int64_t n_tokens_g = (int64_t) inpL->ne[1];

    for (int il = 0; il < n_layer; ++il) {
        // Optional per-block dump for HF↔llama.cpp comparison, gated on DRAGON_DUMP_DIR.
        inpL = dragon_maybe_dump(ctx0, inpL, "block_%02d_in.bin", il);

        ggml_tensor * residual = inpL;

        ggml_tensor * y_mixer;
        if (hparams.is_recr(il)) {
            y_mixer = build_dragon_m_mixer_real(*this, model, inp_rs, mctx_recr,
                                                residual, hparams, n_tokens_g,
                                                ubatch_ref, il);
        } else {
            y_mixer = build_dragon_v_mixer(*this, model, inp_attn, inp_rs, mctx_recr,
                                           residual, hparams, n_tokens_g, il);
        }
        cb(y_mixer, "mixer_out", il);

        // ----- Geodesic-residual #1 -----
        cur = build_dragon_geodesic(ctx0, residual, y_mixer,
                                    model.layers[il].geo_mixer_scale,
                                    model.layers[il].geo_mixer_bias, il);
        cb(cur, "post_mixer", il);

        // ----- MoE + shared expert -----
        residual = cur;

        // n_embd → moe_latent_size bottleneck before the routed experts.
        ggml_tensor * inp_latent = ggml_mul_mat(ctx0, model.layers[il].ffn_latent_down, cur);
        cb(inp_latent, "moe_inp_latent", il);

        // Sigmoid router with per-expert bias (DeepSeek-V3 style). HF does the
        // routing matmul, sigmoid, bias-add, and top-k strictly in fp32; even a
        // single bf16 cast here can flip which experts get selected. Force fp32
        // on the input so the matmul output is fp32 and build_moe_ffn's
        // downstream sigmoid + bias + top-k operate on fp32 values.
        ggml_tensor * cur_f32 = ggml_cont(ctx0, ggml_cast(ctx0, cur, GGML_TYPE_F32));
        ggml_tensor * router_logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur_f32);
        cb(router_logits, "moe_router_logits", il);
        // Optional dump of the top-k selection for HF↔llama.cpp router comparison.
        if (dragon_dump_dir()) {
            ggml_tensor * probs    = ggml_sigmoid(ctx0, router_logits);
            ggml_tensor * probs_b  = ggml_add(ctx0, probs, model.layers[il].ffn_exp_probs_b);
            ggml_tensor * topk_idx = ggml_argsort_top_k(ctx0, probs_b, n_expert_used);
            ggml_tensor * topk_f32 = ggml_cast(ctx0, topk_idx, GGML_TYPE_F32);
            topk_f32 = dragon_maybe_dump(ctx0, topk_f32, "moe_topk_blk%02d.bin", il);
            // Anchor in graph so the dump op isn't optimised away.
            cur = ggml_add(ctx0, cur, ggml_scale(ctx0, ggml_sum_rows(ctx0, topk_f32), 0.0f));
        }

        static const bool no_fused_moe = std::getenv("DRAGON_NO_FUSED_MOE") != nullptr;
        ggml_tensor * routed = nullptr;
        if (no_fused_moe) {
            routed = build_moe_ffn(
                    inp_latent,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    nullptr, // no gate (ungated ReLU²)
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_RELU_SQR, /*norm_w=*/true, hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il, router_logits);
        } else {
            // Fused selection/weights/reduce (3 custom nodes instead of 14
            // small ggml nodes around the two mul_mat_ids).
            const int64_t n_lat = inp_latent->ne[0];
            const int64_t L_moe = inp_latent->ne[1];
            ggml_tensor * targs[2] = { router_logits, model.layers[il].ffn_exp_probs_b };
            ggml_tensor * sel = ggml_custom_4d(ctx0, GGML_TYPE_I32,
                    n_expert_used, L_moe, 1, 1, targs, 2,
                    dragon_moe_topk_kernel, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(sel, "moe_topk_fused");
            auto * mud = (dragon_moe_userdata *) std::malloc(sizeof(dragon_moe_userdata));
            mud->w_scale = hparams.expert_weights_scale;
            ggml_tensor * w_sel = ggml_custom_4d(ctx0, GGML_TYPE_F32,
                    n_expert_used, L_moe, 1, 1, targs, 2,
                    dragon_moe_weights_kernel, GGML_N_TASKS_MAX, mud);
            ggml_set_name(w_sel, "moe_weights_fused");

            ggml_tensor * lat3 = ggml_reshape_3d(ctx0, inp_latent, n_lat, 1, L_moe);
            ggml_tensor * up = ggml_mul_mat_id(ctx0, model.layers[il].ffn_up_exps, lat3, sel);
            ggml_set_name(up, "ffn_moe_up");
            up = ggml_sqr(ctx0, ggml_relu(ctx0, up));
            ggml_tensor * down = ggml_mul_mat_id(ctx0, model.layers[il].ffn_down_exps, up, sel);
            ggml_set_name(down, "ffn_moe_down");
            ggml_tensor * rargs[2] = { down, w_sel };
            routed = ggml_custom_4d(ctx0, GGML_TYPE_F32,
                    n_lat, L_moe, 1, 1, rargs, 2,
                    dragon_moe_reduce_kernel, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(routed, "moe_reduce_fused");
        }
        cb(routed, "moe_routed_pre_up", il);

        // moe_latent_size → n_embd back up.
        routed = ggml_mul_mat(ctx0, model.layers[il].ffn_latent_up, routed);
        cb(routed, "moe_routed", il);

        // Dense shared expert: n_embd → ff_shexp → ReLU² → n_embd, no gate.
        ggml_tensor * shared = build_ffn(cur,
                model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                nullptr,                          nullptr, nullptr,
                model.layers[il].ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(shared, "moe_shared", il);

        ggml_tensor * y_mlp = ggml_add(ctx0, routed, shared);
        cb(y_mlp, "mlp_out", il);

        // ----- Geodesic-residual #2 -----
        cur = build_dragon_geodesic(ctx0, residual, y_mlp,
                                    model.layers[il].geo_mlp_scale,
                                    model.layers[il].geo_mlp_bias, il);

        // Only carry forward the rows we need on the last layer.
        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        cb(cur, "l_out", il);
        inpL = cur;
    }

    cur = inpL;

    // Dump the final hidden state (input to lm_head) if requested.
    cur = dragon_maybe_dump(ctx0, cur, "final_hidden.bin", 0);

    cb(cur, "result_norm", -1); // no final norm in Dragon
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
