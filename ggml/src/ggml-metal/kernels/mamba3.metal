#include "common.h"

// Dragon (Olala) fused ops: GGML_OP_MAMBA3_MIMO (Mamba3-MIMO mixer core, raw
// projections in) and GGML_OP_GEODESIC (geodesic residual).
//
// semantic reference: ggml_compute_forward_mamba3_mimo_f32 / _geodesic_f32 in
// ggml-cpu/ops.cpp; the design mirrors ggml-cuda/mamba3-mimo.cu.
//
// MAMBA3_MIMO: one threadgroup of M3_NTH = 256 threads (8 simdgroups) per
// (chunk, head, seq). Each token step builds its operands in threadgroup memory
// from the raw projections (B/C rms-norm + weight + bias + rotary, x/z MIMO
// scaling, softplus/exp/sigmoid discretization), then updates the state and,
// when producing outputs, applies the D skip, the SiLU(z) gate and the MIMO
// output combine. Simdgroup w owns the state rows p = w, w + 8, ...; lane l owns
// the columns d = l, l + 32, ..., so the state update and the q.S contraction
// are per thread (one simd_sum per (row, rank)) and need no extra barriers.
//
// The per-head state (D_v*D_qk floats, 32 KB for Dragon) does not fit next to
// the staging buffers in the 32 KB of threadgroup memory, so:
//   - fast path (template dims, Dragon: D_qk = 128, D_v = 64, R = 4): the state
//     lives in registers (D_v/8 x D_qk/32 = 32 floats per thread)
//   - generic path (DQK == 0, runtime dims): the state lives in device memory
//     (output state slot / chunk workspace), each element owned by one thread
//
// short ubatches (T < 2*M3_CHUNK tokens per sequence, e.g. decode): one serial
// kernel. long ubatches: the recurrence is affine in S, so with chunks of
// M3_CHUNK tokens S_end(n) = P_n S_end(n-1) + L_n (P_n the product of the
// chunk's alphas, L_n the chunk-local state run from S = 0):
//   angles:  per (head, seq): serial phase cumsum -> per-token angle buffer
//   phase 1: per (chunk, head, seq): L_n and P_n                 (parallel)
//   phase 2: per (state element, head, seq): scan over chunks, leaves the
//            chunk carry-ins in the workspace, writes the final state
//   phase 3: per (chunk, head, seq): outputs from the carry-in   (parallel)

#define M3_NTH     OP_MAMBA3_MIMO_NTH
#define M3_NSG     OP_MAMBA3_MIMO_NSG
#define M3_MAX_R   OP_MAMBA3_MIMO_MAX_R
#define M3_CHUNK   OP_MAMBA3_MIMO_CHUNK

#define M3_MODE_SERIAL 0
#define M3_MODE_PHASE1 1
#define M3_MODE_PHASE3 2

#define M3_PI      3.14159265358979f
#define M3_TWO_PI  6.283185307179586f
#define M3_INV_2PI 0.15915494309189535f

static inline float m3_softplus(float x) {
    return fmax(x, 0.0f) + precise::log(precise::exp(-fabs(x)) + 1.0f);
}

static inline float m3_wrap(float a) {
    return a - M3_TWO_PI*rint(a*M3_INV_2PI);
}

// rms scales of the first n_groups D_qk-groups of the pdyn row pd (one simdgroup per group)
static inline void m3_rms(
        device const float * pd,
        threadgroup  float * rms,
        int    n_groups,
        int    D_qk,
        float  eps,
        ushort sgitg,
        ushort tiisg) {
    for (int g = sgitg; g < n_groups; g += M3_NSG) {
        float s = 0.0f;
        for (int d = tiisg; d < D_qk; d += N_SIMDWIDTH) {
            const float x = pd[g*D_qk + d];
            s += x*x;
        }
        s = simd_sum(s);
        if (tiisg == 0) {
            rms[g] = 1.0f/sqrt(s/D_qk + eps);
        }
    }
}

// builds k (and q, gz if want_qz) of one token into threadgroup memory from the raw
// projections; cs/sn must hold the token's rotary cos/sin, rms the 2R (R) scales.
// gz holds the output gate mimo_o*silu(z) instead of z itself.
// v is always built. The caller puts barriers before and after.
template<short DQK, short DV, short RR>
static inline void m3_build_operands(
        constant ggml_metal_kargs_mamba3_mimo & args,
        device const float * pd,      // pdyn row of the token
        device const float * ps,      // pstat segment of (token, head): [z | x | dt | A | trap]
        device const float * bias_h,  // (D_qk, 2R) of the head: [b_bias | c_bias]
        device const float * mx_h,    // (D_v,  2R) of the head: [mimo_x | mimo_z]
        device const float * mo_h,    // (D_v,  R)  of the head
        device const float * norms,   // (D_qk, 2): [w_b | w_c]
        threadgroup const float * cs,
        threadgroup const float * sn,
        threadgroup const float * rms,
        threadgroup float * k,
        threadgroup float * q,
        threadgroup float * v,
        threadgroup float * gz,
        bool   want_qz,
        ushort tiitg) {
    const int D_qk = DQK > 0 ? DQK : args.D_qk;
    const int D_v  = DQK > 0 ? DV  : args.D_v;
    const int R    = DQK > 0 ? RR  : args.R;

    const int hf      = D_qk/2;
    const int quarter = D_qk/4;

    const int n_kq = (want_qz ? 2 : 1)*R*D_qk;
    for (int j = tiitg; j < n_kq; j += M3_NTH) {
        const int g = j / D_qk;
        const int d = j % D_qk;

        device const float * src = pd + g*D_qk;
        device const float * w   = norms + (g < R ? 0 : D_qk);
        device const float * b   = bias_h + g*D_qk;

        const float sc = rms[g];

        float out;
        if (d < quarter) {
            const float v0 = src[d       ]*sc*w[d       ] + b[d       ];
            const float v2 = src[d + hf]*sc*w[d + hf] + b[d + hf];
            out = v0*cs[d] - v2*sn[d];
        } else if (d >= hf && d < hf + quarter) {
            const int i = d - hf;
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

    for (int j = tiitg; j < R*D_v; j += M3_NTH) {
        const int p = j % D_v;
        v[j] = ps[D_v + p]*mx_h[j];
        if (want_qz) {
            const float zv = ps[p]*mx_h[R*D_v + j];
            gz[j] = mo_h[j]*(zv/(1.0f + precise::exp(-zv)));
        }
    }
}

// MODE: M3_MODE_SERIAL / M3_MODE_PHASE1 / M3_MODE_PHASE3
// DQK, DV, RR: compile-time dims of the register-state fast path, DQK == 0 selects the
// generic path (runtime dims, state in device memory)
template<short MODE, short DQK, short DV, short RR>
kernel void kernel_mamba3_mimo_impl(
        constant ggml_metal_kargs_mamba3_mimo & args,
        device const float * pdyn,
        device const float * pstat,
        device const float * bias,
        device const float * mxz,
        device const float * mimo_o,
        device const float * norms,
        device const float * misc,
        device const float * s_in,
        device       float * y,
        device       float * s_out,
        device       float * ws,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr bool FAST = DQK > 0;

    // fast path register tiling: NP rows per simdgroup, ND columns per lane
    constexpr short NP = FAST ? DV/M3_NSG      : 1;
    constexpr short ND = FAST ? DQK/N_SIMDWIDTH : 1;
    constexpr short NR = FAST ? RR              : 1;

    static_assert(!FAST || (DV % M3_NSG == 0 && DQK % N_SIMDWIDTH == 0 && DQK % 4 == 0), "invalid fast-path dims");

    constexpr bool want_y = MODE != M3_MODE_PHASE1;

    const int D_qk  = FAST ? DQK : args.D_qk;
    const int D_v   = FAST ? DV  : args.D_v;
    const int R     = FAST ? RR  : args.R;
    const int H     = args.H;
    const int n_ang = D_qk/4;
    const int SD    = D_qk*D_v;

    const int nc  = tgpig.x;
    const int h   = tgpig.y;
    const int seq = tgpig.z;

    const int t0 = MODE == M3_MODE_SERIAL ? 0      : nc*M3_CHUNK;
    const int tc = MODE == M3_MODE_SERIAL ? args.T : min(M3_CHUNK, args.T - t0);

    const int64_t tok0 = (int64_t) seq*args.T;

    device const float * s_in_q  = s_in  + seq*args.s_in_s;
    device       float * s_out_q = s_out + seq*args.s_out_s;

    // chunked-path workspace: [ang (n_ang, H, n_tok) | state (SD, n_chunks, H, n_seqs) | decay (n_chunks, H, n_seqs)]
    device const float * ws_ang   = ws;
    device       float * ws_state = ws + args.ws_state_off;
    device       float * ws_decay = ws + args.ws_decay_off;

    device float * W = ws_state + (((int64_t) seq*H + h)*args.n_chunks + nc)*SD;

    // threadgroup: [ k x2 | q | v x2 | gz | cs | sn | ang | rms | coef ]
    threadgroup float * kb  = shmem;
    threadgroup float * qb  = kb  + 2*R*D_qk;
    threadgroup float * vb  = qb  + R*D_qk;
    threadgroup float * zb  = vb  + 2*R*D_v;
    threadgroup float * cs  = zb  + R*D_v;
    threadgroup float * sn  = cs  + n_ang;
    threadgroup float * ang = sn  + n_ang;
    threadgroup float * rms = ang + n_ang;
    threadgroup float * cfb = rms + 2*R;

    device const float * bias_h = bias   + h*2*R*D_qk;
    device const float * mx_h   = mxz    + h*2*R*D_v;
    device const float * mo_h   = mimo_o + h*R*D_v;

    const float dt_bias = misc[h];
    const float D_h     = misc[H + h];

    // generic path state (device memory): the output state slot (serial) or the chunk workspace
    device float * S = MODE == M3_MODE_SERIAL ? s_out_q + h*SD : W;

    // fast path state (registers)
    float st[NP][ND];

    if (FAST) {
        FOR_UNROLL (short j = 0; j < NP; ++j) {
            const int p = sgitg + M3_NSG*j;
            FOR_UNROLL (short i = 0; i < ND; ++i) {
                const int d = tiisg + N_SIMDWIDTH*i;
                if (MODE == M3_MODE_SERIAL) {
                    st[j][i] = s_in_q[h*SD + p*D_qk + d];
                } else if (MODE == M3_MODE_PHASE1) {
                    st[j][i] = 0.0f;
                } else {
                    st[j][i] = W[p*D_qk + d]; // carry-in left by phase 2
                }
            }
        }
    } else {
        // each element is owned by the thread that updates it below: no barrier needed
        if (MODE != M3_MODE_PHASE3) {
            for (int p = sgitg; p < D_v; p += M3_NSG) {
                for (int d = tiisg; d < D_qk; d += N_SIMDWIDTH) {
                    S[p*D_qk + d] = MODE == M3_MODE_SERIAL ? s_in_q[h*SD + p*D_qk + d] : 0.0f;
                }
            }
        }
    }

    // rotary phases (serial path): owned by the lanes of simdgroup 0
    if (MODE == M3_MODE_SERIAL && sgitg == 0) {
        for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
            ang[i] = s_in_q[args.off_A + h*n_ang + i];
        }
    }

    // previous token's k/v: from the cache at the sequence start, else rebuilt
    short prv = 1;
    if (t0 == 0) {
        for (int j = tiitg; j < R*D_qk; j += M3_NTH) {
            kb[prv*R*D_qk + j] = s_in_q[args.off_K + h*R*D_qk + j];
        }
        for (int j = tiitg; j < R*D_v; j += M3_NTH) {
            vb[prv*R*D_v + j] = s_in_q[args.off_V + h*R*D_v + j];
        }
    } else {
        const int64_t ptok = tok0 + t0 - 1;

        device const float * pd = pdyn  + ptok*args.pdyn_s;
        device const float * ps = pstat + ptok*args.pstat_s + h*(2*D_v + 3);

        for (int i = tiitg; i < n_ang; i += M3_NTH) {
            const float a = ws_ang[(ptok*H + h)*n_ang + i];
            cs[i] = precise::cos(a);
            sn[i] = precise::sin(a);
        }
        m3_rms(pd, rms, R, D_qk, args.eps, sgitg, tiisg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        m3_build_operands<DQK, DV, RR>(args, pd, ps, bias_h, mx_h, mo_h, norms, cs, sn, rms,
                kb + prv*R*D_qk, qb, vb + prv*R*D_v, zb, false, tiitg);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pa = 1.0f; // running alpha product (phase 1)

    for (int t = 0; t < tc; ++t) {
        const int64_t tok = tok0 + t0 + t;
        const short   cur = prv ^ 1;

        device const float * pd = pdyn  + tok*args.pdyn_s;
        device const float * ps = pstat + tok*args.pstat_s + h*(2*D_v + 3);

        // token coefficients + rotary phases: simdgroup 0, published through threadgroup memory
        //
        // note: the coef/cs/sn/rms buffers are overwritten here while other simdgroups may still be
        //       in the state update of the previous token - this is safe because those buffers are
        //       only read between the two barriers below (the coefs are copied to registers)
        if (sgitg == 0) {
            const float dt    = m3_softplus(ps[2*D_v + 0] + dt_bias);
            const float a     = fmin(-m3_softplus(ps[2*D_v + 1]), -args.a_floor);
            const float alpha = precise::exp(a*dt);
            const float trap  = 1.0f/(1.0f + precise::exp(-ps[2*D_v + 2]));

            if (tiisg == 0) {
                cfb[0] = alpha;
                cfb[1] = (1.0f - trap)*dt*alpha; // beta
                cfb[2] = trap*dt;                // gamma
            }

            for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
                float a_i;
                if (MODE == M3_MODE_SERIAL) {
                    a_i = m3_wrap(ang[i] + precise::tanh(pd[2*R*D_qk + i])*M3_PI*dt);
                    ang[i] = a_i;
                } else {
                    a_i = ws_ang[(tok*H + h)*n_ang + i];
                }
                cs[i] = precise::cos(a_i);
                sn[i] = precise::sin(a_i);
            }
        }
        m3_rms(pd, rms, want_y ? 2*R : R, D_qk, args.eps, sgitg, tiisg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = cfb[0];
        const float beta  = cfb[1];
        const float gamma = cfb[2];

        if (MODE == M3_MODE_PHASE1) {
            pa *= alpha;
        }

        threadgroup float * k_c = kb + cur*R*D_qk;
        threadgroup float * v_c = vb + cur*R*D_v;
        threadgroup float * k_p = kb + prv*R*D_qk;
        threadgroup float * v_p = vb + prv*R*D_v;

        m3_build_operands<DQK, DV, RR>(args, pd, ps, bias_h, mx_h, mo_h, norms, cs, sn, rms,
                k_c, qb, v_c, zb, want_y, tiitg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * y_t = y + (tok*H + h)*D_v;

        if (FAST) {
            // per-lane operand columns, with beta/gamma folded in
            float kcg[NR][ND];
            float kpb[NR][ND];
            float qq [NR][ND];

            FOR_UNROLL (short r = 0; r < NR; ++r) {
                FOR_UNROLL (short i = 0; i < ND; ++i) {
                    const int d = tiisg + N_SIMDWIDTH*i;
                    kcg[r][i] = gamma*k_c[r*D_qk + d];
                    kpb[r][i] = beta *k_p[r*D_qk + d];
                    qq [r][i] = want_y ? qb[r*D_qk + d] : 0.0f;
                }
            }

            FOR_UNROLL (short j = 0; j < NP; ++j) {
                const int p = sgitg + M3_NSG*j;

                float vc [NR];
                float vp [NR];
                float acc[NR];

                FOR_UNROLL (short r = 0; r < NR; ++r) {
                    vc [r] = v_c[r*D_v + p];
                    vp [r] = v_p[r*D_v + p];
                    acc[r] = 0.0f;
                }

                FOR_UNROLL (short i = 0; i < ND; ++i) {
                    float s = alpha*st[j][i];
                    FOR_UNROLL (short r = 0; r < NR; ++r) {
                        s += vp[r]*kpb[r][i] + vc[r]*kcg[r][i];
                    }
                    st[j][i] = s;

                    if (want_y) {
                        FOR_UNROLL (short r = 0; r < NR; ++r) {
                            acc[r] += s*qq[r][i];
                        }
                    }
                }

                if (want_y) {
                    float yp = 0.0f;
                    FOR_UNROLL (short r = 0; r < NR; ++r) {
                        const float o = simd_sum(acc[r]) + D_h*vc[r];
                        yp += o*zb[r*D_v + p];
                    }
                    if (tiisg == 0) {
                        y_t[p] = yp;
                    }
                }
            }
        } else {
            for (int p = sgitg; p < D_v; p += M3_NSG) {
                float vc [M3_MAX_R];
                float vp [M3_MAX_R];
                float acc[M3_MAX_R];

                for (int r = 0; r < R; ++r) {
                    vc [r] = v_c[r*D_v + p];
                    vp [r] = v_p[r*D_v + p];
                    acc[r] = 0.0f;
                }

                device float * Sp = S + p*D_qk;

                for (int d = tiisg; d < D_qk; d += N_SIMDWIDTH) {
                    float c_acc = 0.0f;
                    float p_acc = 0.0f;
                    for (int r = 0; r < R; ++r) {
                        c_acc += vc[r]*k_c[r*D_qk + d];
                        p_acc += vp[r]*k_p[r*D_qk + d];
                    }
                    const float s = alpha*Sp[d] + beta*p_acc + gamma*c_acc;
                    Sp[d] = s;

                    if (want_y) {
                        for (int r = 0; r < R; ++r) {
                            acc[r] += s*qb[r*D_qk + d];
                        }
                    }
                }

                if (want_y) {
                    float yp = 0.0f;
                    for (int r = 0; r < R; ++r) {
                        const float o = simd_sum(acc[r]) + D_h*vc[r];
                        yp += o*zb[r*D_v + p];
                    }
                    if (tiisg == 0) {
                        y_t[p] = yp;
                    }
                }
            }
        }

        prv = cur;
    }

    // k/v of the last token feed the next ubatch's trapezoid term
    // (written by the barrier-separated build of the last step, so readable by any thread)
    const bool last = MODE == M3_MODE_SERIAL || (MODE == M3_MODE_PHASE3 && nc == args.n_chunks - 1);
    if (last) {
        for (int j = tiitg; j < R*D_qk; j += M3_NTH) {
            s_out_q[args.off_K + h*R*D_qk + j] = kb[prv*R*D_qk + j];
        }
        for (int j = tiitg; j < R*D_v; j += M3_NTH) {
            s_out_q[args.off_V + h*R*D_v + j] = vb[prv*R*D_v + j];
        }
    }

    if (MODE == M3_MODE_SERIAL && sgitg == 0) {
        for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
            s_out_q[args.off_A + h*n_ang + i] = ang[i];
        }
    }

    if (FAST && MODE != M3_MODE_PHASE3) {
        // serial: final state; phase 1: chunk-local state
        device float * dst_s = MODE == M3_MODE_SERIAL ? s_out_q + h*SD : W;
        FOR_UNROLL (short j = 0; j < NP; ++j) {
            const int p = sgitg + M3_NSG*j;
            FOR_UNROLL (short i = 0; i < ND; ++i) {
                const int d = tiisg + N_SIMDWIDTH*i;
                dst_s[p*D_qk + d] = st[j][i];
            }
        }
    }

    if (MODE == M3_MODE_PHASE1 && tiitg == 0) {
        ws_decay[((int64_t) seq*H + h)*args.n_chunks + nc] = pa;
    }
}

typedef decltype(kernel_mamba3_mimo_impl<M3_MODE_SERIAL, 0, 0, 0>) kernel_mamba3_mimo_t;

template [[host_name("kernel_mamba3_mimo_serial_gen")]]        kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_SERIAL, 0,   0,  0>;
template [[host_name("kernel_mamba3_mimo_phase1_gen")]]        kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE1, 0,   0,  0>;
template [[host_name("kernel_mamba3_mimo_phase3_gen")]]        kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE3, 0,   0,  0>;
template [[host_name("kernel_mamba3_mimo_serial_d128_v64_r4")]] kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_SERIAL, 128, 64, 4>;
template [[host_name("kernel_mamba3_mimo_phase1_d128_v64_r4")]] kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE1, 128, 64, 4>;
template [[host_name("kernel_mamba3_mimo_phase3_d128_v64_r4")]] kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE3, 128, 64, 4>;

// chunked path, step 0: per-token wrapped rotary phases (serial cumsum over the tokens of a
// sequence); one simdgroup per (head, seq), lanes own the angles
kernel void kernel_mamba3_mimo_angles(
        constant ggml_metal_kargs_mamba3_mimo & args,
        device const float * pdyn,
        device const float * pstat,
        device const float * misc,
        device const float * s_in,
        device       float * s_out,
        device       float * ws,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int h   = tgpig.x;
    const int seq = tgpig.y;

    const int D_qk  = args.D_qk;
    const int D_v   = args.D_v;
    const int R     = args.R;
    const int H     = args.H;
    const int n_ang = D_qk/4;

    const float dt_bias = misc[h];

    device const float * s_in_q  = s_in  + seq*args.s_in_s;
    device       float * s_out_q = s_out + seq*args.s_out_s;

    for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
        float a = s_in_q[args.off_A + h*n_ang + i];
        for (int t = 0; t < args.T; ++t) {
            const int64_t tok = (int64_t) seq*args.T + t;

            device const float * ps = pstat + tok*args.pstat_s + h*(2*D_v + 3);

            const float dt = m3_softplus(ps[2*D_v + 0] + dt_bias);

            a = m3_wrap(a + precise::tanh(pdyn[tok*args.pdyn_s + 2*R*D_qk + i])*M3_PI*dt);

            ws[(tok*H + h)*n_ang + i] = a;
        }
        s_out_q[args.off_A + h*n_ang + i] = a;
    }
}

// chunked path, phase 2: serial scan over the chunks, one thread per state element.
// leaves each chunk's carry-in in the workspace and writes the final state
kernel void kernel_mamba3_mimo_chunk_scan(
        constant ggml_metal_kargs_mamba3_mimo & args,
        device const float * s_in,
        device       float * s_out,
        device       float * ws,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const int h   = tgpig.y;
    const int seq = tgpig.z;

    const int H  = args.H;
    const int SD = args.D_qk*args.D_v;
    const int i  = tgpig.x*M3_NTH + tiitg;

    if (i >= SD) {
        return;
    }

    const int n_chunks = args.n_chunks;

    device       float * W  = ws + args.ws_state_off + ((int64_t) seq*H + h)*n_chunks*SD + i;
    device const float * pa = ws + args.ws_decay_off + ((int64_t) seq*H + h)*n_chunks;

    float s = s_in[seq*args.s_in_s + h*SD + i];

    for (int n = 0; n < n_chunks; ++n) {
        const float lcl = W[(int64_t) n*SD];
        W[(int64_t) n*SD] = s;
        s = pa[n]*s + lcl;
    }

    s_out[seq*args.s_out_s + h*SD + i] = s;
}

// geodesic residual, one threadgroup per row:
//   g_perp = g - (x.g / |x|^2) x
//   theta  = min((min(|g_perp|/|x|, pi/4) * scale + bias) * inv_depth, pi/4)
//   out    = x cos(theta) + (g_perp/|g_perp|) |x| sin(theta)
kernel void kernel_geodesic_f32(
        constant ggml_metal_kargs_geodesic & args,
        device const char  * x,
        device const char  * g,
        device const float * scale,
        device const float * bias,
        device       char  * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short nsg = (ntg.x + N_SIMDWIDTH - 1)/N_SIMDWIDTH;

    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    const int n = args.ne00;

    device const float * xr = (device const float *) (x   + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device const float * gr = (device const float *) (g   + i1*args.nb11 + i2*args.nb12 + i3*args.nb13);
    device       float * yr = (device       float *) (dst + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    float xx = 0.0f;
    float xg = 0.0f;
    for (int i = tiitg; i < n; i += ntg.x) {
        const float xv = xr[i];
        xx += xv*xv;
        xg += xv*gr[i];
    }
    xx = simd_sum(xx);
    xg = simd_sum(xg);
    if (tiisg == 0) {
        shmem[sgitg]               = xx;
        shmem[N_SIMDWIDTH + sgitg] = xg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    xx = 0.0f;
    xg = 0.0f;
    for (short k = 0; k < nsg; ++k) {
        xx += shmem[k];
        xg += shmem[N_SIMDWIDTH + k];
    }

    const float x_norm_sq = fmax(xx, 1e-12f);
    const float coef      = xg/x_norm_sq;

    // |g - coef*x|^2 accumulated directly (the expanded form is cancellation-prone)
    float pp = 0.0f;
    for (int i = tiitg; i < n; i += ntg.x) {
        const float gp = gr[i] - coef*xr[i];
        pp += gp*gp;
    }
    pp = simd_sum(pp);

    // all threads have read the first partials before they are overwritten
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = pp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pp = 0.0f;
    for (short k = 0; k < nsg; ++k) {
        pp += shmem[k];
    }

    const float pi4      = 0.785398163f;
    const float tan_norm = fmax(sqrt(pp), 1e-8f);
    const float xn       = sqrt(x_norm_sq);

    float theta = fmin(tan_norm/fmax(xn, 1e-6f), pi4);
    theta = fmin((theta*scale[0] + bias[0])*args.inv_depth, pi4);

    const float c = precise::cos(theta);
    const float s = precise::sin(theta)*xn/tan_norm;

    for (int i = tiitg; i < n; i += ntg.x) {
        const float xv = xr[i];
        yr[i] = xv*c + (gr[i] - coef*xv)*s;
    }
}


// ---------------------------------------------------------------------------------------------
// GGML_OP_DRAGON_MOE: the Dragon latent MoE block in 4 dispatches (semantic reference:
// ggml_compute_forward_dragon_moe_f32 in ggml-cpu/ops.cpp). Activations f32, weights T4 =
// float4 / half4 / bfloat4, 256 threads (8 simdgroups) per threadgroup, lanes over 4-wide
// columns (rows must be 16-byte aligned: n % 8 == 0 for f16/bf16). Every dot product first
// issues all of its row loads (up to DMOE_MAXCH float4 per lane, unrolled) so that the memory
// latency is paid once per row group, not once per load: decode is latency-bound, not
// bandwidth-bound.
//   a:  lat = W_ld^T x | logits = W_r^T x (f32 weights) | ush = relu(W_us^T x)^2      -> workspace
//   b1: per (token, expert slot kk, FF slice): routing recomputed from the logits (top-k of
//       sigmoid + bias, ties -> lower index; slice 0 publishes sel / w_k), h = relu(W_up[e]^T lat)^2
//   b2: per (token, kk, n_lat row slice): o[t][kk][rows] = w_k * W_down[e][rows]^T h
//   c:  out = W_lu^T sum_kk o[t][kk] + W_ds^T ush                                       -> dst

#define DMOE_NTH   256
#define DMOE_NSG   8
#define DMOE_TB    4    // max tokens per pass of kernel a (x staged in threadgroup memory)
#define DMOE_MAXCH 16   // max float4 per lane per row: n <= 16*128 = 2048 columns

template<typename T4>
static inline float4 dmoe_load4(device const char * p) {
    return float4(*(device const T4 *) p);
}

// loads the lane's chunks of one weight row: w[i] = row[tiisg*4 + i*128 .. +3], i < nch
template<typename T4, int MAXCH>
static inline void dmoe_load_row(thread float4 * w, device const char * wrow, int nch, ushort tiisg) {
    constexpr int esz = sizeof(T4)/4;
#pragma unroll
    for (int i = 0; i < MAXCH; ++i) {
        if (i < nch) {
            w[i] = dmoe_load4<T4>(wrow + (size_t) (tiisg*4 + i*4*N_SIMDWIDTH)*esz);
        }
    }
}

// dot of a loaded row with a threadgroup vector (float4-aligned), full simdgroup sum
template<int MAXCH>
static inline float dmoe_dot_tg(thread const float4 * w, threadgroup const float4 * x4, int nch, ushort tiisg) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXCH; ++i) {
        if (i < nch) {
            acc += dot(w[i], x4[tiisg + i*N_SIMDWIDTH]);
        }
    }
    return simd_sum(acc);
}

template<typename T4>
kernel void kernel_dragon_moe_a(
        constant ggml_metal_kargs_dragon_moe & args,
        device const float * x,
        device const char  * w_ld,
        device const float * w_r,
        device const char  * w_us,
        device       float * ws,
        threadgroup  float * shm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int esz = sizeof(T4)/4;

    const int n   = args.n_embd;
    const int T   = args.n_tok;
    const int TB  = args.tb;
    const int nch = n/(4*N_SIMDWIDTH);
    const int row = tgpig.x*DMOE_NSG + sgitg;

    const int nA = args.n_lat;
    const int nB = nA + args.n_expert;
    const int nC = nB + args.n_ff_sh;

    device const char  * wt  = w_ld;
    device const float * wf  = w_r;
    device       float * out = ws;
    int  stride = 0;
    bool is_f32 = false;
    bool relu2  = false;

    if (row < nA) {
        wt     = w_ld + (size_t) row*n*esz;
        out    = ws + args.off_lat + row;
        stride = args.n_lat;
    } else if (row < nB) {
        wf     = w_r + (size_t) (row - nA)*n;
        out    = ws + args.off_logits + (row - nA);
        stride = args.n_expert;
        is_f32 = true;
    } else if (row < nC) {
        wt     = w_us + (size_t) (row - nB)*n*esz;
        out    = ws + args.off_ush + (row - nB);
        stride = args.n_ff_sh;
        relu2  = true;
    }

    threadgroup float4 * xs4 = (threadgroup float4 *) shm; // TB x n

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nt = min(TB, T - t0);

        // stage the tokens of this pass
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < nt*(n/4); i += DMOE_NTH) {
            xs4[i] = ((device const float4 *) (x + (size_t) t0*n))[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < nC) {
            float4 w[DMOE_MAXCH];
            if (is_f32) {
                dmoe_load_row<float4, DMOE_MAXCH>(w, (device const char *) wf, nch, tiisg);
            } else {
                dmoe_load_row<T4, DMOE_MAXCH>(w, wt, nch, tiisg);
            }
#pragma unroll
            for (int j = 0; j < DMOE_TB; ++j) {
                if (j < nt) {
                    float s = dmoe_dot_tg<DMOE_MAXCH>(w, xs4 + (size_t) j*(n/4), nch, tiisg);
                    if (tiisg == 0) {
                        if (relu2) {
                            s = max(s, 0.0f);
                            s = s*s;
                        }
                        out[(size_t) (t0 + j)*stride] = s;
                    }
                }
            }
        }
    }
}

// routing of token t: thread e < E owns expert e; k rounds of (max score, lowest index among the
// equal); sel/selv in threadgroup memory. red: 8 floats, redi: 8 ints.
static inline void dmoe_route(
        device const float * lt, device const float * bias, int E, int K,
        threadgroup float * red, threadgroup int * redi, threadgroup float * selv, threadgroup int * sele,
        ushort tiitg, ushort sgitg, ushort tiisg) {
    const int e = tiitg;
    float s = 0.0f;
    float r = -INFINITY;
    bool  taken = true;
    if (e < E) {
        s = 1.0f/(1.0f + exp(-lt[e]));
        r = s + bias[e];
        taken = false;
    }
    for (int i = 0; i < K; ++i) {
        const float m = simd_max(taken ? -INFINITY : r);
        if (tiisg == 0) {
            red[sgitg] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gm = red[0];
        for (int q = 1; q < DMOE_NSG; ++q) {
            gm = max(gm, red[q]);
        }
        const int cand = (!taken && r == gm) ? e : 0x7fffffff;
        const int cm   = simd_min(cand);
        if (tiisg == 0) {
            redi[sgitg] = cm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int sel = redi[0];
        for (int q = 1; q < DMOE_NSG; ++q) {
            sel = min(sel, redi[q]);
        }
        if (sel == 0x7fffffff) {
            sel = i; // all scores -inf/NaN: in-bounds fallback, as the CPU reference
        }
        if (e == sel) {
            taken   = true;
            selv[i] = s;
            sele[i] = sel;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<typename T4>
kernel void kernel_dragon_moe_b1(
        constant ggml_metal_kargs_dragon_moe & args,
        device const float * ws_in,
        device const float * bias,
        device const char  * w_up,
        device       float * ws_out,
        threadgroup  float * shm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int esz = sizeof(T4)/4;

    const int K      = args.k;
    const int nsplit = args.nsplit;
    const int E      = args.n_expert;
    const int n_lat  = args.n_lat;
    const int n_ff   = args.n_ff;
    const int FFS    = n_ff/nsplit;
    const int nch    = n_lat/(4*N_SIMDWIDTH);

    const int p  = tgpig.x;
    const int kk = p/nsplit;
    const int sp = p - kk*nsplit;
    const int t  = tgpig.y;

    // [ xs (n_lat) | red (8) | redi (8) | selv (K) | sele (K) ]
    threadgroup float * xs   = shm;
    threadgroup float * red  = xs + n_lat;
    threadgroup int   * redi = (threadgroup int *) (red + DMOE_NSG);
    threadgroup float * selv = red + 2*DMOE_NSG;
    threadgroup int   * sele = (threadgroup int *) (selv + K);

    dmoe_route(ws_in + args.off_logits + (size_t) t*E, bias, E, K, red, redi, selv, sele, tiitg, sgitg, tiisg);

    const int ex = sele[kk];

    if (sp == 0 && tiitg < K) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += selv[i];
        }
        sum = max(sum, 6.103515625e-5f);
        ws_out[args.off_sel + (size_t) t*K + tiitg] = (float) sele[tiitg];
        ws_out[args.off_wk  + (size_t) t*K + tiitg] = (selv[tiitg]/sum)*args.w_scale;
    }

    device const float * lat = ws_in + args.off_lat + (size_t) t*n_lat;
    for (int i = tiitg; i < n_lat; i += DMOE_NTH) {
        xs[i] = lat[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rows [sp*FFS, (sp+1)*FFS) of W_up[ex] (n_ff rows of n_lat), 4 rows per simdgroup pass
    device const char  * wup = w_up + ((size_t) ex*n_ff + (size_t) sp*FFS)*n_lat*esz;
    device       float * h   = ws_out + args.off_h + ((size_t) t*K + kk)*n_ff + sp*FFS;

    for (int j0 = sgitg*4; j0 < FFS; j0 += DMOE_NSG*4) {
        float4 w[4][8];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (j0 + r < FFS) {
                dmoe_load_row<T4, 8>(w[r], wup + (size_t) (j0 + r)*n_lat*esz, nch, tiisg);
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (j0 + r < FFS) {
                const float s = dmoe_dot_tg<8>(w[r], (threadgroup const float4 *) xs, nch, tiisg);
                if (tiisg == 0) {
                    const float rl = max(s, 0.0f);
                    h[j0 + r] = rl*rl;
                }
            }
        }
    }
}

template<typename T4>
kernel void kernel_dragon_moe_b2(
        constant ggml_metal_kargs_dragon_moe & args,
        device const float * ws_in,
        device const char  * w_down,
        device       float * ws_out,
        threadgroup  float * shm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int esz = sizeof(T4)/4;

    const int K      = args.k;
    const int nsplit = args.nsplit;
    const int n_lat  = args.n_lat;
    const int n_ff   = args.n_ff;
    const int RS     = n_lat/nsplit;
    const int nch    = n_ff/(4*N_SIMDWIDTH);

    const int p  = tgpig.x;
    const int kk = p/nsplit;
    const int sq = p - kk*nsplit;
    const int t  = tgpig.y;

    const int   ex = (int) ws_in[args.off_sel + (size_t) t*K + kk];
    const float wk =       ws_in[args.off_wk  + (size_t) t*K + kk];

    // h[t][kk] (n_ff) -> threadgroup
    threadgroup float * hs = shm;
    device const float * h = ws_in + args.off_h + ((size_t) t*K + kk)*n_ff;
    for (int i = tiitg; i < n_ff; i += DMOE_NTH) {
        hs[i] = h[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rows [sq*RS, (sq+1)*RS) of W_down[ex] (n_lat rows of n_ff), 2 rows per simdgroup pass
    device const char  * wdn = w_down + ((size_t) ex*n_lat + (size_t) sq*RS)*n_ff*esz;
    device       float * o   = ws_out + args.off_o + ((size_t) t*K + kk)*n_lat + sq*RS;

    for (int j0 = sgitg*2; j0 < RS; j0 += DMOE_NSG*2) {
        float4 w[2][8];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (j0 + r < RS) {
                dmoe_load_row<T4, 8>(w[r], wdn + (size_t) (j0 + r)*n_ff*esz, nch, tiisg);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (j0 + r < RS) {
                const float s = dmoe_dot_tg<8>(w[r], (threadgroup const float4 *) hs, nch, tiisg);
                if (tiisg == 0) {
                    o[j0 + r] = wk*s;
                }
            }
        }
    }
}

template<typename T4>
kernel void kernel_dragon_moe_c(
        constant ggml_metal_kargs_dragon_moe & args,
        device const float * ws,
        device const char  * w_lu,
        device const char  * w_ds,
        device       float * dst,
        threadgroup  float * shm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int esz = sizeof(T4)/4;

    const int K     = args.k;
    const int n_lat = args.n_lat;
    const int n_sh  = args.n_ff_sh;
    const int n     = args.n_embd;
    const int nch_l = n_lat/(4*N_SIMDWIDTH);
    const int nch_s = n_sh/(4*N_SIMDWIDTH);
    const int TBc   = args.tbc;
    const int t0    = tgpig.y*TBc;
    const int nt    = min(TBc, args.n_tok - t0);

    // [ routed (TBc x n_lat) | ush (TBc x n_ff_sh) ] - the weights are read once for the TBc tokens
    threadgroup float * routed = shm;
    threadgroup float * ush    = shm + TBc*n_lat;

    for (int j = 0; j < nt; ++j) {
        device const float * o = ws + args.off_o + (size_t) (t0 + j)*K*n_lat;
        for (int d = tiitg; d < n_lat; d += DMOE_NTH) {
            float s = 0.0f;
            for (int i = 0; i < K; ++i) {
                s += o[(size_t) i*n_lat + d];
            }
            routed[j*n_lat + d] = s;
        }
        device const float * u = ws + args.off_ush + (size_t) (t0 + j)*n_sh;
        for (int i = tiitg; i < n_sh; i += DMOE_NTH) {
            ush[j*n_sh + i] = u[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int row = tgpig.x*DMOE_NSG + sgitg;
    if (row >= n) {
        return;
    }

    float4 wl[8];
    float4 wd[DMOE_MAXCH];
    dmoe_load_row<T4, 8>         (wl, w_lu + (size_t) row*n_lat*esz, nch_l, tiisg);
    dmoe_load_row<T4, DMOE_MAXCH>(wd, w_ds + (size_t) row*n_sh*esz,  nch_s, tiisg);

#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (j < nt) {
            const float s = dmoe_dot_tg<8>(wl, (threadgroup const float4 *) (routed + j*n_lat), nch_l, tiisg) +
                            dmoe_dot_tg<DMOE_MAXCH>(wd, (threadgroup const float4 *) (ush + j*n_sh), nch_s, tiisg);
            if (tiisg == 0) {
                dst[(size_t) (t0 + j)*n + row] = s;
            }
        }
    }
}

typedef decltype(kernel_dragon_moe_a<float4>)  kernel_dragon_moe_a_t;
typedef decltype(kernel_dragon_moe_b1<float4>) kernel_dragon_moe_b1_t;
typedef decltype(kernel_dragon_moe_b2<float4>) kernel_dragon_moe_b2_t;
typedef decltype(kernel_dragon_moe_c<float4>)  kernel_dragon_moe_c_t;

template [[host_name("kernel_dragon_moe_a_f32")]]   kernel kernel_dragon_moe_a_t  kernel_dragon_moe_a<float4>;
template [[host_name("kernel_dragon_moe_b1_f32")]]  kernel kernel_dragon_moe_b1_t kernel_dragon_moe_b1<float4>;
template [[host_name("kernel_dragon_moe_b2_f32")]]  kernel kernel_dragon_moe_b2_t kernel_dragon_moe_b2<float4>;
template [[host_name("kernel_dragon_moe_c_f32")]]   kernel kernel_dragon_moe_c_t  kernel_dragon_moe_c<float4>;
template [[host_name("kernel_dragon_moe_a_f16")]]   kernel kernel_dragon_moe_a_t  kernel_dragon_moe_a<half4>;
template [[host_name("kernel_dragon_moe_b1_f16")]]  kernel kernel_dragon_moe_b1_t kernel_dragon_moe_b1<half4>;
template [[host_name("kernel_dragon_moe_b2_f16")]]  kernel kernel_dragon_moe_b2_t kernel_dragon_moe_b2<half4>;
template [[host_name("kernel_dragon_moe_c_f16")]]   kernel kernel_dragon_moe_c_t  kernel_dragon_moe_c<half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_dragon_moe_a_bf16")]]  kernel kernel_dragon_moe_a_t  kernel_dragon_moe_a<bfloat4>;
template [[host_name("kernel_dragon_moe_b1_bf16")]] kernel kernel_dragon_moe_b1_t kernel_dragon_moe_b1<bfloat4>;
template [[host_name("kernel_dragon_moe_b2_bf16")]] kernel kernel_dragon_moe_b2_t kernel_dragon_moe_b2<bfloat4>;
template [[host_name("kernel_dragon_moe_c_bf16")]]  kernel kernel_dragon_moe_c_t  kernel_dragon_moe_c<bfloat4>;
#endif


// ---------------------------------------------------------------------------------------------
// Dragon V-layer helpers (semantic reference: ggml_compute_forward_dragon_{mmv,tpa_prep,diff_gate} in ggml-cpu/ops.cpp)

// y = concat_i(W_i^T x): one output row per simdgroup, x staged for tb tokens (same scheme as kernel_dragon_moe_a)
template<typename T4>
kernel void kernel_dragon_mmv(
        constant ggml_metal_kargs_dragon_mmv & args,
        device const float * x,
        device const char  * w0,
        device const char  * w1,
        device const char  * w2,
        device const char  * w3,
        device const char  * w4,
        device const char  * w5,
        device const char  * w6,
        device const char  * w7,
        device const char  * w8,
        device       float * y,
        threadgroup  float * shm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int esz = sizeof(T4)/4;

    const int n   = args.n_embd;
    const int T   = args.n_tok;
    const int TB  = args.tb;
    const int nch = n/(4*N_SIMDWIDTH);
    const int row = tgpig.x*DMOE_NSG + sgitg;

    device const char * wrow = w0;
    int sec = -1;
    if (row < args.rows_total) {
        for (int i = 0; i < args.n_ws; ++i) {
            if (row >= args.off[i] && row < args.off[i] + args.rows[i]) {
                sec = i;
            }
        }
        device const char * w = sec == 0 ? w0 : sec == 1 ? w1 : sec == 2 ? w2 : sec == 3 ? w3 : sec == 4 ? w4 : sec == 5 ? w5 : sec == 6 ? w6 : sec == 7 ? w7 : w8;
        wrow = w + (size_t) (row - args.off[sec])*n*esz;
    }

    threadgroup float4 * xs4 = (threadgroup float4 *) shm;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nt = min(TB, T - t0);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < nt*(n/4); i += DMOE_NTH) {
            xs4[i] = ((device const float4 *) (x + (size_t) t0*n))[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sec >= 0) {
            float4 w[DMOE_MAXCH];
            dmoe_load_row<T4, DMOE_MAXCH>(w, wrow, nch, tiisg);
#pragma unroll
            for (int j = 0; j < DMOE_TB; ++j) {
                if (j < nt) {
                    const float s = dmoe_dot_tg<DMOE_MAXCH>(w, xs4 + (size_t) j*(n/4), nch, tiisg);
                    if (tiisg == 0) {
                        y[(size_t) (t0 + j)*args.rows_total + row] = s;
                    }
                }
            }
        }
    }
}

// one threadgroup per (q head h, token t); the kv head hn = h / (H/Hn) is prepared by the G threadgroups
// sharing it (only the first writes it). threadgroup memory: [ kk (Dh) | vv (Dh) | red (8) ]
kernel void kernel_dragon_tpa_prep_f32(
        constant ggml_metal_kargs_dragon_tpa_prep & args,
        device const float   * pr,
        device const float   * prev,
        device const int32_t * pos,
        device const float   * qn,
        device const float   * kn,
        device const float   * sc,
        device       float   * out,
        threadgroup  float   * shm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int H  = args.H;
    const int Hn = args.Hn;
    const int R  = args.R;
    const int Dh = args.Dh;
    const int Hs = args.Hs;
    const int T  = args.n_tok;
    const int ROW = args.ROW;

    const int h = tgpig.x;
    const int t = tgpig.y;
    const int G = H/Hn;
    const int hn = h/G;
    const int sub = h - hn*G;
    const int n_seq_t = T/args.n_seqs;
    const int s  = t/n_seq_t;
    const int tt = t - s*n_seq_t;

    const int OAK = H*Dh, OAV = OAK + Hn*R, OBK = OAV + Hn*R, OBV = OBK + R*Dh, OSK = OBV + R*Dh, OSV = OSK + Hn, OLAM = OSV + Hn, OG = OLAM + Hn;

    threadgroup float * kk  = shm;
    threadgroup float * vv  = kk + Dh;
    threadgroup float * red = vv + Dh;

    device const float * p = pr + (size_t) t*ROW;

    float pmask = (float) pos[t];
    pmask = pmask < 0.0f ? 0.0f : (pmask > 1.0f ? 1.0f : pmask);
    const float lp = log(min(max((float) pos[t] + 1.0f, 1.0f), args.wsize));

    const int d = tiitg;
    float kv2 = 0.0f;
    float q2  = 0.0f;
    float q   = 0.0f;

    if (d < Dh) {
        // raw TPA k/v of this token (and of the previous one when the shift source is in the batch)
        float ak = 0.0f, av = 0.0f, pk = 0.0f, pv = 0.0f;
        for (int r = 0; r < R; ++r) {
            ak += p[OBK + r*Dh + d]*p[OAK + hn*R + r];
            av += p[OBV + r*Dh + d]*p[OAV + hn*R + r];
        }
        ak *= 1.0f/(float) R;
        av *= 1.0f/(float) R;
        if (tt == 0) {
            pk = prev[(size_t) s*args.prev_s + hn*Dh + d];
            pv = prev[(size_t) s*args.prev_s + Hn*Dh + hn*Dh + d];
        } else {
            device const float * pp = pr + (size_t) (t - 1)*ROW;
            for (int r = 0; r < R; ++r) {
                pk += pp[OBK + r*Dh + d]*pp[OAK + hn*R + r];
                pv += pp[OBV + r*Dh + d]*pp[OAV + hn*R + r];
            }
            pk *= 1.0f/(float) R;
            pv *= 1.0f/(float) R;
        }
        if (sub == 0 && tt == n_seq_t - 1) {
            out[args.o_last + (size_t) s*2*Hn*Dh + hn*Dh + d]         = ak;
            out[args.o_last + (size_t) s*2*Hn*Dh + Hn*Dh + hn*Dh + d] = av;
        }
        const float alk = (1.0f/(1.0f + exp(-p[OSK + hn])))*pmask;
        const float alv = (1.0f/(1.0f + exp(-p[OSV + hn])))*pmask;
        const float kx = pk*alk + ak*(alk*-1.0f + 1.0f);
        kk[d] = kx;
        vv[d] = pv*alv + av*(alv*-1.0f + 1.0f);
        kv2 = kx*kx;
        q = p[h*Dh + d];
        q2 = q*q;
    }

    kv2 = simd_sum(kv2);
    q2  = simd_sum(q2);
    if (tiisg == 0) {
        red[sgitg] = kv2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ks = 0.0f;
    for (int i = 0; i < DMOE_NSG; ++i) {
        ks += red[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        red[sgitg] = q2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs = 0.0f;
    for (int i = 0; i < DMOE_NSG; ++i) {
        qs += red[i];
    }

    const float rsk = 1.0f/sqrt(ks/(float) Dh + args.eps);
    const float rsq = 1.0f/sqrt(qs/(float) Dh + args.eps);

    if (d < Dh) {
        if (sub == 0) {
            out[args.o_k + ((size_t) t*Hn + hn)*Dh + d] = (kk[d]*rsk)*kn[d];
            out[args.o_v + ((size_t) t*Hn + hn)*Dh + d] = vv[d];
        }
        out[((size_t) t*H + h)*Dh + d] = (((q*rsq)*qn[d])*lp)*sc[h];
    }
    if (h == 0 && tiitg < Hn) {
        out[args.o_lam + (size_t) t*Hn + tiitg] = 1.0f/(1.0f + exp(-p[OLAM + tiitg]));
    }
    // gate: silu(g + gate_b), spread over the H threadgroups of the token
    const int ng = Hs*Dh;
    if (args.GS > 0) {
        for (int i = tiitg; i < args.GS; i += DMOE_NTH) {
            const float g = p[OG + h*args.GS + i] + args.gate_b;
            out[args.o_g + (size_t) t*ng + h*args.GS + i] = g/(1.0f + exp(-g));
        }
    } else if (h == 0) {
        for (int i = tiitg; i < ng; i += DMOE_NTH) {
            const float g = p[OG + i] + args.gate_b;
            out[args.o_g + (size_t) t*ng + i] = g/(1.0f + exp(-g));
        }
    }
}

kernel void kernel_dragon_diff_gate_f32(
        constant ggml_metal_kargs_dragon_diff_gate & args,
        device const float * attn,
        device const float * lam,
        device const float * gs,
        device       float * y,
        uint tpig[[thread_position_in_grid]]) {
    const int Dh = args.Dh, H = args.H, Hn = args.Hn;
    const int Hs = H - Hn;
    const int snr = Hs/Hn;
    const size_t n = (size_t) args.T*Hs*Dh;
    if (tpig >= n) {
        return;
    }
    const int t  = tpig/((size_t) Hs*Dh);
    const int j  = tpig - (size_t) t*Hs*Dh;
    const int hs = j/Dh;
    const int d  = j - hs*Dh;
    const int hn = hs/snr;
    const int k  = hs - hn*snr;
    const float sig = attn[((size_t) t*H + hn*(snr + 1) + k)*Dh + d];
    const float noi = attn[((size_t) t*H + hn*(snr + 1) + snr)*Dh + d];
    y[tpig] = (sig - noi*lam[(size_t) t*args.lam_s + hn])*gs[(size_t) t*args.gs_s + j];
}

typedef decltype(kernel_dragon_mmv<float4>) kernel_dragon_mmv_t;
template [[host_name("kernel_dragon_mmv_f32")]]  kernel kernel_dragon_mmv_t kernel_dragon_mmv<float4>;
template [[host_name("kernel_dragon_mmv_f16")]]  kernel kernel_dragon_mmv_t kernel_dragon_mmv<half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_dragon_mmv_bf16")]] kernel kernel_dragon_mmv_t kernel_dragon_mmv<bfloat4>;
#endif
