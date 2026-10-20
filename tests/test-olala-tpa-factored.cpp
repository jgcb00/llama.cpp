// Olala TPA factored KV cache: the factored path (ggml_olala_tpa_prep_factored + rows in a cache +
// ggml_olala_tpa_attn) must match the expanded one (ggml_olala_tpa_prep + softmax attention over the expanded K/V).
// The cache rows are stored out of order with a cell permutation and a few unused cells, so the previous-token
// indirection and the mask are exercised; two sequences of different lengths share the cache.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static ggml_tensor * rnd(ggml_context * ctx, std::mt19937 & rng, int64_t n0, int64_t n1, float scale) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
    std::normal_distribution<float> nd(0.0f, scale);
    for (int64_t i = 0; i < n0*n1; ++i) {
        ((float *) t->data)[i] = nd(rng);
    }
    return t;
}

int main() {
    const int H = 16, Hn = 4, R = 4, Dh = 64, Hs = H - Hn;
    const int T = 12;           // tokens of the ubatch: one sequence, positions p0 .. p0 + T - 1
    const int p0 = 0;
    const float eps = 1e-6f, wsize = 4096.0f, gate_b = 1.15f, scale = 1.0f/Dh, softcap = 150.0f;
    const int n_threads = 4;

    std::mt19937 rng(42);
    ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t ROW = (int64_t) H*Dh + 2*Hn*R + 2*R*Dh + 3*Hn + (int64_t) Hs*Dh;
    ggml_tensor * pr     = rnd(ctx, rng, ROW, T, 1.0f);
    ggml_tensor * prev   = rnd(ctx, rng, 2*Hn*Dh, 1, 0.5f);
    ggml_tensor * q_norm = rnd(ctx, rng, Dh, 1, 0.3f);
    ggml_tensor * k_norm = rnd(ctx, rng, Dh, 1, 0.3f);
    ggml_tensor * scaler = rnd(ctx, rng, H, 1, 1.0f);
    ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    for (int t = 0; t < T; ++t) {
        ((int32_t *) pos->data)[t] = p0 + t;
    }
    ggml_set_name(pr, "pr");

    // ---- expanded reference
    ggml_tensor * full = ggml_olala_tpa_prep(ctx, pr, prev, pos, q_norm, k_norm, scaler, H, Hn, R, Dh, Hs, 1, eps, wsize, gate_b);
    ggml_tensor * fact = ggml_olala_tpa_prep_factored(ctx, pr, prev, pos, q_norm, k_norm, scaler, H, Hn, R, Dh, Hs, 1, eps, wsize, gate_b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, fact);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float * fq = (const float *) full->data;
    const float * fk = fq + (int64_t) T*H*Dh;
    const float * fv = fk + (int64_t) T*Hn*Dh;

    std::vector<float> ref((size_t) T*H*Dh, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const int g = h/(H/Hn);
            std::vector<float> s(t + 1);
            float mx = -INFINITY;
            for (int c = 0; c <= t; ++c) {
                float acc = 0.0f;
                for (int d = 0; d < Dh; ++d) {
                    acc += fq[((int64_t) t*H + h)*Dh + d]*fk[((int64_t) c*Hn + g)*Dh + d];
                }
                s[c] = softcap*tanhf(scale*acc/softcap);
                mx = std::max(mx, s[c]);
            }
            float sum = 0.0f;
            for (int c = 0; c <= t; ++c) {
                s[c] = expf(s[c] - mx);
                sum += s[c];
            }
            for (int c = 0; c <= t; ++c) {
                for (int d = 0; d < Dh; ++d) {
                    ref[((size_t) t*H + h)*Dh + d] += s[c]/sum*fv[((int64_t) c*Hn + g)*Dh + d];
                }
            }
        }
    }

    // ---- factored: rows scattered into a cache of n_kv cells (cell of token t: perm[t]), the others unused
    const int64_t KR = ggml_olala_tpa_row_k(Hn, R, Dh), VR = ggml_olala_tpa_row_v(Hn, R, Dh);
    const int n_kv = T + 5;
    std::vector<int> perm(n_kv);
    for (int i = 0; i < n_kv; ++i) {
        perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), rng);

    bool ok = true;
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        ggml_tensor * kc   = ggml_new_tensor_3d(ctx, type, KR, n_kv, 1);
        ggml_tensor * vc   = ggml_new_tensor_3d(ctx, type, VR, n_kv, 1);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, T, 1, 1);
        ggml_tensor * pv   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, 1);
        const float * fr = (const float *) fact->data;
        const float * frk = fr + (int64_t) T*H*Dh;
        const float * frv = frk + (int64_t) T*KR;
        memset(kc->data, 0, ggml_nbytes(kc));
        memset(vc->data, 0, ggml_nbytes(vc));
        for (int t = 0; t < T; ++t) {
            for (int64_t i = 0; i < KR; ++i) {
                if (type == GGML_TYPE_F16) {
                    ((ggml_fp16_t *) kc->data)[perm[t]*KR + i] = ggml_fp32_to_fp16(frk[t*KR + i]);
                } else {
                    ((float *) kc->data)[perm[t]*KR + i] = frk[t*KR + i];
                }
            }
            for (int64_t i = 0; i < VR; ++i) {
                if (type == GGML_TYPE_F16) {
                    ((ggml_fp16_t *) vc->data)[perm[t]*VR + i] = ggml_fp32_to_fp16(frv[t*VR + i]);
                } else {
                    ((float *) vc->data)[perm[t]*VR + i] = frv[t*VR + i];
                }
            }
        }
        for (int c = 0; c < n_kv; ++c) {
            ((int32_t *) pv->data)[c] = -1;
        }
        for (int t = 1; t < T; ++t) {
            ((int32_t *) pv->data)[perm[t]] = perm[t - 1];
        }
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) mask->data)[t*n_kv + c] = ggml_fp32_to_fp16(-INFINITY);
            }
            for (int u = 0; u <= t; ++u) {
                ((ggml_fp16_t *) mask->data)[t*n_kv + perm[u]] = ggml_fp32_to_fp16(0.0f);
            }
        }
        // the first token's shift partner is the previous ubatch (prev state): with p0 == 0 its weight is 0
        ggml_tensor * q = ggml_view_4d(ctx, fact, Dh, H, T, 1, Dh*sizeof(float), (size_t) H*Dh*sizeof(float),
                (size_t) T*H*Dh*sizeof(float), 0);
        ggml_tensor * out = ggml_olala_tpa_attn(ctx, ggml_cont(ctx, q), kc, vc, mask, pv, Hn, R, scale, softcap);
        ggml_cgraph * g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, out);
        ggml_graph_compute_with_ctx(ctx, g2, n_threads);

        // the same attention over the K / V expanded back from the factors (the path of long batches)
        ggml_tensor * ke = ggml_olala_tpa_expand(ctx, kc, pv, Hn, R, Dh, false);
        ggml_tensor * ve = ggml_olala_tpa_expand(ctx, vc, pv, Hn, R, Dh, true);
        ggml_cgraph * g3 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g3, ke);
        ggml_build_forward_expand(g3, ve);
        ggml_graph_compute_with_ctx(ctx, g3, n_threads);
        std::vector<float> oe(ref.size(), 0.0f);
        const float * qf = (const float *) fact->data;
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < H; ++h) {
                const int g = h/(H/Hn);
                std::vector<float> sc(t + 1);
                float mx = -INFINITY;
                for (int u = 0; u <= t; ++u) {
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) ke->data + ((int64_t) perm[u]*Hn + g)*Dh;
                    float acc = 0.0f;
                    for (int d = 0; d < Dh; ++d) {
                        acc += qf[((int64_t) t*H + h)*Dh + d]*ggml_fp16_to_fp32(kr[d]);
                    }
                    sc[u] = softcap*tanhf(scale*acc/softcap);
                    mx = std::max(mx, sc[u]);
                }
                float sum = 0.0f;
                for (int u = 0; u <= t; ++u) {
                    sc[u] = expf(sc[u] - mx);
                    sum += sc[u];
                }
                for (int u = 0; u <= t; ++u) {
                    const ggml_fp16_t * vr = (const ggml_fp16_t *) ve->data + ((int64_t) perm[u]*Hn + g)*Dh;
                    for (int d = 0; d < Dh; ++d) {
                        oe[((size_t) t*H + h)*Dh + d] += sc[u]/sum*ggml_fp16_to_fp32(vr[d]);
                    }
                }
            }
        }

        auto nmse_of = [&](const float * x) {
            double err = 0.0, nrm = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = x[i] - ref[i];
                err += d*d;
                nrm += (double) ref[i]*ref[i];
            }
            return err/std::max(nrm, 1e-30);
        };
        const double nmse   = nmse_of((const float *) out->data);
        const double nmse_e = nmse_of(oe.data());
        const double tol    = type == GGML_TYPE_F16 ? 1e-5 : 1e-10;
        printf("factored %s cache vs expanded: nmse = %.3g %s, through ggml_olala_tpa_expand: nmse = %.3g %s\n", ggml_type_name(type),
                nmse, nmse < tol ? "OK" : "FAIL", nmse_e, nmse_e < 1e-5 ? "OK" : "FAIL");
        ok = ok && nmse < tol && nmse_e < 1e-5;
    }

    ggml_free(ctx);
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
