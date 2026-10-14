#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
static bool ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    if (n == 64 || n == 128 || n == 256 || n == 512) {
        return true;
    }

    if (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        return (size_t) n * sizeof(float) <= max_tg_mem;
    }

    return false;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op, size_t max_tg_mem) {
    return ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

// The break-even between the small-batch mat-vec kernels (mul_mv_ext) and the simdgroup-matrix kernel (mul_mm, tiles of
// 32 src1 rows) depends on the matrix size: for small, memory-bound dense matrices (e.g. Dragon 7A1B: 9 rows cost the same
// as 32 on the mm path, decode step x1.8 from 8 to 9 sequences) the mat-vec path wins up to ~16-32 rows. Tunable via env.
static int ggml_metal_env_int(const char * name, int def) {
    const char * s = getenv(name);
    const int v = s ? atoi(s) : def;
    return v > 0 ? v : def;
}

int ggml_metal_op_mul_mat_mm_min(void) {
    static const int v = ggml_metal_env_int("GGML_METAL_MM_MIN", 8);
    return v;
}

int ggml_metal_op_mul_mat_mv_ext_max(void) {
    static const int v = ggml_metal_env_int("GGML_METAL_MV_EXT_MAX", 8);
    return v;
}

// DRAGON_MMV: rows per simdgroup (1 or 2) and simdgroups per threadgroup
int ggml_metal_op_dragon_mmv_nr(void) {
    static const int e = ggml_metal_env_int("GGML_METAL_DRAGON_MMV_NR", 1);
    static const int v = e >= 4 ? 4 : e >= 2 ? 2 : 1;
    return v;
}

// DRAGON_MMV: x staged in threadgroup memory (original kernel, 8 simdgroups, one row each) above this many tokens
bool ggml_metal_op_dragon_mmv_staged(const struct ggml_tensor * op) {
    static const int max_t = ggml_metal_env_int("GGML_METAL_DRAGON_MMV_STAGED_T", 4);
    return op->src[0]->ne[1] > max_t;
}

// DRAGON_MMV: K split across the simdgroups of a threadgroup (1) or one row per simdgroup (0)
int ggml_metal_op_dragon_mmv_ks(void) {
    static const int v = ggml_metal_env_int("GGML_METAL_DRAGON_MMV_KS", 0) != 0;
    return v;
}

// DRAGON_MMV: 4-wide chunks per lane loaded per group (2, 4, 6 or 12)
int ggml_metal_op_dragon_mmv_u(void) {
    static const int e = ggml_metal_env_int("GGML_METAL_DRAGON_MMV_U", 2);
    static const int v = e >= 12 ? 12 : e >= 6 ? 6 : e >= 4 ? 4 : 2;
    return v;
}

int ggml_metal_op_dragon_mmv_nsg(void) {
    static const int v = std::max(1, std::min(32, ggml_metal_env_int("GGML_METAL_DRAGON_MMV_NSG", 4)));
    return v;
}

int ggml_metal_op_dragon_moe_nsplit(const struct ggml_tensor * op) {
    const int64_t n_ff  = op->src[4]->ne[1];
    const int64_t n_lat = op->src[1]->ne[1];
    static const int want = ggml_metal_env_int("GGML_METAL_DRAGON_MOE_NSPLIT", 32);
    for (int ns = (int) std::min<int64_t>(want, n_ff/8); ns >= 1; --ns) {
        if (n_ff % ns == 0 && (n_ff/ns) % 8 == 0 && n_lat % ns == 0) {
            return ns;
        }
    }
    return 1;
}

// n_lat row slices of kernel b2: rows per threadgroup RS = n_lat/nsplit2 with k*RS <= 8*DMOE_B2_MAXI (= 32) items
int ggml_metal_op_dragon_moe_nsplit2(const struct ggml_tensor * op) {
    const int64_t n_lat = op->src[1]->ne[1];
    const int64_t k     = ggml_get_op_params_i32(op, 0);
    static const int want_rs = ggml_metal_env_int("GGML_METAL_DRAGON_MOE_RS", 4);
    for (int64_t rs = std::min<int64_t>(want_rs, 32/std::max<int64_t>(k, 1)); rs >= 1; --rs) {
        if (n_lat % rs == 0) {
            return (int) (n_lat/rs);
        }
    }
    return (int) n_lat;
}

size_t ggml_metal_op_dragon_moe_smem(const struct ggml_tensor * op, int which) {
    const int64_t n_embd  = op->src[0]->ne[0];
    const int64_t n_tok   = op->src[0]->ne[1];
    const int64_t n_lat   = op->src[1]->ne[1];
    const int64_t n_ff    = op->src[4]->ne[1];
    const int64_t n_ff_sh = op->src[7]->ne[1];
    const int64_t k       = ggml_get_op_params_i32(op, 0);
    const int64_t tb      = std::min<int64_t>(4, n_tok); // DMOE_TB
    size_t n = 0;
    switch (which) {
        case 0: n = tb*n_embd + 16 + 4*8;   break; // a : x of the pass | red (fused input geodesic) | K-split partial sums
        case 1: n = n_lat + 16 + 2*k;       break; // b1: lat | red | redi | selv | sele
        case 2: n = n_ff;                   break; // b2: h
        default: n = std::min<int64_t>(2, n_tok)*(n_lat + n_ff_sh); break; // c : routed | ush of the pass
    }
    if (which == 5) {
        // fused kernel: [ y (n_embd) | red (16) | part (32) | phase scratch: routed (lat | h slice), shared us, acc ]
        const int64_t ffs = n_ff/ggml_metal_op_dragon_moe_msplit(op);
        n = n_embd + 48 + std::max<int64_t>(((n_lat + 3) & ~3) + ffs, ggml_metal_op_dragon_moe_sh_rows(op));
        return GGML_PAD(n*sizeof(float), 16);
    }
    if (which == 4) {
        // kernel m: routed [ lat | red | redi | selv | sele | h slice ], shared [ x (T x n_embd) | us (T x sh_rows) ]
        const int64_t ffs = n_ff/ggml_metal_op_dragon_moe_msplit(op);
        n = std::max<int64_t>(((n_lat + 3) & ~3) + ffs, n_tok*(n_embd + ggml_metal_op_dragon_moe_sh_rows(op)));
        return GGML_PAD(n*sizeof(float), 16);
    }
    if (ggml_metal_op_dragon_moe_sh_split(op)) {
        // the shared expert runs in extra threadgroups of b1 (up, as kernel a) and b2 (down, as kernel c)
        switch (which) {
            case 1: n = std::max<int64_t>(n, tb*n_embd); break;
            case 2: n = std::max<int64_t>(n, std::min<int64_t>(2, n_tok)*n_ff_sh); break;
            case 3: n = std::min<int64_t>(2, n_tok)*n_lat; break;
            default: break;
        }
    }
    return GGML_PAD(n*sizeof(float), 16);
}

// kernel m (merged up + down) FF slices of an expert and shared-expert rows per threadgroup; 0: not usable
int ggml_metal_op_dragon_moe_msplit(const struct ggml_tensor * op) {
    const int64_t n_ff = op->src[4]->ne[1];
    static const int want = [] { const char * e = getenv("GGML_METAL_DRAGON_MOE_MSPLIT"); return e ? atoi(e) : 24; }();
    const int64_t n_lat = op->src[1]->ne[1];
    if (n_lat > 2*256) {
        return 0; // DMOE_MD*DMOE_NTH
    }
    for (int ns = (int) std::min<int64_t>(want, n_ff/4); ns >= 1; --ns) {
        if (n_ff % ns == 0 && (n_ff/ns) % 4 == 0 && n_ff/ns <= 64) {
            return ns;
        }
    }
    return 0;
}

int ggml_metal_op_dragon_moe_sh_rows(const struct ggml_tensor * op) {
    const int64_t n_ff_sh = op->src[7]->ne[1];
    static const int want = [] { const char * e = getenv("GGML_METAL_DRAGON_MOE_SH_ROWS"); return e ? atoi(e) : 16; }();
    const int step = 4;
    for (int r = (std::max(step, want)/step)*step; r >= step; r -= step) {
        if (n_ff_sh % r == 0) {
            return r;
        }
    }
    return 0;
}

// kernel m replaces b1 and b2 for T <= GGML_METAL_DRAGON_MOE_MERGE_T (default 2, 0 disables)
bool ggml_metal_op_dragon_moe_merged(const struct ggml_tensor * op) {
    static const int max_t = [] { const char * e = getenv("GGML_METAL_DRAGON_MOE_MERGE_T"); return e ? atoi(e) : 2; }();
    // kernel m reads W_up, W_down, W_us and W_ds with one weight type
    const bool same_type = op->src[4]->type == op->src[7]->type && op->src[4]->type == op->src[8]->type;
    return same_type && op->src[0]->ne[1] <= std::min(max_t, 4) && ggml_metal_op_dragon_moe_msplit(op) > 0 && ggml_metal_op_dragon_moe_sh_rows(op) > 0;
}

// fused single-dispatch kernel (decode): number of persistent threadgroups, 0 when not usable. All the threadgroups
// must be resident at once (they wait for each other): at most 3 per GPU core (measured on M3 Ultra: 240 of 80 cores
// run, 288 do not), so the kernel is not used when the core count is unknown or too small.
// GGML_METAL_DRAGON_MOE_FUSED=0 disables it, GGML_METAL_DRAGON_MOE_FUSED_P overrides the threadgroup count.
int ggml_metal_op_dragon_moe_fused_ntg(const struct ggml_tensor * op, int gpu_cores) {
    static const bool env = [] { const char * e = getenv("GGML_METAL_DRAGON_MOE_FUSED"); return !(e && e[0] == '0'); }();
    static const int  p_env = [] { const char * e = getenv("GGML_METAL_DRAGON_MOE_FUSED_P"); return e ? atoi(e) : 0; }();
    if (!env || op->src[0]->ne[1] != 1 || !ggml_metal_op_dragon_moe_merged(op)) {
        return 0;
    }
    const ggml_type t = op->src[1]->type;
    for (int i : { 4, 5, 6, 7, 8 }) {
        if (op->src[i]->type != t) {
            return 0;
        }
    }
    const int64_t n_embd   = op->src[0]->ne[0];
    const int64_t n_lat    = op->src[1]->ne[1];
    const int64_t n_expert = op->src[2]->ne[1];
    const int64_t n_ff_sh  = op->src[7]->ne[1];
    const int64_t k        = ggml_get_op_params_i32(op, 0);
    const int64_t R        = k*ggml_metal_op_dragon_moe_msplit(op);
    if (n_lat > 1024 || n_embd > 2048 || (n_embd/128) % 2 != 0 || (n_lat + n_expert) % 4 != 0) {
        return 0;
    }
    // needed: rows of [W_ld | W_r] 4 per threadgroup, W_lu rows 16 per threadgroup, one shared-expert threadgroup
    // after the routed ones; wanted: every shared-expert slice on its own threadgroup
    const int64_t need = std::max<int64_t>({ (n_lat + n_expert)/4, (n_embd + 15)/16, R + 1 });
    const int64_t want = R + n_ff_sh/ggml_metal_op_dragon_moe_sh_rows(op);
    const int64_t cap  = 3*(int64_t) gpu_cores;
    const int64_t P    = p_env > 0 ? std::max<int64_t>(need, p_env) : std::min(want, cap);
    if (P < need || (p_env <= 0 && gpu_cores <= 0)) {
        return 0;
    }
    return (int) P;
}

bool ggml_metal_op_dragon_moe_sh_split(const struct ggml_tensor * op) {
    static const int max_t = [] { const char * e = getenv("GGML_METAL_DRAGON_MOE_SH_SPLIT_T"); return e ? atoi(e) : 2; }();
    return op->src[0]->ne[1] <= max_t;
}

bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    return !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > ggml_metal_op_mul_mat_mm_min();
}

bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    bool is_output(const ggml_tensor * t) const {
        if (t == node) {
            return true;
        }
        for (const auto * f : fused) {
            if (t == f) {
                return true;
            }
        }
        return false;
    }
};

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        // only external sources matter: sources produced by the fused group are internal
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = node.node->src[i];
            if (src && !node.is_output(src)) {
                if (!ggml_mem_ranges_add_src(mrs, src)) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                const ggml_tensor * src = fused->src[i];
                if (src && !node.is_output(src)) {
                    if (!ggml_mem_ranges_add_src(mrs, src)) {
                        return false;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        if (!ggml_mem_ranges_add_dst(mrs, node.node)) {
            return false;
        }
        for (const auto * fused : node.fused) {
            if (!ggml_mem_ranges_add_dst(mrs, fused)) {
                return false;
            }
        }

        return true;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = node.node->src[i];
            if (src && !node.is_output(src)) {
                if (!ggml_mem_ranges_check_src(mrs, src)) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                const ggml_tensor * src = fused->src[i];
                if (src && !node.is_output(src)) {
                    if (!ggml_mem_ranges_check_src(mrs, src)) {
                        return false;
                    }
                }
            }
        }

        if (!ggml_mem_ranges_check_dst(mrs, node.node)) {
            return false;
        }
        for (const auto * fused : node.fused) {
            if (!ggml_mem_ranges_check_dst(mrs, fused)) {
                return false;
            }
        }

        return true;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
            case GGML_OP_GEODESIC:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

void ggml_graph_optimize(ggml_cgraph * gf) {
    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        const int f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        for (int k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
