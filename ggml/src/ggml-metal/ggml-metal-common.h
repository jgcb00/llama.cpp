// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

enum ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(int debug);
void ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht (const struct ggml_tensor * op, size_t max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (const struct ggml_tensor * op, bool has_simdgroup_mm);

// GGML_OP_OLALA_MMV: x staged in threadgroup memory (larger batches) or read from device memory
bool ggml_metal_op_olala_mmv_staged(const struct ggml_tensor * op);

// GGML_OP_OLALA_MOE: FF slices per expert (up to 32), each slice a multiple of 8 columns
int ggml_metal_op_olala_moe_nsplit (const struct ggml_tensor * op);
int ggml_metal_op_olala_moe_nsplit2(const struct ggml_tensor * op);
// threadgroup memory of kernel `which` (0: a, 1: b1, 2: b2, 3: c), keep in sync with kernels/mamba3.metal
size_t ggml_metal_op_olala_moe_smem(const struct ggml_tensor * op, int which);
// shared expert off the critical path (in b1/b2) for batches up to 2 tokens
bool ggml_metal_op_olala_moe_sh_split(const struct ggml_tensor * op);
int  ggml_metal_op_olala_moe_msplit (const struct ggml_tensor * op);
int  ggml_metal_op_olala_moe_sh_rows(const struct ggml_tensor * op);
bool ggml_metal_op_olala_moe_merged (const struct ggml_tensor * op);
// shared-expert weight read by kernel which (0 a, 1 b1, 2 b2, 3 c) next to its routed-path weight; -1 for m and f
int  ggml_metal_op_olala_moe_sh_src (int which);
bool ggml_metal_op_olala_moe_types_ok(const struct ggml_tensor * op, bool has_bfloat);
int  ggml_metal_op_olala_moe_fused_ntg(const struct ggml_tensor * op, int gpu_cores);
bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm);

// GGML_OP_OLALA_TPA_ATTN: cells per block, cells per chunk and chunks (one threadgroup per chunk and token), and the
// threadgroup memory of kernel_olala_tpa_attn
struct ggml_metal_olala_tpa_attn_split {
    int nc;
    int chunk;
    int n_chunk;
};
struct ggml_metal_olala_tpa_attn_split ggml_metal_op_olala_tpa_attn_split(const struct ggml_tensor * op, int gpu_cores, bool has_tensor);
// GGML_OP_OLALA_TPA_ATTN with the Metal 4 tensor API (kernel_olala_tpa_attn_tensor_f16): Olala dims, f16 cache
bool ggml_metal_op_olala_tpa_attn_use_tensor(const struct ggml_tensor * op, bool has_tensor);
size_t ggml_metal_op_olala_tpa_attn_smem(const struct ggml_tensor * op, int nc);
bool ggml_metal_op_mul_mat_use_mma    (const struct ggml_tensor * op);
bool ggml_metal_op_mul_mat_may_use_mma(const struct ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
enum ggml_metal_mma_kind { GGML_METAL_MMA_KIND_BLK, GGML_METAL_MMA_KIND_Q5_K, GGML_METAL_MMA_KIND_GEN };
enum ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(enum ggml_type type, int rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
int ggml_metal_mul_mv_mma_rt(const struct ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t ggml_metal_mul_mv_mma_k_step(enum ggml_type type, int rt);

#ifdef __cplusplus
}
#endif
