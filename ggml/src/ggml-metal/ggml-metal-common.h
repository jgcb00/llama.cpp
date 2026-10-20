// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

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

// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op, size_t max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (const struct ggml_tensor * op, bool has_simdgroup_mm);

// GGML_OP_DRAGON_MMV: x staged in threadgroup memory (larger batches) or read from device memory
bool ggml_metal_op_dragon_mmv_staged(const struct ggml_tensor * op);

// GGML_OP_DRAGON_MOE: FF slices per expert (up to 32), each slice a multiple of 8 columns
int ggml_metal_op_dragon_moe_nsplit (const struct ggml_tensor * op);
int ggml_metal_op_dragon_moe_nsplit2(const struct ggml_tensor * op);
// threadgroup memory of kernel `which` (0: a, 1: b1, 2: b2, 3: c), keep in sync with kernels/mamba3.metal
size_t ggml_metal_op_dragon_moe_smem(const struct ggml_tensor * op, int which);
// shared expert off the critical path (in b1/b2) for batches up to 2 tokens
bool ggml_metal_op_dragon_moe_sh_split(const struct ggml_tensor * op);
int  ggml_metal_op_dragon_moe_msplit (const struct ggml_tensor * op);
int  ggml_metal_op_dragon_moe_sh_rows(const struct ggml_tensor * op);
bool ggml_metal_op_dragon_moe_merged (const struct ggml_tensor * op);
// shared-expert weight read by kernel which (0 a, 1 b1, 2 b2, 3 c) next to its routed-path weight; -1 for m and f
int  ggml_metal_op_dragon_moe_sh_src (int which);
bool ggml_metal_op_dragon_moe_types_ok(const struct ggml_tensor * op, bool has_bfloat);
int  ggml_metal_op_dragon_moe_fused_ntg(const struct ggml_tensor * op, int gpu_cores);
bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm);

#ifdef __cplusplus
}
#endif
