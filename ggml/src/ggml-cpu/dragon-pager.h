#pragma once

// Expert-slab paging for the Dragon fused MoE op in low-RAM mode (the expert
// tensors are a read-only file mapping read on demand: llama --lazy-mode on,
// budget DRAGON_EXPERT_CACHE_MB). Kept in its own translation unit so the OS headers
// (windows.h) stay out of ops.cpp. The paging calls are hints: they never
// change the mapped data.

#include <cstddef>
#include <cstdint>

// start reading [p, p+n) from the file in the background, in large requests
// (Linux/macOS MADV_WILLNEED, Windows PrefetchVirtualMemory)
void dragon_pager_prefetch(const char * p, size_t n);

// map [p, p+n) into the page tables in one call, waiting for the reads
// (Linux >= 5.14 MADV_POPULATE_READ; no-op elsewhere, plain page faults)
void dragon_pager_populate(const char * p, size_t n);

// give the pages of [p, p+n) back to the OS, rounded outward to 64 KiB
// (Linux MADV_DONTNEED only if the range is verified to still be a shared file
// mapping, macOS MADV_DONTNEED, Windows VirtualUnlock = remove from the working
// set). The data stays in the OS file cache, which is reclaimed under memory
// pressure and re-read from the file on the next access.
void dragon_pager_release(const char * p, size_t n);

// process-wide LRU of resident slabs: marks the n slabs as most recently used,
// then evicts the least recently used ones until at most budget bytes remain.
// The evicted slabs are returned in out_ptrs/out_sizes (at most max_out; any
// further ones are released directly). Returns the number returned.
int dragon_pager_used(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                      const char ** out_ptrs, size_t * out_sizes, int max_out);
