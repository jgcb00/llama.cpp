#pragma once

// Expert-slab paging for the Dragon fused MoE op in low-RAM mode (the expert
// tensors are a read-only file mapping read on demand, llama --lazy-mode on).
// Kept in its own translation unit so the OS headers (windows.h) stay out of
// ops.cpp. All calls are hints: they never change the mapped data.

#include <cstddef>
#include <cstdint>

// start reading [p, p+n) from the file in the background, in large requests
// (Linux/macOS MADV_WILLNEED, Windows PrefetchVirtualMemory)
void dragon_pager_prefetch(const char * p, size_t n);

// map [p, p+n) into the page tables in one call, waiting for the reads
// (Linux >= 5.14 MADV_POPULATE_READ; no-op elsewhere, plain page faults)
void dragon_pager_populate(const char * p, size_t n);

// give the pages of [p, p+n) back to the OS, rounded outward to 64 KiB
// (Linux MADV_DONTNEED on a verified shared file mapping, macOS MADV_DONTNEED,
// Windows VirtualUnlock = remove from the working set). The data stays in the
// OS file cache, which is reclaimed under memory pressure and re-read from the
// file on the next access. DRAGON_EXPERT_DROP_CACHE=1 (Linux) also drops it
// from the page cache.
void dragon_pager_release(const char * p, size_t n);

// process-wide LRU of resident slabs: marks the n slabs as most recently used,
// then evicts the least recently used ones until at most budget bytes remain.
// The evicted slabs are returned in out_ptrs/out_sizes (at most max_out; any
// further ones are released directly). Returns the number returned.
int dragon_pager_used(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                      const char ** out_ptrs, size_t * out_sizes, int max_out);

// ---- explicit expert cache (default in low-memory mode on Linux and macOS)
//
// Instead of paging the file mapping, the slabs are copied into a cache of
// anonymous memory of at most `budget` bytes (plus the slabs of the current
// call when they do not fit), read with pread from the file behind the mapping
// (macOS F_NOCACHE: the reads bypass the unified buffer cache, so the resident
// size is exactly the cache; DRAGON_EXPERT_PAGE_CACHE=1 keeps the OS file cache).
// The mapping itself is never touched.

struct dragon_pager_read {
    char   * dst;
    int      fd;
    uint64_t off; // file offset
    size_t   n;
};

// Resolves the n slabs (pointers into a read-only file mapping) to their copies
// in the cache: out[i] is the copy of ptrs[i], valid until the next call. The
// misses get a buffer and an entry in reads[] (at most n) that the caller must
// complete (dragon_pager_cache_read) before using the copies. Returns false,
// with nothing changed, if a slab is not in a file the cache can read (the
// caller pages the mapping instead). Not reentrant across concurrent ops.
bool dragon_pager_cache_get(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                            const char ** out, dragon_pager_read * reads, int * n_reads);

// reads bytes [off, off + len) of r
void dragon_pager_cache_read(const dragon_pager_read & r, size_t off, size_t len);

// false if DRAGON_EXPERT_MMAP=1 (page the file mapping, as before) or on
// platforms without the cache (Windows)
bool dragon_pager_cache_enabled();
