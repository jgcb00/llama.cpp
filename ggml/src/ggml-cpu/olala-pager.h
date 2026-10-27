#pragma once

// Expert-slab paging for the Olala fused MoE op in low-RAM mode (the expert
// tensors are a read-only file mapping read on demand: llama --lazy-mode on,
// budget --lazy-cache). Kept in its own translation unit so the OS headers
// (windows.h) stay out of ops.cpp. The paging calls are hints: they never
// change the mapped data.

#include <atomic>
#include <cstddef>
#include <cstdint>

// start reading [p, p+n) from the file in the background, in large requests
// (Linux/macOS MADV_WILLNEED, Windows PrefetchVirtualMemory)
void olala_pager_prefetch(const char * p, size_t n);

// map [p, p+n) into the page tables in one call, waiting for the reads
// (Linux >= 5.14 MADV_POPULATE_READ; no-op elsewhere, plain page faults)
void olala_pager_populate(const char * p, size_t n);

// give the pages of [p, p+n) back to the OS, rounded outward to 64 KiB
// (Linux MADV_DONTNEED only if the range is verified to still be a shared file
// mapping, macOS MADV_DONTNEED, Windows VirtualUnlock = remove from the working
// set). The data stays in the OS file cache, which is reclaimed under memory
// pressure and re-read from the file on the next access.
void olala_pager_release(const char * p, size_t n);

// process-wide LRU of resident slabs: marks the n slabs as most recently used,
// then evicts the least recently used ones until at most budget bytes remain.
// The evicted slabs are returned in out_ptrs/out_sizes (at most max_out; any
// further ones are released directly). Returns the number returned.
int olala_pager_used(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                      const char ** out_ptrs, size_t * out_sizes, int max_out);

// ---- explicit expert cache (low-memory mode on Linux and macOS)
//
// Instead of paging the file mapping, the slabs are copied into a process-wide
// cache of anonymous memory of at most `budget` bytes (more while the slabs in
// use by the running ops do not fit), read with pread from the file behind the
// mapping, bypassing the OS file cache (macOS F_NOCACHE, Linux
// POSIX_FADV_DONTNEED), so the resident size is the cache. The mapping itself
// is never touched. The entries are keyed by file (device, inode) and offset.
//
// Use from one op: get -> complete the reads (any threads) -> wait -> use the
// copies -> release. Several ops (contexts) may do this concurrently.

struct olala_pager_read {
    char                * dst;
    int                   fd;
    uint64_t              off;  // file offset
    size_t                n;
    std::atomic<size_t> * done; // bytes of dst read so far
};

// Resolves the n slabs (pointers into a read-only file mapping) to their copies
// in the cache: out[i] is the copy of ptrs[i], pinned (never evicted) until
// olala_pager_cache_release. The misses get a buffer and an entry in reads[]
// (at most n) that the caller must complete (olala_pager_cache_read). Returns
// false, with nothing pinned, if a slab is not in a file the cache can read
// (the caller pages the mapping instead).
bool olala_pager_cache_get(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                            const char ** out, olala_pager_read * reads, int * n_reads);

// reads bytes [off, off + len) of r
void olala_pager_cache_read(const olala_pager_read & r, size_t off, size_t len);

// waits until the n copies (from olala_pager_cache_get, nullptr entries are
// skipped) are completely read, including the ones another op is reading.
// Call after completing the own reads.
void olala_pager_cache_wait(const char * const * copies, int n);

// unpins the n copies (nullptr entries are skipped); evicts down to the budget
void olala_pager_cache_release(const char * const * copies, int n);

// false on platforms without the cache (Windows, iOS/tvOS/visionOS)
bool olala_pager_cache_enabled();
