#include "dragon-pager.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// a fault (or MADV_POPULATE_READ) also maps the neighbouring pages that are in
// the page cache within the aligned fault-around window (Linux
// fault_around_bytes, 64 KiB by default), i.e. the edges of the adjacent slabs:
// releases are rounded to this so no page of a slab stays mapped for good
static constexpr uintptr_t DRAGON_PAGER_RELEASE_ALIGN = 64*1024;

static uintptr_t dragon_pager_page_size() {
#if defined(_WIN32)
    static const uintptr_t ps = [] {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (uintptr_t) si.dwPageSize;
    }();
    return ps;
#elif defined(__linux__) || defined(__APPLE__)
    static const uintptr_t ps = (uintptr_t) sysconf(_SC_PAGESIZE);
    return ps;
#else
    return 4096;
#endif
}

// [*a, *b) = [p, p+n) rounded outward to align (a power of two)
static void dragon_pager_round(const char * p, size_t n, uintptr_t align, uintptr_t * a, uintptr_t * b) {
    *a = (uintptr_t) p & ~(align - 1);
    *b = ((uintptr_t) p + n + align - 1) & ~(align - 1);
}

// ---------------------------------------------------------------- prefetch

#if defined(_WIN32)
// PrefetchVirtualMemory is Windows 8+: resolved at run time
struct dragon_pager_range_entry {
    PVOID  VirtualAddress;
    SIZE_T NumberOfBytes;
};
typedef BOOL (WINAPI * dragon_pager_prefetch_fn)(HANDLE, ULONG_PTR, dragon_pager_range_entry *, ULONG);

static dragon_pager_prefetch_fn dragon_pager_prefetch_api() {
    static const dragon_pager_prefetch_fn fn = [] {
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        return k32 ? (dragon_pager_prefetch_fn) (void *) GetProcAddress(k32, "PrefetchVirtualMemory") : nullptr;
    }();
    return fn;
}
#endif

void dragon_pager_prefetch(const char * p, size_t n) {
    uintptr_t a, b;
    dragon_pager_round(p, n, dragon_pager_page_size(), &a, &b);
    if (b <= a) {
        return;
    }
#if defined(_WIN32)
    if (dragon_pager_prefetch_fn fn = dragon_pager_prefetch_api()) {
        dragon_pager_range_entry e = { (PVOID) a, (SIZE_T) (b - a) };
        fn(GetCurrentProcess(), 1, &e, 0);
    }
#elif defined(__linux__) || defined(__APPLE__)
    madvise((void *) a, b - a, MADV_WILLNEED);
#endif
}

// ---------------------------------------------------------------- populate

void dragon_pager_populate(const char * p, size_t n) {
#if defined(__linux__)
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
    static std::atomic<bool> unsupported { false };
    if (unsupported.load(std::memory_order_relaxed)) {
        return;
    }
    uintptr_t a, b;
    dragon_pager_round(p, n, dragon_pager_page_size(), &a, &b);
    if (b > a && madvise((void *) a, b - a, MADV_POPULATE_READ) != 0 && errno == EINVAL) {
        unsupported.store(true, std::memory_order_relaxed); // kernel < 5.14
    }
#else
    (void) p;
    (void) n;
#endif
}

// ---------------------------------------------------------------- release

#if defined(__linux__)
// file behind an address, from /proc/self/maps: release checks that a slab
// really lies in a shared file mapping before MADV_DONTNEED (destructive on
// anonymous memory), and can drop its pages from the page cache via the file
// (fadvise), much cheaper than MADV_PAGEOUT (TLB shootdowns page by page)
struct dragon_pager_file_map {
    struct region {
        uintptr_t beg, end;
        uint64_t  off; // file offset of beg
        int       fd;  // -1: not a shared file mapping
    };
    std::mutex          mtx;
    std::vector<region> regions;
    std::unordered_map<std::string, int> fds; // path -> fd

    static dragon_pager_file_map & get() {
        static dragon_pager_file_map m;
        return m;
    }

    void reload() {
        regions.clear();
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) {
            return;
        }
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long beg, end, off, inode;
            char perms[8], dev[32];
            int n_read = 0;
            if (sscanf(line, "%llx-%llx %7s %llx %31s %llu %n", &beg, &end, perms, &off, dev, &inode, &n_read) < 6) {
                continue;
            }
            std::string path = line + n_read;
            while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) {
                path.pop_back();
            }
            int fd = -1;
            if (perms[3] == 's' && inode != 0 && !path.empty() && path[0] == '/') {
                auto it = fds.find(path);
                if (it == fds.end()) {
                    it = fds.emplace(path, open(path.c_str(), O_RDONLY | O_CLOEXEC)).first;
                }
                fd = it->second;
            }
            regions.push_back({ (uintptr_t) beg, (uintptr_t) end, (uint64_t) off, fd });
        }
        fclose(f);
    }

    // fd and file offset of [a, b) if it lies in one shared file mapping (the
    // range may run over several VMAs of that mapping: madvise splits them)
    bool find(uintptr_t a, uintptr_t b, int * fd, uint64_t * off) {
        std::lock_guard<std::mutex> lock(mtx);
        for (int pass = 0; pass < 2; ++pass) {
            for (const region & r : regions) {
                if (a < r.beg || a >= r.end) {
                    continue;
                }
                uintptr_t      cur  = r.end;
                const region * last = &r;
                while (cur < b) {
                    const region * nx = nullptr;
                    for (const region & q : regions) {
                        if (q.beg == cur) {
                            nx = &q;
                            break;
                        }
                    }
                    if (!nx || nx->fd != r.fd || nx->off != last->off + (last->end - last->beg)) {
                        break;
                    }
                    last = nx;
                    cur  = nx->end;
                }
                if (r.fd >= 0 && cur >= b) {
                    *fd  = r.fd;
                    *off = r.off + (a - r.beg);
                    return true;
                }
                break;
            }
            if (pass == 0) {
                reload();
            }
        }
        return false;
    }
};
#endif

void dragon_pager_release(const char * p, size_t n) {
    uintptr_t a, b;
    dragon_pager_round(p, n, std::max(dragon_pager_page_size(), DRAGON_PAGER_RELEASE_ALIGN), &a, &b);
    if (b <= a) {
        return;
    }
#if defined(_WIN32)
    // VirtualUnlock on pages that are not locked removes them from the working
    // set (documented behaviour; it then fails with ERROR_NOT_LOCKED, expected).
    // They go to the standby list: free for any other use, or soft-faulted back
    // without disk I/O if the expert returns before Windows reuses them.
    VirtualUnlock((LPVOID) a, (SIZE_T) (b - a));
#elif defined(__linux__)
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
    // unmap from this process (MADV_DONTNEED, non-destructive on a shared file
    // mapping, which is checked first). The pages stay in the page cache, which
    // the kernel reclaims under pressure (or at a cgroup limit) and which serves
    // the experts that come back soon; DRAGON_EXPERT_DROP_CACHE=1 also drops them
    // from the page cache right away (strict footprint, slower: the kernel drains
    // the per-CPU page lists of every core). Unknown mappings: MADV_PAGEOUT
    // (slow, also non-destructive).
    static const bool drop_cache = [] {
        const char * e = getenv("DRAGON_EXPERT_DROP_CACHE");
        return e != nullptr && atoi(e) != 0;
    }();
    int fd;
    uint64_t off;
    if (dragon_pager_file_map::get().find(a, b, &fd, &off)) {
        madvise((void *) a, b - a, MADV_DONTNEED);
        if (drop_cache) {
            posix_fadvise(fd, (off_t) off, (off_t) (b - a), POSIX_FADV_DONTNEED);
        }
    } else {
        madvise((void *) a, b - a, MADV_PAGEOUT);
    }
#elif defined(__APPLE__)
    // non-destructive on Darwin: the pages leave the process footprint and are
    // re-read from the file if needed
    madvise((void *) a, b - a, MADV_DONTNEED);
#endif
}

// ---------------------------------------------------------------- LRU

namespace {
struct dragon_pager_lru {
    using slab = std::pair<const char *, size_t>;
    std::mutex mtx;
    std::list<slab> lru; // front = most recently used
    std::unordered_map<const char *, std::list<slab>::iterator> pos;
    size_t bytes = 0;

    static dragon_pager_lru & get() {
        static dragon_pager_lru l;
        return l;
    }
};
}

int dragon_pager_used(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                      const char ** out_ptrs, size_t * out_sizes, int max_out) {
    dragon_pager_lru & c = dragon_pager_lru::get();
    std::lock_guard<std::mutex> lock(c.mtx);
    for (int i = 0; i < n; ++i) {
        auto it = c.pos.find(ptrs[i]);
        if (it != c.pos.end()) {
            c.lru.splice(c.lru.begin(), c.lru, it->second);
        } else {
            c.lru.emplace_front(ptrs[i], sizes[i]);
            c.pos[ptrs[i]] = c.lru.begin();
            c.bytes += sizes[i];
        }
    }
    int n_out = 0;
    while (c.bytes > budget && !c.lru.empty()) {
        const dragon_pager_lru::slab s = c.lru.back();
        c.lru.pop_back();
        c.pos.erase(s.first);
        c.bytes -= s.second;
        if (n_out < max_out) {
            out_ptrs[n_out]  = s.first;
            out_sizes[n_out] = s.second;
            ++n_out;
        } else {
            dragon_pager_release(s.first, s.second);
        }
    }
    return n_out;
}
