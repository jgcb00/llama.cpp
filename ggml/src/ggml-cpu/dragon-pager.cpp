#include "dragon-pager.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include <sys/stat.h>
#include <unistd.h>
#endif

// Linux release: needs the file behind an address (/proc/self/maps)
#if defined(__linux__)
#include <sys/sysmacros.h>
#define DRAGON_PAGER_LOCATE 1
#else
#define DRAGON_PAGER_LOCATE 0
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
#if defined(__linux__) && defined(MADV_POPULATE_READ)
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

// ---------------------------------------------------------------- file behind an address (Linux)
//
// A mapping can be freed (model unloaded) and its addresses reused by another
// mapping, so the region behind an address is checked again on every use: the
// VMAs of /proc/self/maps are memoized; a VMA is used only if
// readlink(/proc/self/map_files/<beg>-<end>) still gives its path, else
// /proc/self/maps is read again.

#if DRAGON_PAGER_LOCATE

struct dragon_pager_region {
    uintptr_t   beg    = 0;
    uintptr_t   end    = 0;
    uint64_t    off    = 0; // file offset of beg
    uint64_t    dev    = 0;
    uint64_t    ino    = 0; // 0: not a file mapping
    bool        shared = false;
    std::string path;
};

namespace {
struct dragon_pager_maps {
    std::mutex mtx;
    std::vector<dragon_pager_region> regions; // sorted by address

    static dragon_pager_maps & get() {
        static dragon_pager_maps m;
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
            unsigned int dev_major, dev_minor;
            char perms[8];
            int n_read = 0;
            if (sscanf(line, "%llx-%llx %7s %llx %x:%x %llu %n", &beg, &end, perms, &off, &dev_major, &dev_minor, &inode, &n_read) < 7) {
                continue;
            }
            dragon_pager_region r;
            r.beg    = (uintptr_t) beg;
            r.end    = (uintptr_t) end;
            r.off    = (uint64_t) off;
            r.dev    = (uint64_t) makedev(dev_major, dev_minor);
            r.shared = perms[3] == 's';
            r.path   = line + n_read;
            while (!r.path.empty() && (r.path.back() == '\n' || r.path.back() == ' ')) {
                r.path.pop_back();
            }
            r.ino = !r.path.empty() && r.path[0] == '/' ? (uint64_t) inode : 0;
            regions.push_back(std::move(r));
        }
        fclose(f);
    }

    const dragon_pager_region * find(uintptr_t p) const {
        auto it = std::upper_bound(regions.begin(), regions.end(), p,
            [](uintptr_t v, const dragon_pager_region & r) { return v < r.beg; });
        if (it == regions.begin()) {
            return nullptr;
        }
        --it;
        return p < it->end ? &*it : nullptr;
    }

    // the file VMA r is still mapped at the same place
    static bool still_mapped(const dragon_pager_region & r) {
        if (r.ino == 0) {
            return false;
        }
        char name[64];
        snprintf(name, sizeof(name), "/proc/self/map_files/%llx-%llx", (unsigned long long) r.beg, (unsigned long long) r.end);
        char buf[4096];
        const ssize_t k = readlink(name, buf, sizeof(buf));
        return k > 0 && (size_t) k == r.path.size() && memcmp(buf, r.path.data(), (size_t) k) == 0;
    }

    bool at(uintptr_t p, dragon_pager_region * out) {
        std::lock_guard<std::mutex> lock(mtx);
        const dragon_pager_region * r = find(p);
        if (r == nullptr || !still_mapped(*r)) {
            reload();
            r = find(p);
        }
        if (r == nullptr) {
            return false;
        }
        *out = *r;
        return true;
    }
};
}

// current region that contains p
static bool dragon_pager_region_at(uintptr_t p, dragon_pager_region * r) {
    return dragon_pager_maps::get().at(p, r);
}

struct dragon_pager_file {
    uint64_t    dev = 0, ino = 0;
    uint64_t    off = 0; // file offset of the start of the range
    std::string path;
};

// file and offset behind [a, b), which may run over several regions (madvise
// splits them) as long as they continue the same file. checked: the regions
// already checked in the current batch (nullptr: check them all)
static bool dragon_pager_locate(uintptr_t a, uintptr_t b, bool need_shared, dragon_pager_file * f,
                                std::vector<dragon_pager_region> * checked) {
    uintptr_t cur = a;
    while (cur < b) {
        const dragon_pager_region * r = nullptr;
        dragon_pager_region tmp;
        if (checked) {
            for (const dragon_pager_region & q : *checked) {
                if (cur >= q.beg && cur < q.end) {
                    r = &q;
                    break;
                }
            }
        }
        if (r == nullptr) {
            if (!dragon_pager_region_at(cur, &tmp)) {
                return false;
            }
            if (checked) {
                checked->push_back(tmp);
                r = &checked->back();
            } else {
                r = &tmp;
            }
        }
        if (r->ino == 0 || (need_shared && !r->shared)) {
            return false;
        }
        const uint64_t o = r->off + (cur - r->beg);
        if (cur == a) {
            f->dev  = r->dev;
            f->ino  = r->ino;
            f->off  = o;
            f->path = r->path;
        } else if (r->dev != f->dev || r->ino != f->ino || o != f->off + (cur - a)) {
            return false;
        }
        cur = r->end;
    }
    return true;
}

#endif // DRAGON_PAGER_LOCATE

// ---------------------------------------------------------------- release

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
    // unmap from this process. MADV_DONTNEED is not destructive only on a shared
    // file mapping: do nothing unless the range is still one. The pages stay in
    // the page cache, which the kernel reclaims under pressure.
    dragon_pager_file f;
    if (dragon_pager_locate(a, b, /*need_shared =*/ true, &f, nullptr)) {
        madvise((void *) a, b - a, MADV_DONTNEED);
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
