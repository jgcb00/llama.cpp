// 64-bit off_t for pread/posix_fadvise on 32-bit Linux (checked below)
#if defined(__linux__) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS 64
#endif

#include "olala-pager.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

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

// explicit expert cache: needs the file behind an address (Linux /proc/self/maps, macOS proc_pidinfo)
#if defined(__linux__)
#include <sys/sysmacros.h>
#define OLALA_PAGER_CACHE 1
#elif defined(__APPLE__) && TARGET_OS_OSX
#include <libproc.h>
#define OLALA_PAGER_CACHE 1
#else
#define OLALA_PAGER_CACHE 0
#endif

// a fault (or MADV_POPULATE_READ) also maps the neighbouring pages that are in
// the page cache within the aligned fault-around window (Linux
// fault_around_bytes, 64 KiB by default), i.e. the edges of the adjacent slabs:
// releases are rounded to this so no page of a slab stays mapped for good
static constexpr uintptr_t OLALA_PAGER_RELEASE_ALIGN = 64*1024;

static uintptr_t olala_pager_page_size() {
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
static void olala_pager_round(const char * p, size_t n, uintptr_t align, uintptr_t * a, uintptr_t * b) {
    *a = (uintptr_t) p & ~(align - 1);
    *b = ((uintptr_t) p + n + align - 1) & ~(align - 1);
}

// ---------------------------------------------------------------- prefetch

#if defined(_WIN32)
// PrefetchVirtualMemory is Windows 8+: resolved at run time
struct olala_pager_range_entry {
    PVOID  VirtualAddress;
    SIZE_T NumberOfBytes;
};
typedef BOOL (WINAPI * olala_pager_prefetch_fn)(HANDLE, ULONG_PTR, olala_pager_range_entry *, ULONG);

static olala_pager_prefetch_fn olala_pager_prefetch_api() {
    static const olala_pager_prefetch_fn fn = [] {
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        return k32 ? (olala_pager_prefetch_fn) (void *) GetProcAddress(k32, "PrefetchVirtualMemory") : nullptr;
    }();
    return fn;
}
#endif

void olala_pager_prefetch(const char * p, size_t n) {
    uintptr_t a, b;
    olala_pager_round(p, n, olala_pager_page_size(), &a, &b);
    if (b <= a) {
        return;
    }
#if defined(_WIN32)
    if (olala_pager_prefetch_fn fn = olala_pager_prefetch_api()) {
        olala_pager_range_entry e = { (PVOID) a, (SIZE_T) (b - a) };
        fn(GetCurrentProcess(), 1, &e, 0);
    }
#elif defined(__linux__) || defined(__APPLE__)
    madvise((void *) a, b - a, MADV_WILLNEED);
#endif
}

// ---------------------------------------------------------------- populate

void olala_pager_populate(const char * p, size_t n) {
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    static std::atomic<bool> unsupported { false };
    if (unsupported.load(std::memory_order_relaxed)) {
        return;
    }
    uintptr_t a, b;
    olala_pager_round(p, n, olala_pager_page_size(), &a, &b);
    if (b > a && madvise((void *) a, b - a, MADV_POPULATE_READ) != 0 && errno == EINVAL) {
        unsupported.store(true, std::memory_order_relaxed); // kernel < 5.14
    }
#else
    (void) p;
    (void) n;
#endif
}

// ---------------------------------------------------------------- file behind an address
//
// A mapping can be freed (model unloaded) and its addresses reused by another
// mapping, so a slab is identified by its file (device, inode) and offset, and
// the region behind an address is checked again on every use:
// - macOS: proc_pidinfo(PROC_PIDREGIONPATHINFO) queries the live VM map
// - Linux: the VMAs of /proc/self/maps are memoized; a VMA is used only if
//   readlink(/proc/self/map_files/<beg>-<end>) still gives its path, else
//   /proc/self/maps is read again
// A batch of slabs checks each region once (see olala_pager_locate).

#if OLALA_PAGER_CACHE

struct olala_pager_region {
    uintptr_t   beg    = 0;
    uintptr_t   end    = 0;
    uint64_t    off    = 0; // file offset of beg
    uint64_t    dev    = 0;
    uint64_t    ino    = 0; // 0: not a file mapping
    bool        shared = false;
    std::string path;
};

#if defined(__linux__)
namespace {
struct olala_pager_maps {
    std::mutex mtx;
    std::vector<olala_pager_region> regions; // sorted by address

    static olala_pager_maps & get() {
        static olala_pager_maps m;
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
            olala_pager_region r;
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

    const olala_pager_region * find(uintptr_t p) const {
        auto it = std::upper_bound(regions.begin(), regions.end(), p,
            [](uintptr_t v, const olala_pager_region & r) { return v < r.beg; });
        if (it == regions.begin()) {
            return nullptr;
        }
        --it;
        return p < it->end ? &*it : nullptr;
    }

    // the file VMA r is still mapped at the same place
    static bool still_mapped(const olala_pager_region & r) {
        if (r.ino == 0) {
            return false;
        }
        char name[64];
        snprintf(name, sizeof(name), "/proc/self/map_files/%llx-%llx", (unsigned long long) r.beg, (unsigned long long) r.end);
        char buf[4096];
        const ssize_t k = readlink(name, buf, sizeof(buf));
        return k > 0 && (size_t) k == r.path.size() && memcmp(buf, r.path.data(), (size_t) k) == 0;
    }

    bool at(uintptr_t p, olala_pager_region * out) {
        std::lock_guard<std::mutex> lock(mtx);
        const olala_pager_region * r = find(p);
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
static bool olala_pager_region_at(uintptr_t p, olala_pager_region * r) {
    return olala_pager_maps::get().at(p, r);
}
#else
static bool olala_pager_region_at(uintptr_t p, olala_pager_region * r) {
    struct proc_regionwithpathinfo info;
    if (proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, (uint64_t) p, &info, sizeof(info)) != (int) sizeof(info)) {
        return false;
    }
    r->beg    = (uintptr_t) info.prp_prinfo.pri_address;
    r->end    = r->beg + (uintptr_t) info.prp_prinfo.pri_size;
    r->off    = info.prp_prinfo.pri_offset;
    r->path   = info.prp_vip.vip_path;
    r->dev    = (uint64_t) info.prp_vip.vip_vi.vi_stat.vst_dev;
    r->ino    = !r->path.empty() && r->path[0] == '/' ? (uint64_t) info.prp_vip.vip_vi.vi_stat.vst_ino : 0;
    r->shared = true; // not checked on Darwin (MADV_DONTNEED is not destructive)
    // the call returns the next region if p is not mapped
    return p >= r->beg && p < r->end;
}
#endif

struct olala_pager_file {
    uint64_t    dev = 0, ino = 0;
    uint64_t    off = 0; // file offset of the start of the range
    std::string path;
};

// file and offset behind [a, b), which may run over several regions (madvise
// splits them) as long as they continue the same file. checked: the regions
// already checked in the current batch (nullptr: check them all)
static bool olala_pager_locate(uintptr_t a, uintptr_t b, bool need_shared, olala_pager_file * f,
                                std::vector<olala_pager_region> * checked) {
    uintptr_t cur = a;
    while (cur < b) {
        const olala_pager_region * r = nullptr;
        olala_pager_region tmp;
        if (checked) {
            for (const olala_pager_region & q : *checked) {
                if (cur >= q.beg && cur < q.end) {
                    r = &q;
                    break;
                }
            }
        }
        if (r == nullptr) {
            if (!olala_pager_region_at(cur, &tmp)) {
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

#endif // OLALA_PAGER_CACHE

// ---------------------------------------------------------------- release

void olala_pager_release(const char * p, size_t n) {
    uintptr_t a, b;
    olala_pager_round(p, n, std::max(olala_pager_page_size(), OLALA_PAGER_RELEASE_ALIGN), &a, &b);
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
    olala_pager_file f;
    if (olala_pager_locate(a, b, /*need_shared =*/ true, &f, nullptr)) {
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
struct olala_pager_lru {
    using slab = std::pair<const char *, size_t>;
    std::mutex mtx;
    std::list<slab> lru; // front = most recently used
    std::unordered_map<const char *, std::list<slab>::iterator> pos;
    size_t bytes = 0;

    static olala_pager_lru & get() {
        static olala_pager_lru l;
        return l;
    }
};
}

int olala_pager_used(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                      const char ** out_ptrs, size_t * out_sizes, int max_out) {
    olala_pager_lru & c = olala_pager_lru::get();
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
        const olala_pager_lru::slab s = c.lru.back();
        c.lru.pop_back();
        c.pos.erase(s.first);
        c.bytes -= s.second;
        if (n_out < max_out) {
            out_ptrs[n_out]  = s.first;
            out_sizes[n_out] = s.second;
            ++n_out;
        } else {
            olala_pager_release(s.first, s.second);
        }
    }
    return n_out;
}

// ---------------------------------------------------------------- explicit cache

#if OLALA_PAGER_CACHE

static_assert(sizeof(off_t) == 8, "the expert cache needs a 64-bit off_t for pread");

namespace {
struct olala_pager_key {
    uint64_t dev, ino, off;
    size_t   n;

    bool operator==(const olala_pager_key & o) const {
        return dev == o.dev && ino == o.ino && off == o.off && n == o.n;
    }
};

struct olala_pager_key_hash {
    size_t operator()(const olala_pager_key & k) const {
        uint64_t h = k.dev*0x9e3779b97f4a7c15ULL;
        h = (h ^ k.ino)*0x9e3779b97f4a7c15ULL;
        h = (h ^ k.off)*0x9e3779b97f4a7c15ULL;
        h = (h ^ (uint64_t) k.n)*0x9e3779b97f4a7c15ULL;
        return (size_t) (h ^ (h >> 32));
    }
};

struct olala_pager_cache {
    struct entry {
        char * buf  = nullptr;
        size_t n    = 0;
        int    pins = 0; // ops using it: never evicted while > 0
        std::unique_ptr<std::atomic<size_t>> done; // bytes read
        std::list<olala_pager_key>::iterator it;  // position in lru, valid when pins == 0
    };
    struct file {
        int      fd;
        uint64_t call; // last call that used it
    };

    std::mutex mtx;
    std::unordered_map<olala_pager_key, entry, olala_pager_key_hash> map;
    std::unordered_map<const char *, olala_pager_key> by_buf;
    std::list<olala_pager_key> lru; // unpinned entries, front = most recently used
    std::map<std::pair<uint64_t, uint64_t>, file> files; // (dev, ino) -> fd
    size_t   bytes    = 0;
    size_t   budget   = 0;
    int64_t  n_pinned = 0;
    uint64_t call     = 0;

    // scratch of olala_pager_cache_get
    std::vector<olala_pager_region> checked;
    std::vector<olala_pager_key>    keys;
    std::vector<int>                 key_fd;

    // buffers of evicted slabs, reused by the misses of the same call. Slabs are
    // anonymous mappings, unmapped when evicted: malloc keeps large freed blocks
    // around (macOS), the resident size would not follow the budget
    std::vector<std::pair<char *, size_t>> spare;

    static olala_pager_cache & get() {
        static olala_pager_cache c;
        return c;
    }

    // fd of the file f (opened with the OS file cache bypassed), -1 if f can not be opened
    int open_file(const olala_pager_file & f, bool * opened) {
        auto it = files.find({ f.dev, f.ino });
        if (it != files.end()) {
            it->second.call = call;
            return it->second.fd;
        }
        const int fd = open(f.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return -1;
        }
        // the path can name another file now (replaced on disk)
        struct stat st;
        if (fstat(fd, &st) != 0 || (uint64_t) st.st_dev != f.dev || (uint64_t) st.st_ino != f.ino) {
            close(fd);
            return -1;
        }
#if defined(__APPLE__)
        fcntl(fd, F_NOCACHE, 1);
#endif
        files.emplace(std::make_pair(f.dev, f.ino), file { fd, call });
        *opened = true;
        return fd;
    }

    // close the files not used by this call. Only when no op holds a pin, i.e.
    // no read of another op is in flight.
    void close_unused() {
        if (n_pinned != 0) {
            return;
        }
        for (auto it = files.begin(); it != files.end(); ) {
            if (it->second.call != call) {
                close(it->second.fd);
                it = files.erase(it);
            } else {
                ++it;
            }
        }
    }

    void pin(entry & e) {
        if (e.pins++ == 0) {
            lru.erase(e.it);
        }
        n_pinned++;
    }

    void evict_back() {
        auto it = map.find(lru.back());
        lru.pop_back();
        bytes -= it->second.n;
        spare.emplace_back(it->second.buf, it->second.n);
        by_buf.erase(it->second.buf);
        map.erase(it);
    }

    // evict unpinned entries down to the budget (stays over it if the pinned ones do not fit)
    void trim() {
        while (bytes > budget && !lru.empty()) {
            evict_back();
        }
    }

    void free_spare() {
        for (auto & b : spare) {
            munmap(b.first, b.second);
        }
        spare.clear();
    }

    char * alloc(size_t n) {
        for (size_t i = 0; i < spare.size(); ++i) {
            if (spare[i].second == n) {
                char * b = spare[i].first;
                spare[i] = spare.back();
                spare.pop_back();
                return b;
            }
        }
        void * m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        char * b = m == MAP_FAILED ? nullptr : (char *) m;
        if (b == nullptr) {
            GGML_LOG_ERROR("%s: olala expert cache: failed to allocate %zu bytes\n", __func__, n);
            GGML_ABORT("out of memory");
        }
        return b;
    }
};
}

bool olala_pager_cache_enabled() {
    return true;
}

bool olala_pager_cache_get(const char * const * ptrs, const size_t * sizes, int n, size_t budget,
                            const char ** out, olala_pager_read * reads, int * n_reads) {
    olala_pager_cache & c = olala_pager_cache::get();
    std::lock_guard<std::mutex> lock(c.mtx);

    c.call++;
    c.checked.clear();
    c.keys.resize(n);
    c.key_fd.resize(n);
    bool opened = false;
    for (int i = 0; i < n; ++i) {
        olala_pager_file f;
        if (!olala_pager_locate((uintptr_t) ptrs[i], (uintptr_t) ptrs[i] + sizes[i], /*need_shared =*/ false, &f, &c.checked)) {
            return false;
        }
        c.key_fd[i] = c.open_file(f, &opened);
        if (c.key_fd[i] < 0) {
            return false;
        }
        c.keys[i] = { f.dev, f.ino, f.off, sizes[i] };
    }
    if (opened) {
        c.close_unused();
    }

    c.budget = budget;
    *n_reads = 0;
    for (int i = 0; i < n; ++i) {
        auto it = c.map.find(c.keys[i]);
        if (it != c.map.end()) {
            c.pin(it->second);
            out[i] = it->second.buf;
            continue;
        }
        // make room: least recently used slabs that no op uses
        while (c.bytes + sizes[i] > budget && !c.lru.empty()) {
            c.evict_back();
        }
        olala_pager_cache::entry & e = c.map[c.keys[i]];
        e.buf  = c.alloc(sizes[i]);
        e.n    = sizes[i];
        e.pins = 1;
        e.done = std::make_unique<std::atomic<size_t>>(0);
        c.by_buf[e.buf] = c.keys[i];
        c.bytes += sizes[i];
        c.n_pinned++;
        out[i] = e.buf;
        reads[(*n_reads)++] = { e.buf, c.key_fd[i], c.keys[i].off, sizes[i], e.done.get() };
    }
    c.trim();
    c.free_spare();
    return true;
}

void olala_pager_cache_read(const olala_pager_read & r, size_t off, size_t len) {
    size_t done = 0;
    while (done < len) {
        const ssize_t k = pread(r.fd, r.dst + off + done, len - done, (off_t) (r.off + off + done));
        if (k <= 0) {
            if (k < 0 && errno == EINTR) {
                continue;
            }
            GGML_LOG_ERROR("%s: pread of %zu bytes at file offset %llu failed (%s)\n", __func__,
                len - done, (unsigned long long) (r.off + off + done), k < 0 ? strerror(errno) : "end of file");
            GGML_ABORT("olala expert cache: read failed");
        }
        done += (size_t) k;
    }
#if defined(__linux__)
    // bypass the page cache, so the resident size is the cache
    posix_fadvise(r.fd, (off_t) (r.off + off), (off_t) len, POSIX_FADV_DONTNEED);
#endif
    r.done->fetch_add(len, std::memory_order_release);
}

void olala_pager_cache_wait(const char * const * copies, int n) {
    olala_pager_cache & c = olala_pager_cache::get();
    for (int i = 0; i < n; ++i) {
        if (copies[i] == nullptr) {
            continue;
        }
        std::atomic<size_t> * done;
        size_t size;
        {
            std::lock_guard<std::mutex> lock(c.mtx);
            const auto it = c.by_buf.find(copies[i]);
            GGML_ASSERT(it != c.by_buf.end());
            const olala_pager_cache::entry & e = c.map.at(it->second);
            done = e.done.get();
            size = e.n;
        }
        // pinned by the caller: e stays valid
        while (done->load(std::memory_order_acquire) < size) {
            std::this_thread::yield();
        }
    }
}

void olala_pager_cache_release(const char * const * copies, int n) {
    olala_pager_cache & c = olala_pager_cache::get();
    std::lock_guard<std::mutex> lock(c.mtx);
    for (int i = 0; i < n; ++i) {
        if (copies[i] == nullptr) {
            continue;
        }
        const auto it = c.by_buf.find(copies[i]);
        GGML_ASSERT(it != c.by_buf.end());
        olala_pager_cache::entry & e = c.map.at(it->second);
        GGML_ASSERT(e.pins > 0);
        if (--e.pins == 0) {
            c.lru.push_front(it->second);
            e.it = c.lru.begin();
        }
        c.n_pinned--;
    }
    c.trim();
    c.free_spare();
}

#else // no explicit cache

bool olala_pager_cache_enabled() {
    return false;
}

bool olala_pager_cache_get(const char * const *, const size_t *, int, size_t, const char **, olala_pager_read *, int *) {
    return false;
}

void olala_pager_cache_read(const olala_pager_read &, size_t, size_t) {
}

void olala_pager_cache_wait(const char * const *, int) {
}

void olala_pager_cache_release(const char * const *, int) {
}

#endif
