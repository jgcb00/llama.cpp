// Low-RAM mode of the Dragon fused MoE op (GGML_OP_DRAGON_MOE, expert_cache_mib >= 0):
// the expert tensors live in a read-only file mapping, as with llama --lazy-mode on.
// Checks, at the real Olala dimensions (256 experts, ~150 MB of expert weights):
//   1. the output is bitwise identical to the same op on in-RAM weights
//   2. after the op the expert pages are no longer resident in the process
//      (working set / RSS grows by far less than the experts that were read)
//   3. control: the same mapped weights without paging (expert_cache_mib = -1)
//      do grow the resident set, so the measurement in 2 is meaningful
//   4. a cache budget keeps roughly that much resident
// Runs on Linux (MADV_DONTNEED) and Windows (VirtualUnlock); on macOS the resident
// size excludes file-backed pages, so only the numerics are checked there.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#endif

static const int64_t N_EMBD = 1536, N_LAT = 384, N_EXPERT = 256, N_FF = 768, N_SH = 1536, K = 6;
static const ggml_type T_EXP = GGML_TYPE_Q8_0; // fast to quantize; the paging does not depend on the type

static double resident_mib() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return -1;
    }
    return pmc.WorkingSetSize/1048576.0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) {
        return -1;
    }
    return info.resident_size/1048576.0;
#else
    FILE * f = fopen("/proc/self/statm", "r");
    if (!f) {
        return -1;
    }
    long size = 0, res = 0;
    const int n = fscanf(f, "%ld %ld", &size, &res);
    fclose(f);
    return n == 2 ? res*(double) sysconf(_SC_PAGESIZE)/1048576.0 : -1;
#endif
}

// read-only mapping of a whole file, like llama_mmap
struct file_map {
    void * addr = nullptr;
    size_t size = 0;
#if defined(_WIN32)
    HANDLE hfile = INVALID_HANDLE_VALUE, hmap = nullptr;
#else
    int fd = -1;
#endif

    bool open(const std::string & path, size_t n) {
        size = n;
#if defined(_WIN32)
        hfile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hfile == INVALID_HANDLE_VALUE) return false;
        hmap = CreateFileMappingA(hfile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hmap) return false;
        addr = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
        return addr != nullptr;
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        addr = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) { addr = nullptr; return false; }
        posix_madvise(addr, n, POSIX_MADV_RANDOM); // what llama does for lazy ranges
        return true;
#endif
    }

    ~file_map() {
#if defined(_WIN32)
        if (addr) UnmapViewOfFile(addr);
        if (hmap) CloseHandle(hmap);
        if (hfile != INVALID_HANDLE_VALUE) CloseHandle(hfile);
#else
        if (addr) munmap(addr, size);
        if (fd >= 0) close(fd);
#endif
    }
};

static std::vector<uint8_t> random_quantized(std::mt19937 & rng, ggml_type type, int64_t ne0, int64_t nrows, float scale) {
    std::vector<float> src((size_t) (ne0*nrows));
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto & v : src) v = nd(rng);
    std::vector<uint8_t> dst(ggml_row_size(type, ne0)*nrows);
    ggml_quantize_chunk(type, src.data(), dst.data(), 0, nrows, ne0, nullptr);
    return dst;
}

static ggml_tensor * f32_tensor(ggml_context * ctx, std::mt19937 & rng, int64_t ne0, int64_t ne1, float scale) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    std::normal_distribution<float> nd(0.0f, scale);
    for (int64_t i = 0; i < ne0*ne1; ++i) ((float *) t->data)[i] = nd(rng);
    return t;
}

static ggml_tensor * q_tensor(ggml_context * ctx, std::mt19937 & rng, ggml_type type, int64_t ne0, int64_t ne1, float scale) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    std::vector<uint8_t> q = random_quantized(rng, type, ne0, ne1, scale);
    memcpy(t->data, q.data(), q.size());
    return t;
}

struct weights {
    ggml_tensor * x, * ld, * r, * bias, * lu, * us, * ds;
};

// one forward of the fused op; returns the output (n_embd x T)
static std::vector<float> run_op(const weights & w, ggml_tensor * x, ggml_tensor * up, ggml_tensor * down, int cache_mib, int n_threads) {
    ggml_init_params ip = { (size_t) 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * out = ggml_dragon_moe(ctx, x, w.ld, w.r, w.bias, up, down, w.lu, w.us, w.ds, (int) K, 2.45f, cache_mib);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    std::vector<float> res((const float *) out->data, (const float *) out->data + ggml_nelements(out));
    ggml_free(ctx);
    return res;
}

int main() {
    const int n_threads = (int) std::max(2u, std::min(8u, std::thread::hardware_concurrency()));
    std::mt19937 rng(1234);
    int fails = 0;

    // ---- expert weights -> temporary file (page-aligned tensors)
    const size_t up_bytes   = ggml_row_size(T_EXP, N_LAT)*N_FF*N_EXPERT;
    const size_t down_bytes = ggml_row_size(T_EXP, N_FF)*N_LAT*N_EXPERT;
    const size_t up_off = 0, down_off = GGML_PAD(up_bytes, 65536), file_size = down_off + GGML_PAD(down_bytes, 65536);
    const std::string path = (std::filesystem::temp_directory_path() / ("test-dragon-moe-paging-" + std::to_string(rng()) + ".bin")).string();
    std::vector<uint8_t> up_q   = random_quantized(rng, T_EXP, N_LAT, N_FF*N_EXPERT, 0.05f);
    std::vector<uint8_t> down_q = random_quantized(rng, T_EXP, N_FF, N_LAT*N_EXPERT, 0.05f);
    {
        std::vector<uint8_t> file(file_size, 0);
        memcpy(file.data() + up_off, up_q.data(), up_bytes);
        memcpy(file.data() + down_off, down_q.data(), down_bytes);
        FILE * f = fopen(path.c_str(), "wb");
        if (!f || fwrite(file.data(), 1, file.size(), f) != file.size()) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            return 1;
        }
        fclose(f);
    }
    printf("expert weights: %.1f MiB in %s, %d threads\n", (up_bytes + down_bytes)/1048576.0, path.c_str(), n_threads);

    {
        // ---- small weights in RAM
        ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
        ggml_context * wctx = ggml_init(ip);
        weights w;
        w.ld   = q_tensor(wctx, rng, GGML_TYPE_Q8_0, N_EMBD, N_LAT, 0.03f);
        w.r    = f32_tensor(wctx, rng, N_EMBD, N_EXPERT, 0.02f); // unsaturated sigmoid: diverse routing
        w.bias = f32_tensor(wctx, rng, N_EXPERT, 1, 0.01f);
        w.lu   = q_tensor(wctx, rng, GGML_TYPE_Q8_0, N_LAT, N_EMBD, 0.05f);
        w.us   = q_tensor(wctx, rng, GGML_TYPE_Q8_0, N_EMBD, N_SH, 0.03f);
        w.ds   = q_tensor(wctx, rng, GGML_TYPE_Q8_0, N_SH, N_EMBD, 0.03f);
        const int64_t T = 64;
        ggml_tensor * x  = f32_tensor(wctx, rng, N_EMBD, T, 1.0f);

        // ---- expert tensors pointing into the mapping
        file_map fm;
        if (!fm.open(path, file_size)) {
            fprintf(stderr, "cannot map %s\n", path.c_str());
            return 1;
        }
        ggml_init_params mp = { 2*ggml_tensor_overhead(), nullptr, true };
        ggml_context * mctx = ggml_init(mp);
        ggml_tensor * up_m   = ggml_new_tensor_3d(mctx, T_EXP, N_LAT, N_FF, N_EXPERT);
        ggml_tensor * down_m = ggml_new_tensor_3d(mctx, T_EXP, N_FF, N_LAT, N_EXPERT);
        up_m->data   = (char *) fm.addr + up_off;
        down_m->data = (char *) fm.addr + down_off;

        // ---- 1. reference on in-RAM copies (freed before the measurements)
        std::vector<float> ref;
        {
            ggml_init_params rp = { up_bytes + down_bytes + 4096, nullptr, false };
            ggml_context * rctx = ggml_init(rp);
            ggml_tensor * up_r   = ggml_new_tensor_3d(rctx, T_EXP, N_LAT, N_FF, N_EXPERT);
            ggml_tensor * down_r = ggml_new_tensor_3d(rctx, T_EXP, N_FF, N_LAT, N_EXPERT);
            memcpy(up_r->data, up_q.data(), up_bytes);
            memcpy(down_r->data, down_q.data(), down_bytes);
            ref = run_op(w, x, up_r, down_r, -1, n_threads);
            ggml_free(rctx);
        }
        std::vector<uint8_t>().swap(up_q);
        std::vector<uint8_t>().swap(down_q);
        run_op(w, x, up_m, down_m, 0, n_threads); // warm up allocations

        // ---- 2. paged run: exact, and the experts do not stay resident
        const double r0 = resident_mib();
        std::vector<float> paged = run_op(w, x, up_m, down_m, 0, n_threads);
        for (int step = 0; step < 16; ++step) { // decode-like single tokens
            ggml_tensor * xt = ggml_view_2d(wctx, x, N_EMBD, 1, x->nb[1], (size_t) step*x->nb[1]);
            run_op(w, xt, up_m, down_m, 0, n_threads);
        }
        const double r1 = resident_mib();
        const bool exact = paged.size() == ref.size() && memcmp(paged.data(), ref.data(), ref.size()*sizeof(float)) == 0;
        printf("1. paged output %s the in-RAM reference\n", exact ? "bitwise equal to" : "DIFFERS from");
        fails += !exact;

        // ---- 3. control: same mapping without paging
        std::vector<float> resident = run_op(w, x, up_m, down_m, -1, n_threads);
        const double r2 = resident_mib();
        fails += memcmp(resident.data(), ref.data(), ref.size()*sizeof(float)) != 0;

        // ---- 4. budget: release, then keep at most 32 MiB of experts
        run_op(w, x, up_m, down_m, 0, n_threads);   // drop what the control mapped
        const double r3 = resident_mib();
        run_op(w, x, up_m, down_m, 32, n_threads);
        const double r4 = resident_mib();

        const double touched = (up_bytes + down_bytes)/1048576.0;
        printf("2. resident growth with paging (T=64 + 16 decode steps): %+.1f MiB\n", r1 - r0);
        printf("3. resident growth without paging (control):             %+.1f MiB (experts: %.1f MiB)\n", r2 - r1, touched);
        printf("4. resident growth with a 32 MiB expert budget:          %+.1f MiB\n", r4 - r3);
        bool measurable = r0 >= 0;
#if defined(__APPLE__)
        // the macOS task resident size / footprint does not include clean
        // file-backed pages (they count as reclaimable file cache), so mapped
        // experts never show up in it: only the numerics can be checked here
        if (measurable && r2 - r1 < 0.3*touched) {
            printf("the resident size of this OS does not count file-backed pages: memory checks skipped\n");
            measurable = false;
        }
#endif
        if (!measurable) {
            if (r0 < 0) {
                printf("cannot read the resident set size: memory checks skipped\n");
            }
        } else {
            const bool control_ok = r2 - r1 > 0.3*touched;
            const bool paged_ok   = r1 - r0 < 16.0;
            const bool budget_ok  = r4 - r3 < 32.0 + 16.0;
            printf("   control grows: %s, paged stays flat: %s, budget respected: %s\n",
                   control_ok ? "yes" : "NO", paged_ok ? "yes" : "NO", budget_ok ? "yes" : "NO");
            fails += !control_ok + !paged_ok + !budget_ok;
        }
        ggml_free(mctx);
        ggml_free(wctx);
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);

    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
