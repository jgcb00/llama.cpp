#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#include <mach/mach_time.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy;  // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // GGML_METAL_SPIN_WAIT=1: the last command buffer of each submission signals spin_ev with an increasing value and
    // ggml_metal_synchronize busy-waits on the event instead of sleeping in waitUntilCompleted (saves the ~0.2 ms
    // wake-up latency per token; costs a busy CPU core while waiting)
    bool                 spin;
    id<MTLSharedEvent>   spin_ev;
    uint64_t             spin_seq;
    id<MTLCommandBuffer> spin_cb;     // command buffer that signals spin_val (weak, compared with cmd_buf_last)
    uint64_t             spin_val;

    // GGML_METAL_SPIN_WAIT=2: the blit of get_tensor_async also writes an incrementing byte into flag_buf (shared
    // storage) after the copy, and ggml_metal_synchronize polls that byte (no command buffer completion involved)
    id<MTLBuffer>        flag_buf;
    id<MTLFence>         flag_fence;
    id<MTLCommandBuffer> flag_cb;     // weak, compared with cmd_buf_last
    uint8_t              flag_val;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
        if (queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

        {
            const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        res->cmd_buf_last = nil;

        {
            const char * e = getenv("GGML_METAL_SPIN_WAIT");
            res->spin = e && e[0] == '1';
            res->flag_buf = (e && e[0] == '2') ? [(id<MTLDevice>) ggml_metal_device_get_obj(dev) newBufferWithLength:64 options:MTLResourceStorageModeShared] : nil;
            if (res->flag_buf) {
                memset([res->flag_buf contents], 0, 64);
                GGML_LOG_INFO("%s: busy-wait synchronization on a shared-memory flag (GGML_METAL_SPIN_WAIT=2)\n", __func__);
            }
            res->flag_fence = res->flag_buf ? [(id<MTLDevice>) ggml_metal_device_get_obj(dev) newFence] : nil;
            res->flag_cb  = nil;
            res->flag_val = 0;

            res->spin_ev  = res->spin ? [ggml_metal_device_get_obj(dev) newSharedEvent] : nil;
            res->spin_seq = 0;
            res->spin_cb  = nil;
            res->spin_val = 0;
            if (res->spin) {
                GGML_LOG_INFO("%s: busy-wait synchronization (GGML_METAL_SPIN_WAIT)\n", __func__);
            }
        }

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->flag_buf) {
        [ctx->flag_buf release];
        [ctx->flag_fence release];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

// spin mode: make cmd_buf (about to be committed, and the last one queued) signal the next event value
static void ggml_metal_spin_signal(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    if (!ctx->spin) {
        return;
    }
    [cmd_buf encodeSignalEvent:ctx->spin_ev value:++ctx->spin_seq];
    ctx->spin_cb  = cmd_buf;
    ctx->spin_val = ctx->spin_seq;
}

static double ggml_metal_host_time_s(void);

// GGML_METAL_GAPLOG=1 (debug): host-side timeline of the GPU idle gap between graphs, without extra waits:
// GPU end -> wake (synchronize returns), wake -> next graph_compute entry, entry -> GPU start of its first command buffer
static int    g_gl = -1;
static double g_gl_wake = 0.0, g_gl_entry = 0.0, g_gl_t_wake = 0.0, g_gl_t_entry = 0.0, g_gl_start = 0.0, g_gl_gpu = 0.0, g_gl_end_prev = 0.0;
static int64_t g_gl_n = 0;
static id<MTLCommandBuffer> g_gl_first = nil;
static void ggml_metal_gaplog_print(void) {
    const double n = MAX(g_gl_n, 1);
    fprintf(stderr, "[metal-gap] %lld graphs: GPU end -> wake %.1f us, wake -> entry %.1f us, entry -> GPU start %.1f us, GPU span %.1f us\n",
            (long long) g_gl_n, g_gl_wake/n*1e6, g_gl_entry/n*1e6, g_gl_start/n*1e6, g_gl_gpu/n*1e6);
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    bool spun = false;
    if (ctx->cmd_buf_last) {
        if (ctx->flag_buf && ctx->flag_cb == ctx->cmd_buf_last) {
            volatile const uint8_t * f = (volatile const uint8_t *) [ctx->flag_buf contents];
            const uint8_t want = ctx->flag_val;
            int64_t it = 0;
            while (*f != want) {
                if ((++it & 0xffff) == 0 && [ctx->cmd_buf_last status] >= MTLCommandBufferStatusCompleted) {
                    break; // completed (or failed) without the flag becoming visible
                }
            }
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            // the flag only covers the last copy: command buffers without a dependency between them can complete
            // out of order, so the other pending copies (several outputs or ubatches per graph) are waited for
            for (NSUInteger i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
                id<MTLCommandBuffer> cb = ctx->cmd_bufs_ext[i];
                if (cb != ctx->flag_cb && [cb status] < MTLCommandBufferStatusCompleted) {
                    [cb waitUntilCompleted];
                }
            }
            spun = true;
        } else if (ctx->spin && ctx->spin_cb == ctx->cmd_buf_last) {
            while ([ctx->spin_ev signaledValue] < ctx->spin_val) {
                // busy-wait
            }
            spun = true;
        } else {
            [ctx->cmd_buf_last waitUntilCompleted];
        }
        if (g_gl < 0) { g_gl = getenv("GGML_METAL_GAPLOG") != NULL; if (g_gl) atexit(ggml_metal_gaplog_print); }
        if (g_gl && !spun && g_gl_first) {
            const double t_wake = ggml_metal_host_time_s();
            const double t_end  = [ctx->cmd_buf_last GPUEndTime];
            const double t_beg  = [g_gl_first GPUStartTime];
            if (g_gl_end_prev > 0 && t_beg > 0 && t_end > 0) {
                g_gl_wake  += t_wake - t_end;
                g_gl_entry += g_gl_t_entry - g_gl_t_wake;
                g_gl_start += t_beg - g_gl_t_entry;
                g_gl_gpu   += t_end - t_beg;
                g_gl_n++;
            }
            g_gl_end_prev = t_end;
            g_gl_t_wake   = t_wake;
            [g_gl_first release];
            g_gl_first = nil;
        }
        ctx->cmd_buf_last = nil;
    }
    // flag_cb / spin_cb are weak references compared by address: once their command buffer is released (below, or
    // with the next graph), a new command buffer can get the same address and would pass the check with a stale
    // flag value, so they are only valid until this synchronization
    ctx->flag_cb = nil;
    ctx->spin_cb = nil;

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if ((spun || ctx->spin || ctx->flag_buf) ? status == MTLCommandBufferStatusError : status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if ((spun || ctx->spin || ctx->flag_buf) ? status == MTLCommandBufferStatusError : status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        ggml_metal_spin_signal(ctx, cmd_buf);
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        if (ctx->flag_buf) {
            // the flag is written by a second encoder that waits for the copy (fence)
            [encoder updateFence:ctx->flag_fence];
            [encoder endEncoding];

            ctx->flag_val = ctx->flag_val == 255 ? 1 : ctx->flag_val + 1;
            id<MTLBlitCommandEncoder> enc_flag = [cmd_buf blitCommandEncoder];
            [enc_flag waitForFence:ctx->flag_fence];
            [enc_flag fillBuffer:ctx->flag_buf range:NSMakeRange(0, 1) value:ctx->flag_val];
            [enc_flag endEncoding];
            ctx->flag_cb = cmd_buf;
        } else {
            [encoder endEncoding];
        }
        ggml_metal_spin_signal(ctx, cmd_buf);
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

// GGML_METAL_PROFILE=1: debug-only per-op GPU timing. Each encoded op (fused groups count as one) goes into its own
// command buffer, committed and waited for, and its GPU time is accumulated per (graph size, op, node name without the
// layer suffix). Tables are printed at exit. Serializes the GPU completely: use only to rank the hot spots.
struct ggml_metal_prof_entry { char key[160]; int n_graph_nodes; double t_us; int64_t n; };
static struct ggml_metal_prof_entry * g_metal_prof = NULL;
static int g_metal_prof_n = 0, g_metal_prof_cap = 0;
static int64_t g_metal_prof_graphs[64][2]; // (n_nodes, count)
static int g_metal_prof_n_graphs = 0;

static int ggml_metal_prof_cmp(const void * a, const void * b) {
    const struct ggml_metal_prof_entry * x = a, * y = b;
    if (x->n_graph_nodes != y->n_graph_nodes) return x->n_graph_nodes - y->n_graph_nodes;
    return x->t_us < y->t_us ? 1 : (x->t_us > y->t_us ? -1 : 0);
}

static void ggml_metal_prof_print(void) {
    if (g_metal_prof_n == 0) return;
    qsort(g_metal_prof, g_metal_prof_n, sizeof(*g_metal_prof), ggml_metal_prof_cmp);
    int top = getenv("GGML_METAL_PROFILE_TOP") ? atoi(getenv("GGML_METAL_PROFILE_TOP")) : 40;
    for (int i = 0; i < g_metal_prof_n; ) {
        const int ng = g_metal_prof[i].n_graph_nodes;
        int64_t calls = 0;
        for (int k = 0; k < g_metal_prof_n_graphs; ++k) if (g_metal_prof_graphs[k][0] == ng) calls = g_metal_prof_graphs[k][1];
        double tot = 0; int j = i; int64_t nops = 0;
        for (; j < g_metal_prof_n && g_metal_prof[j].n_graph_nodes == ng; ++j) { tot += g_metal_prof[j].t_us; nops += g_metal_prof[j].n; }
        fprintf(stderr, "\n[metal-profile] graph with %d nodes: %lld evaluations, %.1f us GPU per evaluation, %.1f encoded ops per evaluation\n",
                ng, (long long) calls, tot / MAX(calls, 1), (double) nops / MAX(calls, 1));
        fprintf(stderr, "  %10s %6s %8s %7s  %s\n", "us/eval", "%", "ops/eval", "us/op", "op / name");
        for (int k = i; k < j && k < i + top; ++k) {
            fprintf(stderr, "  %10.1f %5.1f%% %8.1f %7.2f  %s\n", g_metal_prof[k].t_us / MAX(calls, 1), 100.0 * g_metal_prof[k].t_us / tot,
                    (double) g_metal_prof[k].n / MAX(calls, 1), g_metal_prof[k].t_us / g_metal_prof[k].n, g_metal_prof[k].key);
        }
        i = j;
    }
}

static void ggml_metal_prof_add(int n_graph_nodes, const char * key, double t_us) {
    for (int i = 0; i < g_metal_prof_n; ++i) {
        if (g_metal_prof[i].n_graph_nodes == n_graph_nodes && strcmp(g_metal_prof[i].key, key) == 0) {
            g_metal_prof[i].t_us += t_us; g_metal_prof[i].n++; return;
        }
    }
    if (g_metal_prof_n == g_metal_prof_cap) {
        if (g_metal_prof_cap == 0) atexit(ggml_metal_prof_print);
        g_metal_prof_cap = g_metal_prof_cap ? 2*g_metal_prof_cap : 256;
        g_metal_prof = realloc(g_metal_prof, g_metal_prof_cap * sizeof(*g_metal_prof));
    }
    struct ggml_metal_prof_entry * e = &g_metal_prof[g_metal_prof_n++];
    snprintf(e->key, sizeof(e->key), "%s", key); e->n_graph_nodes = n_graph_nodes; e->t_us = t_us; e->n = 1;
}

static void ggml_metal_prof_key(const struct ggml_tensor * node, int n_fused, char * out, size_t sz) {
    // strip the "-<layer>" suffix of the node name
    char name[GGML_MAX_NAME]; snprintf(name, sizeof(name), "%s", node->name);
    char * dash = strrchr(name, '-');
    if (dash && dash[1] >= '0' && dash[1] <= '9') *dash = 0;
    snprintf(out, sz, "%s%s [%s] ne=%lld,%lld,%lld", ggml_op_desc(node), n_fused > 1 ? "+fused" : "", name,
             (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
}

// GGML_METAL_SPAN=1: per graph size, average GPU span (first command buffer start -> last command buffer end) and the
// GPU idle gap since the previous graph ended, printed at exit (debug)
static double g_span_sum[64], g_gap_sum[64]; static int64_t g_span_n[64]; static int g_span_nodes[64]; static int g_span_k = 0;
static double g_span_last_end = 0.0;
// same time base as MTLCommandBuffer GPUStartTime (host time in seconds)
static double ggml_metal_host_time_s(void) {
    static mach_timebase_info_data_t tb; if (tb.denom == 0) mach_timebase_info(&tb);
    return (double) mach_absolute_time()*tb.numer/tb.denom*1e-9;
}
static double g_wake_sum = 0.0;
static double g_entry_gap_sum = 0.0, g_entry_to_commit_sum = 0.0; static int64_t g_entry_n = 0;
static double g_span_gap_avg(int k) { return g_gap_sum[k]/MAX(g_span_n[k] - 1, 1)*1e6; }
static void ggml_metal_span_print(void) {
    for (int k = 0; k < g_span_k; ++k) {
        fprintf(stderr, "[metal-span] prev GPU end -> graph_compute entry %.1f us (of which wake-up after GPU end %.1f us), entry -> GPU start %.1f us (avg)\n",
                g_entry_gap_sum/MAX(g_entry_n, 1)*1e6, g_wake_sum/MAX(g_entry_n + 1, 1)*1e6, g_entry_to_commit_sum/MAX(g_entry_n, 1)*1e6);
        fprintf(stderr, "[metal-span] graph %d nodes: %lld evals, GPU span %.1f us, idle gap before %.1f us\n", g_span_nodes[k],
                (long long) g_span_n[k], g_span_sum[k]/MAX(g_span_n[k], 1)*1e6, g_span_gap_avg(k));
    }
}

static enum ggml_status ggml_metal_graph_compute_profile(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
    if (ctx->cmd_buf_last) { [ctx->cmd_buf_last waitUntilCompleted]; }
    {
        int k = 0;
        for (; k < g_metal_prof_n_graphs; ++k) if (g_metal_prof_graphs[k][0] == gf->n_nodes) break;
        if (k == g_metal_prof_n_graphs && k < 64) { g_metal_prof_graphs[k][0] = gf->n_nodes; g_metal_prof_graphs[k][1] = 0; g_metal_prof_n_graphs++; }
        if (k < 64) g_metal_prof_graphs[k][1]++;
    }
    int i = 0;
    while (i < gf->n_nodes) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            ggml_metal_op_t ctx_op = ggml_metal_op_init(ctx->dev, cmd_buf, gf, ctx->finfo, i, gf->n_nodes,
                                                        ctx->use_concurrency, false, ctx->debug_graph);
            if (ggml_metal_op_n_nodes(ctx_op) == 0) { ggml_metal_op_free(ctx_op); break; }
            const int res = ggml_metal_op_encode(ctx_op, 0);
            ggml_metal_op_free(ctx_op);
            // map "res non-empty nodes" back to a raw node index
            int first = -1, cnt = 0, j = i;
            for (; j < gf->n_nodes && cnt < res; ++j) {
                if (!ggml_op_is_empty(gf->nodes[j]->op) && !ggml_is_empty(gf->nodes[j])) { if (first < 0) first = j; cnt++; }
            }
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
            if ([cmd_buf status] != MTLCommandBufferStatusCompleted) { return GGML_STATUS_FAILED; }
            const double t_us = ([cmd_buf GPUEndTime] - [cmd_buf GPUStartTime]) * 1e6;
            char key[160];
            ggml_metal_prof_key(gf->nodes[first >= 0 ? first : i], res, key, sizeof(key));
            ggml_metal_prof_add(gf->n_nodes, key, t_us);
            static int seq_left = -1;
            if (seq_left < 0) { seq_left = getenv("GGML_METAL_PROFILE_SEQ") ? atoi(getenv("GGML_METAL_PROFILE_SEQ")) : 0; }
            if (seq_left > 0 && gf->n_nodes < 1000) {
                seq_left--;
                fprintf(stderr, "[seq] %4d %8.2f us  %s <- %s, %s\n", first, t_us, key,
                        gf->nodes[first >= 0 ? first : i]->src[0] ? gf->nodes[first >= 0 ? first : i]->src[0]->name : "-",
                        gf->nodes[first >= 0 ? first : i]->src[1] ? gf->nodes[first >= 0 ? first : i]->src[1]->name : "-");
            }
            i = MAX(j, i + 1);
        }
    }
    ctx->cmd_buf_last = nil;
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    static int use_profile = -1;
    if (use_profile < 0) { use_profile = getenv("GGML_METAL_PROFILE") != NULL; }
    if (use_profile) {
        ggml_metal_device_rsets_keep_alive(ctx->dev);
        return ggml_metal_graph_compute_profile(ctx, gf);
    }

    // number of nodes encoded by the main thread (empirically determined)
    // (GGML_METAL_N_MAIN overrides: a smaller first command buffer starts the GPU earlier)
    static int n_main_env = -2;
    if (n_main_env == -2) { const char * e = getenv("GGML_METAL_N_MAIN"); n_main_env = e ? atoi(e) : -1; }
    const int n_main = n_main_env > 0 ? n_main_env : MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    const double t_entry = ggml_metal_host_time_s();

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        if (ctx->n_cb == 0) {
            // single-threaded encoding: the whole graph is encoded by one command buffer
            ctx->n_nodes_0      = gf->n_nodes;
            ctx->n_nodes_1      = 0;
            ctx->n_nodes_per_cb = 0;
        } else {
            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        if (g_gl > 0) {
            g_gl_t_entry = t_entry;
            [g_gl_first release];
            g_gl_first = [ctx->cmd_bufs[n_cb].obj retain];
        }

        static int use_span = -1;
        if (use_span < 0) { use_span = getenv("GGML_METAL_SPAN") != NULL; if (use_span) atexit(ggml_metal_span_print); }
        if (use_span) {
            // wait here (debug only) and measure
            id<MTLCommandBuffer> first = ctx->cmd_bufs[n_cb].obj;
            id<MTLCommandBuffer> last  = ctx->cmd_buf_last;
            [last waitUntilCompleted];
            for (int i = 0; i < n_cb; ++i) { [ctx->cmd_bufs[i].obj waitUntilCompleted]; }
            const double t_wake = ggml_metal_host_time_s();
            const double t0 = [first GPUStartTime], t1 = [last GPUEndTime];
            g_wake_sum += t_wake - t1;
            int k = 0; for (; k < g_span_k; ++k) if (g_span_nodes[k] == gf->n_nodes) break;
            if (k == g_span_k && k < 64) { g_span_nodes[k] = gf->n_nodes; g_span_k++; }
            if (g_span_last_end > 0) { g_entry_gap_sum += t_entry - g_span_last_end; g_entry_to_commit_sum += t0 - t_entry; g_entry_n++; }
            if (k < 64) {
                g_span_sum[k] += t1 - t0; g_span_n[k]++;
                if (g_span_last_end > 0 && g_span_n[k] > 1) g_gap_sum[k] += t0 - g_span_last_end;
            }
            g_span_last_end = t1;
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        // the last command buffer of the graph signals the busy-wait event (GGML_METAL_SPIN_WAIT)
        if (cb_idx == (n_cb_l > 0 ? n_cb_l - 1 : n_cb_l) && (cb_idx < 2 || ctx->abort_callback == NULL)) {
            ggml_metal_spin_signal(ctx, cmd_buf);
        }

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
