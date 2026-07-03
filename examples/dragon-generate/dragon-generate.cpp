// Multi-token greedy generation for Dragon, implemented as repeated prefill.
//
// Decode-mode recurrent-state plumbing isn't wired through Dragon's M-layer
// kernel yet (state in/out are zeroed every call), so we work around it by
// re-prefilling the whole sequence on every generation step. O(L²) total work
// but numerically correct.
//
// Usage:
//   dragon-generate -m model.gguf [-n N] [-ngl K] [-t T] -p "prompt"
//
// Output: the generated text on stdout. Stderr carries timings.
//
// Token sampling is greedy (argmax). For temperature / top-p use a real
// sampling pipeline; this example is intentionally minimal.

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt    = "Hello world";
    int n_threads     = -1;
    int n_gpu_layers  = 0;
    int n_generate    = 16;
    bool incremental  = false;  // false → repeated-prefill (workaround). true → real decode mode.

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n_generate = atoi(argv[++i]);
        } else if ((!strcmp(argv[i], "-ngl") || !strcmp(argv[i], "--n-gpu-layers")) && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--incremental")) {
            incremental = true;
        } else {
            prompt = argv[i];
            for (int j = i + 1; j < argc; ++j) {
                prompt += " ";
                prompt += argv[j];
            }
            break;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-n N] [-ngl K] [-t T] -p \"prompt\"\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize the initial prompt.
    const int n_max = (int) prompt.size() + 8;
    std::vector<llama_token> tokens(n_max);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                  tokens.data(), n_max, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "tokenize failed (%d)\n", n_prompt);
        return 1;
    }
    tokens.resize(n_prompt);

    // Context big enough for prompt + everything we'll generate.
    const int n_ctx_max = n_prompt + n_generate;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx_max;
    cp.n_batch   = n_ctx_max;
    cp.no_perf   = true;
    if (std::getenv("DRAGON_FA")) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (n_threads > 0) {
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

    fprintf(stderr, "prompt: %d tokens, generating %d more (total ctx %d, mode=%s)\n",
            n_prompt, n_generate, n_ctx_max, incremental ? "incremental" : "repeated-prefill");

    auto t0 = std::chrono::steady_clock::now();

    // In incremental mode: prefill the whole prompt once, then issue single-token
    // batches. Each step's batch is just the newly-sampled token; the recurrent
    // state from the prior step lives in the cache.
    int n_already_in_cache = 0;
    if (incremental) {
        llama_memory_clear(mem, /*data=*/true);
        llama_batch batch = llama_batch_init(n_prompt, /*embd*/0, /*n_seq_max*/1);
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[i]    = tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0]= 0;
            batch.logits[i]   = (i == n_prompt - 1) ? 1 : 0;
        }
        batch.n_tokens = n_prompt;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "initial prefill decode failed\n");
            llama_batch_free(batch);
            return 1;
        }
        const float * row = llama_get_logits_ith(ctx, n_prompt - 1);
        int best = 0;
        float best_v = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > best_v) { best_v = row[v]; best = v; }
        }
        tokens.push_back((llama_token) best);
        char piece[256];
        int n = llama_token_to_piece(vocab, (llama_token) best, piece, sizeof(piece), 0, /*special=*/false);
        if (n > 0) { fwrite(piece, 1, n, stdout); fflush(stdout); }
        n_already_in_cache = n_prompt;
        llama_batch_free(batch);
    }

    for (int step = 0; step < n_generate; ++step) {
        const int n_now = (int) tokens.size();

        // First incremental token was emitted above; skip step 0.
        if (incremental && step == 0) {
            continue;
        }

        llama_batch batch;
        int sample_pos;  // batch position whose logits to read

        if (incremental) {
            // Append a single new token (the last-sampled one) to the running sequence.
            const int new_pos = n_already_in_cache;
            batch = llama_batch_init(1, /*embd*/0, /*n_seq_max*/1);
            batch.token[0]    = tokens[new_pos];
            batch.pos[0]      = new_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0]= 0;
            batch.logits[0]   = 1;
            batch.n_tokens    = 1;
            sample_pos = 0;
            ++n_already_in_cache;
        } else {
            // Repeated-prefill: clear cache and re-prefill the full sequence.
            llama_memory_clear(mem, /*data=*/true);
            batch = llama_batch_init(n_now, /*embd*/0, /*n_seq_max*/1);
            for (int i = 0; i < n_now; ++i) {
                batch.token[i]    = tokens[i];
                batch.pos[i]      = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0]= 0;
                batch.logits[i]   = (i == n_now - 1) ? 1 : 0;
            }
            batch.n_tokens = n_now;
            sample_pos = n_now - 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step);
            llama_batch_free(batch);
            return 1;
        }

        // Greedy: argmax over the relevant position's logits.
        const float * row = llama_get_logits_ith(ctx, sample_pos);
        if (!row) {
            fprintf(stderr, "null logits at step %d\n", step);
            llama_batch_free(batch);
            return 1;
        }
        int best = 0;
        float best_v = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > best_v) { best_v = row[v]; best = v; }
        }

        tokens.push_back((llama_token) best);
        fprintf(stderr, "  step=%d best=%d v=%.4f\n", step, best, best_v);
        llama_batch_free(batch);

        // Stream the token's piece to stdout.
        char piece[256];
        int n = llama_token_to_piece(vocab, (llama_token) best, piece, sizeof(piece), 0, /*special=*/false);
        if (n > 0) {
            fwrite(piece, 1, n, stdout);
            fflush(stdout);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    fputc('\n', stdout);
    fprintf(stderr, "generated %d tokens in %lld ms (%.1f tok/s, %s)\n",
            n_generate, (long long) ms, n_generate / (ms / 1000.0),
            incremental ? "incremental" : "repeated prefill");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
