// Tokenize a prompt, run a single forward pass, and dump per-token logits to
// a binary file. Used for numerical comparison against a reference (e.g. HF).
//
// Usage:
//   dragon-logits -m model.gguf -o out.bin [-t threads] "prompt text"
//
// Output file format (little-endian):
//   int32  n_tokens
//   int32  n_vocab
//   float  logits[n_tokens * n_vocab]
//
// If DRAGON_DUMP_DIR=<dir> is set in the environment, the Dragon graph also
// writes per-block hidden states to <dir>/block_XX_in.bin for debugging.

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path;
    std::string out_path  = "/tmp/llama_logits.bin";
    std::string prompt    = "Hello world";
    int n_threads     = -1;
    int n_gpu_layers  = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if ((!strcmp(argv[i], "-ngl") || !strcmp(argv[i], "--n-gpu-layers")) && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            // Consume only the next arg as the prompt; trailing flags (-n, -ngl, ...)
            // still get parsed by the outer loop.
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            // Accepted for CLI parity with llama-cli; ignored (we always run a
            // single prefill forward pass and dump logits — no generation).
            ++i;
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
        fprintf(stderr, "usage: %s -m model.gguf -o out.bin [-t threads] [-ngl N] \"prompt\"\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize.
    const int n_max = (int) prompt.size() + 8;
    std::vector<llama_token> tokens(n_max);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                  tokens.data(), n_max, true, true);
    if (n_tokens <= 0) {
        fprintf(stderr, "tokenize failed (%d)\n", n_tokens);
        return 1;
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "n_tokens=%d, n_vocab=%d\n", n_tokens, n_vocab);
    for (int i = 0; i < n_tokens; ++i) fprintf(stderr, " %d", tokens[i]);
    fprintf(stderr, "\n");

    // Context with logits_all so we keep every position's logits.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_tokens;
    cp.n_batch   = n_tokens;
    cp.no_perf   = true;
    if (n_threads > 0) {
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    // Build a batch with logits=true on every position.
    llama_batch batch = llama_batch_init(n_tokens, /*embd*/0, /*n_seq_max*/1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = 1; // keep logits
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    // Dump logits: for each token position, llama_get_logits_ith returns the row.
    std::ofstream out(out_path, std::ios::binary);
    int32_t hdr[2] = { (int32_t) n_tokens, (int32_t) n_vocab };
    out.write((const char *) hdr, sizeof(hdr));
    for (int i = 0; i < n_tokens; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        if (!row) {
            fprintf(stderr, "logits row %d is null\n", i);
            return 1;
        }
        out.write((const char *) row, n_vocab * sizeof(float));
    }
    out.close();
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
