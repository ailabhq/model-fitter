// mf-score: score a Fit against a top-k Reference pack, prefill only.
//
// Usage: llama-mf-score --mf-in IN --mf-out OUT [any llama.cpp common args: -m, -c, -fa, --fit, -ub, ...]
//
// IN  (little-endian int32): n_items, then per item: n_tokens, k, ids[n_tokens], ref_ids[(n_tokens-1)*k],
//     text_len, text bytes (padded to 4).
// OUT per item: lq[(n_tokens-1)*k] float32  log-prob the Fit gives each reference top-k token
//               amax[n_tokens-1]   int32    the Fit's argmax token
//               n_tok int32, tok[n_tok] int32  the Fit's tokenization of the item text (parse special, no BOS)
// Every item is its own sequence in a cleared context; all positions but the last produce logits.

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"

#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool rd(FILE * f, void * p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string in_path, out_path;
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mf-in") && i + 1 < argc) { in_path = argv[++i]; continue; }
        if (!strcmp(argv[i], "--mf-out") && i + 1 < argc) { out_path = argv[++i]; continue; }
        rest.push_back(argv[i]);
    }
    if (in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s --mf-in IN --mf-out OUT [llama.cpp args]\n", argv[0]);
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    params.n_parallel = 1;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    llama_backend_init();
    llama_numa_init(params.numa);
    const auto t_load0 = std::chrono::steady_clock::now();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model or context\n");
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    FILE * fi = fopen(in_path.c_str(), "rb");
    FILE * fo = fopen(out_path.c_str(), "wb");
    if (!fi || !fo) {
        LOG_ERR("cannot open in/out files\n");
        return 1;
    }
    int32_t n_items = 0;
    rd(fi, &n_items, 4);
    fprintf(stdout, "{\"ready\": true, \"load_s\": %.2f, \"n_vocab\": %d, \"n_ctx\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"items\": %d}\n",
            load_s, n_vocab, n_ctx, n_batch, (int) llama_n_ubatch(ctx), n_items);
    fflush(stdout);

    common_batch batch(ctx);
    double total_s = 0;
    long total_pos = 0;
    for (int it = 0; it < n_items; it++) {
        int32_t n, k;
        rd(fi, &n, 4);
        rd(fi, &k, 4);
        std::vector<int32_t> ids(n), ref((size_t) (n - 1) * k);
        rd(fi, ids.data(), 4 * (size_t) n);
        rd(fi, ref.data(), 4 * ref.size());
        int32_t tl;
        rd(fi, &tl, 4);
        std::string text((size_t) tl, '\0');
        rd(fi, text.data(), (size_t) tl);
        if (tl % 4) { char pad[4]; rd(fi, pad, 4 - tl % 4); }

        if (n > n_ctx) {
            LOG_ERR("item %d has %d tokens, context is %d\n", it, n, n_ctx);
            return 1;
        }
        std::vector<float> lq(ref.size());
        std::vector<int32_t> amax(n - 1);

        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int s = 0; s < n; s += n_batch) {
            const int e = std::min(n, s + n_batch);
            batch.clear();
            for (int j = s; j < e; j++) {
                batch.add(ids[j], j, 0, j < n - 1);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed at item %d pos %d\n", it, s);
                return 1;
            }
            int out_i = 0;
            for (int j = s; j < e && j < n - 1; j++, out_i++) {
                const float * lg = llama_get_logits_ith(ctx, out_i);
                float mx = -INFINITY;
                int am = 0;
                for (int v = 0; v < n_vocab; v++) {
                    if (lg[v] > mx) { mx = lg[v]; am = v; }
                }
                double se = 0;
                for (int v = 0; v < n_vocab; v++) {
                    se += std::exp((double) lg[v] - mx);
                }
                const float lse = mx + (float) std::log(se);
                amax[j] = am;
                for (int c = 0; c < k; c++) {
                    const int32_t t = ref[(size_t) j * k + c];
                    lq[(size_t) j * k + c] = (t >= 0 && t < n_vocab) ? lg[t] - lse : -INFINITY;
                }
            }
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        total_s += dt;
        total_pos += n;

        std::vector<llama_token> tok = common_tokenize(vocab, text, false, true);
        fwrite(lq.data(), 4, lq.size(), fo);
        fwrite(amax.data(), 4, amax.size(), fo);
        int32_t nt = (int32_t) tok.size();
        fwrite(&nt, 4, 1, fo);
        fwrite(tok.data(), 4, tok.size(), fo);
        fflush(fo);
        fprintf(stdout, "{\"item\": %d, \"tokens\": %d, \"s\": %.2f, \"tok_s\": %.1f}\n", it, n, dt, n / dt);
        fflush(stdout);
    }
    fclose(fi);
    fclose(fo);
    fprintf(stdout, "{\"done\": true, \"positions\": %ld, \"score_s\": %.1f, \"tok_s\": %.1f}\n", total_pos, total_s, total_pos / total_s);
    llama_perf_context_print(ctx);
    common_memory_breakdown_print(ctx);
    llama_backend_free();
    return 0;
}
