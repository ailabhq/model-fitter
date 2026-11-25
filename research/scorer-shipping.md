# Scorer shipping facts

For [How does sarto's Quality loss scorer ship with its pinned llama.cpp?](https://github.com/ailabhq/sarto/issues/33). Source: ggml-org/llama.cpp `master` `beb9d493` (contains `51831d47d`), GitHub API and homebrew-core, checked 2026-11-25.

## 1. Symbol split
- libllama (`include/llama.h`): llama_backend_init/free, llama_numa_init (l.516), llama_model_get_vocab, llama_vocab_n_tokens, llama_n_ctx, llama_n_ubatch, llama_get_memory, llama_memory_clear (l.782), llama_get_logits_ith (l.1157), llama_perf_context_print, llama_process + `enum llama_process_type`/LLAMA_PROCESS_TYPE_DECODE (l.308-310).
- llama-common (`common/`): common_params, common_params_parse + LLAMA_EXAMPLE_PERPLEXITY (arg.h), common_init_from_params/common_init, common_batch, common_tokenize, common_memory_breakdown_print, fit.h.
- Equivalents: arg parsing → none in libllama. common_init_from_params → llama_model_load_from_file + llama_init_from_model. common_tokenize → llama_tokenize. common_batch → llama_batch / new llama_batch_ext. `--fit` → **no public llama_params_fit**; only `common_fit_params()` in `common/fit.h` (l.30, `common_params_fit_status`). common_memory_breakdown_print → none (llama.h has no equivalent found).
- `llama_process`/LLAMA_PROCESS_TYPE_DECODE introduced fc343a84 (2026-09-24, "llama: add llama_batch_ext", #24669). Old `llama_decode`/`llama_encode` deprecated pointing to llama_process (llama.h l.1000, 1017); batch API deprecated a06044b9 (2026-11-16, #32306).
- `common/fit.h` exists; history in shallow clone goes back to ≥2026-05 (83eebe9d 2026-05-24 "server: add margin for draft model for fit" #23485) — original introduction predates clone depth (unknown exact).

## 2. Install / packaging
- Root CMakeLists.txt l.285-322: `install(TARGETS llama LIBRARY PUBLIC_HEADER)` installs libllama + `llama.h` + `llama-cpp.h`; installs `lib/cmake/llama/llama-config.cmake` + `llama-config-version.cmake` (COMPATIBILITY SameMajorVersion; semver introduced 680a9ae6, #26839) and `lib/pkgconfig/llama.pc`.
- `if (LLAMA_BUILD_COMMON) install(TARGETS llama-common LIBRARY)` — library only; **no common headers installed** (no PUBLIC_HEADER set).
- common/CMakeLists.txt: `add_library(${TARGET} ...)` without STATIC/SHARED → follows BUILD_SHARED_LIBS (PIC + WINDOWS_EXPORT_ALL_SYMBOLS, "TODO: make fine-grained exports"); a helper lib at l.42 is explicitly STATIC. BUILD_SHARED_LIBS default ON on most platforms (l.56-79).
- Homebrew formula (homebrew-core Formula/l/llama.cpp.rb): `-DBUILD_SHARED_LIBS=ON -DLLAMA_USE_SYSTEM_GGML=ON -DLLAMA_BUILD_TESTS=OFF`, `cmake --install`; so installs libllama, llama.h, cmake config, pc, libllama-common dylib (no common headers). Its own test builds `test-sampling.cpp` with `target_link_libraries(... llama)` via find_package — proving out-of-tree libllama linking.

## 3. API churn (2026-05-25 → 2026-11-25, `git log --since`)
- include/llama.h: 47 commits. Breaking/deprecating examples: a06044b9 deprecate old batch API (#32306); fc343a84 llama_batch_ext + llama_process (#24669); a6aa6f54 sampler remove "full-context windows" (#26524); 4f3a4beb deprecate llama_set_warmup (#24009). Most others additive.
- common/common.h: 108 commits (no stability promise; struct common_params changes constantly). common/arg.h: 10.

## 4. perplexity --kl-divergence
- Base file (`--save-all-logits`/`--kl-divergence-base`): per position, full-vocab quantized log-probs: `nv = 2*((n_vocab+1)/2)+4` uint16 = 2 floats header (max_logit, min_log_prob) + n_vocab uint16 (tools/perplexity/perplexity.cpp l.79-104, 147, 172). ≈2·n_vocab bytes/position (e.g. ~300 KB for 150k vocab).
- No top-k, no response-only/position mask, no per-position selection (scores second half of each n_ctx chunk). Flags all LLAMA_EXAMPLE_PERPLEXITY in common/arg.cpp l.2452-2515.
- Upstream searches (`gh search prs/issues` "kl-divergence top-k", "kld top-k logits") returned nothing. Unknown beyond that.

## 5. Maintainer stance
- CONTRIBUTING.md l.46-48: "Features must begin with an issue, not a PR … niche features may only land as an example/tool, or on a private fork." "New CLI or public API additions carry a **higher bar** … justify why an existing mechanism doesn't suffice." "Meeting all of the above still doesn't guarantee a merge."
- New tool/example dirs added since 2026-05-25 (diff-filter=A on tools/*/examples/* CMakeLists): none found in the 3 hits (all unrelated). Rejected-tool examples: not searched exhaustively (unknown).

## 6. Output buffer
- `llama_context::output_reserve(n_outputs)` (src/llama-context.cpp l.2617+): `n_outputs_max = max(n_outputs, n_seq_max)`; `logits.size = n_vocab*n_outputs_max` — sized by outputs requested (batch.logits flags), grows/reallocs on demand (l.2677-2702), never shrinks.
- Cap: `llama_context_params.n_outputs_max` ("max outputs in a ubatch (0 = n_batch)", llama.h l.383) + n_outputs_max_per_seq; graph reserve uses min(n_tokens, n_outputs_max) (l.853,1083). So flagging fewer positions + setting n_outputs_max shrinks both host buffer and graph output size.
- Backend sampling adds `2*n_vocab*n_outputs_max` floats + `(1+n_vocab)*n_outputs_max` tokens.
- GPU-side: [EXPERIMENTAL] `llama_set_sampler` / `llama_context_params.samplers` with backend_init (l.1432-1473); readback via `llama_get_sampled_token_ith`, `llama_get_sampled_probs_ith` (l.1242-1247). No public top-k log-softmax-of-arbitrary-ids API found; KLD needs full logits or a custom backend sampler.

## 7. Out-of-tree linking
- Yes: `find_package(llama)` → target `llama` (installed llama-config.cmake) or pkg-config `llama`. Homebrew's formula test does exactly this.
- llama-cpp-python: git submodule `vendor/llama.cpp` pinned to commit f5e6915c (gh api repos/abetlen/llama-cpp-python/contents/vendor); builds in-tree via its CMake.
- ollama: vendors llama.cpp source with patches (Makefile.sync path 404'd now; layout not re-verified — [INFERENCE] pinned commit + patch dir).

## 8. Server prompt logprobs; perplexity vs server args
- Stock master: **no**. tools/server/server-common.cpp l.1268-1269: `"echo"` → throws "Only no echo is supported". n_probs/post_sampling_probs only for generated tokens (`populate_token_probs`, server-context.cpp l.2250+; top n_probs only, no arbitrary ids).
- Open PRs: #27537 "server: return prompt-token logprobs when echo=true" (https://github.com/ggml-org/llama.cpp/pull/27537, fixes #27174; chunks by new `llama_n_outputs_max`, top-n only); #15189 "v1/completions echo logprobs support" (https://github.com/ggml-org/llama.cpp/pull/15189, open).
- Arg parsing: same `common_params_parse` with different example enum (perplexity.cpp l.1998 LLAMA_EXAMPLE_PERPLEXITY). Options with no `set_examples` are global: `--fit` (arg.cpp l.2864-2876) and `-ctk/--cache-type-k` (l.2421-2432) accepted by both. `-np/--parallel` at l.2519-2527 is `{LLAMA_EXAMPLE_SERVER}` only → rejected by perplexity (a second -np def at l.2530 exists for other examples; scope not confirmed). Server-only flags (host/port, slots, etc.) rejected; perplexity-only flags (--kl-divergence etc.) rejected by server.
