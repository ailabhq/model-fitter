qwen-q3kxl-llamacpp-fit (default --fit, default wired limit): stopped by hand during the first 8K generation.
- Prefill of 8,000 tokens: 410 s (19.5 tok/s), from the server log.
- Generation at 8K depth: n_decoded went from 50 to 72 in 61 s on the server's /slots endpoint, i.e. **0.36 tok/s**.
- At the time: system free memory 6%, swap used 6.1 GiB; weights partly CPU-mapped and re-read from SSD every token.
- The server was sent SIGTERM (ignored under thrash), then SIGKILL; the harness recorded the dropped connection as the error.
