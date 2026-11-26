# sarto-ref plus a patch that dumps target sizing's per-tensor error tables (MF_TABLE_DUMP) and adds the remaining Metal types.
FROM sarto-ref:22b99122e
COPY tables.patch /tmp/tables.patch
RUN cd /llama.cpp && git apply /tmp/tables.patch && cmake --build build --target llama-quantize -j 4
