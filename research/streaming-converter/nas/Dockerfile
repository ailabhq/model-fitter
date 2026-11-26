# Reference toolchain for the identity check, on the NAS (x86_64): llama.cpp at the pinned commit, its
# convert_hf_to_gguf.py requirements, llama-quantize and a shared libggml-base for stream_convert.py.
# GGML_NATIVE=OFF: baseline x86-64, so the compiler cannot fuse multiply-adds into FMA instructions.
FROM python:3.11-slim
ARG LLAMA_COMMIT=22b99122e
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp && cd /llama.cpp && git checkout $LLAMA_COMMIT
RUN pip install --no-cache-dir -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt psutil requests huggingface_hub
RUN cd /llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --target llama-quantize -j 4
ENV MF_LLAMA=/llama.cpp MF_GGML_LIB=/llama.cpp/build/bin/libggml-base.so
