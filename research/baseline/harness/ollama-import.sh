#!/bin/sh
# ollama-import.sh GGUF TAG — import a GGUF into Ollama without a second copy on disk.
# APFS-clones the file into Ollama's blob store under its sha256 name, so `ollama create`
# finds the blob already present and skips the upload.
set -eu
gguf=$(cd "$(dirname "$1")" && pwd)/$(basename "$1") tag=$2
digest=$(shasum -a 256 "$gguf" | cut -d' ' -f1)
blob="$HOME/.ollama/models/blobs/sha256-$digest"
[ -e "$blob" ] || cp -c "$gguf" "$blob"
printf 'FROM %s\n' "$gguf" > "$(dirname "$gguf")/Modelfile"
ollama create "$tag" -f "$(dirname "$gguf")/Modelfile"
