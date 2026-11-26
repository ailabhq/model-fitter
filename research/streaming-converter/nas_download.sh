#!/usr/bin/env bash
# Download a Hugging Face repo at a fixed revision with plain curl (resumable, retried), then check every
# file's size and, for LFS files, its sha256. For a machine without pip (the NAS).
#
# Usage: nas_download.sh REPO REVISION DEST
# Needs the manifest DEST.manifest.tsv (path, size, sha256 or "-") next to DEST; make it with:
#   curl -s https://huggingface.co/api/models/REPO/tree/REVISION?recursive=true | python3 -c '...'
set -uo pipefail
repo=$1 rev=$2 dest=$3
manifest=$dest.manifest.tsv
mkdir -p "$dest"
echo "$(date +%s) start $repo@$rev"
while IFS=$'\t' read -r path size sha; do
  out=$dest/$path
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then continue; fi
  for attempt in 1 2 3 4 5 6 7 8; do
    curl -fsSL -C - --retry 10 --retry-delay 5 -o "$out" "https://huggingface.co/$repo/resolve/$rev/$path" && break
    echo "$(date +%s) retry $attempt $path"; sleep 10
  done
  echo "$(date +%s) got $path $(stat -c %s "$out")/$size"
done < "$manifest"
echo "$(date +%s) verify"
bad=0
while IFS=$'\t' read -r path size sha; do
  out=$dest/$path
  [ "$(stat -c %s "$out" 2>/dev/null)" = "$size" ] || { echo "SIZE MISMATCH $path"; bad=1; continue; }
  if [ "$sha" != "-" ]; then
    [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ] || { echo "SHA MISMATCH $path"; bad=1; }
  fi
done < "$manifest"
echo "$(date +%s) done bad=$bad"
