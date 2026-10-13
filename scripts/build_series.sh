#!/bin/bash
# Apply the series to ai-janitor/llama-metal @ 72ba023 and build every step in its own worktree.
# Usage: SRC=/path/to/llama-metal-clone PATCHES=/path/to/patches ./build_series.sh
set -eu
SRC=${SRC:?clone of https://github.com/ai-janitor/llama-metal}; PATCHES=${PATCHES:?directory with 0001..0008}
git -C "$SRC" checkout -q -B d700-fixes 72ba023
git -C "$SRC" am -q "$PATCHES"/000*.patch
n=0
for sha in $(git -C "$SRC" rev-list --reverse 72ba023..d700-fixes); do n=$((n+1)); W="$SRC-step$n"
  rm -rf "$W"; git -C "$SRC" worktree add -q --detach "$W" "$sha"
  cmake -S "$W" -B "$W/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
  cmake --build "$W/build" -j6 --target test-backend-ops llama-bench
done
