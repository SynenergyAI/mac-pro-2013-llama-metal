#!/bin/bash
# One card (MTL0), Qwen2.5-Coder-7B-Instruct Q4_K_M, 3 repetitions; each length in its own process
# (a crash at one length must not hide the others). Usage: ./bench.sh /path/to/step/build/bin /path/to/model.gguf
set -u; BIN=${1:?build/bin directory}; M=${2:?model .gguf}; DEV="MTL0:AMD Radeon HD - FirePro D700"
for a in "-p 256 -n 0" "-p 512 -n 0" "-p 0 -n 128"; do
  GGML_METAL_DEVICES=2 "$BIN/llama-bench" -m "$M" -ngl 99 -dev "$DEV" -fa 0 -t 4 $a -r 3 -o json; echo "exit $?"
done
