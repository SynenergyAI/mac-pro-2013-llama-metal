#!/bin/bash
# Correctness on Metal device 0 for one built step. Usage: ./correctness.sh /path/to/step/build/bin
set -eu; BIN=${1:?build/bin directory}
for op in MUL_MAT MUL_MAT_ID; do
  GGML_METAL_DEVICES=1 "$BIN/test-backend-ops" test -b MTL0 -o $op 2>&1 | sed -E "s/\x1b\[[0-9;]*m//g" | tee "correctness-$op.log" | grep -E "tests passed"
done
