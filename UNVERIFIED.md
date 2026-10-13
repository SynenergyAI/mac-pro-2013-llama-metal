# Inferred, not measured

- **Cause of `ggml-metal-context.m:319: fatal error`.** The vector path aborts at pp512 (patches 0002 and 0003, `logs/bench_series.log:6,9`). The log does not show why; we have not established the cause.
- **Cause of the NoCopy nil.** Apple documents that `newBufferWithBytesNoCopy` needs a page-aligned pointer and length. We did not instrument which condition fails on this GPU.
- **`logs/correctness-tests-without-q6K-loader-fix-MUL_MAT.log` (929/941) comes from a separate build.** That build was fork 72ba023 + the build fix + the wave-64 fix + tiled dispatch + the 72 new tests, without the loader fix. It was assembled from whole files, not by applying this patch series. Its wave-64 kernel file is the 0006 variant, not the 0003 variant.
- **Other GPUs.** We tested only the FirePro D700 (GCN1, 64-wide SIMD) on macOS 12.7.6. Other GCN or Polaris cards, and other macOS versions, are untested.
- **Third-party numbers.** Linux / Mesa RADV results on the same Mac Pro model come from a third-party blog (matthewgribben.com, 2026-05-26). We did not reproduce them.
- **Qwen3.6-35B-A3B MTP file identity.** The blog names the model "Qwen3.6-35B-A3B-MTP-GGUF Q4_K_M" but no provider. We assume `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, because that is the only Q4_K_M in a repository with that exact name. This is an inference.
- **Run-to-run variation.** Each speed number is the median of 3 repetitions in one session. We did not repeat the series on other days.
- **Blog expert placement and threads.** The blog's expert placement (-ngl/-ncmoe) is unknown and its run uses -t 2; our headline uses -t 8, so the numbers are not a matched pair.
