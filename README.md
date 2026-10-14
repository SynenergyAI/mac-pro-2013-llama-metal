# llama.cpp Metal on AMD FirePro D700 (Mac Pro 2013, macOS 12)

**[Qwen3.6-35B on a 2013 Mac Pro (Vulkan): reads 66 tok/s, writes ~11 tok/s.](#qwen36-35b-a3b-moe-mtp-on-vulkanmoltenvk)**

Eight patches that make llama.cpp's Metal backend compute correctly on the AMD FirePro D700 (GCN1, 64-wide SIMD). Every number below points to a log line in `logs/`.

Article: [English](https://synenergy.ai/research/mac-pro-2013-35b-local/) · [Українською](https://synenergy.ai/ua/research/mac-pro-2013-35b-local/)

## The problem

- **Mainline ggml-org/llama.cpp (7cf1c54) runs no matmuls on these GPUs.** It lists both cards as Metal devices, but every MUL_MAT and MUL_MAT_ID case is reported as `not supported [MTL0]`. Out of 1649 and 904 cases, 0/0 are tested (`logs/mainline-7cf1c54-mulmat-not-supported.txt`). The reason is that SIMD-group reductions are enabled only for Apple7/Metal3 GPUs.
- **With `-ngl 99`, mainline aborts** at `GGML_ASSERT(buf_src) failed` (`logs/mainline-7cf1c54-ngl99-abort.txt`). This matches [ggml-org/llama.cpp#26949](https://github.com/ggml-org/llama.cpp/issues/26949).
- **The fork does run matmuls on these GPUs, but at 72ba023 it does not work on the D700.** [ai-janitor/llama-metal](https://github.com/ai-janitor/llama-metal) (MIT) adds Metal support for non-Apple-Silicon GPUs. These patches apply on top of it.

## Base

[ai-janitor/llama-metal](https://github.com/ai-janitor/llama-metal) at commit `72ba023` (its HEAD on 2026-10-10). Thanks to its author: without that fork there would be no GPU path on this machine.

## Patches

**0001–0005: correctness fixes.**

| Patch | What it fixes | MUL_MAT | MUL_MAT_ID | Log |
|---|---|---|---|---|
| 0001 | build: missing `#include <string>` (macOS 12 CLT) | 433/869 | 162/616 | `logs/series_run.log:1` |
| 0002 | `newBufferWithBytesNoCopy` returns nil → fall back to a copied buffer | 433/869 | 162/616 | `logs/series_run.log:2` |
| 0003 | `chunks_per_thread = 32 / NW` is 0 at NW = 64 | **869/869** | **616/616** | `logs/series_run.log:3` |
| 0004 | quantized matmuls use the existing tiled kernels (batch ≥ 16) | 869/869 | 616/616 | `logs/series_run.log:4` |
| 0005 | q6_K tiled loader unpacking + 72 test cases with m = 128 | **941/941** | 616/616 | `logs/series_run.log:5` |

**0006–0008: optional speed changes.** Correctness stays at 941/941 and 616/616 (`logs/series_run.log:6,9,10`).

- **0006:** the q4_K and q6_K vector kernels use all 64 lanes; N_R0_Q6_K 2 → 4.
- **0007:** q4_K vector kernel rewritten for 64-wide GCN; N_R0_Q4_K 2 → 8.
- **0008:** q6_K vector kernel rewritten; N_R0_Q6_K 4 → 8.

**More detail:**
- Before 0003, failures cover f16 (192/262 failed) and f32 (143/198), not only quantized types (`logs/per-type-failures-patch1.txt`).
- The q6_K bug in 0005 is invisible to the existing tests, because they use m = 16 and the tiled kernel needs m ≥ 64. With the new cases and without the loader fix, the result is 929/941; all 12 failures are q6_K (`logs/correctness-tests-without-q6K-loader-fix-MUL_MAT.log`).

## Speed per patch

One card, Qwen2.5-Coder-7B-Instruct Q4_K_M, `llama-bench -ngl 99 -fa 0 -t 4 -r 3`, each length in its own process. Values are the median of three; all three repetitions are in `logs/bench_series.log`.

| Patch | pp256 tok/s | pp512 tok/s | tg128 tok/s | Lines in `logs/bench_series.log` |
|---|---|---|---|---|
| 0001 | exit 134 · `GGML_ASSERT(buf_dst) failed` | exit 134 · same | exit 134 · same | 2–4 |
| 0002 | 9.95 | exit 134 · `ggml-metal-context.m:319: fatal error` | 67.03 ⚠ | 5–7 |
| 0003 | 9.95 | exit 134 · `ggml-metal-context.m:319: fatal error` | 7.45 | 8–10 |
| 0004 | 49.14 | 52.58 | 7.42 | 11–13 |
| 0005 | 48.84 | 52.61 | 7.42 | 14–16 |
| 0006 | 48.97 | 52.08 | 11.20 | 17–19 |
| 0007 | 48.63 | 52.45 | 12.21 | 20–22 |
| 0008 | 49.23 | 52.58 | 12.89 | 23–25 |

**⚠ Fake benchmark.** After 0002 llama-bench reports **67.03 tok/s** generation while 436 of 869 MUL_MAT cases fail (`logs/series_run.log:2`). The kernels skip work, so the speed is meaningless. Never read a speed number on this GPU without the correctness table of the same binary.

**Honest pair.** The patches make prompt processing ~5x faster: 9.95 → 49.23 tok/s (pp256), generation 7.45 → 12.89 tok/s. This compares the first correct state (0003) against the full series (0008):
- prompt processing pp256: **9.95 → 49.23 tok/s** (lines 8 and 23);
- generation tg128: **7.45 → 12.89 tok/s** (lines 10 and 25).

## Qwen3.6-35B-A3B (MoE, MTP) on Vulkan/MoltenVK

Prompt processing: 66 tok/s (pp512). Best measured: ~11 tok/s generation (10.99, tg128, -t 8) on a 2013 Mac Pro. The best published figure we could find for this model on this hardware is 7.04 tok/s ([matthewgribben.com](https://matthewgribben.com), 2026-05-26). The blog lists only generation for this model and publishes no prompt-processing figure for it. That run used Linux with the open RADV driver. Ours runs on the original macOS 12, which on this hardware has less freedom (no open driver, mainline Metal does not compute matmul on these GPUs). All logs and commands are here; reproduce or beat it.

Mainline 7cf1c54, Vulkan via MoltenVK. Two cards, `-sm layer -ts 1/1 -fa 0 -p 512 -n 128 -r 3`. File: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Experts placed on the GPUs are split evenly between the two cards with `-ot`.

| Expert placement | Threads | pp512 tok/s | tg128 tok/s | Log |
|---|---|---|---|---|
| all expert layers on CPU (`-ncmoe 41`) | 2 | 60.77 | 7.95 | `logs/moltenvk_35b.log` |
| 10 expert layers on the GPUs | 2 | 66.14 | **9.01** | `logs/moltenvk_35b_experts_on_gpu.log` |
| all expert layers on CPU (`-ncmoe 41`) | 8 | 60.94 | 10.13 | `logs/moltenvk_35b.log` |
| 10 expert layers on the GPUs | 8 | 66.39 | **10.99** | `logs/moltenvk_35b_experts_on_gpu.log` |
| Blog, Linux/RADV | 2 | — | 7.04 | [matthewgribben.com](https://matthewgribben.com), 2026-05-26 |

## Reproduce

1. `scripts/build_series.sh` applies the patches with `git am` and builds every step in its own worktree.
2. `scripts/correctness.sh` runs test-backend-ops.
3. `scripts/bench.sh` runs llama-bench.

Hardware and toolchain: `logs/hardware.txt`.

## Related upstream threads

- [#26949](https://github.com/ggml-org/llama.cpp/issues/26949): NoCopy / `buf_src` on Intel Mac + AMD;
- [#19563](https://github.com/ggml-org/llama.cpp/issues/19563): garbage output on AMD discrete GPUs;
- [#15228](https://github.com/ggml-org/llama.cpp/issues/15228): Metal on Intel Mac + Radeon request;
- [#19527](https://github.com/ggml-org/llama.cpp/pull/19527): open PR, Metal on AMD discrete GPUs.

We do not claim to have found the NoCopy issue first.

See `UNVERIFIED.md` for what is inferred rather than measured.
