# Benchmark Results

Hardware: (fill in: GPU model, VRAM, peak BW)  
Compiler: nvcc (version), -arch=sm_XX -O2  

---

## Project 01: Matrix Multiplication

| Version | N | Time (ms) | GFLOP/s | Speedup vs v1 |
|---------|---|-----------|---------|---------------|
| v1 naive | 1024 | 972.18 | 2.21 | 1.0x |
| v2 transposed B | 1024 | 630.24 | 3.41 | 1.5x |
| v3 tiled (tile=16, best) | 1024 | 343.06 | 6.26 | 2.8x |
| GPU naive (bonus, Phase 2 preview) | 1024 | 5.00 | 429.11 | ~194x |
| GPU best (shared-memory tiled) | 1024 | — | — | — |

---

## Project 02: Prefix Scan

| Version | N | Time (ms) | GB/s | Speedup vs v1 |
|---------|---|-----------|------|---------------|
| v1 sequential | 1M | — | — | 1.0x |
| v2 parallel (Blelloch) | 1M | — | — | — |

---

## Project 03: Image Convolution

| Version | Image | Kernel | Time (ms) | Speedup vs v1 |
|---------|-------|--------|-----------|---------------|
| v1 naive | 1024×1024 | K=15 | — | 1.0x |
| v2 separable | 1024×1024 | K=15 | — | — |

---

## Project 04: SAXPY

| Version | N | Time (ms) | GB/s | Speedup vs v1 |
|---------|---|-----------|------|---------------|
| v1 vector add | 100M | — | — | 1.0x |
| v2 saxpy | 100M | — | — | — |
| v3 pinned memory | 100M | — | — | — |

---

## Project 05: Histogram

| Version | N | Time (ms) | Elements/s | Speedup vs v1 |
|---------|---|-----------|------------|---------------|
| v1 global atomics | 10M | — | — | 1.0x |
| v2 shared memory | 10M | — | — | — |

---

## Project 06: Heat Diffusion

| Version | Grid | Iters/s | GB/s | Speedup vs v1 |
|---------|------|---------|------|---------------|
| v1 naive GPU | 2048×2048 | — | — | 1.0x |
| v2 shared memory + halo | 2048×2048 | — | — | — |
| v3 + convergence check | 2048×2048 | — | — | — |
| v4 + async streams | 2048×2048 | — | — | — |

---

## Project 07: Parallel Reduction

| Version | N | Time (ms) | GB/s | Speedup vs v0 |
|---------|---|-----------|------|---------------|
| v0 interleaved (divergent) | 64M | — | — | 1.0x |
| v1 interleaved (no bank conflict) | 64M | — | — | — |
| v2 sequential addressing | 64M | — | — | — |
| v3 first add during load | 64M | — | — | — |
| v4 unroll last warp | 64M | — | — | — |
| v5 completely unrolled | 64M | — | — | — |
| v6 warp shuffle | 64M | — | — | — |

---

## Project 08: Memory Allocator

| Version | Allocs | Time (ms) | Speedup vs malloc |
|---------|--------|-----------|-------------------|
| malloc baseline | 10M | — | 1.0x |
| pool allocator (CPU) | 10M | — | — |
| free-list allocator | 10M | — | — |
| pinned GPU pool | 10M | — | — |

---

## Project 09: SpMV

| Version | Matrix | NNZ | Time (ms) | GB/s | Speedup vs v1 |
|---------|--------|-----|-----------|------|---------------|
| v1 CPU CSR | — | — | — | — | 1.0x |
| v2 GPU naive (thread/row) | — | — | — | — | — |
| v3 GPU warp/row | — | — | — | — | — |

---

## Project 10: Softmax

| Version | Seq len | Time (ms) | GB/s | Speedup vs v1 |
|---------|---------|-----------|------|---------------|
| v1 naive (3 kernels) | 8192 | — | — | 1.0x |
| v2 online (1 pass) | 8192 | — | — | — |
| v3 tiled online | 8192 | — | — | — |
