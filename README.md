# cuda-kernels

Progressive C++ and CUDA implementations — from CPU baselines to optimized GPU kernels.

A self-directed learning project working through CPU and CUDA performance
programming from first principles. Each project shows the full optimization
journey: naive → cache-aware → GPU-accelerated.

**Status:** Projects 01–03 (CPU baselines) complete; bonus GPU matmul preview done.
Phase 2 (04–07) and Phase 3 (08–10) are in progress — see the table below for
what's done vs. planned.

## Structure

| Phase | Projects | Focus |
|-------|----------|-------|
| [Phase 1 — C++ Baselines](phase1-cpp-baselines/) | 01–03 | Memory layout, cache behavior, CPU performance |
| [Phase 2 — CUDA Core](phase2-cuda-core/) | 04–07 | Host/device model, shared memory, warps, reductions |
| [Phase 3 — Advanced](phase3-advanced/) | 08–10 | Allocators, irregular access, fused kernels |

## Benchmark Summary

| Project | Problem Size | CPU Baseline | GPU Naive | GPU Best | Speedup |
|---------|-------------|--------------|-----------|----------|---------|
| 01 Matmul | N=1024 | 343.06 ms (tiled, best) | 5.00 ms | — | ~69x |
| 02 Prefix Scan | N=1M | — | — | — | — |
| 03 Image Convolution | 1024×1024, K=15 | — | — | — | — |
| 04 SAXPY | N=100M | — | — | — | — |
| 05 Histogram | N=10M | — | — | — | — |
| 06 Heat Diffusion | 2048×2048 | — | — | — | — |
| 07 Parallel Reduction | N=64M | — | — | — | — |
| 08 Memory Allocator | 10M allocs | — | — | — | — |
| 09 SpMV | varies | — | — | — | — |
| 10 Softmax | seq=8192 | — | — | — | — |

Full results: [benchmarks/results.md](benchmarks/results.md)

## Hardware

GPU: (fill in after `nvidia-smi`)  
CUDA: (fill in after `nvcc --version`)  
CPU: (fill in)
