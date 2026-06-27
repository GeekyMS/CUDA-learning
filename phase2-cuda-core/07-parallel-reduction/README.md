# Project 07: Parallel Reduction

**Phase:** 2 — CUDA Core  
**Concept:** 7-version optimization journey following NVIDIA's canonical guide

## Problem

Sum all elements of a large float array on GPU. Minimize time to peak memory bandwidth.

## Why This Matters

Reduction is the building block of everything — dot products, norms, convergence checks,
softmax denominators. Working through all 7 Harris versions teaches every shared memory
and warp-level optimization in a single, tightly scoped problem.

**Required reading:** "Optimizing Parallel Reduction in CUDA" — Mark Harris (NVIDIA)

## Versions

| Version | Optimization Applied |
|---------|----------------------|
| v0 | Interleaved addressing — divergent branching |
| v1 | Interleaved addressing — bank-conflict-free |
| v2 | Sequential addressing |
| v3 | First add during global load |
| v4 | Unroll last warp |
| v5 | Completely unrolled |
| v6 | Multiple elements per thread + `__shfl_down_sync` |

## CUDA Concepts Targeted

- Shared memory bank conflicts (v0 → v1)
- Warp divergence in conditionals (v0)
- Why the last warp doesn't need `__syncthreads()` (v4)
- `__shfl_down_sync` warp intrinsics (v6)
- Roofline model: is this kernel memory-bound or compute-bound?

## Build

```bash
make SM=86
make bench SM=86    # all 7 versions, N=64M
make profile SM=86  # NCU roofline for v6
```

## Benchmark Target

v6 should approach peak memory bandwidth of your GPU.

## Key Questions to Answer Before Moving On

1. What is a "bank conflict" in shared memory and how does v1 fix v0?
2. Why can we "unroll the last warp" without `__syncthreads()`?
3. What does `0xffffffff` mean as the mask in `__shfl_down_sync`?
