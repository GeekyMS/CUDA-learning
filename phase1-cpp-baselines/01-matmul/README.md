# Project 01: Matrix Multiplication

**Phase:** 1 — C++ Baselines  
**Concept:** Row-major memory layout, cache behavior, loop tiling

## Problem

Multiply two NxN matrices: `C = A * B`.

This is the canonical GPU computing problem. You will rewrite this 4–5 more times in CUDA.
Understanding *why* the naive version is slow on CPU is the prerequisite to understanding
why shared memory helps on GPU.

## Versions

| Version | Key idea | Expected speedup |
|---------|----------|-----------------|
| v1 — naive ijk | Baseline, column-strided access on B | 1.0x |
| v2 — transposed B | Transpose B first → sequential access | ~2–4x |
| v3 — tiled | Block the computation to fit in L1/L2 | ~4–8x |

## C++ Concepts Targeted

- Heap allocation: `new float[N*N]`, flat 1D arrays
- Row-major indexing: `A[i*N + k]` vs `A[i][k]`
- Cache lines and spatial locality (CS:APP §6.6)
- `std::chrono` timing
- GFLOP/s measurement: `2*N^3 / time_s / 1e9`

## Build

```bash
make        # builds all three versions
make bench  # runs benchmark for N=256,512,1024
```

## Correctness Check

All three versions compared against reference for N=64. Max absolute error < 1e-4.

## Benchmark Target

For N=1024: v3 at least 3x faster than v1.

## Key Questions to Answer Before Moving On

1. Why does changing loop order (ijk → ikj) improve performance?
2. What is a cache line and how wide is it on your machine?
3. Why is `A[i*N + k]` not the same as `A[i][k]` for a 2D array in C?
