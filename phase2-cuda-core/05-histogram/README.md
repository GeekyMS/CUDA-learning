# Project 05: Histogram Computation

**Phase:** 2 — CUDA Core  
**Concept:** Atomic operations, shared memory per-block histograms

## Problem

Compute a 256-bin histogram of a large array of integers on GPU.

## Why This Matters

Teaches atomic operations and the shared memory optimization pattern that appears in
reduction, sort, and scan. The naive version has severe atomic contention — fixing it
with per-block shared histograms is a fundamental GPU programming pattern.

## Versions

| Version | Key idea | Bottleneck |
|---------|----------|-----------|
| v1 — global atomics | `atomicAdd(&hist[data[i]], 1)` | All threads contend on 256 global locations |
| v2 — shared memory | Per-block local histogram, then merge | Contention reduced by 1/blockSize |

## CUDA Concepts Targeted

- `atomicAdd` semantics and necessity
- `__shared__` memory allocation
- `__syncthreads()` between phases (zero → accumulate → merge)
- Race conditions and why they matter
- NCU profiling: L2 hit rate, atomic throughput

## Build

```bash
make SM=86
make bench SM=86   # N=10M random bytes, v1 vs v2
make profile SM=86 # runs ncu on both versions
```

## Key Questions to Answer Before Moving On

1. What is a race condition and why does `atomicAdd` prevent it?
2. Why is shared memory atomic contention lower than global memory contention?
3. What happens if you remove `__syncthreads()` between phases?
