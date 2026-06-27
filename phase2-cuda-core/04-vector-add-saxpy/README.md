# Project 04: Vector Addition → SAXPY

**Phase:** 2 — CUDA Core  
**Concept:** Host/device memory pipeline, bandwidth measurement, pinned memory

## Problem

GPU vector addition and SAXPY: `out[i] = a * x[i] + y[i]`

## Why This Matters

This is the "Hello World" of CUDA. Forces you to set up the full host↔device memory
transfer pipeline before doing anything interesting.

## Versions

| Version | Key idea |
|---------|----------|
| v1 — vector add | Baseline: `cudaMalloc`, `cudaMemcpy`, kernel, verify |
| v2 — SAXPY | Extend to fused multiply-add |
| v3 — bandwidth measurement | Measured GB/s vs theoretical peak |
| v4 — pinned memory | `cudaHostAlloc` for 2–3x faster H↔D transfers |

## CUDA Concepts Targeted

- `cudaMalloc` / `cudaFree` / `cudaMemcpy`
- Kernel launch: `kernel<<<gridDim, blockDim>>>(args)`
- Thread indexing: `int i = blockIdx.x * blockDim.x + threadIdx.x`
- Guard pattern: `if (i < N)`
- `cudaDeviceSynchronize()` before timing
- Pinned vs. pageable host memory

## Build

```bash
# Set SM version for your GPU (e.g., sm_86 for RTX 30xx)
make SM=86
make bench SM=86
```

## Key Questions to Answer Before Moving On

1. What happens if you launch with 1 thread? 1 block? More threads than N?
2. Why does the `if (i < N)` guard matter?
3. What is "warp divergence" and does vecAdd have any?
