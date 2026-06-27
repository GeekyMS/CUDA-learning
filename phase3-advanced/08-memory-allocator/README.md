# Project 08: Custom Memory Allocator

**Phase:** 3 — Advanced  
**Concept:** Pool allocation, RAII, GPU pinned memory management

## Problem

Build a pool allocator in C++, then extend it to manage GPU pinned memory.

## Why This Matters

`cudaMalloc` is expensive — it synchronizes the device. Production GPU code pools
allocations. Understanding allocators is the difference between someone who uses cuBLAS
and someone who understands why it's fast.

## Versions

| Version | Key idea |
|---------|----------|
| v1 — CPU bump-pointer pool | Pre-allocated slab, O(1) alloc, reset-to-free |
| v2 — CPU free-list allocator | Per-block free, first-fit and best-fit, fragmentation |
| v3 — GPU pinned memory pool | `cudaHostAlloc` slab, same interface, DMA-friendly |

## C++ Concepts Targeted

- `placement new`, RAII, `std::aligned_alloc`
- Move semantics, `noexcept`
- Linked list for free-list management
- Why `cudaMalloc` is slow (device synchronization)
- Pinned vs. pageable memory and DMA

## Build

```bash
make SM=86
make bench SM=86    # pool vs malloc for 10M small allocs; pinned vs per-transfer overhead
```

## Benchmark Target

Pool allocator at least 10x faster than `malloc` for 10M small allocations.

## Key Questions to Answer Before Moving On

1. Why is a bump-pointer allocator O(1) but also limited?
2. What is external fragmentation and why does first-fit produce more than best-fit?
3. Why does `cudaMalloc` synchronize the device?
