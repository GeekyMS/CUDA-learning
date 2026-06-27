# Project 02: Parallel Prefix Sum (Scan)

**Phase:** 1 — C++ Baselines  
**Concept:** Blelloch scan, work vs. span complexity, `std::thread`

## Problem

Compute inclusive and exclusive prefix sums over an array of N floats.
`out[i] = in[0] + in[1] + ... + in[i]`

## Why This Matters

Prefix scan is one of the most important parallel primitives — used inside radix sort,
stream compaction, and histogram equalization. CUDA has `cub::DeviceScan`.
Understanding it from scratch makes those abstractions transparent.
The parallel version here is *exactly* what the CUDA shared-memory scan (Project 05) looks like.

## Versions

| Version | Key idea |
|---------|----------|
| v1 — sequential | O(N) work, simple loop |
| v2 — parallel Blelloch | Upsweep (reduce tree) + downsweep (distribute); O(N) work, O(log N) span |
| v3 — `std::exclusive_scan` | C++17 reference implementation |

## C++ Concepts Targeted

- `std::thread`, lambda captures
- `std::atomic` for shared counters
- Work complexity vs. span (parallel) complexity
- Inclusive vs. exclusive scan semantics

## Build

```bash
make        # builds all versions
make bench  # N=1M random floats, sequential vs parallel timing
```

## Correctness Check

Compare v2 output against v1 for N=1M random floats. Max absolute error < 1e-4.

## Key Questions to Answer Before Moving On

1. What is the work complexity vs. span complexity of Blelloch scan vs. naive parallel scan?
2. Why does the sequential scan have better cache behavior than the parallel one on a single core?
