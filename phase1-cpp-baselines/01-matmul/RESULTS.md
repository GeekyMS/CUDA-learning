# Matrix Multiplication — CPU Baseline Results

N=1024, `-O2`, single-threaded, row-major flat storage.

## Correctness (N=64)

| Comparison | Max abs diff | Expected |
|---|---|---|
| naive vs transposed | 2.38419e-06 | < 1e-4 |
| naive vs tiled | 0 | < 1e-4 |

All three implementations agree (within float rounding tolerance).

## Benchmark (N=1024)

| Version | Time | GFLOP/s |
|---|---|---|
| v1 naive | 972.175 ms | 2.21 |
| v2 transposed | 630.244 ms | 3.41 |
| v3 tiled(8) | 413.524 ms | 5.19 |
| **v3 tiled(16)** | **343.064 ms** | **6.26** |
| v3 tiled(32) | 414.314 ms | 5.18 |
| v3 tiled(64) | 605.489 ms | 3.55 |

Best result: **tiled, tile=16 → ~2.8x faster than naive**.

## Why each version behaves the way it does

### v1 — naive (ijk loop order)
```
C[i][j] = sum_k A[i][k] * B[k][j]
```
`A` is read row-wise (stride-1, cache-friendly), but `B` is read column-wise
(`B[k*N+j]` — each step of `k` jumps `N` floats ahead in memory). For N=1024
that's a 4KB stride per access, so almost every read of `B` is a cache miss.
This is the reason naive is the slowest here — the CPU is spending most of
its time waiting on memory, not actually computing.

### v2 — transposed (ikj loop order + pre-transposed B)
I transpose `B` once into `Bt` so that `Bt[j*N+k]` accesses memory the same
way `A[i*N+k]` does — sequentially, stride-1. Both operands now stream
through memory in the order the prefetcher expects, cutting cache misses
drastically. Got ~1.5x speedup from this alone, with the exact same amount
of arithmetic — good confirmation that for this workload, memory access
pattern matters more than instruction count.

### v3 — tiled (cache-blocked)
Even with sequential access, a full row of `A`, `B`, or `C` at N=1024
(1024 * 4 bytes = 4KB) times three matrices doesn't comfortably fit in a
small L1 cache. Tiling restricts each pass to a `tile x tile` block of all
three matrices, so the "working set" for a tile stays resident in L1/L2
cache while it's being reused across the innermost loops, instead of getting
evicted and re-fetched from RAM.

The tile-size curve is the part I found most interesting to actually see
play out:
- **tile=8** — blocks small enough to easily fit cache, but loop overhead
  (incrementing/branching relative to actual work done) starts to matter.
- **tile=16** — the sweet spot I landed on. Three 16x16 float tiles = 3 * 1KB
  = 3KB, comfortably inside a typical 32KB L1 cache, while still doing
  enough work per tile to amortize loop overhead.
- **tile=32 / tile=64** — working set per tile grows (64x64 tile = 16KB per
  matrix, 48KB total for three matrices), starts exceeding L1 capacity, so
  performance degrades back toward transposed-only levels as the CPU evicts
  and re-fetches cache lines mid-tile.

This matches the shape I'd expect for cache-blocking experiments in
general: performance rises then falls as tile size crosses the L1 capacity
boundary. Worth remembering that the actual optimal tile size is going to
depend on whatever CPU's cache size I'm running on, not a fixed constant I
can reuse blindly next time.
