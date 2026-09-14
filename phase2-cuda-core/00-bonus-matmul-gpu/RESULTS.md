# Bonus: Matmul on GPU — Results

Run on UMass Unity, `nvcc -O3 -arch=native`, CUDA 13.1 module (Pascal-class
GPU node — `-arch=native` initially resolved to `compute_61`, which CUDA
13.1's `nvcc` no longer supports; had to reconfirm the node's actual
architecture is supported by whichever CUDA module is loaded).

## Correctness (N=1024)

```
Correctness: max|gpu-cpu| = 1.90735e-05 (expect < 1e-2)
```

Passes, well under tolerance. I noticed the diff is larger than the pure
CPU-vs-CPU diffs from Projects 1-2 (`~1e-6` to `~1e-5` there) and had to
think about why — most likely GPUs commonly fuse multiply-add into a single
FMA instruction with different intermediate rounding than the CPU's separate
multiply-then-add step, on top of the usual summation-order non-associativity
effect I already ran into in Project 1 (see
[floating-point-nonassociativity.md](../../additional-learnings/floating-point-nonassociativity.md)).

## Benchmark (N=1024)

| Version | Time | GFLOP/s | Speedup vs CPU naive |
|---|---|---|---|
| CPU naive (Project 1) | 972.18 ms | 2.21 | 1x |
| CPU transposed (Project 1) | 630.24 ms | 3.41 | 1.5x |
| CPU tiled, best (Project 1, tile=16) | 343.06 ms | 6.26 | 2.8x |
| **GPU naive (this kernel)** | **5.00 ms** | **429.11** | **~194x** |

The naive, unoptimized GPU kernel — no shared memory, no tiling, just one
thread per output element — beats the *best hand-tuned CPU version* by ~69x,
and beats CPU naive by ~194x. Genuinely surprised me the first time I saw
this number; wanted to actually understand why before chalking it up to
"GPUs are just fast."

## Why a "naive" GPU kernel already crushes hand-tuned CPU code

All the CPU-side tiling work I did in Project 1 existed to work around a
handful of cores and a small cache. The GPU sidesteps that problem
differently: it launches roughly a million threads (`1024x1024`), each doing
its own independent dot product, and the sheer volume of concurrent work
keeps the memory system saturated even though any single thread's own access
pattern (`B[k*N+col]`, strided as `k` increases — the exact "bad" pattern
that made CPU `matmul_naive` slow) is just as cache-unfriendly per-thread as
the CPU version. Latency doesn't need to disappear, it just needs to be
hidden behind other threads' useful work, and there's enough concurrent work
here to do that — same latency-hiding idea I worked out in
[why-does-blelloch-need-gpu-hardware.md](../../additional-learnings/why-does-blelloch-need-gpu-hardware.md)
for the scan-vs-thread-overhead case.

## Memory access pattern — who reads what, and how much sharing happens

Worked this out by hand to make sure I actually understood what the kernel
was doing, not just that it was fast. With `blockDim = 16x16` (256
threads/block, 4096 blocks total for N=1024):

- **`C[row*N+col]`** — written by exactly one thread each. No contention.
- **`A[row*N+k]`** — depends only on `row` and `k`, not `col`. Every thread
  sharing the same `row` (i.e. the 16 threads with the same `threadIdx.y` in
  a block) reads the *identical* address at the same loop iteration `k`,
  since all threads execute in lockstep with no data-dependent branching.
  The hardware serves this as a **broadcast** — one memory transaction feeds
  every thread that asked for the same address, instead of fetching it once
  per thread.
- **`B[k*N+col]`** — depends only on `k` and `col`, not `row`. Threads with
  *adjacent* `threadIdx.x` (i.e. adjacent `col`) read **consecutive
  addresses** in memory at the same `k`. When a warp's 32 threads request
  contiguous addresses simultaneously, the memory controller merges them
  into one wide transaction — **coalesced access**. This is the same
  underlying strided-per-thread pattern that hurt CPU `matmul_naive`, but on
  GPU what matters is that *many threads together, at one instant*, are
  reading contiguous addresses — not what any single thread's own access
  sequence looks like over time.
- **Redundancy across blocks**: this naive kernel has no caching of its own.
  64 different blocks all cover the same `row`-band (splitting the 1024
  columns into 16-wide slices), and each independently re-reads the same row
  of `A` from global memory as `k` advances — 64x redundant traffic on `A`,
  and symmetrically on `B`. Only the GPU's own L2 cache incidentally
  mitigates this; the kernel itself does nothing to prevent it.

## Deferred: shared-memory tiled version

That last point — redundant re-fetching of the same `A`/`B` data across
blocks — is exactly what a **tiled, shared-memory GPU kernel** eliminates:
each block cooperatively loads a tile of `A` and `B` into fast on-chip shared
memory once, then every thread in that block reuses it from shared memory
instead of re-hitting global memory. I stubbed out a `matmul_tiled_gpu`
kernel with a full TODO writeup in `src/main.cu`, but deliberately **haven't
implemented or wired it into `main()` yet** — holding off until the formal
Phase 2 sequence reaches shared memory (this bonus project was me jumping
ahead of Project 4 early because I had cluster access, not meant to
front-run the whole phase).
