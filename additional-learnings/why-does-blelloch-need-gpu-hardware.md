# Why Blelloch Scan Is Designed For GPU Hardware, Not CPU

Came up while benchmarking `scan_parallel_blelloch` against sequential scan in
[phase1-cpp-baselines/02-prefix-scan](../phase1-cpp-baselines/02-prefix-scan/)
— see that project's `RESULTS.md` for the actual numbers (Blelloch lost to
sequential scan at both N=1M and N=100M).

## Work vs. span — the framing this project is built around

- **Work (W)**: total operations summed across all processors.
- **Span (S)**: length of the longest chain of operations that must happen in
  order — the theoretical minimum time given infinite processors.

Sequential scan: W = O(N), S = O(N) — one unbroken dependency chain, no
parallelism possible.

Blelloch scan: W = O(N) (same order, just a different constant — see the
geometric-series argument for why upsweep+downsweep doesn't cost O(N log N)),
S = O(log N) — reshapes the computation into a binary-tree-shaped dependency
graph, so with enough processors, the critical path is only `log2(N)` levels
deep instead of N steps deep.

On paper, O(log N) span is a huge asymptotic win over O(N). In practice, on a
CPU, it lost by 4-5x. Why the theory didn't translate to wall-clock time is
the interesting part.

## The two costs Blelloch pays that sequential scan doesn't

1. **A synchronization barrier between every tree level.** Level d+1 reads
   values level d just wrote, so all threads at level d must finish
   (`join()`) before level d+1 can start. This is mandatory for correctness,
   not optional.
2. **Strided memory access that gets worse near the root.** Early levels
   touch memory with small strides (cache-friendly); later levels have
   strides approaching N (cache-hostile), same problem as reading a matrix
   column-major.

Sequential scan pays neither cost — one thread, one linear pass, zero
synchronization, perfect cache behavior.

## Why this exact algorithm works well on GPU despite paying "the same" costs

The algorithm structure (per-level barrier, strided tree access) is identical
whether you run it on CPU threads or GPU threads. What's different is the
hardware's cost model for those two things:

- **Synchronization**: GPU's `__syncthreads()` is a hardware barrier
  instruction *within a thread block* — threads are already resident on the
  same streaming multiprocessor, so this is a nanosecond-scale wait, not a
  syscall. `std::thread::join()` on a CPU involves the OS scheduler, real
  kernel threads, and is orders of magnitude more expensive per
  synchronization point.
- **Memory access**: a GPU doesn't make individual scattered/strided memory
  accesses fast — a cache-missing global memory read still costs hundreds of
  cycles, same as on a CPU. What it does instead is keep thousands of threads
  in flight (high occupancy), so when one warp stalls on memory, the
  scheduler switches to a different warp with ready work. The latency is
  still there; it's hidden behind other useful work. A CPU has a handful of
  threads (matched to physical core count) — when one stalls, there usually
  isn't a large pool of alternate ready work to hide behind, so the stall
  shows up directly in wall-clock time.
- **Thread count/weight**: CPU threads are heavyweight OS entities (own
  stack, scheduler bookkeeping, expensive to create). GPU threads are
  extremely lightweight (register-based, essentially free to schedule in
  bulk) — spinning up thousands of them for one kernel launch is normal and
  cheap, the opposite of spinning up hundreds of `std::thread`s.

So Blelloch scan isn't "better" or "worse" as an algorithm in the abstract —
it trades more total work/complexity for a shorter dependency chain, and
whether that trade pays off depends entirely on how expensive synchronization
and non-sequential memory access are on the hardware actually running it. CPU:
expensive sync + cache-dependent access + few heavyweight threads → the trade
loses. GPU: cheap sync + latency-hidden access + thousands of lightweight
threads → the same trade wins.

## Nuance: real GPU scan implementations aren't naively "the same tree, just on GPU"

Production implementations (`cub::DeviceScan`, the classic GPU Gems Blelloch
scan) don't run the log-depth strided tree pattern all the way up against
global memory either. Within a thread block, they do the tree scan in
**shared memory** (on-chip, ~100x faster than global memory), carefully
padding indices to avoid shared-memory bank conflicts — the GPU-specific
version of "watch your stride pattern." Only the combination *across* blocks
falls back to a pass over global memory. So even GPU implementations respect
a version of the same lesson (minimize expensive-memory-tier access) — they
just have a faster on-chip tier to exploit that a CPU's cache hierarchy
serves a similar but less controllable role for.

## Forward note

This is the throughline connecting Project 1 (matmul) and Project 2 (scan):
**the "right" algorithm is inseparable from the memory/synchronization cost
model of the hardware it runs on.** This CPU exercise exists specifically to
build the work/span framework and show it failing to pay off on CPU hardware,
so that when this same Blelloch structure resurfaces as an actual CUDA kernel
in phase 2 of the project plan, the payoff (and the shared-memory bank-conflict
caveat) will make sense from first principles instead of being taken on faith.
