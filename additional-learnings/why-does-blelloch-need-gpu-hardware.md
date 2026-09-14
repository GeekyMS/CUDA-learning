# Why Blelloch Scan Is Designed For GPU Hardware, Not CPU

Ran into this benchmarking `scan_parallel_blelloch` against sequential scan in
[phase1-cpp-baselines/02-prefix-scan](../phase1-cpp-baselines/02-prefix-scan/)
— see that project's `RESULTS.md` for the actual numbers. Blelloch lost to
plain sequential scan at both N=1M and N=100M, which was not what I expected
going in, so I wanted to actually understand why before moving on.

## Work vs. span — the framing I built this project around

- **Work (W)**: total operations summed across all processors.
- **Span (S)**: length of the longest chain of operations that must happen in
  order — the theoretical minimum time given infinite processors.

Sequential scan: W = O(N), S = O(N) — one unbroken dependency chain, no
parallelism possible.

Blelloch scan: W = O(N) (same order, just a different constant — the
geometric-series argument is why upsweep+downsweep doesn't cost O(N log N)),
S = O(log N) — reshapes the computation into a binary-tree-shaped dependency
graph, so with enough processors the critical path is only `log2(N)` levels
deep instead of N steps deep.

On paper O(log N) span looked like a clear win over O(N). On my CPU it lost
by 4-5x. Figuring out why the theory didn't translate to wall-clock time was
the actual point of this exercise.

## The two costs Blelloch pays that sequential scan doesn't

1. **A synchronization barrier between every tree level.** Level d+1 reads
   values level d just wrote, so all threads at level d must finish
   (`join()`) before level d+1 can start. Mandatory for correctness, not
   optional.
2. **Strided memory access that gets worse near the root.** Early levels
   touch memory with small strides (cache-friendly); later levels have
   strides approaching N (cache-hostile) — same problem as the column-major
   read that made `matmul_naive` slow in Project 1.

Sequential scan pays neither cost — one thread, one linear pass, zero
synchronization, perfect cache behavior. Once I laid it out this way the 4-5x
loss stopped being surprising.

## Why I think this same algorithm should win on GPU despite paying "the same" costs

The algorithm structure (per-level barrier, strided tree access) doesn't
change between CPU threads and GPU threads. What's different is the
hardware's cost model for those two things — this is the part I had to
reason through rather than just observe:

- **Synchronization**: GPU's `__syncthreads()` is a hardware barrier
  instruction *within a thread block* — threads are already resident on the
  same streaming multiprocessor, so it's a nanosecond-scale wait, not a
  syscall. `std::thread::join()` on my CPU goes through the OS scheduler,
  real kernel threads — orders of magnitude more expensive per sync point.
- **Memory access**: a GPU doesn't make scattered/strided memory accesses
  fast — a cache-missing global memory read still costs hundreds of cycles,
  same as on CPU. What it does instead is keep thousands of threads in
  flight, so when one warp stalls on memory, the scheduler switches to a
  different warp with ready work. The latency doesn't go away, it gets
  hidden behind other useful work. My CPU only has a handful of threads —
  when one stalls there usually isn't a big pool of alternate work to hide
  behind, so the stall shows up directly in the wall-clock time I measured.
- **Thread count/weight**: CPU threads are heavyweight OS entities (own
  stack, scheduler bookkeeping, expensive to create). GPU threads are
  register-based and essentially free to schedule in bulk — spinning up
  thousands of them per kernel launch is normal, the opposite of spinning up
  hundreds of `std::thread`s like my Blelloch implementation does.

So I don't think Blelloch scan is "better" or "worse" as an algorithm in the
abstract — it trades more total work/complexity for a shorter dependency
chain, and whether that trade pays off depends entirely on how expensive
synchronization and non-sequential memory access are on the hardware
actually running it. On my CPU: expensive sync + cache-dependent access +
few heavyweight threads → the trade loses. On GPU it should be the reverse.
I'll get to actually verify this once the CUDA version of this shows up in
Phase 2.

## Nuance I want to remember: real GPU scan implementations aren't naively "the same tree, just on GPU"

Production implementations (`cub::DeviceScan`, the classic GPU Gems Blelloch
scan) don't run the log-depth strided tree pattern all the way up against
global memory either. Within a thread block they do the tree scan in
**shared memory** (on-chip, ~100x faster than global memory), carefully
padding indices to avoid shared-memory bank conflicts — the GPU-specific
version of "watch your stride pattern." Only the combination *across* blocks
falls back to a pass over global memory. Worth remembering: even the "real"
GPU version respects the same lesson about minimizing expensive-memory-tier
access, it just has a faster on-chip tier available that my CPU's cache
hierarchy doesn't give me the same control over.

## Note to self connecting this back to Project 1

This is the throughline between the matmul project and this one:
**the "right" algorithm is inseparable from the memory/synchronization cost
model of the hardware it runs on.** I built the work/span framework here
specifically so I'd have it in hand when this same Blelloch structure shows
up again as an actual CUDA kernel in Phase 2 — I want the payoff (and the
shared-memory bank-conflict caveat) to click from first principles instead
of just taking it on faith because a textbook said so.
