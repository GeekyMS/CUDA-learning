# Prefix Scan — CPU Baseline Results

Single-threaded and multi-threaded (`std::thread`), `-O2 -pthread`.

## Correctness (N=1024)

| Comparison | Max abs diff | Expected |
|---|---|---|
| sequential vs blelloch | 1.52588e-05 | < 1e-4 |
| sequential vs std::inclusive_scan | 0 | < 1e-4 |

All three agree within float rounding tolerance. The nonzero seq-vs-blelloch
diff is expected: Blelloch sums the same values in a different grouping
(tree-shaped instead of linear), and float addition isn't associative — see
[floating-point-nonassociativity.md](../../additional-learnings/floating-point-nonassociativity.md).
The seq-vs-std diff is exactly 0 because std::inclusive_scan (unthreaded here)
walks the array in the same left-to-right order as scan_sequential.

## Benchmark

| N | v1 sequential | v2 blelloch | v3 std::scan |
|---|---|---|---|
| 1,000,000 | 1.34 ms | 6.07 ms (0.22x) | 1.95 ms (0.69x) |
| 100,000,000 | 84.08 ms | 311.52 ms (0.27x) | 73.01 ms (1.15x) |

**Blelloch lost at every N tested** — both at 1M and 100M, it's slower than
plain sequential, and the gap doesn't meaningfully close as N grows 100x.
std::inclusive_scan roughly ties sequential (slightly worse at 1M, slightly
better at 100M, likely because it can vectorize the linear pass).

## Why Blelloch loses here

Two costs stack against it, and neither goes away with bigger N:

1. **Thread-spawn/join overhead.** `log2(N)` levels x 2 phases (upsweep +
   downsweep), each spawning up to `hardware_concurrency()` threads and
   joining them before the next level can start (level d+1 depends on level
   d's writes, so this barrier is mandatory for correctness — see
   [why-does-blelloch-need-gpu-hardware.md](../../additional-learnings/why-does-blelloch-need-gpu-hardware.md)
   for why this is cheap on GPU and expensive here). At N=100M that's ~27
   levels x 2 phases = ~54 level-passes, each creating/joining several OS
   threads — hundreds of syscalls of pure overhead before any useful work
   happens in some of them.

2. **Strided memory access.** Early tree levels touch memory with small
   strides (cache-friendly); later levels near the root have strides
   approaching N, jumping across large swaths of memory — same cache-miss
   problem as `matmul_naive`'s column-major read of B in Project 1. Sequential
   scan never has this problem: one linear pass, fully cache/prefetcher
   friendly, close to memory-bandwidth-bound (about as fast as this problem
   shape can go on a CPU).

Sequential scan's O(N) span (long dependency chain) sounds like the "worse"
algorithm on paper, but on this hardware it has zero synchronization cost and
optimal memory access pattern — both costs Blelloch pays repeatedly. The
theoretical span reduction (O(N) -> O(log N)) doesn't translate into wall-clock
wins unless the hardware makes synchronization and scattered access cheap,
which a CPU does not.

## What would actually help (not implemented here)

- **Skip threading at levels with little work** — many tree levels (near the
  root) have only a handful of independent updates, fewer than the thread
  count; spawning threads there is guaranteed pure overhead. A threshold
  guard that runs those levels single-threaded would cut a large fraction of
  wasted thread creation.
- **Persistent thread pool** — spin up N threads once, reuse them across all
  ~54 level-passes via a work queue, instead of creating/destroying threads
  every level. Removes the repeated creation cost; more code (condition
  variables, task queue) than this CPU-baseline exercise called for.

## Repo notes

The compiled `prefix_scan` binary is not committed — build artifact,
regenerated via `make`. Already covered by the root `.gitignore`.
