# Project 10: Fused Softmax Kernel

**Phase:** 3 — Advanced  
**Concept:** Online (single-pass) numerically stable softmax, kernel fusion, Flash Attention foundation

## Problem

Compute row-wise softmax over a matrix of logits.

```
softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
```

## Why This Matters

This is the gateway to Flash Attention. Online softmax requires a non-trivial algorithm
insight — maintaining a running (max, sum_exp) pair that lets you correct previous
estimates as you scan. Fusing the passes into one kernel directly demonstrates the
memory bandwidth savings that make transformer inference fast.

**Required background:** Read the Flash Attention paper (Dao et al., 2022) introduction.

## Versions

| Version | Passes over data | Key idea |
|---------|-----------------|---------|
| v1 — naive (3 kernels) | 3 | Separate max-reduce, exp-sum, normalize kernels |
| v2 — online (1 pass) | 1 | Running `(m, s)` pair with rescaling; warp shuffle |
| v3 — tiled online | 1 | Tiles in shared memory, two-level reduction |

## Online Softmax Update Rule

When scanning and encountering a new value that exceeds the current max:
```
new_max = max(old_max, x_new)
sum     = sum * exp(old_max - new_max) + exp(x_new - new_max)
max     = new_max
```

This rescales the running sum without needing to revisit previous elements.

## CUDA Concepts Targeted

- Kernel fusion for memory bandwidth savings
- Online algorithms and numerically stable reductions
- `__shfl_down_sync` for warp-level (max, sum) reduction
- Correctness validation against PyTorch `F.softmax`

## Build

```bash
make SM=86
make bench SM=86    # seq_len = 1024, 4096, 8192, 16384
make verify SM=86   # compare against PyTorch reference
make profile SM=86
```

## Correctness

```bash
python3 bench/verify.py   # compares kernel output to torch.nn.functional.softmax
# Max absolute error must be < 1e-5
```

## Benchmark Target

v2 approaches 2–3x GB/s improvement over v1 for seq_len > 8192.

## Key Questions to Answer Before Moving On

1. Why does naive softmax need 3 passes? What would go wrong with 1 pass naively?
2. Derive the online update rule: when you see a new max, why multiply sum by `exp(old - new)`?
3. Why is kernel fusion valuable even when the fused kernel isn't "faster" per FLOP?
