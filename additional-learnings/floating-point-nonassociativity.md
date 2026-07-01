# Floating-Point Non-Associativity (why matmul variants gave slightly different results)

Came up while comparing `matmul_naive` vs `matmul_transposed` vs `matmul_tiled`
in [phase1-cpp-baselines/01-matmul](../phase1-cpp-baselines/01-matmul/) — see
`RESULTS.md` there for the actual numbers.

## The observation

- naive vs transposed: `max|C1-C2| = 2.38419e-06` (not exactly 0)
- naive vs tiled: `max|C1-C3| = 0` (exact match)

All three compute the same mathematical result, so why isn't the diff always 0?

## Why float addition isn't associative

A `float` stores `sign * mantissa * 2^exponent` with a fixed 23-bit mantissa.
Adding two floats requires aligning exponents, adding mantissas, then rounding
back to 23 bits — and rounding loses information differently depending on
operand magnitudes. So:

```
(a + b) + c   !=   a + (b + c)     // not guaranteed equal, bit-for-bit
```

Addition is still commutative (`a+b == b+a` exactly), but **not associative** —
the grouping/order of operations changes the rounding path and can shift the
last bit or two of the result.

## Why naive and tiled matched exactly, but transposed didn't

- `matmul_tiled` only changes the order in which *output elements* `(i,j)` are
  computed — for any single `(i,j)`, it still sums `k = 0..N-1` in increasing
  order, identical to naive. Same summation order → bit-identical result.
- `matmul_transposed` also loops `k` in the same logical order, but at `-O2`
  the compiler can auto-vectorize (SIMD, e.g. 4/8 floats per instruction),
  which sums partial vector lanes in a different order than the scalar loop.
  Different summation grouping → tiny rounding differences, even though the
  math is "the same."

## Takeaway for correctness checks

This is why the harness uses `max_abs_diff < 1e-4` instead of exact equality
(`==`) — bit-exact reproducibility across different loop orders/vectorization
is not a reasonable expectation for floating point, so correctness checks need
a tolerance based on expected accumulated rounding error, not zero-diff.

## When this kind of error matters a lot, and what to do about it

Relevant for: iterative solvers (errors compound over iterations), scientific
simulations needing cross-hardware reproducibility, financial calculations,
and multi-GPU neural net training (non-reproducible loss curves across GPU
counts).

Mitigations:
1. **Kahan summation (compensated summation)** — tracks the rounding error
   from each addition and feeds it back into the next one, greatly reducing
   accumulated error in long sums.
2. **Higher-precision accumulators** — accumulate in `double` even when
   inputs/outputs are `float` (common "mixed precision" pattern in real
   BLAS/matmul kernels).
3. **Deterministic reduction order** — enforce the same summation tree every
   run instead of letting it vary (cuBLAS offers deterministic modes for this,
   at a performance cost).
4. **Avoid aggressive float reordering flags** — e.g. `-ffast-math` lets the
   compiler reassociate float ops for speed; leaving it off (GCC/Clang
   default) keeps ordering closer to source code.
5. **Error-bound analysis over bit-exactness** — bound how much error can
   accumulate (roughly `N * epsilon_machine` scale) and set tolerance
   accordingly, rather than expecting exact equality.

## Forward note for CUDA work

This becomes directly relevant once doing **parallel reductions** on GPU —
summing partial results from many threads. Thread completion order is often
non-deterministic, so two runs of the same kernel can give slightly different
floating-point answers. Normal and expected; handle with mixed-precision
accumulation (accumulate in `float`/`double` even if I/O is `half`) plus a
tolerance-based correctness check, same pattern as the `1e-4` check used here.
