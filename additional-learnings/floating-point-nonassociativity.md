# Floating-Point Non-Associativity (why my matmul variants gave slightly different results)

Ran into this comparing `matmul_naive` vs `matmul_transposed` vs `matmul_tiled`
in [phase1-cpp-baselines/01-matmul](../phase1-cpp-baselines/01-matmul/) — see
`RESULTS.md` there for the actual numbers.

## What I saw

- naive vs transposed: `max|C1-C2| = 2.38419e-06` (not exactly 0)
- naive vs tiled: `max|C1-C3| = 0` (exact match)

All three are supposed to compute the same thing, so my first reaction was
"did I break something in transposed?" Turned out no — this is expected, and
here's why.

## Why float addition isn't associative

A `float` stores `sign * mantissa * 2^exponent` with a fixed 23-bit mantissa.
Adding two floats means aligning exponents, adding mantissas, then rounding
back to 23 bits — and that rounding loses information differently depending
on operand magnitudes. So:

```
(a + b) + c   !=   a + (b + c)     // not guaranteed equal, bit-for-bit
```

Addition is still commutative (`a+b == b+a` exactly), but **not associative**
— the grouping/order of operations changes the rounding path and can shift
the last bit or two of the result. I knew this in the abstract before, but
this was the first time I actually watched it happen in my own numbers.

## Why naive and tiled matched exactly, but transposed didn't

Took me a minute to figure out why only *one* of the two variants disagreed:

- `matmul_tiled` only changes the order in which *output elements* `(i,j)`
  get computed — for any single `(i,j)`, it still sums `k = 0..N-1` in
  increasing order, identical to naive. Same summation order → bit-identical
  result. Makes sense in hindsight.
- `matmul_transposed` loops `k` in the same logical order too, but at `-O2`
  the compiler can auto-vectorize it (SIMD, 4/8 floats per instruction),
  which sums partial vector lanes in a different order than the scalar loop
  would. Different summation grouping → tiny rounding differences, even
  though the math is "the same." I hadn't thought about auto-vectorization
  having this side effect until I saw the diff.

## Takeaway for my correctness checks

This is why I used `max_abs_diff < 1e-4` instead of exact equality (`==`) in
the harness — bit-exact reproducibility across different loop orders/
vectorization isn't realistic for floating point. Lesson for future
projects: pick tolerance based on expected accumulated rounding error, not
zero-diff, and don't panic when a "correct" refactor doesn't match bit-for-bit.

## Where this could actually bite me later

Noting these down so I remember to think about it when it matters more than
it did here: iterative solvers (errors compound over iterations), anything
needing cross-hardware reproducibility, financial calculations, multi-GPU
training (non-reproducible loss curves across GPU counts).

If it ever does matter:
1. **Kahan summation (compensated summation)** — tracks the rounding error
   from each addition and feeds it back into the next one, cuts accumulated
   error in long sums.
2. **Higher-precision accumulators** — accumulate in `double` even when
   inputs/outputs are `float` (mixed-precision pattern real BLAS/matmul
   kernels use).
3. **Deterministic reduction order** — force the same summation tree every
   run instead of letting it vary (cuBLAS has deterministic modes for this,
   at a perf cost).
4. **Avoid aggressive float reordering flags** — `-ffast-math` lets the
   compiler reassociate float ops for speed; leaving it off keeps ordering
   closer to what I actually wrote.
5. **Error-bound analysis over bit-exactness** — bound how much error can
   accumulate (roughly `N * epsilon_machine`) and set tolerance from that,
   instead of expecting exact equality.

## Note to self for when I get to CUDA reductions

This is going to come up again, harder, once I do **parallel reductions** on
GPU — summing partial results from many threads whose completion order isn't
deterministic, so two runs of the same kernel can give slightly different
answers. Expected, not a bug — remember the pattern from here: mixed-
precision accumulation plus a tolerance-based check instead of expecting
identical output every run.
