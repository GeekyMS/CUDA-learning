# Project 09: Sparse Matrix-Vector Multiply (SpMV)

**Phase:** 3 — Advanced  
**Concept:** CSR format, irregular memory access, warp-per-row reduction

## Problem

Multiply a sparse matrix A (CSR format) by a dense vector x: `y = A * x`.

## Why This Matters

Irregular memory access patterns are the hard part of GPU programming. SpMV is
memory-bound with non-coalesced access — optimizing it teaches everything naive matmul
doesn't. Appears in graph algorithms, FEM solvers, and ML sparse attention.

## Versions

| Version | Key idea | Problem |
|---------|----------|---------|
| v1 — CPU CSR | Reference implementation | — |
| v2 — GPU thread/row | One thread per row | Low occupancy, variable work |
| v3 — GPU warp/row | One warp per row, `__shfl_down_sync` | Better load balance |

## Data Format — CSR (Compressed Sparse Row)

```
values[nnz]   — non-zero values
col_idx[nnz]  — column index of each value
row_ptr[N+1]  — row_ptr[i]..row_ptr[i+1]-1 are the non-zeros in row i
```

## C++/CUDA Concepts Targeted

- CSR format construction and iteration
- Load imbalance from variable row lengths
- Warp-level reduction with `__shfl_down_sync`
- NCU: low occupancy diagnosis, memory access pattern analysis

## Build

```bash
make SM=86
make bench SM=86     # requires a .mtx matrix in bench/
make profile SM=86
```

## Test Matrix

Download from SuiteSparse Matrix Collection: https://sparse.tamu.edu  
Recommended: `bcsstk17.mtx` (10,974 × 10,974, 428,650 non-zeros)

## Key Questions to Answer Before Moving On

1. Why is SpMV harder to optimize than dense matmul?
2. What causes load imbalance in the thread-per-row version?
3. How does warp-per-row reduce but not eliminate the imbalance?
