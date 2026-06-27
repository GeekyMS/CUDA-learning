# Project 03: Image Convolution / Gaussian Blur

**Phase:** 1 — C++ Baselines  
**Concept:** 2D stencil computation, separable kernels, image I/O

## Problem

Apply a 2D Gaussian blur to a grayscale image using convolution.

```
output[i][j] = sum_{ki,kj} input[i+ki][j+kj] * kernel[ki][kj]
```

## Why This Matters

2D convolution is the foundation of CNNs and image processing. On GPU, tiled convolution
with shared memory is a canonical optimization. This CPU version becomes the correctness
reference for future CUDA ports.

## Versions

| Version | Key idea | Complexity per pixel |
|---------|----------|---------------------|
| v1 — naive 2D conv | Direct sum over K×K window | O(K²) |
| v2 — separable conv | Horizontal pass then vertical pass | O(2K) |
| v3 — Gaussian kernel gen | `exp(-r² / 2σ²)`, normalized | — |

## C++ Concepts Targeted

- 2D array indexing on flat 1D buffer
- Zero-padding for border pixels
- Separability of Gaussian kernel: G(x,y) = G(x)·G(y)
- PGM/PPM image I/O (no external libs)

## Build

```bash
make            # builds all versions
make bench      # 1024×1024 image, K=15, v1 vs v2
make test       # identity kernel and box blur correctness checks
```

## Benchmark Target

For 1024×1024, K=15: v2 ~3–5x faster than v1.

## Key Questions to Answer Before Moving On

1. Why is the Gaussian kernel separable but a Sobel edge kernel is not?
2. What goes wrong at image borders and how does zero-padding address it?
