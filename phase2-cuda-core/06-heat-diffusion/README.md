# Project 06: 2D Heat Diffusion

**Phase:** 2 — CUDA Core  
**Concept:** 2D thread indexing, shared memory with halo cells, ping-pong buffers, async streams

## Problem

Finite difference simulation of heat spreading on a 2D plate.

**Physics:** `u_t = α * (u_xx + u_yy)` — heat equation, explicit Euler discretization.

```
u_next[x,y] = u[x,y] + α * (u[x+1,y] + u[x-1,y] + u[x,y+1] + u[x,y-1] - 4*u[x,y])
```

## Why This Matters

This is the **anchor project**. It teaches every core CUDA pattern in one problem:
2D indexing, shared memory with halos, iterative simulation, convergence, async streams.
~60% of practical CUDA patterns appear here. Take your time.

## Versions

| Version | Key idea | Expected speedup |
|---------|----------|-----------------|
| v1 — naive GPU | Global memory stencil, ping-pong buffers | baseline |
| v2 — shared memory + halos | Load tile + ghost cells, 4x fewer global reads | ~2–4x |
| v3 — convergence check | `cub::DeviceReduce::Max` on `|u_next - u|` | — |
| v4 — async streams | Overlap compute with H→D frame copies | — |

## CUDA Concepts Targeted

- 2D thread blocks: `dim3 block(16,16)`, `dim3 grid(W/16, H/16)`
- Ping-pong buffers (why you need two)
- Halo / ghost cells for stencils in shared memory
- CFL stability condition: `α * dt / dx² ≤ 0.25`
- `cudaMemcpyAsync` + `cudaStreamSynchronize`
- Occupancy analysis via block size choice

## Build

```bash
make SM=86
make bench SM=86     # 512x512, 1024x1024, 2048x2048
make animate SM=86   # outputs frames for Python visualization
make profile SM=86
```

## Visualization

```bash
python3 bench/animate.py  # generates heat_diffusion.gif
```

## Benchmark Target

v2 at least 2–4x faster than v1 for grids ≥ 2048×2048.

## Key Questions to Answer Before Moving On

1. What is the CFL stability condition and why does your time step need to satisfy it?
2. Why do we need two buffers (ping-pong)? What goes wrong with one?
3. What are "halo cells" and why does shared memory require them for stencils?
4. What is occupancy and how does block size affect it?
