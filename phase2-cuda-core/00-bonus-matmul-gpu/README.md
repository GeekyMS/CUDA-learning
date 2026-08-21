# Bonus: Matmul on GPU (early preview)

**Status:** out-of-sequence bonus, done ahead of the formal Phase 2 start
because Unity cluster access was available early. The numbered Phase 2
sequence (Project 4: SAXPY onward) still starts from first principles.

## Why this exists

Project 1 built `matmul_naive` on CPU and taught cache/memory-access lessons.
This is the same algorithm, ported to a real `__global__` CUDA kernel, run on
an actual NVIDIA GPU — a first hands-on look at the execution model (thread
hierarchy, host/device memory) before Phase 2 formally introduces it.

## Requirements

Must be built and run on a machine with an NVIDIA GPU + the CUDA toolkit —
**not the local Mac** (Apple Silicon has no CUDA support at all). This repo's
workflow: edit locally, push to GitHub, then on Unity `git pull` and build
there. See the `unity-cluster-workflow` note for the general pattern.

## What to implement

`src/main.cu` has one TODO: the `matmul_naive_gpu` kernel. Everything else
(host-side allocation, `cudaMemcpy` transfers, launch configuration, timing,
CPU-reference correctness check) is provided — read through `main()` to see
what the kernel is expected to produce.

## Build & run on Unity

```bash
ssh <you>@unity.rc.umass.edu
cd cuda-kernels && git pull        # sync latest code from GitHub
cd phase2-cuda-core/00-bonus-matmul-gpu

# Option A: interactive session (good for iterating while writing the kernel)
srun --partition=gpu --gres=gpu:1 --time=00:15:00 --pty bash
module load cuda
make
./matmul_gpu
exit                                # leave the interactive allocation when done

# Option B: batch job (good once the kernel is working, for logging results)
sbatch slurm/run.sbatch
squeue -u $USER                     # check job status
cat matmul_gpu_<jobid>.out          # see output once it finishes
```

Verify partition name and CUDA module version for your account with `sinfo`
and `module avail cuda` before running — placeholders in `slurm/run.sbatch`
may need adjusting.

## Correctness target

`main()` compares GPU output against a CPU reference; expect
`max|gpu-cpu| < 1e-2` (looser tolerance than the CPU-only Projects 1-2 checks,
since GPU and CPU floating-point summation order — and therefore rounding —
differ, same non-associativity issue as
[floating-point-nonassociativity.md](../../additional-learnings/floating-point-nonassociativity.md)).
