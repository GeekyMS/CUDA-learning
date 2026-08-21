#!/bin/bash
#SBATCH --job-name=matmul_gpu
#SBATCH --partition=gpu          # VERIFY: run `sinfo` and pick a real GPU partition name
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=matmul_gpu_%j.out

# Run from the project root: sbatch slurm/run.sbatch

module load cuda   # VERIFY: run `module avail cuda` and pin an exact version, e.g. `module load cuda/12.4`

# Build directly on the GPU node so `ARCH=native` (the Makefile default)
# can correctly detect this node's actual GPU.
make clean && make

./matmul_gpu
