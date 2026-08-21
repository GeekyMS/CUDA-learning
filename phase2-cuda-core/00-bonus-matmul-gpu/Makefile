# Bonus GPU matmul — must be built where nvcc + an NVIDIA GPU exist (Unity),
# not on the local Mac.
#
# ARCH=native (default) tells nvcc to target whatever GPU is present on the
# machine doing the compile — only works if you compile ON a GPU node
# (e.g. inside an `srun --gres=gpu:1` session or a GPU-partition sbatch job),
# since it queries the actual hardware. If you ever need to cross-compile for
# a specific GPU from a non-GPU node instead, override explicitly, e.g.:
#   make ARCH=sm_80      # A100
#   make ARCH=sm_70      # V100
# Find the right sm_XX for a given card at https://developer.nvidia.com/cuda-gpus
# or by running `nvidia-smi --query-gpu=compute_cap --format=csv` on a GPU node.

NVCC     := nvcc
ARCH     := native
NVCCFLAGS := -O3 -arch=$(ARCH)
TARGET   := matmul_gpu

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: clean
