// Bonus / early preview: Matrix Multiplication — first real CUDA kernel
//
// Out-of-sequence port of Project 1's matmul_naive to the GPU, done ahead of
// the formal Phase 2 start. Must be compiled and run on a machine with an
// NVIDIA GPU (UMass Unity, not the local Mac) — see ../README.md.
//
// Implement the TODO kernel below. Do not change its signature — main()
// depends on it. The CPU reference and all host-side boilerplate
// (allocation, transfer, timing, correctness check) are provided.
//
// Build: nvcc -O3 -arch=sm_XX -o matmul_gpu src/main.cu   (see ../Makefile)
// Run:   ./matmul_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA error checking — every CUDA API call can fail (bad launch config, out
// of device memory, etc.), and unlike C++ exceptions, CUDA reports failures
// via a returned error code that is silently ignorable if you don't check
// it. Wrap every cuda* call in this macro so failures abort with a message
// immediately instead of corrupting results silently.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// TODO: implement this kernel.
//
// C[row][col] = sum_k A[row][k] * B[k][col]      (row-major, flat NxN)
//
// This runs once PER GPU THREAD, not once per matrix. Every thread is
// responsible for computing exactly one output element C[row][col]. There is
// no outer i/j loop like the CPU version had — the "loop over output
// elements" is implicit in how you launch the kernel (grid of thread blocks,
// set up in main() below), not written in this function.
//
// What you need inside this function:
//   1. Figure out which (row, col) THIS thread is responsible for. CUDA
//      gives every thread three built-in variables: blockIdx (which block
//      this thread's block is, within the grid), blockDim (how many threads
//      per block, in each dimension), and threadIdx (this thread's position
//      within its own block). Each has .x and .y components since we're
//      launching a 2D grid of 2D blocks (matches the 2D structure of a
//      matrix). You need to combine block index + block dimension + thread
//      index to get this thread's *global* row and column in the matrix —
//      think about how you'd convert a (block, offset-within-block) pair
//      into a single flat position, same idea as the tiling loops in
//      CPU matmul_tiled, just with the tiling now done by hardware.
//   2. Guard against out-of-bounds threads: because N might not divide
//      evenly by block size, some launched threads may compute a (row, col)
//      that's >= N. Those threads must do nothing (just return) — writing
//      out of bounds corrupts memory.
//   3. Do the same triple-nested-loop-collapsed-to-one accumulation as
//      matmul_naive's inner k-loop: walk k from 0 to N-1, accumulate
//      A[row*N+k] * B[k*N+col], and write the final sum to C[row*N+col].
// ---------------------------------------------------------------------------
__global__ void matmul_naive_gpu(const float* A, const float* B, float* C, int N) {
    // TODO: implement.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N){
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < N; k++){
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// Support code below.
// ---------------------------------------------------------------------------

static void matmul_cpu_reference(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

static void fill_random(float* M, int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N * N; i++) M[i] = dist(rng);
}

static float max_abs_diff(const float* X, const float* Y, int N) {
    float worst = 0.0f;
    for (int i = 0; i < N * N; i++) worst = std::max(worst, std::fabs(X[i] - Y[i]));
    return worst;
}

int main() {
    const int N = 1024;
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    // --- Host allocation + init ---
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_C_ref(N * N);
    fill_random(h_A.data(), N, 1);
    fill_random(h_B.data(), N, 2);

    // --- CPU reference (for correctness check) ---
    matmul_cpu_reference(h_A.data(), h_B.data(), h_C_ref.data(), N);

    // --- Device allocation ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // --- Host -> Device transfer ---
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // --- Kernel launch configuration ---
    // 16x16 = 256 threads per block, a common default (multiple of warp size
    // 32, small enough to allow many blocks resident per SM at once).
    dim3 blockDim(16, 16);
    // Ceiling division so we launch enough blocks to cover all N x N output
    // elements even when N isn't a multiple of 16 (out-of-bounds threads are
    // guarded inside the kernel, per the TODO comment above).
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // --- Timing via CUDA events (more accurate than CPU-side chrono for
    //     measuring device-side kernel execution time) ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());  // catches bad launch configuration
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // --- Device -> Host transfer ---
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // --- Correctness + performance report ---
    float diff = max_abs_diff(h_C.data(), h_C_ref.data(), N);
    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;

    printf("N=%d\n", N);
    printf("Correctness: max|gpu-cpu| = %g (expect < 1e-2)\n", diff);
    printf("GPU kernel time: %.4f ms, %.2f GFLOP/s\n", ms, gflops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
