// Project 1: Matrix Multiplication — CPU Baselines
//
// Implement the three functions marked TODO below. Do not change the
// function signatures — main() and the correctness checker depend on them.
//
// Build: make        (see Makefile)
// Run:   ./main

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>

// ---------------------------------------------------------------------------
// v1 — Naive matmul (ijk loop order)
//
// C[i][j] = sum_k A[i][k] * B[k][j]
//
// A, B, C are flat NxN row-major matrices: element (i,j) lives at index i*N+j.
// C is assumed to be zero-initialized by the caller.
// ---------------------------------------------------------------------------
void matmul_naive(const float* A, const float* B, float* C, int N) {
    // TODO: implement using ijk loop order.
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v2a — Transpose helper
//
// Bt[j][i] = B[i][j]   (i.e. Bt is B transposed)
// Bt must be a separate buffer from B — do not transpose in place.
// ---------------------------------------------------------------------------
void transpose(const float* B, float* Bt, int N) {
    // TODO: implement.
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            Bt[i * N + j] = B[j * N + i];
        }
    }
}

// ---------------------------------------------------------------------------
// v2b — Transposed-B matmul (ikj loop order)
//
// Same result as matmul_naive, but takes Bt (B already transposed) and
// accesses it as Bt[j*N+k] instead of B[k*N+j], so both operands are read
// with sequential (stride-1) access in the inner loop.
// ---------------------------------------------------------------------------
void matmul_transposed(const float* A, const float* Bt, float* C, int N) {
    // TODO: implement using ikj loop order, reading Bt[j*N+k].
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                C[i * N + j] += A[i * N + k] * Bt[j * N + k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v3 — Tiled (cache-blocked) matmul
//
// Process the matrix in tile x tile blocks so that the working set for each
// block fits in L1/L2 cache. Must produce the same result as matmul_naive
// for any N, including N not divisible evenly by `tile`.
//
// Hint: you'll have up to 6 nested loops (3 for tile coordinates, 3 for the
// within-tile multiply). Clamp inner bounds with std::min(start+tile, N) so
// you don't read/write out of bounds when N % tile != 0.
// ---------------------------------------------------------------------------
void matmul_tiled(const float* A, const float* B, float* C, int N, int tile) {
    // TODO: implement.
    for(int ii = 0; ii < N; ii += tile){
        for(int jj = 0; jj < N; jj += tile){
            for(int kk = 0; kk < N; kk += tile){
                int i_max = std::min(ii + tile, N);
                int j_max = std::min(jj + tile, N);
                int k_max = std::min(kk + tile, N);
                for(int i = ii; i < i_max; i++){
                    for(int j = jj; j < j_max; j++){
                        for(int k = kk; k < k_max; k++){
                            C[i * N + j] += A[i * N + k] * B[k * N + j]; 
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Support code below — you shouldn't need to modify this, but read it so you
// understand what main() expects from your functions above.
// ---------------------------------------------------------------------------

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

template <typename Fn>
static double time_ms(Fn&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double gflops(int N, double ms) {
    return (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
}

int main() {
    // --- Correctness check on a small matrix ---
    {
        const int N = 64;
        float* A = new float[N * N];
        float* B = new float[N * N];
        float* Bt = new float[N * N];
        float* C1 = new float[N * N]{};
        float* C2 = new float[N * N]{};
        float* C3 = new float[N * N]{};

        fill_random(A, N, 1);
        fill_random(B, N, 2);

        matmul_naive(A, B, C1, N);

        transpose(B, Bt, N);
        matmul_transposed(A, Bt, C2, N);

        matmul_tiled(A, B, C3, N, 16);

        std::cout << "Correctness (N=" << N << "):\n";
        std::cout << "  max|C1-C2| = " << max_abs_diff(C1, C2, N) << " (expect < 1e-4)\n";
        std::cout << "  max|C1-C3| = " << max_abs_diff(C1, C3, N) << " (expect < 1e-4)\n";

        delete[] A; delete[] B; delete[] Bt; delete[] C1; delete[] C2; delete[] C3;
    }

    // --- Benchmark ---
    {
        const int N = 1024;
        float* A = new float[N * N];
        float* B = new float[N * N];
        float* Bt = new float[N * N];
        float* C = new float[N * N]{};

        fill_random(A, N, 1);
        fill_random(B, N, 2);

        std::cout << "\nBenchmark (N=" << N << "):\n";

        double t1 = time_ms([&] { matmul_naive(A, B, C, N); });
        std::cout << "  v1 naive:      " << t1 << " ms, " << gflops(N, t1) << " GFLOP/s\n";

        transpose(B, Bt, N);
        double t2 = time_ms([&] { matmul_transposed(A, Bt, C, N); });
        std::cout << "  v2 transposed: " << t2 << " ms, " << gflops(N, t2) << " GFLOP/s\n";

        for (int tile : {8, 16, 32, 64}) {
            double t3 = time_ms([&] { matmul_tiled(A, B, C, N, tile); });
            std::cout << "  v3 tiled(" << tile << "):  " << t3 << " ms, " << gflops(N, t3) << " GFLOP/s\n";
        }

        delete[] A; delete[] B; delete[] Bt; delete[] C;
    }

    return 0;
}
