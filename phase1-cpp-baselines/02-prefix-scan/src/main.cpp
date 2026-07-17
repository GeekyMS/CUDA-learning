// Project 2: Parallel Prefix Sum (Scan) — CPU Baselines
//
// Implement the three functions marked TODO below. Do not change the
// function signatures — main() and the correctness checker depend on them.
//
// Build: make
// Run:   ./prefix_scan [N]   (default N = 1,000,000)

#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <random>
#include <numeric>

// ---------------------------------------------------------------------------
// v1 — Sequential inclusive scan
//
// out[i] = in[0] + in[1] + ... + in[i]
// O(N) work, single pass, simple running sum.
// ---------------------------------------------------------------------------
void scan_sequential(const float* in, float* out, int N) {
    // TODO: implement.
    float res = 0.0f;

    for(int i = 0; i < N; i++){
        res += in[i];
        out[i] = res;
    }
}

// ---------------------------------------------------------------------------
// v2 — Parallel Blelloch scan (work-efficient, O(N) work / O(log N) span)
//
// Two phases over a conceptual balanced binary tree laid over the array:
//   Upsweep (reduce):   for d = 0 .. log2(N)-1, for each node at that level,
//                        combine pairs spaced 2^(d+1) apart into the right slot.
//   Downsweep:          set last element to 0 (identity for exclusive scan),
//                        then walk back down, swapping/combining to distribute
//                        partial sums to all positions.
//
// This function should produce an INCLUSIVE scan matching scan_sequential's
// output. A common approach: run the classic exclusive Blelloch scan, then
// shift by adding in[i] to each out[i] (exclusive -> inclusive).
//
// Requires N to be a power of 2 for the classic tree layout — pad with 0s
// internally if the caller passes a non-power-of-2 N, then truncate output.
//
// Use std::thread to parallelize each level of the upsweep/downsweep across
// the independent node updates at that level (there are N/2^(d+1) independent
// updates at level d — split them across a fixed pool of threads).
// ---------------------------------------------------------------------------
int next_pow2(int N) {
    int p = 1;
    while (p < N) p <<= 1;
    return p;
}
void upsweep(float* arr, int N){
    int levels = static_cast<int>(std::log2(N));
    for (int d = 0; d < levels; d++){
        int stride = 2 << d;
        int num_threads = std::thread::hardware_concurrency();  // e.g. 8 on your machine
        // for each level d of the upsweep:
        int num_updates = N / stride;   // number of independent (i, i+stride/2, i+stride) triples at this level
        int chunk = (num_updates + num_threads - 1) / num_threads;  // ceil division

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            int start = t * chunk;
            int end = std::min(start + chunk, num_updates);
            if (start >= end) continue;   // fewer updates than threads at small levels — skip empty ranges
            threads.emplace_back([&, start, end, stride]() {
                for (int u = start; u < end; u++) {
                    int i = u * stride;
                    arr[i + stride - 1] += arr[i + stride/2 - 1];
                }
            });
        }
        for (auto& th: threads) th.join();
    }
}

void downsweep(float* arr, int N){
    arr[N-1] = 0;
    int level = std::log2(N) - 1;
    for(int d = level; d >= 0; d--){
        int stride = 2 << d;
        int num_threads = std::thread::hardware_concurrency();  // e.g. 8 on your machine
        int num_updates = N / stride;   // number of independent (i, i+stride/2, i+stride) triples at this level
        int chunk = (num_updates + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++){
            int start = t * chunk;
            int end = std::min(start + chunk, num_updates);
            if (start >= end) continue;
            threads.emplace_back([&, start, end, stride]() {
                for (int u = start; u < end; u++) {
                    int i = u * stride;
                    int right = i + stride -1;
                    int left = i + stride/2 - 1;
                    float temp = arr[left];
                    arr[left] = arr[right];
                    arr[right] += temp;
                }
            });
        }
        for (auto& th: threads) th.join();
    }
}
void scan_parallel_blelloch(const float* in, float* out, int N) {
    // TODO: implement.
    int next = next_pow2(N);
    std::vector<float> buf(next, 0.0f);
    std::copy(in, in + N, buf.data()); 
    upsweep(buf.data(), next);
    downsweep(buf.data(), next);
    for (int i = 0; i < N; i++) buf[i] += in[i];
    std::copy(buf.begin(), buf.begin() + N, out);
}

// ---------------------------------------------------------------------------
// v3 — std::inclusive_scan reference (C++17 standard library)
//
// Use <numeric>'s std::inclusive_scan as a correctness + performance
// reference point. This is what a real production codebase would reach for
// unless it needed GPU execution (cub::DeviceScan) or custom fusion.
// ---------------------------------------------------------------------------
void scan_std(const float* in, float* out, int N) {
    // TODO: implement using std::inclusive_scan.
    std::inclusive_scan(in, in + N,out);
}

// ---------------------------------------------------------------------------
// Support code below — you shouldn't need to modify this, but read it so you
// understand what main() expects from your functions above.
// ---------------------------------------------------------------------------

static void fill_random(float* v, int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) v[i] = dist(rng);
}

static float max_abs_diff(const float* X, const float* Y, int N) {
    float worst = 0.0f;
    for (int i = 0; i < N; i++) worst = std::max(worst, std::fabs(X[i] - Y[i]));
    return worst;
}

template <typename Fn>
static double time_ms(Fn&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;

    // --- Correctness check on a small array ---
    {
        const int n = 1 << 10; // 1024, power of 2 for the classic Blelloch layout
        std::vector<float> in(n), out1(n), out2(n), out3(n);
        fill_random(in.data(), n, 1);

        scan_sequential(in.data(), out1.data(), n);
        scan_parallel_blelloch(in.data(), out2.data(), n);
        scan_std(in.data(), out3.data(), n);

        std::cout << "Correctness (N=" << n << "):\n";
        std::cout << "  max|seq-blelloch| = " << max_abs_diff(out1.data(), out2.data(), n) << " (expect < 1e-4)\n";
        std::cout << "  max|seq-std|      = " << max_abs_diff(out1.data(), out3.data(), n) << " (expect < 1e-4)\n";
    }

    // --- Benchmark ---
    {
        std::vector<float> in(N), out(N);
        fill_random(in.data(), N, 1);

        std::cout << "\nBenchmark (N=" << N << "):\n";

        double t1 = time_ms([&] { scan_sequential(in.data(), out.data(), N); });
        std::cout << "  v1 sequential: " << t1 << " ms\n";

        double t2 = time_ms([&] { scan_parallel_blelloch(in.data(), out.data(), N); });
        std::cout << "  v2 blelloch:   " << t2 << " ms (speedup " << t1 / t2 << "x)\n";

        double t3 = time_ms([&] { scan_std(in.data(), out.data(), N); });
        std::cout << "  v3 std::scan:  " << t3 << " ms (speedup " << t1 / t3 << "x)\n";
    }

    return 0;
}
