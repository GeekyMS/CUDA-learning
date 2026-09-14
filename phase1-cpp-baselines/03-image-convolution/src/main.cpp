// Project 3: Image Convolution / Gaussian Blur — CPU Baselines
//
// Implement the functions marked TODO below. Do not change function
// signatures — main() and the correctness checker depend on them.
//
// Build: make
// Run:   ./imgconv <image.pgm> <kernel_size> <sigma> [--verify]

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <cstdlib>
#include <stdexcept>

// ---------------------------------------------------------------------------
// PGM (P5, binary grayscale) image I/O — support code, not a TODO.
// ---------------------------------------------------------------------------
struct Image {
    int width = 0, height = 0;
    std::vector<unsigned char> pixels;  // row-major, width*height, 0-255
};

static Image read_pgm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string magic;
    f >> magic;
    if (magic != "P5") throw std::runtime_error("not a binary PGM (P5): " + path);
    int w, h, maxval;
    f >> w >> h >> maxval;
    f.get();  // consume the single whitespace char after maxval
    Image img;
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<size_t>(w) * h);
    f.read(reinterpret_cast<char*>(img.pixels.data()), img.pixels.size());
    if (!f) throw std::runtime_error("truncated PGM data: " + path);
    return img;
}

static void write_pgm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.pixels.data()), img.pixels.size());
}

// ---------------------------------------------------------------------------
// v3 — Gaussian kernel generation
//
// Generate a normalized 1D Gaussian kernel of length K (K is odd; center
// index is K/2). G(x) = exp(-x^2 / (2*sigma^2)) for x = -(K/2) .. (K/2),
// then divide every entry by the sum so the kernel sums to 1.0 (preserves
// overall image brightness).
// ---------------------------------------------------------------------------
std::vector<float> gaussian_kernel_1d(int K, float sigma) {
    int center = K / 2;
    std::vector<float> kernel(K, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        int x = i - center;
        float g = std::exp(-(x * x) / (2.0f * sigma * sigma));
        kernel[i] = g;
        sum += g;
    }
    for (int i = 0; i < K; i++) kernel[i] /= sum;
    return kernel;
}

// ---------------------------------------------------------------------------
// v1 — Naive 2D convolution
//
// output[i][j] = sum_{ki=0..K-1} sum_{kj=0..K-1}
//                  input[i+ki-K/2][j+kj-K/2] * kernel2d[ki*K+kj]
//
// kernel2d is a flat K*K buffer (see build_kernel2d below — outer product of
// the 1D kernel with itself). Zero-pad at borders: any input pixel with a
// row/col outside [0,W)x[0,H) contributes 0 to the sum.
// ---------------------------------------------------------------------------
void conv2d_naive(const float* input, float* output, int W, int H,
                   const float* kernel2d, int K) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int row = i + ki - K / 2;
                    int col = j + kj - K / 2;
                    if (row >= 0 && row < H && col >= 0 && col < W) {
                        sum += input[row * W + col] * kernel2d[ki * K + kj];
                    }
                }
            }
            output[i * W + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// v2 — Separable 2D convolution
//
// Same result as conv2d_naive, but exploits G(x,y) = G(x)*G(y): first run
// the 1D kernel horizontally across every row into a temp buffer (same W,H),
// then run the 1D kernel vertically down every column of that temp buffer
// into output. O(2K) work per pixel instead of O(K^2). Zero-pad borders in
// both passes, same as conv2d_naive.
// ---------------------------------------------------------------------------
void conv2d_separable(const float* input, float* output, int W, int H,
                       const float* kernel1d, int K) {
    std::vector<float> temp(static_cast<size_t>(W) * H);

    // Horizontal pass: input -> temp
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int col = j + k - K / 2;
                if (col >= 0 && col < W) {
                    sum += input[i * W + col] * kernel1d[k];
                }
            }
            temp[i * W + j] = sum;
        }
    }

    // Vertical pass: temp -> output
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int row = i + k - K / 2;
                if (row >= 0 && row < H) {
                    sum += temp[row * W + j] * kernel1d[k];
                }
            }
            output[i * W + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Support code below — you shouldn't need to modify this, but read it so you
// understand what main() expects from your functions above.
// ---------------------------------------------------------------------------

static std::vector<float> build_kernel2d(const std::vector<float>& k1d, int K) {
    std::vector<float> k2d(static_cast<size_t>(K) * K);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            k2d[i * K + j] = k1d[i] * k1d[j];
    return k2d;
}

static std::vector<float> to_float(const Image& img) {
    std::vector<float> out(img.pixels.size());
    for (size_t i = 0; i < img.pixels.size(); i++) out[i] = static_cast<float>(img.pixels[i]);
    return out;
}

static Image to_image(const std::vector<float>& buf, int W, int H) {
    Image img;
    img.width = W;
    img.height = H;
    img.pixels.resize(buf.size());
    for (size_t i = 0; i < buf.size(); i++) {
        float v = buf[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        img.pixels[i] = static_cast<unsigned char>(v + 0.5f);
    }
    return img;
}

static float max_abs_diff(const float* X, const float* Y, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) worst = std::max(worst, std::fabs(X[i] - Y[i]));
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
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <image.pgm> <kernel_size> <sigma> [--verify]\n";
        return 1;
    }
    std::string path = argv[1];
    int K = std::atoi(argv[2]);
    float sigma = std::atof(argv[3]);
    bool verify = (argc > 4) && std::string(argv[4]) == "--verify";

    Image img = read_pgm(path);
    int W = img.width, H = img.height;
    std::vector<float> input = to_float(img);
    std::vector<float> out_naive(input.size());
    std::vector<float> out_sep(input.size());

    std::vector<float> k1d = gaussian_kernel_1d(K, sigma);
    std::vector<float> k2d = build_kernel2d(k1d, K);

    std::cout << "Image: " << W << "x" << H << ", K=" << K << ", sigma=" << sigma << "\n";

    if (verify) {
        // Identity check: a kernel that is 1.0 at the center and 0 elsewhere
        // should reproduce the input exactly (minus rounding).
        std::vector<float> identity1d(K, 0.0f);
        identity1d[K / 2] = 1.0f;
        std::vector<float> identity2d = build_kernel2d(identity1d, K);
        std::vector<float> out_identity(input.size());
        conv2d_naive(input.data(), out_identity.data(), W, H, identity2d.data(), K);
        float id_diff = max_abs_diff(input.data(), out_identity.data(), input.size());
        std::cout << "Identity-kernel check: max|input-output| = " << id_diff
                   << " (expect < 1e-4)\n";
    }

    conv2d_naive(input.data(), out_naive.data(), W, H, k2d.data(), K);
    conv2d_separable(input.data(), out_sep.data(), W, H, k1d.data(), K);
    float diff = max_abs_diff(out_naive.data(), out_sep.data(), input.size());
    // Separable does two accumulation passes over pixel-scale values (up to
    // ~255), so it accumulates more float rounding error than the naive
    // single-pass sum as K grows — same non-associativity as
    // additional-learnings/floating-point-nonassociativity.md, just a wider
    // tolerance since the operand magnitudes here are much larger than the
    // matmul case that doc was written for.
    std::cout << "Correctness: max|naive-separable| = " << diff << " (expect < 5e-4)\n";

    double t1 = time_ms([&] { conv2d_naive(input.data(), out_naive.data(), W, H, k2d.data(), K); });
    std::cout << "  v1 naive:     " << t1 << " ms\n";

    double t2 = time_ms([&] { conv2d_separable(input.data(), out_sep.data(), W, H, k1d.data(), K); });
    std::cout << "  v2 separable: " << t2 << " ms (speedup " << t1 / t2 << "x)\n";

    write_pgm("output.pgm", to_image(out_sep, W, H));
    std::cout << "Wrote output.pgm\n";

    return 0;
}
