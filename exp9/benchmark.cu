// benchmark.cu - Add two vectors three ways and compare timings:
//   1) plain sequential CPU loop
//   2) multi-core CPU with OpenMP
//   3) GPU with CUDA
// Then check that all three produce the same answer.

#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using Clock = chrono::high_resolution_clock;   // high-resolution timer

// GPU kernel: one thread adds one element. 'long long' is used because n is large.
__global__ void addKernel(const float* a, const float* b, float* c, long long n) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const long long n = 1000000;
    // a = 1.0 everywhere, b = 2.0 everywhere, so every result should be 3.0.
    vector<float> a(n, 1.0f), b(n, 2.0f), seq(n), ompResult(n), gpu(n);

    // ---- 1) Sequential CPU ----
    auto t0 = Clock::now();
    for (long long i = 0; i < n; ++i) seq[i] = a[i] + b[i];
    auto t1 = Clock::now();

    // ---- 2) OpenMP (splits the loop across CPU cores) ----
    auto t2 = Clock::now();
    #pragma omp parallel for
    for (long long i = 0; i < n; ++i) ompResult[i] = a[i] + b[i];
    auto t3 = Clock::now();

    // ---- 3) CUDA GPU ----
    float *da, *db, *dc;
    if (cudaMalloc(&da, n*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, n*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, n*sizeof(float)) != cudaSuccess) return 1;

    auto t4 = Clock::now();                               // start: includes H2D copy
    cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256, blocks = (n + threads - 1) / threads;
    addKernel<<<blocks, threads>>>(da, db, dc, n);
    cudaDeviceSynchronize();                              // wait for kernel
    auto t5 = Clock::now();                               // kernel done
    cudaMemcpy(gpu.data(), dc, n*sizeof(float), cudaMemcpyDeviceToHost);
    auto t6 = Clock::now();                               // after copy back

    // Lambda that returns elapsed milliseconds between two time points.
    auto ms = [](auto x, auto y) { return chrono::duration<double, milli>(y-x).count(); };
    cout << "Sequential (ms): " << ms(t0,t1) << '\n';
    cout << "OpenMP (ms): " << ms(t2,t3) << '\n';
    cout << "CUDA kernel + H2D (ms): " << ms(t4,t5) << '\n';
    cout << "CUDA total incl. transfers (ms): " << ms(t4,t6) << '\n';

    // Verify all three methods agree.
    bool ok = true;
    for (long long i = 0; i < n; ++i)
        if (seq[i] != ompResult[i] || seq[i] != gpu[i]) { ok = false; break; }
    cout << "Results match: " << (ok ? "YES" : "NO") << '\n';

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return ok ? 0 : 1;
}
