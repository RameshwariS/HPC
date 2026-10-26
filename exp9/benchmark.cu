#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using Clock = chrono::high_resolution_clock;

__global__ void addKernel(const float* a, const float* b, float* c, long long n) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const long long n = 1000000;
    vector<float> a(n, 1.0f), b(n, 2.0f), seq(n), ompResult(n), gpu(n);
    auto t0 = Clock::now();
    for (long long i = 0; i < n; ++i) seq[i] = a[i] + b[i];
    auto t1 = Clock::now();
    auto t2 = Clock::now();
    #pragma omp parallel for
    for (long long i = 0; i < n; ++i) ompResult[i] = a[i] + b[i];
    auto t3 = Clock::now();
    float *da, *db, *dc;
    if (cudaMalloc(&da, n*sizeof(float)) != cudaSuccess || cudaMalloc(&db, n*sizeof(float)) != cudaSuccess || cudaMalloc(&dc, n*sizeof(float)) != cudaSuccess) return 1;
    auto t4 = Clock::now();
    cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256, blocks = (n + threads - 1) / threads;
    addKernel<<<blocks, threads>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    auto t5 = Clock::now();
    cudaMemcpy(gpu.data(), dc, n*sizeof(float), cudaMemcpyDeviceToHost);
    auto t6 = Clock::now();
    auto ms = [](auto x, auto y) { return chrono::duration<double, milli>(y-x).count(); };
    cout << "Sequential (ms): " << ms(t0,t1) << '\n';
    cout << "OpenMP (ms): " << ms(t2,t3) << '\n';
    cout << "CUDA kernel + H2D (ms): " << ms(t4,t5) << '\n';
    cout << "CUDA total incl. transfers (ms): " << ms(t4,t6) << '\n';
    bool ok = true;
    for (long long i = 0; i < n; ++i) if (seq[i] != ompResult[i] || seq[i] != gpu[i]) { ok = false; break; }
    cout << "Results match: " << (ok ? "YES" : "NO") << '\n';
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return ok ? 0 : 1;
}
