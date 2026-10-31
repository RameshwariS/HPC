// vector_add.cu - Add two large vectors on the GPU and verify the result.
// This version uses a CUDA_CHECK macro so every CUDA call is error-checked.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Helper macro: run a CUDA call, and if it fails, print the error and exit.
// "do { ... } while(0)" is the standard way to make a macro behave like one statement.
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { cerr << "CUDA error: " << cudaGetErrorString(e) << '\n'; return 1; } \
} while (0)

// Kernel: one thread adds one pair of elements.
__global__ void addVectors(const float* a, const float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // global index
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const int n = 1000000, threads = 256;            // 1 million elements
    // a filled with 1.5, b filled with 2.5, so every c should be 4.0.
    vector<float> a(n, 1.5f), b(n, 2.5f), c(n);

    // Allocate GPU memory.
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));

    // Copy inputs CPU -> GPU.
    CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch: round the block count up so all n elements are covered.
    int blocks = (n + threads - 1) / threads;
    addVectors<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());        // did the launch itself fail?
    CUDA_CHECK(cudaDeviceSynchronize());   // wait for the kernel to finish

    // Copy result back and check correctness (allow tiny float rounding error).
    CUDA_CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (fabs(c[i] - (a[i] + b[i])) > 1e-6f) { correct = false; break; }

    cout << "Elements: " << n << "\nFirst result: " << c[0]
         << "\nCorrectness: " << (correct ? "PASS" : "FAIL") << '\n';

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return correct ? 0 : 1;
}
