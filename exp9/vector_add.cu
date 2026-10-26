#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { cerr << "CUDA error: " << cudaGetErrorString(e) << '\n'; return 1; } } while (0)

__global__ void addVectors(const float* a, const float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const int n = 1000000, threads = 256;
    vector<float> a(n, 1.5f), b(n, 2.5f), c(n);
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int blocks = (n + threads - 1) / threads;
    addVectors<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) if (fabs(c[i] - (a[i] + b[i])) > 1e-6f) { correct = false; break; }
    cout << "Elements: " << n << "\nFirst result: " << c[0] << "\nCorrectness: " << (correct ? "PASS" : "FAIL") << '\n';
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return correct ? 0 : 1;
}
