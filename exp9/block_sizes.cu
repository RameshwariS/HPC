#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

__global__ void addVectors(const float* a, const float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const int n = 1000000;
    vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    float *da, *db, *dc;
    if (cudaMalloc(&da, n*sizeof(float)) != cudaSuccess || cudaMalloc(&db, n*sizeof(float)) != cudaSuccess || cudaMalloc(&dc, n*sizeof(float)) != cudaSuccess) {
        cerr << "Device allocation failed\n"; return 1;
    }
    cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    for (int threads : {64, 128, 256, 512}) {
        int blocks = (n + threads - 1) / threads;
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        addVectors<<<blocks, threads>>>(da, db, dc, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms = 0; cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(c.data(), dc, n*sizeof(float), cudaMemcpyDeviceToHost);
        cout << "Threads/block: " << threads << ", blocks: " << blocks << ", kernel time (ms): " << ms << ", first result: " << c[0] << '\n';
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}
