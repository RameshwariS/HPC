// block_sizes.cu - Measure how the number of threads per block affects the
// kernel runtime for the same vector-add work. Uses CUDA events to time the GPU.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

// Kernel: one thread adds one pair of elements.
__global__ void addVectors(const float* a, const float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const int n = 1000000;
    vector<float> a(n, 1.0f), b(n, 2.0f), c(n);   // expect every c = 3.0

    // Allocate GPU memory once; reuse it for every block size we test.
    float *da, *db, *dc;
    if (cudaMalloc(&da, n*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, n*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, n*sizeof(float)) != cudaSuccess) {
        cerr << "Device allocation failed\n"; return 1;
    }
    cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    // Try several "threads per block" values and time each one.
    for (int threads : {64, 128, 256, 512}) {
        int blocks = (n + threads - 1) / threads;

        // CUDA events are GPU timestamps used to measure kernel time accurately.
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);                          // mark "before kernel"
        addVectors<<<blocks, threads>>>(da, db, dc, n);
        cudaEventRecord(stop);                           // mark "after kernel"
        cudaEventSynchronize(stop);                      // wait until stop is reached

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);          // time between the two marks
        cudaMemcpy(c.data(), dc, n*sizeof(float), cudaMemcpyDeviceToHost);

        cout << "Threads/block: " << threads << ", blocks: " << blocks
             << ", kernel time (ms): " << ms << ", first result: " << c[0] << '\n';

        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}
