// histogram.cu - Count how many values fall into each "bin" on the GPU.
// Many threads may want to update the SAME bin at once, so we use atomicAdd
// to make those updates safe (no two threads overwrite each other).

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

// Kernel: each thread reads one input value and increments its bin counter.
__global__ void hist(const int *in, int *out, int n, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index
    if (i < n) {
        int v = in[i];
        if (v >= 0 && v < bins) {
            // atomicAdd: read-modify-write that cannot be interrupted by other
            // threads. Essential because many threads hit the same out[v].
            atomicAdd(&out[v], 1);
        }
    }
}

int main() {
    const int n = 100000, bins = 16, th = 256;
    vector<int> h(n), cpu(bins, 0), gpu(bins, 0);

    // Build input: values cycle 0..15, so every bin should get the same count.
    for (int i = 0; i < n; i++) {
        h[i] = i % bins;
    }
    // Reference histogram computed on the CPU for checking.
    for (int v : h) {
        cpu[v]++;
    }

    // Allocate GPU memory for input (di) and the bin counters (dh).
    int *di, *dh;
    cudaMalloc(&di, n * sizeof(int));
    cudaMalloc(&dh, bins * sizeof(int));
    cudaMemcpy(di, h.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dh, 0, bins * sizeof(int));           // start all counters at 0

    // One thread per input element.
    hist<<<(n + th - 1) / th, th>>>(di, dh, n, bins);
    cudaDeviceSynchronize();

    // Copy the GPU histogram back and compare with the CPU one.
    cudaMemcpy(gpu.data(), dh, bins * sizeof(int), cudaMemcpyDeviceToHost);

    bool ok = cpu == gpu;
    cout << "Bin 0 count=" << gpu[0] << " Correct=" << (ok ? "PASS" : "FAIL")
         << "\n";

    cudaFree(di);
    cudaFree(dh);
    return ok ? 0 : 1;
}
