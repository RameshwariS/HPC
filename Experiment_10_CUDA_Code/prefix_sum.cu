// prefix_sum.cu - Inclusive prefix sum (scan) of an array on the GPU.
// "Inclusive scan" means out[i] = in[0] + in[1] + ... + in[i].
// Uses the Hillis-Steele algorithm inside a single block (max 512 elements).

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

// Simple inclusive Hillis-Steele scan for one block; maximum 512 elements.
__global__ void scan(const int *in, int *out, int n) {
    __shared__ int d[512];       // shared buffer the whole block works on
    int t = threadIdx.x;         // this thread's index

    // Load input into shared memory.
    if (t < n) {
        d[t] = in[t];
    }
    __syncthreads();

    // Repeatedly add the element 'off' positions to the left, doubling 'off'
    // each step: after the loop, each position holds the running total.
    for (int off = 1; off < n; off *= 2) {
        int v = 0;
        if (t < n && t >= off) {
            v = d[t - off];      // read first (before anyone overwrites)
        }
        __syncthreads();
        if (t < n && t >= off) {
            d[t] += v;           // then add
        }
        __syncthreads();
    }

    // Write results back to global memory.
    if (t < n) {
        out[t] = d[t];
    }
}

int main() {
    // Input array; cpu = reference scan, gpu = result from the kernel.
    vector<int> h{3, 1, 7, 0, 4, 2}, cpu(h.size()), gpu(h.size());

    // CPU reference: running sum.
    int s = 0;
    for (size_t i = 0; i < h.size(); i++) {
        s += h[i];
        cpu[i] = s;
    }

    int *di, *doo;
    size_t z = h.size() * sizeof(int);
    cudaMalloc(&di, z);
    cudaMalloc(&doo, z);
    cudaMemcpy(di, h.data(), z, cudaMemcpyHostToDevice);

    // One block, one thread per element.
    scan<<<1, h.size()>>>(di, doo, (int)h.size());
    cudaDeviceSynchronize();
    cudaMemcpy(gpu.data(), doo, z, cudaMemcpyDeviceToHost);

    cout << "CPU scan: ";
    for (int x : cpu) cout << x << " ";
    cout << "\nCUDA scan: ";
    for (int x : gpu) cout << x << " ";
    cout << "\nCorrect=" << (cpu == gpu ? "PASS" : "FAIL") << "\n";

    cudaFree(di);
    cudaFree(doo);
    return cpu == gpu ? 0 : 1;
}
