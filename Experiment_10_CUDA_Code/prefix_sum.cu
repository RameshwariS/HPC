#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace std;

// Simple inclusive Hillis-Steele scan for one block; maximum 512 elements.
__global__ void scan(const int *in, int *out, int n) {
    __shared__ int d[512];
    int t = threadIdx.x;

    if (t < n) {
        d[t] = in[t];
    }
    __syncthreads();

    for (int off = 1; off < n; off *= 2) {
        int v = 0;
        if (t < n && t >= off) {
            v = d[t - off];
        }
        __syncthreads();
        if (t < n && t >= off) {
            d[t] += v;
        }
        __syncthreads();
    }

    if (t < n) {
        out[t] = d[t];
    }
}

int main() {
    vector<int> h{3, 1, 7, 0, 4, 2}, cpu(h.size()), gpu(h.size());
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
    scan<<<1, h.size()>>>(di, doo, (int)h.size());
    cudaDeviceSynchronize();
    cudaMemcpy(gpu.data(), doo, z, cudaMemcpyDeviceToHost);

    cout << "CPU scan: ";
    for (int x : cpu) {
        cout << x << " ";
    }
    cout << "\\nCUDA scan: ";
    for (int x : gpu) {
        cout << x << " ";
    }
    cout << "\\nCorrect=" << (cpu == gpu ? "PASS" : "FAIL") << "\\n";

    cudaFree(di);
    cudaFree(doo);
    return cpu == gpu ? 0 : 1;
}
