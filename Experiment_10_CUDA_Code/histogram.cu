#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace std;

__global__ void hist(const int *in, int *out, int n, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = in[i];
        if (v >= 0 && v < bins) {
            atomicAdd(&out[v], 1);
        }
    }
}

int main() {
    const int n = 100000, bins = 16, th = 256;
    vector<int> h(n), cpu(bins, 0), gpu(bins, 0);

    for (int i = 0; i < n; i++) {
        h[i] = i % bins;
    }
    for (int v : h) {
        cpu[v]++;
    }

    int *di, *dh;
    cudaMalloc(&di, n * sizeof(int));
    cudaMalloc(&dh, bins * sizeof(int));
    cudaMemcpy(di, h.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dh, 0, bins * sizeof(int));
    hist<<<(n + th - 1) / th, th>>>(di, dh, n, bins);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu.data(), dh, bins * sizeof(int), cudaMemcpyDeviceToHost);

    bool ok = cpu == gpu;
    cout << "Bin 0 count=" << gpu[0] << " Correct=" << (ok ? "PASS" : "FAIL")
         << "\\n";

    cudaFree(di);
    cudaFree(dh);
    return ok ? 0 : 1;
}
