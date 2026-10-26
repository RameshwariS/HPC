#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

__global__ void reduceSum(const int *in, int *out, int n) {
    extern __shared__ int s[];
    unsigned t = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (unsigned d = blockDim.x / 2; d > 0; d >>= 1) {
        if (t < d) {
            s[t] += s[t + d];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(out, s[0]);
    }
}

int main() {
    const int n = 1024, th = 256;
    vector<int> h(n);
    iota(h.begin(), h.end(), 1);
    long long cpu = accumulate(h.begin(), h.end(), 0LL);

    int *di, *doo, zero = 0;
    cudaMalloc(&di, n * sizeof(int));
    cudaMalloc(&doo, sizeof(int));
    cudaMemcpy(di, h.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(doo, &zero, sizeof(int), cudaMemcpyHostToDevice);
    reduceSum<<<(n + th - 1) / th, th, th * sizeof(int)>>>(di, doo, n);
    cudaDeviceSynchronize();

    int gpu = 0;
    cudaMemcpy(&gpu, doo, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "CPU sum=" << cpu << " CUDA sum=" << gpu << " Correct="
         << (cpu == gpu ? "PASS" : "FAIL") << "\\n";

    cudaFree(di);
    cudaFree(doo);
    return cpu == gpu ? 0 : 1;
}
