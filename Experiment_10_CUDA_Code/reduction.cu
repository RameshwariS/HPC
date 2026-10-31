// reduction.cu - Sum all elements of an array on the GPU (parallel reduction).
// Each block sums its own chunk in shared memory using a tree pattern, then adds
// its partial sum into a single global total with atomicAdd.

#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

__global__ void reduceSum(const int *in, int *out, int n) {
    // Dynamically-sized shared array (its size is given at launch, 3rd <<<>>> arg).
    extern __shared__ int s[];
    unsigned t = threadIdx.x;                       // index within the block
    unsigned i = blockIdx.x * blockDim.x + t;       // global index

    // Each thread loads one value (or 0 if past the end) into shared memory.
    s[t] = (i < n) ? in[i] : 0;
    __syncthreads();

    // Tree reduction: each step halves the number of active threads, with the
    // first half adding in the second half. After the loop s[0] = block's sum.
    for (unsigned d = blockDim.x / 2; d > 0; d >>= 1) {
        if (t < d) {
            s[t] += s[t + d];
        }
        __syncthreads();
    }

    // Thread 0 adds this block's partial sum into the global total safely.
    if (t == 0) {
        atomicAdd(out, s[0]);
    }
}

int main() {
    const int n = 1024, th = 256;
    vector<int> h(n);
    iota(h.begin(), h.end(), 1);                    // fill with 1, 2, 3, ... n
    long long cpu = accumulate(h.begin(), h.end(), 0LL);   // CPU reference sum

    int *di, *doo, zero = 0;
    cudaMalloc(&di, n * sizeof(int));
    cudaMalloc(&doo, sizeof(int));                  // single int holds the total
    cudaMemcpy(di, h.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(doo, &zero, sizeof(int), cudaMemcpyHostToDevice);   // total starts at 0

    // 3rd launch argument = bytes of dynamic shared memory per block.
    reduceSum<<<(n + th - 1) / th, th, th * sizeof(int)>>>(di, doo, n);
    cudaDeviceSynchronize();

    int gpu = 0;
    cudaMemcpy(&gpu, doo, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "CPU sum=" << cpu << " CUDA sum=" << gpu << " Correct="
         << (cpu == gpu ? "PASS" : "FAIL") << "\n";

    cudaFree(di);
    cudaFree(doo);
    return cpu == gpu ? 0 : 1;
}
