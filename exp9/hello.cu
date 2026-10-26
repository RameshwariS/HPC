#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU: block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();
    hello<<<1, 16>>>();
    cudaDeviceSynchronize();
    hello<<<2, 8>>>();
    cudaDeviceSynchronize();
    hello<<<4, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
