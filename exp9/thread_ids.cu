#include <cuda_runtime.h>
#include <cstdio>

__global__ void showIds() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block ID: %d, Thread ID: %d, Global ID: %d\n", blockIdx.x, threadIdx.x, id);
}

int main() {
    showIds<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
