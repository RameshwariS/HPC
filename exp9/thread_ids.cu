// thread_ids.cu - Show how a thread's GLOBAL id is built from its
// block id and thread id. This is the key formula used in almost every kernel.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void showIds() {
    // Global id = (block number * threads per block) + position in block.
    // This gives every thread in the whole grid a unique number.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block ID: %d, Thread ID: %d, Global ID: %d\n",
           blockIdx.x, threadIdx.x, id);
}

int main() {
    // 2 blocks x 4 threads = 8 threads, with global ids 0..7.
    showIds<<<2, 4>>>();
    cudaDeviceSynchronize();   // wait for GPU prints to finish
    return 0;
}
