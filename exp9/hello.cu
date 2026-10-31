// hello.cu - Launch the same GPU kernel with different grid/block sizes
// to see how blocks and threads are numbered.

#include <cuda_runtime.h>
#include <cstdio>

// Kernel: every thread prints which block and which thread it is.
__global__ void hello() {
    // blockIdx.x  = this thread's block number
    // threadIdx.x = this thread's position inside that block
    printf("Hello from GPU: block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // <<<blocks, threadsPerBlock>>>

    hello<<<1, 8>>>();            // 1 block  x 8 threads  = 8 threads
    cudaDeviceSynchronize();      // wait so output prints before next launch

    hello<<<1, 16>>>();           // 1 block  x 16 threads = 16 threads
    cudaDeviceSynchronize();

    hello<<<2, 8>>>();            // 2 blocks x 8 threads  = 16 threads
    cudaDeviceSynchronize();

    hello<<<4, 8>>>();            // 4 blocks x 8 threads  = 32 threads
    cudaDeviceSynchronize();

    return 0;
}
