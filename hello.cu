// hello.cu - The simplest CUDA program: run code on the GPU and print from it.

#include <stdio.h>
#include <cuda_runtime.h>

// __global__ marks a "kernel": a function that runs ON THE GPU but is CALLED
// from the CPU. Every GPU thread that is launched runs this same function.
__global__ void helloCUDA()
{
    // threadIdx.x = the index of THIS thread inside its block (0,1,2,...).
    // Each thread prints its own line, so order on screen may vary.
    printf("Hello from GPU! Thread = %d\n", threadIdx.x);
}

int main()
{
    // <<<blocks, threads>>> is the launch configuration.
    // Here: 1 block containing 8 threads  ->  8 threads run helloCUDA().
    helloCUDA<<<1, 8>>>();

    // The GPU runs asynchronously. This makes the CPU WAIT until the GPU
    // finishes, so all the printf output appears before the program exits.
    cudaDeviceSynchronize();

    return 0;
}
