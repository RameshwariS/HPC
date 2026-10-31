// vector_add.cu - Add two arrays C = A + B on the GPU.
// Shows the standard CUDA workflow: allocate GPU memory, copy data in,
// run a kernel, copy the result back, and free the memory.

#include <stdio.h>
#include <cuda_runtime.h>

// GPU kernel: each thread computes ONE element of the output array.
__global__ void vectorAdd(int *A, int *B, int *C, int N)
{
    // Global index of this thread across the whole grid:
    //   blockIdx.x  = which block this thread is in
    //   blockDim.x  = how many threads per block
    //   threadIdx.x = this thread's position inside its block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // We may launch more threads than elements, so skip extra threads.
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 10;                  // number of elements
    int size = N * sizeof(int);  // number of bytes to allocate/copy

    // Host (CPU) arrays. Prefix h_ = "host".
    int h_A[10] = {1,2,3,4,5,6,7,8,9,10};
    int h_B[10] = {10,20,30,40,50,60,70,80,90,100};
    int h_C[10];

    // Device (GPU) pointers. Prefix d_ = "device".
    int *d_A, *d_B, *d_C;

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data CPU -> GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Choose launch configuration.
    // threads per block = 256. blocks = enough to cover all N elements.
    // The (N + threads - 1) / threads trick rounds the division UP.
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch the kernel on the GPU.
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Wait for the GPU to finish before reading the result.
    cudaDeviceSynchronize();

    // Copy the result GPU -> CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Display result
    printf("Result:\n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_C[i]);
    }

    printf("\n");

    // Always free GPU memory when done.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
