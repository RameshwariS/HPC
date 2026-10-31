// array_square.cu - Square each element of an array on the GPU: out[i] = a[i]*a[i].

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

// Kernel: one thread squares one element.
__global__ void squareArray(const float *a, float *out, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (id < n)                                       // stay inside the array
        out[id] = a[id] * a[id];
}

int main()
{
    const int n = 10;
    vector<float> a(n), out(n);          // CPU-side input and output
    for (int i = 0; i < n; ++i)
        a[i] = float(i + 1);             // a = 1, 2, 3, ... 10

    // Allocate GPU memory for input (da) and output (dout).
    float *da = nullptr, *dout = nullptr;
    if (cudaMalloc(&da, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dout, n * sizeof(float)) != cudaSuccess)
        return 1;                        // allocation failed -> stop

    // Copy input CPU -> GPU.
    cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch enough blocks of 128 threads to cover all n elements.
    int threads = 128, blocks = (n + threads - 1) / threads;
    squareArray<<<blocks, threads>>>(da, dout, n);

    // Wait for the GPU and check it ran without error.
    if (cudaDeviceSynchronize() != cudaSuccess)
        return 1;

    // Copy result GPU -> CPU and print it.
    cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        cout << a[i] << "^2 = " << out[i] << '\n';

    // Free GPU memory.
    cudaFree(da);
    cudaFree(dout);
    return 0;
}
