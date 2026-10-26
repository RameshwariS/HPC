#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

__global__ void squareArray(const float *a, float *out, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = a[id] * a[id];
}

int main()
{
    const int n = 10;
    vector<float> a(n), out(n);
    for (int i = 0; i < n; ++i)
        a[i] = float(i + 1);
    float *da = nullptr, *dout = nullptr;
    if (cudaMalloc(&da, n * sizeof(float)) != cudaSuccess || cudaMalloc(&dout, n * sizeof(float)) != cudaSuccess)
        return 1;

    cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 128, blocks = (n + threads - 1) / threads;
    squareArray<<<blocks, threads>>>(da, dout, n);
    
    if (cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        cout << a[i] << "^2 = " << out[i] << '\n';
    cudaFree(da);
    cudaFree(dout);
    return 0;
}
