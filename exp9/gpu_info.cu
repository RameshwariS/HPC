#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        cout << "No CUDA-capable GPU available: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        cout << "GPU: " << p.name << '\n';
        cout << "Global memory (MB): " << p.totalGlobalMem / (1024 * 1024) << '\n';
        cout << "Compute capability: " << p.major << "." << p.minor << '\n';
        cout << "Multiprocessors: " << p.multiProcessorCount << '\n';
        cout << "Max threads per block: " << p.maxThreadsPerBlock << '\n';
        cout << "Max grid dimensions: " << p.maxGridSize[0] << ", " << p.maxGridSize[1] << ", " << p.maxGridSize[2] << '\n';
    }
    return 0;
}
