// gpu_info.cu - Query and print the properties of every CUDA GPU in the system.
// Useful to know limits like max threads per block before writing kernels.

#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main() {
    // Find how many CUDA-capable GPUs are present.
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        cout << "No CUDA-capable GPU available: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    // Print details for each GPU.
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;                    // struct that holds all properties
        cudaGetDeviceProperties(&p, i);      // fill it for GPU number i

        cout << "GPU: " << p.name << '\n';
        cout << "Global memory (MB): " << p.totalGlobalMem / (1024 * 1024) << '\n';
        cout << "Compute capability: " << p.major << "." << p.minor << '\n';      // GPU "version"
        cout << "Multiprocessors: " << p.multiProcessorCount << '\n';              // number of SMs
        cout << "Max threads per block: " << p.maxThreadsPerBlock << '\n';
        cout << "Max grid dimensions: " << p.maxGridSize[0] << ", "
             << p.maxGridSize[1] << ", " << p.maxGridSize[2] << '\n';
    }
    return 0;
}
