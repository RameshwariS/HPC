#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

__global__ void matmul(const float *A, const float *B, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < n && c < n) {
        float s = 0;
        for (int k = 0; k < n; k++) {
            s += A[r * n + k] * B[k * n + c];
        }
        C[r * n + c] = s;
    }
}

int main() {
    const int n = 1024, t = 16;
    size_t z = size_t(n) * n * sizeof(float);
    vector<float> A(n * n, 1), B(n * n, 2), C(n * n), R(n * n);

    auto a = chrono::high_resolution_clock::now();
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            float s = 0;
            for (int k = 0; k < n; k++) {
                s += A[r * n + k] * B[k * n + c];
            }
            R[r * n + c] = s;
        }
    }
    auto b = chrono::high_resolution_clock::now();

    float *da, *db, *dc;
    cudaMalloc(&da, z);
    cudaMalloc(&db, z);
    cudaMalloc(&dc, z);
    cudaMemcpy(da, A.data(), z, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), z, cudaMemcpyHostToDevice);
    dim3 block(t, t), grid((n + t - 1) / t, (n + t - 1) / t);

    auto g = chrono::high_resolution_clock::now();
    matmul<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    auto h = chrono::high_resolution_clock::now();
    cudaMemcpy(C.data(), dc, z, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < C.size(); i++) {
        if (fabs(C[i] - R[i]) > 1e-3) {
            ok = false;
            break;
        }
    }

    cout << "N=" << n
         << " CPU ms=" << chrono::duration<double, milli>(b - a).count()
         << " CUDA kernel ms=" << chrono::duration<double, milli>(h - g).count()
         << " Correct=" << (ok ? "PASS" : "FAIL") << "\\n";

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return ok ? 0 : 1;
}
