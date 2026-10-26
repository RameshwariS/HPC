#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
using Clock = chrono::high_resolution_clock;

#define CUDA_OK(call)                                                        \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";        \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define CUBLAS_OK(call)                                                      \
    do {                                                                     \
        cublasStatus_t s = (call);                                           \
        if (s != CUBLAS_STATUS_SUCCESS) {                                    \
            cerr << "cuBLAS error code: " << s << "\n";                    \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void matmulBasic(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

#ifndef TILE
#define TILE 32
#endif

__global__ void matmulTiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int m = 0; m < (n + TILE - 1) / TILE; m++) {
        int ac = m * TILE + threadIdx.x;
        int br = m * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < n && ac < n) ? A[row * n + ac] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (br < n && col < n) ? B[br * n + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

double ms(Clock::time_point a, Clock::time_point b) {
    return chrono::duration<double, milli>(b - a).count();
}

int main() {
    cout << fixed << setprecision(3);

    for (int n : {256, 512, 1024}) {
        size_t bytes = size_t(n) * n * sizeof(float);
        vector<float> A(n * n), B(n * n), cpu(n * n), basic(n * n), tiled(n * n),
            blas(n * n);

        for (int i = 0; i < n * n; i++) {
            A[i] = float((i % 13) + 1) / 13.0f;
            B[i] = float((i % 7) + 1) / 7.0f;
        }

        auto c0 = Clock::now();
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                float s = 0;
                for (int k = 0; k < n; k++) {
                    s += A[r * n + k] * B[k * n + c];
                }
                cpu[r * n + c] = s;
            }
        }
        auto c1 = Clock::now();
        double cpuMs = ms(c0, c1);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_OK(cudaMalloc(&dA, bytes));
        CUDA_OK(cudaMalloc(&dB, bytes));
        CUDA_OK(cudaMalloc(&dC, bytes));

        auto e0 = Clock::now();
        CUDA_OK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
        auto b0 = Clock::now();
        matmulBasic<<<grid, block>>>(dA, dB, dC, n);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto b1 = Clock::now();
        CUDA_OK(cudaMemcpy(basic.data(), dC, bytes, cudaMemcpyDeviceToHost));
        auto e1 = Clock::now();
        double basicKernel = ms(b0, b1);
        double basicEnd = ms(e0, e1);

        auto t0 = Clock::now();
        matmulTiled<<<grid, block>>>(dA, dB, dC, n);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        CUDA_OK(cudaMemcpy(tiled.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double tiledKernel = ms(t0, t1);

        cublasHandle_t handle;
        CUBLAS_OK(cublasCreate(&handle));
        // Row-major C=A*B is computed as column-major C^T=B^T*A^T.
        const float alpha = 1.0f, beta = 0.0f;
        auto l0 = Clock::now();
        CUBLAS_OK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                              dB, n, dA, n, &beta, dC, n));
        CUDA_OK(cudaDeviceSynchronize());
        auto l1 = Clock::now();
        CUDA_OK(cudaMemcpy(blas.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double blasMs = ms(l0, l1);
        cublasDestroy(handle);

        auto check = [&](const vector<float> &x) {
            for (size_t i = 0; i < x.size(); i++) {
                if (fabs(x[i] - cpu[i]) > 1e-2f) {
                    return false;
                }
            }
            return true;
        };
        double gflops = 2.0 * double(n) * n * n / (max(blasMs, 1e-9) * 1e6);

        cout << "\nMatrix " << n << " x " << n << "\nCPU time (ms): " << cpuMs
             << "\nBasic CUDA kernel (ms): " << basicKernel
             << "\nBasic CUDA end-to-end (ms): " << basicEnd
             << "\nTiled CUDA kernel (ms): " << tiledKernel
             << "\ncuBLAS call + sync (ms): " << blasMs
             << "\nApprox. cuBLAS GFLOPS: " << gflops
             << "\nCorrectness CPU/basic/tiled/cuBLAS: "
             << (check(basic) && check(tiled) && check(blas) ? "PASS" : "FAIL")
             << "\nSpeedup CPU/basic kernel: "
             << cpuMs / max(basicKernel, 1e-9)
             << "\nSpeedup CPU/tiled kernel: " << cpuMs / max(tiledKernel, 1e-9)
             << "\nSpeedup CPU/cuBLAS: " << cpuMs / max(blasMs, 1e-9) << "\n";

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    return 0;
}
