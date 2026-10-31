// cpu_basic_tiled.cu - Compare FOUR ways to multiply matrices (C = A * B):
//   1) CPU triple loop
//   2) basic CUDA kernel (global memory only)
//   3) tiled CUDA kernel (shared memory)
//   4) cuBLAS (NVIDIA's highly optimized library)
// For sizes 256, 512, 1024 it prints timings, checks correctness, and speedups.
// Matrices are 1D row-major arrays: element (r,c) = M[r*n + c].

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

// Error-check macro for CUDA runtime calls: print and exit on failure.
#define CUDA_OK(call)                                                        \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";         \
            return 1;                                                        \
        }                                                                    \
    } while (0)

// Error-check macro for cuBLAS calls.
#define CUBLAS_OK(call)                                                      \
    do {                                                                     \
        cublasStatus_t s = (call);                                           \
        if (s != CUBLAS_STATUS_SUCCESS) {                                    \
            cerr << "cuBLAS error code: " << s << "\n";                      \
            return 1;                                                        \
        }                                                                    \
    } while (0)

// ---- Method 2: basic kernel. One thread computes one output element. ----
__global__ void matmulBasic(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {                 // dot product row . column
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

#ifndef TILE
#define TILE 32
#endif

// ---- Method 3: tiled kernel. Loads TILE x TILE squares into fast shared
// memory and reuses them, cutting slow global-memory reads. ----
__global__ void matmulTiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // Move across A's row and down B's column one tile at a time.
    for (int m = 0; m < (n + TILE - 1) / TILE; m++) {
        int ac = m * TILE + threadIdx.x;
        int br = m * TILE + threadIdx.y;

        // Load this tile into shared memory (0 for out-of-range padding).
        As[threadIdx.y][threadIdx.x] = (row < n && ac < n) ? A[row * n + ac] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (br < n && col < n) ? B[br * n + col] : 0.0f;

        __syncthreads();                              // tile fully loaded
        for (int k = 0; k < TILE; k++) {              // multiply the two tiles
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();                              // done before reloading
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Elapsed milliseconds between two time points.
double ms(Clock::time_point a, Clock::time_point b) {
    return chrono::duration<double, milli>(b - a).count();
}

int main() {
    cout << fixed << setprecision(3);                 // 3 decimals for timings

    for (int n : {256, 512, 1024}) {                  // test three matrix sizes
        size_t bytes = size_t(n) * n * sizeof(float);
        // cpu/basic/tiled/blas hold the result from each of the four methods.
        vector<float> A(n * n), B(n * n), cpu(n * n), basic(n * n), tiled(n * n),
            blas(n * n);

        // Fill A and B with simple repeating fractions (just some test data).
        for (int i = 0; i < n * n; i++) {
            A[i] = float((i % 13) + 1) / 13.0f;
            B[i] = float((i % 7) + 1) / 7.0f;
        }

        // ---- Method 1: CPU triple loop (the reference answer) ----
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

        // Allocate GPU memory, shared by all GPU methods below.
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_OK(cudaMalloc(&dA, bytes));
        CUDA_OK(cudaMalloc(&dB, bytes));
        CUDA_OK(cudaMalloc(&dC, bytes));

        auto e0 = Clock::now();                       // start of end-to-end timing
        CUDA_OK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

        // 16x16 threads per block; grid covers the whole matrix.
        dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);

        // ---- Method 2: basic kernel ----
        auto b0 = Clock::now();
        matmulBasic<<<grid, block>>>(dA, dB, dC, n);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto b1 = Clock::now();
        CUDA_OK(cudaMemcpy(basic.data(), dC, bytes, cudaMemcpyDeviceToHost));
        auto e1 = Clock::now();
        double basicKernel = ms(b0, b1);              // kernel-only time
        double basicEnd = ms(e0, e1);                 // incl. memory transfers

        // ---- Method 3: tiled kernel ----
        auto t0 = Clock::now();
        matmulTiled<<<grid, block>>>(dA, dB, dC, n);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        CUDA_OK(cudaMemcpy(tiled.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double tiledKernel = ms(t0, t1);

        // ---- Method 4: cuBLAS ----
        cublasHandle_t handle;
        CUBLAS_OK(cublasCreate(&handle));
        // cuBLAS is column-major, our data is row-major. Computing C=A*B in row
        // major is the same as computing C^T = B^T * A^T in column major, which
        // is why B and A are swapped in the call below.
        const float alpha = 1.0f, beta = 0.0f;        // C = alpha*A*B + beta*C
        auto l0 = Clock::now();
        CUBLAS_OK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                              dB, n, dA, n, &beta, dC, n));
        CUDA_OK(cudaDeviceSynchronize());
        auto l1 = Clock::now();
        CUDA_OK(cudaMemcpy(blas.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double blasMs = ms(l0, l1);
        cublasDestroy(handle);

        // Compare a result array against the CPU reference (allow small error).
        auto check = [&](const vector<float> &x) {
            for (size_t i = 0; i < x.size(); i++) {
                if (fabs(x[i] - cpu[i]) > 1e-2f) {
                    return false;
                }
            }
            return true;
        };
        // GFLOPS = billions of floating-point ops per second. A matmul does
        // about 2*n^3 operations; divide by time to get the rate.
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
