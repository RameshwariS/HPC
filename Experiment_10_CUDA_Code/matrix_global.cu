// matrix_global.cu - Multiply two N x N matrices (C = A * B) on the GPU using
// only global memory, and compare the time against a plain CPU triple loop.
// Matrices are stored as 1D arrays in row-major order: element (r,c) = M[r*n + c].

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

// Kernel: each thread computes ONE output element C[r][c].
// Threads are laid out in a 2D grid so (r,c) maps naturally to the matrix.
__global__ void matmul(const float *A, const float *B, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's row
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column

    if (r < n && c < n) {
        float s = 0;
        // Dot product of row r of A with column c of B.
        for (int k = 0; k < n; k++) {
            s += A[r * n + k] * B[k * n + c];
        }
        C[r * n + c] = s;
    }
}

int main() {
    const int n = 1024, t = 16;                      // t = threads per block side
    size_t z = size_t(n) * n * sizeof(float);        // bytes for one matrix
    // A filled with 1, B filled with 2. C = GPU result, R = CPU reference.
    vector<float> A(n * n, 1), B(n * n, 2), C(n * n), R(n * n);

    // ---- CPU reference multiplication (and its timing) ----
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

    // ---- GPU multiplication ----
    float *da, *db, *dc;
    cudaMalloc(&da, z);
    cudaMalloc(&db, z);
    cudaMalloc(&dc, z);
    cudaMemcpy(da, A.data(), z, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), z, cudaMemcpyHostToDevice);

    // 2D block of t x t threads; 2D grid big enough to cover the whole matrix.
    dim3 block(t, t), grid((n + t - 1) / t, (n + t - 1) / t);

    auto g = chrono::high_resolution_clock::now();
    matmul<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();                          // wait for kernel to finish
    auto h = chrono::high_resolution_clock::now();
    cudaMemcpy(C.data(), dc, z, cudaMemcpyDeviceToHost);

    // Compare GPU result with CPU reference (allow small float error).
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
         << " Correct=" << (ok ? "PASS" : "FAIL") << "\n";

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return ok ? 0 : 1;
}
