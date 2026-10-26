#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

#ifndef TILE
#define TILE 16
#endif

__global__ void tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float a[TILE][TILE];
    __shared__ float b[TILE][TILE];

    int r = blockIdx.y * TILE + threadIdx.y;
    int c = blockIdx.x * TILE + threadIdx.x;
    float s = 0;

    for (int q = 0; q < (n + TILE - 1) / TILE; q++) {
        int ac = q * TILE + threadIdx.x;
        int br = q * TILE + threadIdx.y;
        a[threadIdx.y][threadIdx.x] =
            (r < n && ac < n) ? A[r * n + ac] : 0;
        b[threadIdx.y][threadIdx.x] =
            (br < n && c < n) ? B[br * n + c] : 0;

        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            s += a[threadIdx.y][k] * b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (r < n && c < n) {
        C[r * n + c] = s;
    }
}

int main() {
    const int n = 256;
    size_t z = size_t(n) * n * sizeof(float);
    vector<float> A(n * n, 1), B(n * n, 2), C(n * n);

    float *da, *db, *dc;
    cudaMalloc(&da, z);
    cudaMalloc(&db, z);
    cudaMalloc(&dc, z);
    cudaMemcpy(da, A.data(), z, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), z, cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    auto x = chrono::high_resolution_clock::now();
    tiled<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    auto y = chrono::high_resolution_clock::now();
    cudaMemcpy(C.data(), dc, z, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (float v : C) {
        if (fabs(v - 512) > 1e-3) {
            ok = false;
            break;
        }
    }

    cout << "Tile=" << TILE
         << " kernel ms=" << chrono::duration<double, milli>(y - x).count()
         << " Correct=" << (ok ? "PASS" : "FAIL") << "\\n";

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return ok ? 0 : 1;
}
