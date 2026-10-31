// matrix_tiled.cu - Matrix multiply (C = A * B) using SHARED MEMORY TILING.
// Idea: instead of reading A and B from slow global memory for every multiply,
// each block loads a small TILE x TILE square into fast shared memory, reuses it
// for many multiplications, then moves to the next tile. This reduces slow
// global-memory reads and speeds things up.

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

#ifndef TILE
#define TILE 16                 // tile (and block) side length
#endif

__global__ void tiled(const float *A, const float *B, float *C, int n) {
    // Shared memory: visible to all threads in the same block, very fast.
    __shared__ float a[TILE][TILE];
    __shared__ float b[TILE][TILE];

    int r = blockIdx.y * TILE + threadIdx.y;   // global row this thread handles
    int c = blockIdx.x * TILE + threadIdx.x;   // global column this thread handles
    float s = 0;                               // running dot-product sum

    // Walk across the row of A / down the column of B, one tile at a time.
    for (int q = 0; q < (n + TILE - 1) / TILE; q++) {
        int ac = q * TILE + threadIdx.x;       // column in A for this tile
        int br = q * TILE + threadIdx.y;       // row in B for this tile

        // Each thread loads one element of A and one of B into shared memory.
        // Out-of-range elements are loaded as 0 (padding).
        a[threadIdx.y][threadIdx.x] = (r < n && ac < n) ? A[r * n + ac] : 0;
        b[threadIdx.y][threadIdx.x] = (br < n && c < n) ? B[br * n + c] : 0;

        __syncthreads();                       // wait until the whole tile is loaded

        // Multiply the two tiles together and accumulate.
        for (int k = 0; k < TILE; k++) {
            s += a[threadIdx.y][k] * b[k][threadIdx.x];
        }
        __syncthreads();                       // wait before overwriting the tile
    }

    if (r < n && c < n) {
        C[r * n + c] = s;                      // write the final result
    }
}

int main() {
    const int n = 256;
    size_t z = size_t(n) * n * sizeof(float);
    // A = 1 everywhere, B = 2 everywhere. Each C element = sum of n*(1*2) = 2*256 = 512.
    vector<float> A(n * n, 1), B(n * n, 2), C(n * n);

    float *da, *db, *dc;
    cudaMalloc(&da, z);
    cudaMalloc(&db, z);
    cudaMalloc(&dc, z);
    cudaMemcpy(da, A.data(), z, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), z, cudaMemcpyHostToDevice);

    // Block = one tile; grid covers the whole matrix.
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    auto x = chrono::high_resolution_clock::now();
    tiled<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    auto y = chrono::high_resolution_clock::now();
    cudaMemcpy(C.data(), dc, z, cudaMemcpyDeviceToHost);

    // Every element should equal 512 (see note above).
    bool ok = true;
    for (float v : C) {
        if (fabs(v - 512) > 1e-3) {
            ok = false;
            break;
        }
    }

    cout << "Tile=" << TILE
         << " kernel ms=" << chrono::duration<double, milli>(y - x).count()
         << " Correct=" << (ok ? "PASS" : "FAIL") << "\n";

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return ok ? 0 : 1;
}
