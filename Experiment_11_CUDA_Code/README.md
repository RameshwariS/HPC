# Experiment 11 — CUDA Performance Optimization and cuBLAS

This source bundle follows the uploaded Experiment 11 PDF: CPU matrix multiplication, basic CUDA, shared-memory tiled CUDA, and cuBLAS using `cublasSgemm()` for matrix sizes 256, 512, and 1024.

## Requirements
- NVIDIA CUDA Toolkit (`nvcc`)
- NVIDIA CUDA-capable GPU
- cuBLAS (included with the CUDA Toolkit)

## Build
Run `bash build.sh`, or compile each file:
```
nvcc -O2 -std=c++14 cpu_basic_tiled.cu -lcublas -o experiment11
```
The single combined program runs CPU, basic CUDA, tiled CUDA, and cuBLAS implementations, verifies results, and prints timing/speedup/GFLOPS tables. Timings are hardware-dependent; record your own run results.

## Notes
- cuBLAS uses column-major conventions. This example uses `cublasSgemm` with operands reversed to produce the intended row-major `C = A * B` result.
- Kernel timing and end-to-end timing are both reported for CUDA implementations where practical.
- Performance graphs can be generated from the printed results; use actual measured timings rather than invented values.
