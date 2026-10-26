# Experiment 9 — CUDA Code Files

This folder contains source files for the main programs in Experiment 9:

- `gpu_info.cu` — prints CUDA device properties
- `hello.cu` — launches kernels with different block/thread configurations
- `thread_ids.cu` — prints block, local thread and global thread IDs
- `vector_add.cu` — vector addition with device memory allocation, transfers and correctness check
- `block_sizes.cu` — compares kernel time for 64, 128, 256 and 512 threads per block
- `array_square.cu` — squares array elements on the GPU
- `benchmark.cu` — compares sequential, OpenMP and CUDA vector addition

## Requirements

A system with an NVIDIA CUDA-capable GPU, NVIDIA driver, CUDA Toolkit (`nvcc`), and an OpenMP-capable host compiler is needed. If your laptop does not have an NVIDIA CUDA GPU, run these on the college lab machine or a suitable CUDA-enabled cloud environment.

## Compile and run

```bash
nvcc gpu_info.cu -o gpu_info && ./gpu_info
nvcc hello.cu -o hello && ./hello
nvcc thread_ids.cu -o thread_ids && ./thread_ids
nvcc vector_add.cu -o vector_add && ./vector_add
nvcc block_sizes.cu -o block_sizes && ./block_sizes
nvcc array_square.cu -o array_square && ./array_square
nvcc benchmark.cu -Xcompiler -fopenmp -o benchmark && ./benchmark
```

Execution timings and GPU information depend on the machine; record actual outputs in the report. These programs have not been executed on a CUDA GPU in this environment.
