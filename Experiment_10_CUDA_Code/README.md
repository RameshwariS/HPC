# Experiment 10 — CUDA Optimization and Parallel Algorithms

Files: matrix_global.cu, matrix_tiled.cu, reduction.cu, prefix_sum.cu, histogram.cu.

Requirements: NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable NVIDIA GPU.

Compile and run:
```
nvcc -O2 matrix_global.cu -o matrix_global && ./matrix_global
nvcc -O2 matrix_tiled.cu -o matrix_tiled && ./matrix_tiled
nvcc -O2 reduction.cu -o reduction && ./reduction
nvcc -O2 prefix_sum.cu -o prefix_sum && ./prefix_sum
nvcc -O2 histogram.cu -o histogram && ./histogram
```
These are teaching implementations. Prefix sum is single-block; reduction uses block-level shared memory and atomics to combine block results. Record timings from your own hardware; no performance results are fabricated.
