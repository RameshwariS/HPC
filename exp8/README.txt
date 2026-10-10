Experiment 8 - Comparative Study of OpenMP and MPI

The files implement the six source-code deliverables specified in the experiment:
1. Sequential Matrix-Matrix Multiplication
2. OpenMP Matrix-Matrix Multiplication
3. MPI Matrix-Matrix Multiplication
4. Sequential 2D Convolution
5. OpenMP 2D Convolution
6. MPI 2D Convolution

Compilation:

Sequential matrix multiplication:
g++ -O2 01_sequential_matrix_multiplication.cpp -o seq_matmul

OpenMP matrix multiplication:
g++ -O2 -fopenmp 02_openmp_matrix_multiplication.cpp -o omp_matmul

MPI matrix multiplication:
mpic++ -O2 03_mpi_matrix_multiplication.cpp -o mpi_matmul

Sequential convolution:
g++ -O2 04_sequential_2d_convolution.cpp -o seq_conv

OpenMP convolution:
g++ -O2 -fopenmp 05_openmp_2d_convolution.cpp -o omp_conv

MPI convolution:
mpic++ -O2 06_mpi_2d_convolution.cpp -o mpi_conv

Example execution:

./seq_matmul 500
./omp_matmul 500 2
./omp_matmul 500 4
./omp_matmul 500 8

mpirun -np 2 ./mpi_matmul 500
mpirun -np 4 ./mpi_matmul 500
mpirun -np 8 ./mpi_matmul 500

./seq_conv 1000
./omp_conv 1000 4 static
./omp_conv 1000 4 dynamic
./omp_conv 1000 4 guided

mpirun -np 2 ./mpi_conv 1000
mpirun -np 4 ./mpi_conv 1000
mpirun -np 8 ./mpi_conv 1000

For the assignment's larger experiments, repeat with matrix/input sizes:
Matrix multiplication: 500, 1000, 2000
2D convolution: 1000, 2000, 4000

OpenMP: 2, 4, 8 and maximum available threads.
MPI: 2, 4, 8 processes.

Speedup = Sequential Time / Parallel Time
Efficiency = Speedup / Number of Threads or Processes

The MPI convolution implementation uses row-wise domain decomposition and exchanges neighboring rows as halo data, as required by the experiment.
