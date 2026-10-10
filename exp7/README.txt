MPI Assignment 7 - Code Files

Compile:
mpic++ -O2 01_broadcast.cpp -o broadcast
mpic++ -O2 02_scatter_gather.cpp -o scatter_gather
mpic++ -O2 03_reduce.cpp -o reduce
mpic++ -O2 04_matrix_vector.cpp -o matrix_vector
mpic++ -O2 05_matrix_matrix.cpp -o matrix_matrix
mpic++ -O2 06_decomposition_comparison.cpp -o decomposition
mpic++ -O2 07_scatterv_matrix_vector.cpp -o scatterv_matrix_vector

Run examples:
mpirun -np 4 ./broadcast
mpirun -np 4 ./scatter_gather
mpirun -np 4 ./reduce
mpirun -np 4 ./matrix_vector 500
mpirun -np 4 ./matrix_matrix 500

Decomposition comparison:
mpirun -np 4 ./decomposition 400 row
mpirun -np 4 ./decomposition 400 column
mpirun -np 4 ./decomposition 400 block

Scatterv:
mpirun -np 3 ./scatterv_matrix_vector 10

