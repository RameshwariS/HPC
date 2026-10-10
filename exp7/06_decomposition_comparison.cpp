#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
using namespace std;

void fillMatrix(vector<double>& M, int N, int type) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (type == 0) {
                M[i * N + j] = (i + j) % 10 + 1;
            } else {
                M[i * N + j] = (i * j) % 10 + 1;
            }
        }
    }
}

double rowWise(int N, int rank, int size) {
    int rows = N / size;
    vector<double> A, B(N * N), C;
    vector<double> localA(rows * N), localC(rows * N);

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        fillMatrix(A, N, 0);
        fillMatrix(B, N, 1);
    }

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(rank == 0 ? A.data() : nullptr, rows * N, MPI_DOUBLE,
                localA.data(), rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += localA[i * N + k] * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }

    MPI_Gather(localC.data(), rows * N, MPI_DOUBLE,
               rank == 0 ? C.data() : nullptr, rows * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double localTime = MPI_Wtime() - start;
    double totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return totalTime;
}

double columnWise(int N, int rank, int size) {
    int cols = N / size;
    vector<double> A(N * N), B, CTransposed;
    vector<double> localBT(N * cols), localCT(N * cols);

    if (rank == 0) {
        fillMatrix(A, N, 0);
        B.resize(N * N);
        CTransposed.resize(N * N);
        fillMatrix(B, N, 1);
    }

    MPI_Bcast(A.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> packedBT;
    if (rank == 0) {
        packedBT.resize(N * N);
        for (int p = 0; p < size; p++) {
            for (int j = 0; j < cols; j++) {
                int col = p * cols + j;
                for (int i = 0; i < N; i++) {
                    packedBT[p * N * cols + j * N + i] = B[i * N + col];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(rank == 0 ? packedBT.data() : nullptr, N * cols, MPI_DOUBLE,
                localBT.data(), N * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * localBT[j * N + k];
            }
            localCT[i * cols + j] = sum;
        }
    }

    MPI_Gather(localCT.data(), N * cols, MPI_DOUBLE,
               rank == 0 ? CTransposed.data() : nullptr, N * cols, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double localTime = MPI_Wtime() - start;
    double totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return totalTime;
}

double blockWise(int N, int rank, int size) {
    int q = static_cast<int>(sqrt(size));
    int block = N / q;
    int blockSize = block * block;

    vector<double> A, B;
    vector<double> packedA, packedB;

    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        fillMatrix(A, N, 0);
        fillMatrix(B, N, 1);

        packedA.resize(N * N);
        packedB.resize(N * N);

        for (int p = 0; p < size; p++) {
            int br = p / q;
            int bc = p % q;

            for (int i = 0; i < block; i++) {
                for (int j = 0; j < block; j++) {
                    packedA[p * blockSize + i * block + j] =
                        A[(br * block + i) * N + (bc * block + j)];

                    packedB[p * blockSize + i * block + j] =
                        B[(br * block + i) * N + (bc * block + j)];
                }
            }
        }
    }

    vector<double> localA(blockSize);
    vector<double> localB(blockSize);
    vector<double> localC(blockSize);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(rank == 0 ? packedA.data() : nullptr, blockSize, MPI_DOUBLE,
                localA.data(), blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(rank == 0 ? packedB.data() : nullptr, blockSize, MPI_DOUBLE,
                localB.data(), blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < block; i++) {
        for (int j = 0; j < block; j++) {
            double sum = 0;
            for (int k = 0; k < block; k++) {
                sum += localA[i * block + k] * localB[k * block + j];
            }
            localC[i * block + j] = sum;
        }
    }

    vector<double> gatheredC;

    if (rank == 0) {
        gatheredC.resize(N * N);
    }

    MPI_Gather(localC.data(), blockSize, MPI_DOUBLE,
               rank == 0 ? gatheredC.data() : nullptr, blockSize, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double localTime = MPI_Wtime() - start;
    double totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return totalTime;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 400;
    string mode = "row";

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) mode = argv[2];

    if (N <= 0 || N % size != 0) {
        if (rank == 0) {
            cout << "For row/column decomposition, N must be divisible by the number of processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    double time = 0;

    if (mode == "row") {
        time = rowWise(N, rank, size);
    } else if (mode == "column") {
        time = columnWise(N, rank, size);
    } else if (mode == "block") {
        int q = static_cast<int>(sqrt(size));
        if (q * q != size || N % q != 0) {
            if (rank == 0) {
                cout << "Block decomposition requires a perfect-square process count and N divisible by sqrt(processes)." << endl;
            }
            MPI_Finalize();
            return 0;
        }
        time = blockWise(N, rank, size);
    } else {
        if (rank == 0) {
            cout << "Mode must be row, column, or block." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        cout << fixed << setprecision(6);
        cout << "Decomposition = " << mode << endl;
        cout << "MPI time = " << time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
