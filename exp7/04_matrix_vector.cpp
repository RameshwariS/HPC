#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 500;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N <= 0 || N % size != 0) {
        if (rank == 0) {
            cout << "N must be positive and divisible by the number of processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int rows = N / size;

    vector<double> A;
    vector<double> X(N);
    vector<double> Y;
    double sequentialTime;

    if (rank == 0) {
        A.resize(N * N);
        Y.resize(N);

        for (int i = 0; i < N; i++) {
            X[i] = 1.0;
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i + j) % 10 + 1;
            }
        }

        double start = MPI_Wtime();

        for (int i = 0; i < N; i++) {
            Y[i] = 0.0;
            for (int j = 0; j < N; j++) {
                Y[i] += A[i * N + j] * X[j];
            }
        }

         sequentialTime = MPI_Wtime() - start;

        cout << fixed << setprecision(6);
        cout << "Sequential time = " << sequentialTime << " seconds" << endl;
    }

    vector<double> localA(rows * N);
    vector<double> localY(rows);

    MPI_Bcast(X.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(
        rank == 0 ? A.data() : nullptr,
        rows * N,
        MPI_DOUBLE,
        localA.data(),
        rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < rows; i++) {
        localY[i] = 0.0;
        for (int j = 0; j < N; j++) {
            localY[i] += localA[i * N + j] * X[j];
        }
    }

    MPI_Gather(
        localY.data(),
        rows,
        MPI_DOUBLE,
        rank == 0 ? Y.data() : nullptr,
        rows,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localTime = MPI_Wtime() - start;
    double mpiTime = 0.0;

    MPI_Reduce(&localTime, &mpiTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "MPI time = " << mpiTime << " seconds" << endl;
        cout << "Speedup = " << (sequentialTime / mpiTime) << endl;
        cout << "Efficiency = " << (sequentialTime / mpiTime) / size << endl;
        cout << "First result value = " << Y[0] << endl;
    }

    MPI_Finalize();
    return 0;
}
