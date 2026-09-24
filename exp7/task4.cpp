#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000;

    if (N % size != 0) {
        if (rank == 0)
            cout << "N must be divisible by number of processes." << endl;
        MPI_Finalize();
        return 0;
    }

    int rows = N / size;

    double* A = nullptr;
    double* X = new double[N];
    double* Y = nullptr;

    double* localA = new double[rows * N];
    double* localY = new double[rows];

    if (rank == 0) {
        A = new double[N * N];
        Y = new double[N];

        for (int i = 0; i < N; i++) {
            X[i] = 1.0;

            for (int j = 0; j < N; j++)
                A[i * N + j] = 1.0;
        }
    }

    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(
        A,
        rows * N,
        MPI_DOUBLE,
        localA,
        rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    for (int i = 0; i < rows; i++) {
        localY[i] = 0;

        for (int j = 0; j < N; j++)
            localY[i] += localA[i * N + j] * X[j];
    }

    MPI_Gather(
        localY,
        rows,
        MPI_DOUBLE,
        Y,
        rows,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        cout << "MPI Matrix-Vector Time = "
             << end - start << " seconds" << endl;

        cout << "First 10 elements:" << endl;

        for (int i = 0; i < 10; i++)
            cout << Y[i] << " ";

        cout << endl;
    }

    delete[] X;
    delete[] localA;
    delete[] localY;

    if (rank == 0) {
        delete[] A;
        delete[] Y;
    }

    MPI_Finalize();
    return 0;
}