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
            cout << "N must be positive and divisible by the number of MPI processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int rows = N / size;

    vector<double> A;
    vector<double> B(N * N);
    vector<double> C;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i + j) % 10 + 1;
                B[i * N + j] = (i * j) % 10 + 1;
            }
        }
    }

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> localA(rows * N);
    vector<double> localC(rows * N, 0.0);

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
        for (int k = 0; k < N; k++) {
            double aik = localA[i * N + k];
            for (int j = 0; j < N; j++) {
                localC[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    MPI_Gather(
        localC.data(),
        rows * N,
        MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localTime = MPI_Wtime() - start;
    double mpiTime = 0.0;

    MPI_Reduce(&localTime, &mpiTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << fixed << setprecision(6);
        cout << "Matrix size = " << N << " x " << N << endl;
        cout << "MPI processes = " << size << endl;
        cout << "MPI time = " << mpiTime << " seconds" << endl;
        cout << "C[0][0] = " << C[0] << endl;
    }

    MPI_Finalize();
    return 0;
}
