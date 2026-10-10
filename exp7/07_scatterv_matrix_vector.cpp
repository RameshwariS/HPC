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

    int N = 10;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N <= 0) {
        if (rank == 0) {
            cout << "N must be positive." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    vector<int> rows(size);
    vector<int> displs(size);
    vector<int> sendcounts(size);

    int base = N / size;
    int extra = N % size;

    int offset = 0;

    for (int i = 0; i < size; i++) {
        rows[i] = base + (i < extra ? 1 : 0);
        sendcounts[i] = rows[i] * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int localRows = rows[rank];

    vector<double> A;
    vector<double> X(N);
    vector<double> Y;

    if (rank == 0) {
        A.resize(N * N);
        Y.resize(N);

        for (int i = 0; i < N; i++) {
            X[i] = 1.0;
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i + j) % 10 + 1;
            }
        }
    }

    MPI_Bcast(X.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> localA(localRows * N);
    vector<double> localY(localRows);

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        localA.data(),
        localRows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < localRows; i++) {
        localY[i] = 0.0;
        for (int j = 0; j < N; j++) {
            localY[i] += localA[i * N + j] * X[j];
        }
    }

    MPI_Gatherv(
        localY.data(),
        localRows,
        MPI_DOUBLE,
        rank == 0 ? Y.data() : nullptr,
        rows.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localTime = MPI_Wtime() - start;
    double totalTime = 0;

    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << fixed << setprecision(6);
        cout << "N = " << N << endl;
        cout << "Processes = " << size << endl;
        cout << "MPI_Scatterv computation time = " << totalTime << " seconds" << endl;

        cout << "Row distribution:" << endl;
        for (int i = 0; i < size; i++) {
            cout << "Process " << i << " -> " << rows[i] << " rows" << endl;
        }

        cout << "Result:" << endl;
        for (int i = 0; i < N; i++) {
            cout << Y[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
