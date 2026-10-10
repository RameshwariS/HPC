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

    int N = 1000;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N < 3 || N < size) {
        if (rank == 0) {
            cout << "Invalid matrix size." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    vector<int> rows(size);
    vector<int> starts(size);
    vector<int> counts(size);

    int base = N / size;
    int extra = N % size;
    int offset = 0;

    for (int p = 0; p < size; p++) {
        rows[p] = base + (p < extra ? 1 : 0);
        starts[p] = offset;
        counts[p] = rows[p] * N;
        offset += rows[p];
    }

    int localRows = rows[rank];

    vector<double> input;
    vector<double> output;

    if (rank == 0) {
        input.resize(N * N);
        output.resize(N * N, 0.0);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                input[i * N + j] = (i + j) % 256;
            }
        }
    }

    double kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    int haloRows = localRows + 2;
    vector<double> local(haloRows * N, 0.0);
    vector<double> localOutput(localRows * N, 0.0);

    MPI_Datatype rowType;
    MPI_Type_contiguous(N, MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    vector<int> recvcounts(size);
    vector<int> displacements(size);

    for (int p = 0; p < size; p++) {
        recvcounts[p] = counts[p];
        displacements[p] = starts[p] * N;
    }

    vector<double> owned(localRows * N);

    MPI_Scatterv(
        rank == 0 ? input.data() : nullptr,
        recvcounts.data(),
        displacements.data(),
        MPI_DOUBLE,
        owned.data(),
        localRows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < N; j++) {
            local[(i + 1) * N + j] = owned[i * N + j];
        }
    }

    if (rank > 0) {
        MPI_Sendrecv(
            local.data() + N,
            N,
            MPI_DOUBLE,
            rank - 1,
            0,
            local.data(),
            N,
            MPI_DOUBLE,
            rank - 1,
            1,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    if (rank < size - 1) {
        MPI_Sendrecv(
            local.data() + localRows * N,
            N,
            MPI_DOUBLE,
            rank + 1,
            1,
            local.data() + (localRows + 1) * N,
            N,
            MPI_DOUBLE,
            rank + 1,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    for (int i = 1; i <= localRows; i++) {
        int globalRow = starts[rank] + i - 1;

        if (globalRow == 0 || globalRow == N - 1) {
            continue;
        }

        for (int j = 1; j < N - 1; j++) {
            double sum = 0.0;

            for (int u = -1; u <= 1; u++) {
                for (int v = -1; v <= 1; v++) {
                    sum += local[(i + u) * N + (j + v)] * kernel[u + 1][v + 1];
                }
            }

            localOutput[(i - 1) * N + j] = sum;
        }
    }

    MPI_Gatherv(
        localOutput.data(),
        localRows * N,
        MPI_DOUBLE,
        rank == 0 ? output.data() : nullptr,
        recvcounts.data(),
        displacements.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localTime = MPI_Wtime() - start;
    double mpiTime = 0.0;

    MPI_Reduce(&localTime, &mpiTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << fixed << setprecision(6);
        cout << "Input size = " << N << " x " << N << endl;
        cout << "MPI processes = " << size << endl;
        cout << "MPI convolution time = " << mpiTime << " seconds" << endl;
        cout << "Output[1][1] = " << output[N + 1] << endl;
    }

    MPI_Type_free(&rowType);
    MPI_Finalize();
    return 0;
}
