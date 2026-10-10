#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 16;

    if (N % size != 0) {
        if (rank == 0) {
            cout << "Number of processes must divide 16." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int data[N];
    int localN = N / size;
    int* local = new int[localN];

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            data[i] = i + 1;
        }
    }

    MPI_Scatter(data, localN, MPI_INT, local, localN, MPI_INT, 0, MPI_COMM_WORLD);

    int localSum = 0;
    int localMax = local[0];
    int localMin = local[0];

    for (int i = 0; i < localN; i++) {
        localSum += local[i];
        if (local[i] > localMax) localMax = local[i];
        if (local[i] < localMin) localMin = local[i];
    }

    int globalSum = 0;
    int globalMax = 0;
    int globalMin = 0;

    MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    cout << "Process " << rank
         << ": local sum = " << localSum
         << ", local max = " << localMax
         << ", local min = " << localMin << endl;

    if (rank == 0) {
        cout << "Global sum = " << globalSum << endl;
        cout << "Global maximum = " << globalMax << endl;
        cout << "Global minimum = " << globalMin << endl;
    }

    delete[] local;

    MPI_Finalize();
    return 0;
}
