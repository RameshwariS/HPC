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
        if (rank == 0)
            cout << "Number of processes must divide 16." << endl;
        MPI_Finalize();
        return 0;
    }

    int arr[N];
    int localSize = N / size;
    int local[localSize];

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i + 1;
    }

    MPI_Scatter(
        arr,
        localSize,
        MPI_INT,
        local,
        localSize,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    int localSum = 0;
    int localMax = local[0];
    int localMin = local[0];

    for (int i = 0; i < localSize; i++) {
        localSum += local[i];

        if (local[i] > localMax)
            localMax = local[i];

        if (local[i] < localMin)
            localMin = local[i];
    }

    int globalSum;
    int globalMax;
    int globalMin;

    MPI_Reduce(
        &localSum,
        &globalSum,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localMax,
        &globalMax,
        1,
        MPI_INT,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localMin,
        &globalMin,
        1,
        MPI_INT,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        cout << "Global Sum = " << globalSum << endl;
        cout << "Global Maximum = " << globalMax << endl;
        cout << "Global Minimum = " << globalMin << endl;
    }

    MPI_Finalize();
    return 0;
}