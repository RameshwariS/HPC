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

    for (int i = 0; i < localSize; i++)
        localSum += local[i];

    int sums[size];

    MPI_Gather(
        &localSum,
        1,
        MPI_INT,
        sums,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    cout << "Process " << rank << " local sum = " << localSum << endl;

    if (rank == 0) {
        int total = 0;

        for (int i = 0; i < size; i++)
            total += sums[i];

        cout << "Global sum = " << total << endl;
    }

    MPI_Finalize();
    return 0;
}