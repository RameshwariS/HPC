#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value;

    if (rank == 0)
        value = 100;

    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Process " << rank << " received value = " << value << endl;

    MPI_Finalize();
    return 0;
}