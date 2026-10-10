#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            cout << "Run exactly 2 processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int sendValue = rank;
    int receiveValue = -1;

    if (rank == 0) {
        MPI_Send(&sendValue, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&receiveValue, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&receiveValue, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&sendValue, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    cout << "Process " << rank
         << " sent " << sendValue
         << " and received " << receiveValue << endl;

    MPI_Finalize();
    return 0;
}
