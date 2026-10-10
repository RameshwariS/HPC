#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            cout << "Run with at least 2 processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        int value = 100;

        MPI_Send(
            &value,
            1,
            MPI_INT,
            1,
            0,
            MPI_COMM_WORLD
        );

        cout << "Process 0 sent value = " << value << endl;
    } else if (rank == 1) {
        int value;

        MPI_Recv(
            &value,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        cout << "Process 1 received value = " << value << endl;
    }

    MPI_Finalize();
    return 0;
}
