#include <mpi.h>
#include <iostream>

using namespace std;

int main() {
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int value = 100;

        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        int value;

        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Received Value = " << value << endl;
    }

    MPI_Finalize();
    return 0;
}