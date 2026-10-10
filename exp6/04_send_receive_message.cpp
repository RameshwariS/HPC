#include <mpi.h>
#include <iostream>
#include <cstring>
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
        char message[] = "Hello from Process 0";
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 100, MPI_COMM_WORLD);
    } else if (rank == 1) {
        char message[100];
        MPI_Status status;

        MPI_Recv(
            message,
            100,
            MPI_CHAR,
            0,
            100,
            MPI_COMM_WORLD,
            &status
        );

        cout << "Process 1 received: " << message << endl;
    }

    MPI_Finalize();
    return 0;
}
