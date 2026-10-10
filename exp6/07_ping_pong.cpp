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

    if (size != 2) {
        if (rank == 0) {
            cout << "Run exactly 2 processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int messageSize = 1024;
    int iterations = 1000;



    if (messageSize <= 0 || iterations <= 0) {
        if (rank == 0) {
            cout << "Message size and iterations must be positive." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    vector<char> buffer(messageSize, 'A');

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    for (int i = 0; i < iterations; i++) {
        if (rank == 0) {
            MPI_Send(
                buffer.data(),
                messageSize,
                MPI_CHAR,
                1,
                0,
                MPI_COMM_WORLD
            );

            MPI_Recv(
                buffer.data(),
                messageSize,
                MPI_CHAR,
                1,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        } else {
            MPI_Recv(
                buffer.data(),
                messageSize,
                MPI_CHAR,
                0,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            MPI_Send(
                buffer.data(),
                messageSize,
                MPI_CHAR,
                0,
                0,
                MPI_COMM_WORLD
            );
        }
    }

    double totalTime = MPI_Wtime() - start;

    if (rank == 0) {
        double rtt = totalTime / iterations;

        cout << fixed << setprecision(9);
        cout << "Message size = " << messageSize << " bytes" << endl;
        cout << "Iterations = " << iterations << endl;
        cout << "Total time = " << totalTime << " seconds" << endl;
        cout << "Average RTT = " << rtt << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
