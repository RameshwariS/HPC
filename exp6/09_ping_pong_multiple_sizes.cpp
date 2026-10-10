#include <mpi.h>
#include <iostream>
#include <vector>
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

    vector<int> sizes = {
        1024,
        10 * 1024,
        100 * 1024,
        1024 * 1024,
        10 * 1024 * 1024
    };

    const int iterations = 100;

    if (rank == 0) {
        cout << fixed << setprecision(9);
        cout << "Message Size(bytes)\tTotal Time(s)\tAverage RTT(s)" << endl;
    }

    for (int messageSize : sizes) {
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

            cout << messageSize << "\t\t"
                 << totalTime << "\t"
                 << rtt << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
