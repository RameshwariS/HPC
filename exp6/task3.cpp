#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int iterations = 1000;
    const int message_size = 1024;

    vector<char> buffer(message_size);

    MPI_Barrier(MPI_COMM_WORLD);

    double start = 0;

    if (rank == 0) {
        start = MPI_Wtime();

        for (int i = 0; i < iterations; i++) {
            MPI_Send(buffer.data(), message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

            MPI_Recv(buffer.data(), message_size, MPI_CHAR, 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double end = MPI_Wtime();

        double total_time = end - start;
        double rtt = total_time / iterations;

        cout << "Iterations = " << iterations << endl;
        cout << "Total Time = " << total_time << " seconds" << endl;
        cout << "Average RTT = " << rtt << " seconds" << endl;
    }
    else if (rank == 1) {
        for (int i = 0; i < iterations; i++) {
            MPI_Recv(buffer.data(), message_size, MPI_CHAR, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}