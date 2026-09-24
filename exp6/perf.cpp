#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<int> data;

    int sizes[] = {
        1024 / sizeof(int),
        10 * 1024 / sizeof(int),
        100 * 1024 / sizeof(int),
        1024 * 1024 / sizeof(int),
        10 * 1024 * 1024 / sizeof(int)
    };

    for (int s = 0; s < 5; s++) {
        int count = sizes[s];

        if (rank == 0) {
            data.resize(count, 1);

            MPI_Barrier(MPI_COMM_WORLD);

            double start = MPI_Wtime();

            MPI_Send(data.data(), count, MPI_INT, 1, 0, MPI_COMM_WORLD);

            double end = MPI_Wtime();

            cout << "Message size: " << count * sizeof(int)
                 << " bytes, Communication time: "
                 << (end - start) << " seconds" << endl;
        }
        else if (rank == 1) {
            data.resize(count);

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Recv(data.data(), count, MPI_INT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Finalize();
    return 0;
}