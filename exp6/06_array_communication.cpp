#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
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

    const int N = 5;

    if (rank == 0) {
        int data[N] = {10, 20, 30, 40, 50};

        MPI_Send(
            data,
            N,
            MPI_INT,
            1,
            10,
            MPI_COMM_WORLD
        );

        cout << "Process 0 sent array: ";
        for (int x : data) {
            cout << x << " ";
        }
        cout << endl;
    } else if (rank == 1) {
        int data[N];

        MPI_Recv(
            data,
            N,
            MPI_INT,
            0,
            10,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        int sum = 0;
        int maximum = data[0];

        for (int i = 0; i < N; i++) {
            sum += data[i];
            maximum = max(maximum, data[i]);
        }

        double average = static_cast<double>(sum) / N;

        cout << "Process 1 received array: ";
        for (int x : data) {
            cout << x << " ";
        }
        cout << endl;

        cout << "Sum = " << sum << endl;
        cout << "Average = " << average << endl;
        cout << "Maximum = " << maximum << endl;
    }

    MPI_Finalize();
    return 0;
}
