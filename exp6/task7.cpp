#include <mpi.h>
#include <iostream>

using namespace std;

int main() {
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int arr[5];

    if (rank == 0) {
        arr[0] = 10;
        arr[1] = 20;
        arr[2] = 30;
        arr[3] = 40;
        arr[4] = 50;

        MPI_Send(arr, 5, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        MPI_Recv(arr, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int sum = 0;
        int maximum = arr[0];

        for (int i = 0; i < 5; i++) {
            sum += arr[i];

            if (arr[i] > maximum)
                maximum = arr[i];
        }

        double average = static_cast<double>(sum) / 5;

        cout << "Received Array: ";

        for (int i = 0; i < 5; i++)
            cout << arr[i] << " ";

        cout << endl;
        cout << "Sum = " << sum << endl;
        cout << "Average = " << average << endl;
        cout << "Maximum = " << maximum << endl;
    }

    MPI_Finalize();
    return 0;
}