#include <mpi.h>
#include <iostream>

using namespace std;

int main() {
    MPI_Init(NULL, NULL);

    cout << "Hello from MPI" << endl;

    MPI_Finalize();
    return 0;
}