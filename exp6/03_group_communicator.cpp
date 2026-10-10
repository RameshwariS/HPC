#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Group worldGroup, newGroup;
    MPI_Comm newComm = MPI_COMM_NULL;

    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

    vector<int> selected;

    for (int i = 0; i < worldSize; i += 2) {
        selected.push_back(i);
    }
//A group is essentially a collection of MPI processes.
    MPI_Group_incl( // creates group
        worldGroup,
        selected.size(),
        selected.data(),
        &newGroup
    );
//A communicator provides the communication context for those processes.
    MPI_Comm_create_group( // creates communicatoion
        MPI_COMM_WORLD,
        newGroup,
        0,
        &newComm
    );

    if (newComm != MPI_COMM_NULL) {
        int localRank, localSize;
        MPI_Comm_rank(newComm, &localRank);
        MPI_Comm_size(newComm, &localSize);

        cout << "Global rank = " << worldRank
             << ", New communicator rank = " << localRank
             << ", New communicator size = " << localSize << endl;
    }

    MPI_Group_free(&newGroup);
    MPI_Group_free(&worldGroup);

    if (newComm != MPI_COMM_NULL) {
        MPI_Comm_free(&newComm);
    }

    MPI_Finalize();
    return 0;
}
