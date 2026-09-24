#include <mpi.h>
#include <iostream>

using namespace std;

int main() {
    MPI_Init(NULL, NULL);

    int world_rank, world_size;

    // world_rank This tells the current process its rank in MPI_COMM_WORLD.
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // world_size This tells how many processes are running.
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Group world_group, new_group; //collection of MPI processes.
    MPI_Comm new_comm;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int ranks[] = {0, 2, 4, 6};

    if (world_size >= 7) {
        MPI_Group_incl(world_group, 4, ranks, &new_group);

        MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

        if (new_comm != MPI_COMM_NULL) {
            int new_rank;
            MPI_Comm_rank(new_comm, &new_rank);

            cout << "Global Rank: " << world_rank
                 << " -> New Communicator Rank: " << new_rank << endl;

            MPI_Comm_free(&new_comm);
        }

        MPI_Group_free(&new_group);
    }

    MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}