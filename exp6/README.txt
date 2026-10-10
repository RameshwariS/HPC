Experiment 6 - Introduction to MPI

Source files correspond to the coding tasks in the assignment.

1. Hello MPI
   mpic++ -O2 01_hello_mpi.cpp -o hello_mpi
   mpirun -np 4 ./hello_mpi

2. MPI Rank and Size
   mpic++ -O2 02_rank_and_size.cpp -o rank_size
   mpirun -np 2 ./rank_size
   mpirun -np 4 ./rank_size
   mpirun -np 8 ./rank_size

3. MPI Groups and Communicators
   mpic++ -O2 03_group_communicator.cpp -o group_comm
   mpirun -np 8 ./group_comm

   The program selects even global ranks: 0, 2, 4, 6.

4. Send and Receive a Message
   mpic++ -O2 04_send_receive_message.cpp -o send_receive
   mpirun -np 2 ./send_receive

5. Integer Communication
   mpic++ -O2 05_integer_communication.cpp -o integer_comm
   mpirun -np 2 ./integer_comm

6. Array Communication
   mpic++ -O2 06_array_communication.cpp -o array_comm
   mpirun -np 2 ./array_comm

7. Ping-Pong Communication
   mpic++ -O2 07_ping_pong.cpp -o ping_pong
   mpirun -np 2 ./ping_pong 1024 1000

   Message size is in bytes.
   RTT = Total Time / Number of Iterations.

8. Safe Communication Ordering Demo
   mpic++ -O2 08_deadlock_safe_demo.cpp -o deadlock_safe
   mpirun -np 2 ./deadlock_safe

9. Ping-Pong for Required Message Sizes
   mpic++ -O2 09_ping_pong_multiple_sizes.cpp -o ping_pong_sizes
   mpirun -np 2 ./ping_pong_sizes

