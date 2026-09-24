#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    int square[1000];

#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();

#pragma omp for
        for (int i = 0; i < 1000; i++)
        {
            square[i] = i * i;
        }
// A critical section allows only one thread at a time to execute that block. 
// to make it more readable
#pragma omp critical 
        {
            cout<<"Thread ID :"<<id<<endl;
            cout<<"Total Threads :" << total<<endl;
        }
    }

   cout<<"first 100 squares"<<endl;

    for (int i = 0; i < 100; i++)
        printf("%d ", square[i]);

    printf("\n");

    return 0;
}
