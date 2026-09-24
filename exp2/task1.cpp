#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main()
{
    int n = 500;

    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> C(n, vector<int>(n));

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
        {
            A[i][j]=i+j;
            B[i][j]=i-j;
        }

    // omp_set_num_threads(4); // OMP_NUM_THREADS=4 ./program also works


#pragma omp parallel
    {
        int id=omp_get_thread_num();
        int total=omp_get_num_threads();

#pragma omp single //Only ONE thread executes this block critical run all thread but one at time
        cout<<"Total Threads : "<<total<<endl;

        cout<<"Thread ID : "<<id<<endl;

#pragma omp for
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                C[i][j]=A[i][j]+B[i][j];
    }

    cout<<"Matrix Addition Completed"<<endl;

    return 0;
}
