#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main()
{
    int n=1000;

    vector<vector<double>> A(n,vector<double>(n));
    vector<vector<double>> B(n,vector<double>(n));
    vector<vector<double>> C(n,vector<double>(n));

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
        {
            A[i][j]=1;
            B[i][j]=1;
        }

    double start,end;

    start=omp_get_wtime();

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
        {
            C[i][j]=0;

            for(int k=0;k<n;k++)
                C[i][j]+=A[i][k]*B[k][j];
        }

    end=omp_get_wtime();

    double seq=end-start;

    cout<<"Sequential Time = "<<seq<<endl;

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            C[i][j]=0;

    start=omp_get_wtime();
// 
// #pragma omp parallel for schedule(guided)
// #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                C[i][j]+=A[i][k]*B[k][j];

    end=omp_get_wtime();

    double par=end-start;

    cout<<"Parallel Time = "<<par<<endl;

    cout<<"Speedup = "<<seq/par<<endl;

    cout<<"Efficiency = "<<(seq/par)/omp_get_max_threads()<<endl;

    return 0;
}

// Keeping the thread on the same core can make its execution more predictable and can sometimes improve performance.
// OMP_PLACES=cores tells it to consider the physical cores as the places where threads should run.
// "Once a thread is assigned to a CPU place, keep it there."

// Without binding, the operating system may move threads between CPUs.

// collapse(2) tells OpenMP:

// "Treat these two nested loops as one large loop."

// So instead of:

// 4 × 5

// OpenMP sees:

// 20 iterations

// Conceptually:

// (i=0,j=0)
// (i=0,j=1)
// (i=0,j=2)
// ...
// (i=3,j=4)

// Now the 20 iterations can be distributed among your threads.