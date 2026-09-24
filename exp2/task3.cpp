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
// #pragma omp parallel for collapse(4) schedule(static)
#pragma omp parallel for 
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

//static  The iterations are divided before execution starts.
//dynamic iterations are given to threads as they become free.
//guided Guided is similar to dynamic, but the chunk sizes change.
// Initially, threads receive large chunks.
// As the loop progresses, the chunks become smaller.