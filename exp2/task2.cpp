#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main()
{
    long long N=10000000;

    vector<long long> a(N);

    for(long long i=0;i<N;i++)
        a[i]=1;

    double start,end;

    long long sum=0;

    start=omp_get_wtime();

    for(long long i=0;i<N;i++)
        sum+=a[i];

    end=omp_get_wtime();

    double seq=end-start;

    cout<<"Sequential Sum = "<<sum<<endl;
    cout<<"Sequential Time = "<<seq<<endl;

    sum=0;

    start=omp_get_wtime();

#pragma omp parallel for reduction(+:sum)
    for(long long i=0;i<N;i++)
        sum+=a[i];

    end=omp_get_wtime();

    double par=end-start;

    cout<<"Parallel Sum = "<<sum<<endl;
    cout<<"Parallel Time = "<<par<<endl;

    cout<<"Threads = "<<omp_get_max_threads()<<endl;

    cout<<"Speedup = "<<seq/par<<endl;

    return 0;
    
}
