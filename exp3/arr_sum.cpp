#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main(){
    int n =10000000;

    vector<int> arr(n);
    for(int i=0;i<n;i++)
        arr[i]=1;
    int sum=0;
    double start,end;

    start = omp_get_wtime();
    for(int i=0;i<n;i++)
        sum+=arr[i];
    end = omp_get_wtime();

    double seq = end-start;
    cout<<"Sequential Time = "<<seq<<endl;
    sum=0;
    double parR,parC;

    start = omp_get_wtime();
    #pragma omp parallel 
    {
        #pragma omp for reduction(+:sum)
        for(int i=0;i<n;i++)
            sum+=arr[i];
    }
    end = omp_get_wtime();
    parR = end-start;
    cout<<"Parallel Time using reduction = "<<parR<<endl;

    sum=0;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp parallel for
        for(int i=0;i<n;i++){
            #pragma omp critical
            sum+=arr[i];
        }
    }
    end = omp_get_wtime();
    parC = end-start;
    cout<<"Parallel Time using Critical Section = "<<parC<<endl;
    
}
