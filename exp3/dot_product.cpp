#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main(){
    int n = 10000000;
    vector<int> arr(n,4);
    vector<int> arr2(n,3);
    double start,end;
    start = omp_get_wtime();
    long long dot_product = 0;

    
    for(int i=0;i<n;i++)
        dot_product += arr[i]*arr2[i];
    end = omp_get_wtime();
    double seq = end-start;
    cout<<"Sequential Time = "<<seq<<endl;


    dot_product = 0;
    double parR,parC;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for reduction(+:dot_product)
        for(int i=0;i<n;i++)
            dot_product += arr[i]*arr2[i];
    }
    end = omp_get_wtime();


    parR = end-start;
    cout<<"Parallel Time using reduction = "<<parR<<endl;
      cout << "Speedup: "
             << seq / parR << endl;

        cout << "Efficiency: "
             << (seq / parR) / omp_get_max_threads()
             << endl;
}