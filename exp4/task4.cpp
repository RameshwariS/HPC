#include<bits/stdc++.h>
#include <omp.h>
using namespace std;

void mergeSort(vector<int>& a, int l, int r) {
    if (l >= r) return;

    int m = (l + r) / 2;

    #pragma omp task shared(a) //means that all tasks use the same vector a.
    mergeSort(a, l, m);

    #pragma omp task shared(a)
    mergeSort(a, m + 1, r);

    #pragma omp taskwait  //makes the program wait until both sorting tasks are completed

    inplace_merge(a.begin() + l, a.begin() + m + 1, a.begin() + r + 1);
}

int main() {
    int  n = 1000;
    // cin >> n;

    vector<int> a(n);

    for (int i = 0; i < n; i++)
        a[i] = rand();

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(a, 0, n - 1);
    }

    cout << "Parallel Merge Sort Time: "
         << omp_get_wtime() - start << " seconds\n";

    return 0;
}
