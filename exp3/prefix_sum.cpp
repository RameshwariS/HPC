#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

void sequentialPrefixSum(const vector<int>& a, vector<long long>& prefix) {
    prefix[0] = a[0];

    for (int i = 1; i < (int)a.size(); i++)
        prefix[i] = prefix[i - 1] + a[i];
}

void parallelPrefixSum(const vector<int>& a, vector<long long>& prefix) {
    int n = a.size();
    vector<long long> temp(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        prefix[i] = a[i];

    for (int d = 1; d < n; d *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (i >= d)
                temp[i] = prefix[i] + prefix[i - d];
            else
                temp[i] = prefix[i];
        }

        prefix.swap(temp);
    }
}

int main() {
    int n = 10000000;

    vector<int> a(n, 1);
    vector<long long> prefix(n);

    double start = omp_get_wtime();

    sequentialPrefixSum(a, prefix);

    double end = omp_get_wtime();

    cout << "Sequential Time: " << end - start << " seconds\n";
    cout << "Last Prefix Sum: " << prefix[n - 1] << endl;

    start = omp_get_wtime();

    parallelPrefixSum(a, prefix);

    end = omp_get_wtime();

    cout << "Parallel Time: " << end - start << " seconds\n";
    cout << "Last Prefix Sum: " << prefix[n - 1] << endl;

    return 0;
}