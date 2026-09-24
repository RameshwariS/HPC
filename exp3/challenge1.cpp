#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

long long criticalSum(const vector<int>& a) {
    long long sum = 0;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (int)a.size(); i++) {
            #pragma omp critical
            sum += a[i];
        }
    }

    return sum;
}

long long atomicSum(const vector<int>& a) {
    long long sum = 0;

    #pragma omp parallel for
    for (int i = 0; i < (int)a.size(); i++) {
        #pragma omp atomic //single shared update
        sum += a[i];
    }

    return sum;
}

long long reductionSum(const vector<int>& a) {
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)a.size(); i++)
        sum += a[i];

    return sum;
}

int main() {
    int n = 10000000;

    vector<int> a(n, 1);

    double start, end;

    start = omp_get_wtime();
    long long c = criticalSum(a);
    end = omp_get_wtime();
    double criticalTime = end - start;

    start = omp_get_wtime();
    long long at = atomicSum(a);
    end = omp_get_wtime();
    double atomicTime = end - start;

    start = omp_get_wtime();
    long long r = reductionSum(a);
    end = omp_get_wtime();
    double reductionTime = end - start;

    cout << "Critical Result: " << c << endl;
    cout << "Atomic Result: " << at << endl;
    cout << "Reduction Result: " << r << endl;



    cout << "Critical Time: "
         << criticalTime << " seconds\n";

    cout << "Atomic Time: "
         << atomicTime << " seconds\n";

    cout << "Reduction Time: "
         << reductionTime << " seconds\n";

    return 0;
}