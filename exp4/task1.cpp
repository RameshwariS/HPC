#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    int n=  100;
    

    vector<vector<double>> A(n, vector<double>(n, 1.0));
    vector<double> x(n, 1.0), y(n);

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < n; j++)
            y[i] += A[i][j] * x[j];
    }

    double seq = omp_get_wtime() - start;

    start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < n; j++)
            y[i] += A[i][j] * x[j];
    }

    double par = omp_get_wtime() - start;

    cout << "Sequential Time: " << seq << endl;
    cout << "Parallel Time: " << par << endl;
    cout << "Speedup: " << seq / par << endl;
    cout<< "Efficiency: "<< (seq / par) / omp_get_max_threads() << endl;

    return 0;
}
