#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void multiply(vector<vector<double>>& A,
              vector<vector<double>>& B,
              vector<vector<double>>& C,
              int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

int main() {
    int n = 50;
   

    vector<vector<double>> A(n, vector<double>(n, 2));
    vector<vector<double>> B(n, vector<double>(n, 4));
    vector<vector<double>> C(n, vector<double>(n));

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];

    double seq = omp_get_wtime() - start;

    start = omp_get_wtime();
      C.assign(n, vector<double>(n, 0));
    multiply(A, B, C, n);
    double par = omp_get_wtime() - start;

    cout << "Sequential Time: " << seq << endl;
    cout << "Parallel Time: " << par << endl;
    cout << "Speedup: " << seq / par << endl;
    cout << "Efficiency: " << (seq / par) / omp_get_max_threads() << endl;

    return 0;
}
