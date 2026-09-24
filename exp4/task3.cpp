#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
using namespace std;

void rowWise(const vector<vector<double>>& A,
             const vector<vector<double>>& B,
             vector<vector<double>>& C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

void columnWise(const vector<vector<double>>& A,
                const vector<vector<double>>& B,
                vector<vector<double>>& C, int n) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

void blockWise(const vector<vector<double>>& A,
               const vector<vector<double>>& B,
               vector<vector<double>>& C, int n, int blockSize) {
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int i = ii; i < min(ii + blockSize, n); i++) {
                for (int j = jj; j < min(jj + blockSize, n); j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < n; k++)
                        C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}

int main() {
    int n = 50, blockSize = 10;

  
    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (i + j) % 10;
            B[i][j] = (i * j + 1) % 10;
        }
    }

    double start, end;

    start = omp_get_wtime();
    rowWise(A, B, C, n);
    end = omp_get_wtime();
    double rowTime = end - start;

    start = omp_get_wtime();
    columnWise(A, B, C, n);
    end = omp_get_wtime();
    double columnTime = end - start;

    start = omp_get_wtime();
    blockWise(A, B, C, n, blockSize);
    end = omp_get_wtime();
    double blockTime = end - start;

    cout << fixed << setprecision(6);

    cout << "\nMatrix Size: " << n << " x " << n << endl;
    cout << "Block Size: " << blockSize << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;

    cout << "\nExecution Time:\n";
    cout << "Row-wise    : " << rowTime << " seconds\n";
    cout << "Column-wise : " << columnTime << " seconds\n";
    cout << "Block-wise  : " << blockTime << " seconds\n";

    return 0;
}