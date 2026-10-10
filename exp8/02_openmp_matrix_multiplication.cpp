#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
using namespace std;

int main(int argc, char** argv) {
    int N = 500;
    int threads = omp_get_max_threads();

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (argc > 2) {
        threads = atoi(argv[2]);
    }

    if (N <= 0 || threads <= 0) {
        cout << "Invalid input." << endl;
        return 0;
    }

    vector<double> A(N * N);
    vector<double> B(N * N);
    vector<double> C(N * N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i + j) % 10 + 1;
            B[i * N + j] = (i * j) % 10 + 1;
        }
    }

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double aik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    double time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);
    cout << "Matrix size = " << N << " x " << N << endl;
    cout << "OpenMP threads = " << threads << endl;
    cout << "OpenMP time = " << time << " seconds" << endl;
    cout << "C[0][0] = " << C[0] << endl;

    return 0;
}
