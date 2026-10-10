#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
using namespace std;

int main(int argc, char** argv) {
    int N = 500;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N <= 0) {
        cout << "N must be positive." << endl;
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

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double aik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double>(end - start).count();

    cout << fixed << setprecision(6);
    cout << "Matrix size = " << N << " x " << N << endl;
    cout << "Sequential time = " << time << " seconds" << endl;
    cout << "C[0][0] = " << C[0] << endl;

    return 0;
}
