#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
using namespace std;

int main(int argc, char** argv) {
    int N = 1000;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N < 3) {
        cout << "N must be at least 3." << endl;
        return 0;
    }

    vector<double> input(N * N);
    vector<double> output(N * N, 0.0);

    double kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            input[i * N + j] = (i + j) % 256;
        }
    }

    auto start = chrono::high_resolution_clock::now();

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            double sum = 0.0;

            for (int u = -1; u <= 1; u++) {
                for (int v = -1; v <= 1; v++) {
                    sum += input[(i + u) * N + (j + v)] * kernel[u + 1][v + 1];
                }
            }

            output[i * N + j] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double>(end - start).count();

    cout << fixed << setprecision(6);
    cout << "Input size = " << N << " x " << N << endl;
    cout << "Sequential convolution time = " << time << " seconds" << endl;
    cout << "Output[1][1] = " << output[N + 1] << endl;

    return 0;
}
