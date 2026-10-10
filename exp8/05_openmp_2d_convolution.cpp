#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
using namespace std;

int main(int argc, char** argv) {
    int N = 1000;
    int threads = omp_get_max_threads();
    string scheduleType = "static";


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

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    if (scheduleType == "dynamic") {
        #pragma omp parallel for schedule(dynamic)
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
    } else if (scheduleType == "guided") {
        #pragma omp parallel for schedule(guided)
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
    } else {
        #pragma omp parallel for schedule(static)
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
    }

    double time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);
    cout << "Input size = " << N << " x " << N << endl;
    cout << "OpenMP threads = " << threads << endl;
    cout << "Schedule = " << scheduleType << endl;
    cout << "OpenMP convolution time = " << time << " seconds" << endl;
    cout << "Output[1][1] = " << output[N + 1] << endl;

    return 0;
}
