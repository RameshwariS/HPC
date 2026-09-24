#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;

int main()
{
    int n = 5000000;
    vector<double> a(n), b(n);
    for (int i = 0; i < n; i++)
        a[i] = i % 100;

    double s = omp_get_wtime();

    for (int i = 0; i < n; i++)
        a[i] = a[i] / 100.0;
    for (int i = 0; i < n; i++)
        b[i] = a[i] * a[i];
        
    double sum = 0;
    for (double x : b)
        sum += x;
    double sequential = omp_get_wtime() - s;

    fill(a.begin(), a.end(), 0);
    fill(b.begin(), b.end(), 0);
    sum = 0;

    s = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task
            {
                for (int i = 0; i < n; i++)
                    a[i] = i % 100 / 100.0;
            }

#pragma omp task
            {
                for (int i = 0; i < n; i++)
                    b[i] = (i % 100) * (i % 100);
            }

#pragma omp taskwait

#pragma omp task
            {
                double local = 0;
                for (double x : b)
                    local += x;
                sum = local;
            }

#pragma omp taskwait
        }
    }

    double parallel = omp_get_wtime() - s;

    cout << "Result: " << sum << endl;
    cout << "Sequential Time: " << sequential << " seconds\n";
    cout << "Task Time: " << parallel << " seconds\n";
    cout << "Speedup: " << sequential / parallel << endl;
}