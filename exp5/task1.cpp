#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main()
{
    int n = 10000000;
    vector<int> a(n);
    for (int i = 0; i < n; i++)
        a[i] = i % 100;

    long long sum1 = 0, sum2 = 0;
    int max1 = a[0], min1 = a[0];
    double avg1 = 0;

    double s = omp_get_wtime();
    for (int x : a)
        sum1 += x;
    for (int x : a)
        max1 = max(max1, x);
    for (int x : a)
        min1 = min(min1, x);
    avg1 = (double)sum1 / n;
    double sequential = omp_get_wtime() - s;

    long long sum2a = 0;
    int max2 = a[0], min2 = a[0];
    double avg2 = 0;

    s = omp_get_wtime();

#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int x : a)
                sum2a += x;
        }

#pragma omp section
        {
            for (int x : a)
                max2 = max(max2, x);
        }

#pragma omp section
        {
            for (int x : a)
                min2 = min(min2, x);
        }

#pragma omp section
        {
            long long local = 0;
            for (int x : a)
                local += x;
            avg2 = (double)local / n;
        }
    }

    double sections = omp_get_wtime() - s;

    sum2 = sum2a;

    cout << "Sequential Results: " << sum1 << " " << max1 << " " << min1 << " " << avg1 << endl;
    cout << "Sections Results: " << sum2 << " " << max2 << " " << min2 << " " << avg2 << endl;
    cout << "Sequential Time: " << sequential << " seconds\n";
    cout << "Sections Time: " << sections << " seconds\n";
    cout << "Speedup: " << sequential / sections << endl;
}