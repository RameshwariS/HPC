#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main()
{
    int n = 1000;
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> Add(n, vector<int>(n));
    vector<vector<int>> Sub(n, vector<int>(n));
    vector<vector<int>> TA(n, vector<int>(n));
    vector<vector<int>> TB(n, vector<int>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    double s = omp_get_wtime();

#pragma omp parallel sections
    {
#pragma omp section
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Add[i][j] = A[i][j] + B[i][j];

#pragma omp section
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Sub[i][j] = A[i][j] - B[i][j];

#pragma omp section
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                TA[j][i] = A[i][j];

#pragma omp section
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                TB[j][i] = B[i][j];
    }

    double parallel = omp_get_wtime() - s;

    cout << "Matrix operations completed.\n";
    cout << "Sections Time: " << parallel << " seconds\n";
}
