#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main()
{
    int n = 10;

    vector<vector<int>> A(n, vector<int>(n, 2));
    vector<vector<int>> B(n, vector<int>(n, 3));
    vector<vector<int>> C(n, vector<int>(n, 0));

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         for(int k=0;k<n;k++){
    //             C[i][j] += A[i][k] * B[k][j];
    //         }
    //     }
    // }

#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    #pragma omp parallel
    {
        #pragma omp for()

    }
}