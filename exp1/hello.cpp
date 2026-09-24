#include <iostream>
#include <omp.h>
using namespace std;


int main(){
    #pragma omp parallel
    {
        cout<<"hello "
        <<omp_get_thread_num()
        <<endl;
    }
    return 0;
}