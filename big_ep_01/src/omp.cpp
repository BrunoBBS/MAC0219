#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const uint64_t NUMERO = 1e8;

double a[NUMERO];
double b[NUMERO];
double c[NUMERO];

int main()
{
    srand(time(0));
    for (int i = 0; i < NUMERO; i++)
    {
        a[i] = rand() / 1000.f;
        b[i] = rand() / 1000.f;
    }

#pragma omp parallel for
    for (int i = 0; i < NUMERO; i++)
    {
        c[i] = a[i] * 8 + b[i];
    }

    // #pragma omp parallel for num_threads(3)
    // for (int i = 0; i < NUMERO; i++)
    // {
    //     for (int j = 0; j < NUMERO; j++)
    //         printf("%d %d\n", i, j);
    // }
}