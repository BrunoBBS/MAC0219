#include "typedef.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>

uint64_t cols = 50000;
uint64_t rows = 50000;

int main()
{
    std::ofstream A_file;
    std::ofstream B_file;
    A_file.open("A");
    B_file.open("B");
    mat A(rows, cols);
    mat B(rows, cols);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 100.0);

#pragma omp prallel for
    for (uint64_t row = 0; row < rows; row++)
    {
#pragma omp prallel for
        for (uint64_t col = 0; col < cols; col++)
        {
            A[row][col] = distribution(generator);
            B[row][col] = distribution(generator);
        }
    }

    A_file << A.rows() << " " << A.cols() << std::endl;
    B_file << B.rows() << " " << B.cols() << std::endl;

    for (uint64_t row = 0; row < rows; row++)
        for (uint64_t col = 0; col < cols; col++)
        {
            A_file << row + 1 << " " << col + 1 << " " << A[row][col]
                   << std::endl;
            B_file << row + 1 << " " << col + 1 << " " << B[row][col]
                   << std::endl;
        }
    A_file.close();
    B_file.close();
}