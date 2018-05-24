#include "typedef.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>

uint64_t cols = 5000;
uint64_t rows = 5000;

int main()
{
    std::ofstream A_file;
    std::ofstream B_file;
    A_file.open("A");
    B_file.open("B");

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 100);

    A_file << rows << " " << cols << std::endl;
    B_file << rows << " " << cols << std::endl;

    for (uint64_t row = 0; row < rows; row++)
    {
        for (uint64_t col = 0; col < cols; col++)
        {
            A_file << row + 1 << " " << col + 1 << " " << distribution(generator) << std::endl;
            B_file << row + 1 << " " << col + 1 << " " << distribution(generator) << std::endl;
        }
    }

    A_file.close();
    B_file.close();
}
