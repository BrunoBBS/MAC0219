#include <fstream>
#include <iostream>
#include <string>

#include <pthread.h>
#include <stdlib.h>

#include "typedef.hpp"
#include "util.hpp"
#include "methods.hpp"

using namespace std;

/**********************************************
 * Load B in the right format for computation *
 **********************************************/
void load_B(mat &M, std::ifstream &M_file)
{
    uint64_t row, col;
    double val;

    // Read remaining lines from file
    while (M_file >> row >> col >> val)
    {
        row--;
        col--;

        if (!M.valid(row, col))
            error(format("Invalid coordinates (%lld, %lld) in matrix B",
                         row + 1, col + 1));

        // Transpose and space M
        M[col][row * 2] = val;
    }
}

/******************************************************
 * Reads matrix dimensions from file and returns them *
 ******************************************************/
dim read_dimensions(std::ifstream &matrix_file)
{
    dim dimensions;
    if (!(matrix_file >> dimensions.first >> dimensions.second))
        error(format("Values couldn't be read! Maybe wrong format?"));

    return dimensions;
}

/******************************
 * Entry point of the program *
 ******************************/
int main(int argc, char **argv)
{
    // Print help if wrong number of arguments
    if (argc != 5)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << argv[0]
                  << " <implementation> <matrix A filename> <matrix B "
                     "filename> <matrix C filename>"
                  << std::endl;
        return 0;
    }

    /*****************
     * File loading  *
     *****************/
    std::ifstream A_file, B_file;
    std::ofstream C_file;

    // Read execution mode
    char exec_mode = argv[1][0];

    uint64_t m, p, n;

    // Open files
    A_file.open(argv[2]);
    B_file.open(argv[3]);
    C_file.open(argv[4]);

    if (!A_file.is_open())
        error(format("File '%s' couldn't be opened!", argv[2]));
    if (!B_file.is_open())
        error(format("File '%s' couldn't be opened!", argv[3]));
    if (!C_file.is_open())
        error(format("File '%s' couldn't be opened!", argv[4]));

    // Read matrix dimensions
    dim a_dimensions = read_dimensions(A_file);
    dim b_dimensions = read_dimensions(B_file);

    // If the dimensions are incompatible, send error
    if (a_dimensions.second != b_dimensions.first)
        error(format("Matrices of dimensions "
                     "%llux%llu and %llux%llu can't be multiplied",
                     a_dimensions.first, a_dimensions.second,
                     b_dimensions.first, a_dimensions.second));

    // Read values
    m = a_dimensions.first;
    p = b_dimensions.first;
    n = b_dimensions.second;

    // Allocate Matrices (A will be loaded on the fly)
    mat B(n, 2 * p);
    mat C(m, n);

    // Load B from file to M
    load_B(B, B_file);

    // Now the modified B is loaded, C is created. Now we just load the computed
    // values into C
    if (exec_mode == 'p')
        run_pthreads(A_file, B, C, p);
    else
        run_openmp(A_file, B, C, p);


    // Write result to output file

    // Size
    C_file << C.rows() << " " << C.cols() << std::endl;

    for (uint64_t row = 0; row < C.rows(); row++)
        for (uint64_t col = 0; col < C.cols(); col++)
            //if (C[row][col] != 0)
                C_file << row + 1 << " " << col + 1 << " " << C[row][col] << std::endl;

    // Close Matrix Files
    A_file.close();
    B_file.close();
    C_file.close();

    return 0;
}