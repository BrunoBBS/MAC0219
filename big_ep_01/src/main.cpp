#include <iostream>
#include <fstream>
#include <string>

#include <stdlib.h>
#include <stdarg.h>

#include "util.hpp"

// Read matrix from file [ONLY] after reading matrix dimensions
void loadMatrix(double **M, uint64_t l, uint64_t c, std::ifstream &M_file, std::string M_name = "[?]")
{
    for (uint64_t i = 0; i < l; i++)
        for (uint64_t j = 0; j < c; j++)
            M[i][j] = 0;

    uint64_t i, j;
    double val;

    while (M_file >> i >> j >> val)
    {
        i--;
        j--;

        if (i < 0 || i >= l || j < 0 || j >= c)
            error(format("Invalid coordinates (%lld, %lld) in matrix %s", i, j, M_name.c_str()));
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << argv[0] << " <implementation> <matrix A filename> <matrix B filename> <matrix C filename>" << std::endl;
        return 0;
    }

    std::ifstream A_file, B_file;
    std::ofstream C_file;

    uint64_t m, p, n;
    
    // Open files
    A_file.open(argv[2]);
    B_file.open(argv[3]);
    C_file.open(argv[4]);

    if (!A_file.is_open()) error(format("File '%s' couldn't be opened!", argv[2]));
    if (!B_file.is_open()) error(format("File '%s' couldn't be opened!", argv[3]));
    if (!C_file.is_open()) error(format("File '%s' couldn't be opened!", argv[4]));

    // Read matrix dimensions
    try
    {
        // Temporary variables for matrix dimensions
        uint64_t tmp_m, tmp_pa, tmp_pb, tmp_n;

        // Try reading values from files
        if (A_file >> tmp_m >> tmp_pa && B_file >> tmp_pb >> tmp_n)
        {
            // Check if they can be multiplied
            if (tmp_pa != tmp_pb) throw std::string("Can't multiply! Incompatible sizes!");

            m = tmp_m;
            p = tmp_pa;
            n = tmp_n;
        } else throw std::string("Values couldn't be read! Maybe wrong format?");
    }
    catch (std::string e) { error(e); }

    // Allocate Matrices
    double **A, **B;
    double **C;

    A = new double*[m];
    for (uint64_t i = 0; i < m; i++) A[i] = new double[p];

    B = new double*[p];
    for (uint64_t i = 0; i < p; i++) B[i] = new double[n];
    
    C = new double*[m];
    for (uint64_t i = 0; i < m; i++) C[i] = new double[n];

    // Load matrices from file
    loadMatrix(A, m, p, A_file, "A");
    loadMatrix(B, p, n, B_file, "B");

    // TODO: Do things here

    // Deallocate Matrices
    for (uint64_t i = 0; i < m; i++) delete A[i];
    delete A;

    for (uint64_t i = 0; i < p; i++) delete B[i];
    delete B;

    for (uint64_t i = 0; i < m; i++) delete C[i];
    delete C;

    // Close Matrix Files
    A_file.close();
    B_file.close();
    C_file.close();
}