/*! \file main.cpp
    \brief Main file of the program.
*/

#include "util.hpp"
#include "kernel.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

/*! Reads all matrices from file and stores them in the given matrix.
 * @param in_file This informs which file to load from.
 * @param num_mat This informs the number of matrices the file contains.
 * @param matrix Matrix to fill.
 * @return a struct matrices.
 */
void load_matrices(ifstream &in_file, std::vector<int32_t> &matrix)
{
    std::string ast;

    int32_t n = matrix.size() / 9;

    for (int mat = 0; mat < n; mat++)
    {
        in_file >> ast;

        // Reads actual numbers from lines
        in_file >> matrix[mat + 0 * n] >> matrix[mat + 1 * n] >> matrix[mat + 2 * n];
        in_file >> matrix[mat + 3 * n] >> matrix[mat + 4 * n] >> matrix[mat + 5 * n];
        in_file >> matrix[mat + 6 * n] >> matrix[mat + 7 * n] >> matrix[mat + 8 * n];
    }
}

int main(int argc, char *argv[])
{
    std::ifstream in_file;
    in_file.open(argv[1]);
    if (!in_file)
        error(format("File '%s' couldn't be opened!", argv[1]));

    int32_t num_mat;
    in_file >> num_mat;

    std::vector<int32_t> mat(9 * num_mat);

    load_matrices(in_file, mat);

    // Allocate array in GPU
    void *device_array;
    size_t data_size = sizeof(int32_t) * mat.size();
    cudaMalloc(&device_array, data_size);

    // Copy data to device
    cudaMemcpy(device_array, (const void *) mat.data(), data_size, cudaMemcpyHostToDevice);

    reduce(num_mat, device_array);

    in_file.close();
}