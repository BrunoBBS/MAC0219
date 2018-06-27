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
void load_matrices(ifstream &in_file, int itemcnt, std::vector<int32_t> &matrix)
{
    std::string ast;

    int32_t n = matrix.size() / 9;

    for (int mat = 0; mat < n; mat++)
    {
        if (!(in_file >> ast))
            error(format("Failed reading matrix %d!", mat));

        // Reads actual numbers from lines
        for (int k = 0; k < itemcnt; k++)
            if (!(in_file >> matrix[mat + k * n]))
                error(format("Failed reading matrix %d!", mat));
    }
}


int main(int argc, char *argv[])
{
    if (!(argc == 2 || argc == 4))
    {
        std::cout << "Usage: " << argv[0] << " <matrix definition file> [i] [j]" << std::endl;
        return 0;
    }

    int w = 3;
    int h = 3;

    // Read matrix dimensions if given
    if (argc == 4)
    {
        try {
            h = std::stoi(argv[2]);
            w = std::stoi(argv[3]);
        } catch (std::invalid_argument e) {
            error("Given matrix dimensions are invalid!");
        } catch (std::out_of_range e) {
            error("Given matrix dimensions are too big!");
        }
    }


    const int itemcnt = w * h;

    // Try opening file
    std::ifstream in_file;
    in_file.open(argv[1]);
    if (!in_file)
        error(format("File '%s' couldn't be opened!", argv[1]));

    // Read number of matrices
    int32_t num_mat;
    if (!(in_file >> num_mat))
        error(format("Failed to read number of matrices! Is '%s' a valid file?", argv[1]));

    // Allocate and load matrices from file
    std::vector<int32_t> mat(itemcnt * num_mat);

    load_matrices(in_file, itemcnt, mat);

    in_file.close();

    // Allocate array in GPU
    void *device_array;
    size_t data_size = sizeof(int32_t) * mat.size();
    cudaMalloc(&device_array, data_size);

    // Copy data to device
    cudaMemcpy(device_array, (const void *) mat.data(), data_size, cudaMemcpyHostToDevice);

    reduce(num_mat, device_array, itemcnt);

    // Retrieve values
    std::vector<int32_t> out(itemcnt);
    cudaMemcpy((void *) out.data(), device_array, sizeof(int32_t) * itemcnt, cudaMemcpyDeviceToHost);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++)
            std::cout << out[i * w + j] << ((j == w - 1) ? "" : " ");
        std::cout << std::endl;
    }
}