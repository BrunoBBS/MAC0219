/*! \file main.cpp
    \brief Main file of the program.
*/

#include "util.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef vector<vector<int32_t>> mat_t;

/*! Reads all matrices from file and stores them in the given matrix.
 * @param in_file This informs which file to load from.
 * @param num_mat This informs the number of matrices the file contains.
 * @param matrix Matrix to fill.
 * @return a struct matrices.
 */
void load_matrices(ifstream &in_file, int32_t num_mat, mat_t &matrix)
{
    std::string ast;
    for (int mat = 0; mat < num_mat; mat++)
    {
        in_file >> ast;
        // Reads actual numbers from lines
        in_file >> matrix[0][mat] >> matrix[1][mat] >> matrix[2][mat];
        in_file >> matrix[3][mat] >> matrix[4][mat] >> matrix[5][mat];
        in_file >> matrix[6][mat] >> matrix[7][mat] >> matrix[8][mat];
    }
}

int main(int agrc, char *argv[])
{
    std::ifstream in_file;
    in_file.open(argv[1]);
    if (!in_file)
        error(format("File '%s' couldn't be opened!", argv[1]));

    
    int32_t num_mat;
    in_file >> num_mat;

    
    mat_t mat(9);
    for (int i = 0; i < 9; i++)
        mat[i].resize(num_mat);

    load_matrices(in_file, num_mat, mat);
    
    for (auto item : mat)
    {
        cout << "[";
        for (auto number : item)
        {
            cout << number << " ";
        }
        cout << "]" << endl;
    }
    in_file.close();
}