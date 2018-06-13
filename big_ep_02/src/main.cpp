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
 * @param in_file This informs qhich file to load from.
 * @param num_mat This informs the nummber of matrices the file contains.
 * @param matrix Mtartix to fill.
 * @return a struct matrices.
 */
void load_matrices(ifstream &in_file, int num_mat, mat_t &matrix)
{
    for (int mat = 0; mat < num_mat; mat++)
    {
        // ignores line of *
        in_file.ignore();
        // Reads actual numbers from lines
        in_file >> matrix[0][mat] >> matrix[1][mat] >> matrix[2][mat];
        cout << matrix[0][mat] << " " << matrix[1][mat];
        in_file >> matrix[4][mat] >> matrix[5][mat] >> matrix[6][mat];
        in_file >> matrix[7][mat] >> matrix[8][mat] >> matrix[8][mat];
    }
}

int main(int agrc, char *argv[])
{
    std::ifstream in_file;
    in_file.open(argv[1]);
    if (!in_file.is_open())
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
}