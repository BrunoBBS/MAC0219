#include "typedef.hpp"

Matrix::Matrix(uint64_t rows, uint64_t cols)
{
    dimensions.first  = rows;
    dimensions.second = cols;

    matrix = new double*[rows];
    for (uint64_t row = 0; row < rows; row++)
    {
        matrix[row] = new double[cols];
        for (uint64_t col = 0; col < cols; col++)
            matrix[row][col] = 0;

    }
}

Matrix::~Matrix()
{
    for (uint64_t row = 0; row < rows(); row++)
        delete matrix[row];
    delete matrix;
}

uint64_t Matrix::rows() const
{
    return dimensions.first;
}

uint64_t Matrix::cols() const
{
    return dimensions.second;
}

bool Matrix::valid(uint64_t rows, uint64_t cols) const
{
    return rows < Matrix::rows() && cols < Matrix::cols();
}

double*& Matrix::operator[] (uint64_t row)
{
    return matrix[row];
}