#include "typedef.hpp"

Matrix::Matrix(uint64_t rows, uint64_t cols)
{
    dimensions.first  = rows;
    dimensions.second = cols;

    matrix = new double[rows * cols];
    
    for (uint64_t index = 0; index < rows * cols; index++)
        matrix[index] = 0;
}

Matrix::~Matrix()
{
    delete matrix;
}

void Matrix::print() const
{
    for (uint64_t i = 0; i < rows(); i++)
    {
        for (uint64_t j = 0; j < cols(); j++)
            std::cout << matrix[i * Matrix::cols() + j] << " ";
        std::cout << std::endl;
    }
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

double* Matrix::operator[] (uint64_t row)
{
    return &matrix[row * dimensions.second];
}