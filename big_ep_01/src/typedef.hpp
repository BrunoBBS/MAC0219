#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <iostream>

using namespace std;

// Matrix dimensions type
typedef std::pair<uint64_t, uint64_t> dim;

// Class representing a matrix
class Matrix {
public:
    Matrix(uint64_t rows, uint64_t cols);
    ~Matrix();

    inline uint64_t rows() const;
    inline uint64_t cols() const;

    inline bool valid(uint64_t row, uint64_t col) const;

    double*& operator[] (uint64_t row);

private:
    dim dimensions;
    double **matrix;
};

typedef Matrix mat;

#endif
