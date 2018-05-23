#ifndef __TYPEDEF__HPP__

using namespace std;

// Class representing a matrix
class Matrix {
public:
    Matrix(uint64_t rows, uint64_t cols);

    uint64_t rows() const;
    uint64_t cols() const;

    double*& operator[] (uint64_t row);

private:
    double **matrix;
};

typedef Matrix mat;

#endif
