#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <iostream>

using namespace std;

// Matrix dimensions type
typedef std::pair<uint64_t, uint64_t> dim;

// Class representing a matrix
class Matrix
{
  public:
    Matrix(uint64_t rows, uint64_t cols);
    ~Matrix();

    void print() const;

    uint64_t rows() const;
    uint64_t cols() const;

    bool valid(uint64_t row, uint64_t col) const;

    double *&operator[](uint64_t row);

  private:
    dim dimensions;
    double **matrix;
};

typedef Matrix mat;

/****************************
 * Pthreads argument structs*
 ****************************/
struct worker_args_t
{
    uint64_t line;
    double *row;
    uint64_t row_length;
    // place in a line of the C matrix where the worker shourld put the result
    double *place;
};

struct prepper_args_t
{
    prepper_args_t() {}
    prepper_args_t(mat *B, double value_from_a, uint64_t column)
        : B(B), value_from_a(value_from_a), column(column)
    {
    }
    mat *B;
    double value_from_a;
    uint64_t column;
};
#endif
