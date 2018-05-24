#include "methods.hpp"

void run_openmp(std::ifstream &file_A, mat &M, mat &C, uint64_t p)
{
    // For each row of A
    double *row_A = new double[p];
    for (uint64_t i = 0; i < p; i++)
    {
        for (uint64_t j = 0; j < p; j++)
            row_A[j] = 0;
        // Loading phase
        loadRow(file_A, p, i, row_A);

        #pragma omp parallel for
        for (uint64_t v_index = 0; v_index < p; v_index++)
        {
            // My hope is that these values get loaded directly into the registers
            double   val  = row_A[v_index];
            uint64_t col  = v_index * 2 + 1;
            uint64_t rows = M.rows();

            for (uint64_t row = 0; row < rows; row++)
                M[row][col] = val;
        }

        // Reducing phase
        #pragma omp parallel for
        for (uint64_t v_index = 0; v_index < C.cols(); v_index++)
        {
            // My hope is (again) that these values are loaded directly into the registers
            double    sum = 0;
            double   *row = M[v_index];
            uint64_t  col = M.cols();
            
            for (uint64_t index = 0; index < col; index += 2)
                sum += row[index] * row[index + 1];

            C[i][v_index] = sum;
        }
    }
    delete row_A;
}