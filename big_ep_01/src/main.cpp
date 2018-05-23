#include <fstream>
#include <iostream>
#include <string>

#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>

#include "typedef.hpp"
#include "util.hpp"
#include "methods.hpp"

using namespace std;

/******************************************************************
 * Function for the preparation threads. It will write values from
 * the matrix A into the special matrix B.
 ******************************************************************/
void *prep(void *args_p)
{
    prepper_args_t &args = *((prepper_args_t *)args_p);
    for (uint64_t i = 0; i < args.B->rows(); i++)
    {
        *(args.B)[i][args.column] = args.value_from_a;
    }
    return NULL;
}

/******************************************************************
 * Function for the reduce threads. It will multiply and sum the
 * values from the lines of the previously prepared matrix B to
 * put as values in lines of the matrix C.
 ******************************************************************/
void *reduce(void *args_p)
{
    double sum          = 0;
    worker_args_t &args = *((worker_args_t *)args_p);
    for (uint64_t i = 0; i < args.row_length; i += 2)
    {
        sum += args.row[i] * args.row[i + 1];
    }
    args.place = sum;
    return NULL;
}

/********************************************************************
 * This function reads a marix line from a given file and returns it.
 ********************************************************************/
double *loadRow(ifstream &file_M, uint64_t M_size, uint64_t line_no)
{
    static uint64_t i, j;
    static double val;
    static bool last = false;

    double *M_line = new double[M_size];

    if (last && line_no == i) M_line[j] = val;

    while (file_M >> i >> j >> val)
    {
        i--;
        j--;

        if (i < 0 || i >= M_size)
            error(format("Invalid coordinates (%lld, %lld) in matrix", i, j));

        if (i != line_no)
        {
            last = true;
            break;
        }

        M_line[j] = val;
    }
    return M_line;
}

/*********************************************************************
 * This function prepares the transposed matrix B with spaces between
 * items by placing values from the lines of A in these free spaces.
 *********************************************************************/
void prep_b_lines(ifstream &file_A, mat &B)
{
    static uint64_t A_line = 0;
    uint64_t num_threads   = (B.cols() / 2);
    double *line_A         = loadRow(file_A, B.cols() / 2, A_line);
    pthread_t *preppers    = new pthread_t[num_threads];
    prepper_args_t *args   = new prepper_args_t[num_threads];
    A_line++;
    for (uint64_t i = 0; i < num_threads; i++)
    {
        args[i] = prepper_args_t(&B, line_A[i], i * 2 + 1);
        pthread_create(&preppers[i], NULL, &prep, (void *)&args[i]);
    }
    void **ret;
    for (uint64_t i = 0; i < num_threads; i++)
    {
        pthread_join(preppers[i], (void **)&ret);
    }
    delete line_A;
    delete preppers;
    delete args;
}

/*********************************************************************
 * This function is wrapper that do some preparation and calls the
 * other preparatory and reduction functions.
 *********************************************************************/
void generate_next_C_line(mat &B, ifstream &file_A, uint64_t n_lines_a,
                          double *line_C)
{
    pthread_t *workers = new pthread_t[B.rows()];

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, B->size());

    prep_b_lines(file_A, B);

    for (uint64_t i = 0; i < B.rows(); i++)
    {
        worker_args_t args{i, B[i], B.cols(), line_C[i], barrier};
        pthread_create(&workers[i], NULL, &reduce, (void *)&args);
    }
    void **ret;
    for (uint64_t i = 0; i < B.rows(); i++)
    {
        pthread_join(workers[i], (void **)&ret);
    }
    delete workers;
}

/**********************************************
 * Load B in the right format for computation *
 **********************************************/
void load_B(mat &M, std::ifstream &M_file)
{
    uint64_t row, col;
    double val;

    // Read remaining lines from file
    while (M_file >> row >> col >> val)
    {
        row--;
        col--;

        if (M.valid(row, col))
            error(format("Invalid coordinates (%lld, %lld) in matrix B",
                         row + 1, col + 1));

        // Transpose and space M
        M[col][row * 2] = val;
    }
}

/******************************************************
 * Reads matrix dimensions from file and returns them *
 ******************************************************/
dim read_dimensions(std::ifstream &matrix_file)
{
    dim dimensions;
    if (!(matrix_file >> dimensions.first >> dimensions.second))
        error(format("Values couldn't be read! Maybe wrong format?"));

    return dimensions;
}

/******************************
 * Entry point of the program *
 ******************************/
int main(int argc, char **argv)
{
    // Print help if wrong number of arguments
    if (argc != 5)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << argv[0]
                  << " <implementation> <matrix A filename> <matrix B "
                     "filename> <matrix C filename>"
                  << std::endl;
        return 0;
    }

    /*****************
     * File loading  *
     *****************/
    std::ifstream A_file, B_file;
    std::ofstream C_file;

    uint64_t m, p, n;

    // Open files
    A_file.open(argv[2]);
    B_file.open(argv[3]);
    C_file.open(argv[4]);

    if (!A_file.is_open())
        error(format("File '%s' couldn't be opened!", argv[2]));
    if (!B_file.is_open())
        error(format("File '%s' couldn't be opened!", argv[3]));
    if (!C_file.is_open())
        error(format("File '%s' couldn't be opened!", argv[4]));

    // Read matrix dimensions
    dim a_dimensions = read_dimensions(A_file);
    dim b_dimensions = read_dimensions(B_file);

    // If the dimensions are incompatible, send error
    if (a_dimensions.second != b_dimensions.first)
        error(format("Matrices of dimensions "
                     "%llux%llu and %llux%llu can't be multiplied",
                     a_dimensions.first, a_dimensions.second,
                     b_dimensions.first, a_dimensions.second));

    // Read values
    m = a_dimensions.first;
    p = b_dimensions.first;
    n = b_dimensions.second;

    // Allocate Matrices (A will be loaded on the fly)
    mat M(n, 2 * p);
    mat C(m, n);

    // Load B from file to M
    load_B(M, B_file);

    // Now the modified B is loaded, C is created. Now we just load the computed
    // values into C
    // TODO: DO THINGS

    // Close Matrix Files
    A_file.close();
    B_file.close();
    C_file.close();

    return 0;
}