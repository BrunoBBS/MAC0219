#include "methods.hpp"

/******************************************************************
 * Function for the preparation threads. It will write values from
 * the matrix A into the special matrix B.
 ******************************************************************/
void *prep(void *args_p)
{
    prepper_args_t &args = *((prepper_args_t *)args_p);
    uint64_t b_rows = args.B->rows();
 
    for (uint64_t i = 0; i < b_rows; i++)
    {
        (*args.B)[i][args.column] = args.value_from_a;
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
    
    *args.place = sum;
    return NULL;
}

/*********************************************************************
 * This function prepares the transposed matrix B with spaces between
 * items by placing values from the lines of A in these free spaces.
 *********************************************************************/
void prep_b_lines(ifstream &file_A, mat &B, pthread_t *preppers, prepper_args_t *args, double *line_A)
{
    static uint64_t A_line = 0;
    uint64_t num_threads   = (B.cols() / 2);
    loadRow(file_A, B.cols() / 2, A_line, line_A);
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
}

/*********************************************************************
 * This function is wrapper that do some preparation and calls the
 * other preparatory and reduction functions.
 *********************************************************************/
void generate_next_C_line(mat &B, ifstream &file_A, double *line_C)
{
    uint64_t p = B.cols() / 2;
    uint64_t n = B.rows();

    double *line_A = new double[p];

    pthread_t *preppers    = new pthread_t[p];
    prepper_args_t *prep_args   = new prepper_args_t[p];

    pthread_t *workers = new pthread_t[n];
    worker_args_t *work_args   = new worker_args_t[n];

    prep_b_lines(file_A, B, preppers, prep_args, line_A);

    for (uint64_t i = 0; i < B.rows(); i++)
    {
        work_args[i] = worker_args_t{i, B[i], B.cols(), &line_C[i]};
        pthread_create(&workers[i], NULL, &reduce, (void *)&work_args[i]);
    }

    void **ret;
    for (uint64_t i = 0; i < B.rows(); i++)
    {
        pthread_join(workers[i], (void **)&ret);
    }

    delete line_A;

    delete preppers;
    delete prep_args;
    
    delete workers;
    delete work_args;
}

void run_pthreads(std::ifstream &file_A, mat &B, mat &C)
{
    // For each line of C
    for (uint64_t row = 0; row < C.rows(); row++)
    {
        generate_next_C_line(B, file_A, C[row]);
    }
}