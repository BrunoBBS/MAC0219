#include "methods.hpp"

/******************************************************************
 * Function for the preparation threads. It will write values from
 * the matrix A into the special matrix B.
 ******************************************************************/
void *prep(void *args_p)
{
    prepper_args_t &args = *((prepper_args_t *)args_p);
    for (uint64_t i = 0; i < args.B->rows(); i++)
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
    args.place = sum;
    return NULL;
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
    pthread_barrier_init(&barrier, NULL, B.rows());

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

void run_pthreads(std::ifstream &file_A, mat &B, mat &C, uint64_t p)
{
    // For each line of C
   for(uint64_t row = 0; row < C.rows(); row++)
       generate_next_C_line(B, file_A, p, C[row]);
}