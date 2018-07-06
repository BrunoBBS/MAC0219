#include "cpu.hpp"
#include "gpu.hpp"
#include "util.hpp"

#include <chrono>
#include <iostream>
#include <mpi.h>
#include <string>
#include <thread>

using namespace std::chrono;
// Main functions of the program
int main(int argc, char *argv[])
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Argument parsing
    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <N> <k> <M>" << std::endl;
        return 0;
    }

    std::string N_str = argv[1];
    std::string k_str = argv[2];
    std::string M_str = argv[3];

    uint64_t N;
    int64_t k, M;
    int int_res = 0;

    read(N, N_str, "N");
    read(k, k_str, "k");
    read(M, M_str, "M");

    if (abs(k) <= abs(M) && M >= 0)
        int_res = 1;
    else if (abs(k) <= abs(M) && M < 0)
        int_res = -1;

    uint64_t gpu_ops, cpu_ops;

    int cpu_ep_ops, gpu_ep_ops;
    if (world_rank == 1)
    {
        cpu_ep_ops = cpu_probing(100000, M, k);

        // receive how many operations per second the gpu can do
        MPI_Recv(&gpu_ep_ops, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        int sum_ep_ops = cpu_ep_ops + gpu_ep_ops;
        double alpha   = 1.1;

        gpu_ops = floor(alpha * N * gpu_ep_ops / sum_ep_ops);
        gpu_ops = (gpu_ops > N) ? N : gpu_ops;
        cpu_ops = N - gpu_ops;

        MPI_Send(&gpu_ops, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    if (world_rank == 0)
    {
        int gpu_ep_ops_loc = gpu_probing(1000000, M, k);
        MPI_Send(&gpu_ep_ops_loc, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }

    double mean, std_dev, mean_sq;
    double aprx_sum, aprx_sub, err_sum, err_sub;

    /*********************************
     * Balanced
     **********************************/

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Calculation

    double *gpu_sums;
    std::vector<double> cpu_sums;

    if (world_rank == 1)
    {
        cpu_sums = cpu_integration(cpu_ops, M, k, 'm');
        gpu_sums = (double *)malloc(2 * sizeof(double));
        MPI_Recv(gpu_sums, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    if (world_rank == 0)
    {
        MPI_Recv(&gpu_ops, 1, MPI_UNSIGNED_LONG_LONG, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        std::vector<double> gpu_sums_vec = gpu_integration(gpu_ops, M, k);

        MPI_Send(gpu_sums_vec.data(), 2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 1)
    {
        // Calculation of mean and standard deviation
        mean    = (gpu_sums[0] + cpu_sums[0]) / N;
        mean_sq = (gpu_sums[1] + cpu_sums[1]) / N;
        std_dev = sqrt((mean_sq - mean * mean) / N);

        aprx_sum = 2 * 0.5 * (mean + std_dev);
        aprx_sub = 2 * 0.5 * (mean - std_dev);
        double err_sum  = int_res - aprx_sum;
        double err_sub  = int_res - aprx_sub;

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> dur = duration_cast<duration<double>>(t2 - t1);

        printf("Tempo com balanceamento de carga em segundos: %lf \n"
               "Erro no calculo com a soma: %lf \n"
               "Erro no calculo com a subtracao: %lf \n\n",
               dur, err_sum, err_sub);
    }
    /*********************************
     * Full GPU
     **********************************/
    // Calculation
    if (world_rank == 0)
    {
        t1                           = high_resolution_clock::now();
        std::vector<double> gpu_sums = gpu_integration(N, M, k);

        // Calculation of mean and standard deviation
        mean    = (gpu_sums[0]) / N;
        mean_sq = (gpu_sums[1]) / N;
        std_dev = sqrt((mean_sq - mean * mean) / N);

        aprx_sum = 2 * 0.5 * (mean + std_dev);
        aprx_sub = 2 * 0.5 * (mean - std_dev);
        double err_sum  = int_res - aprx_sum;
        double err_sub  = int_res - aprx_sub;

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> dur = duration_cast<duration<double>>(t2 - t1);

        printf("Tempo na GPU com uma thread na CPU em segundos: %lf \n"
               "Erro no calculo com a soma: %lf \n"
               "Erro no calculo com a subtracao: %lf \n\n",
               dur, err_sum, err_sub);
    }
    /*********************************
     * CPU Multithreaded
     **********************************/
    // Calculation
    if (world_rank == 1)
    {
        t1       = high_resolution_clock::now();
        cpu_sums = cpu_integration(N, M, k, 'm');

        // Calculation of mean and standard deviation
        mean    = (cpu_sums[0]) / N;
        mean_sq = (cpu_sums[1]) / N;
        std_dev = sqrt((mean_sq - mean * mean) / N);

        aprx_sum = 2 * 0.5 * (mean + std_dev);
        aprx_sub = 2 * 0.5 * (mean - std_dev);
        double err_sum  = int_res - aprx_sum;
        double err_sub  = int_res - aprx_sub;

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> dur = duration_cast<duration<double>>(t2 - t1);

        printf("Tempo na CPU com %d threads em segundos: %lf \n"
               "Erro no calculo com a soma: %lf \n"
               "Erro no calculo com a subtracao: %lf \n\n",
               std::thread::hardware_concurrency() / 2, dur, err_sum,
               err_sub);
    }
    /*********************************
     * CPU Singlethreaded (Sequential)
     **********************************/
    // Calculation
    if (world_rank == 1)
    {
        t1       = high_resolution_clock::now();
        cpu_sums = cpu_integration(N, M, k, 's');

        // Calculation of mean and standard deviation
        mean    = (cpu_sums[0]) / N;
        mean_sq = (cpu_sums[1]) / N;
        std_dev = sqrt((mean_sq - mean * mean) / N);

        aprx_sum = 2 * 0.5 * (mean + std_dev);
        aprx_sub = 2 * 0.5 * (mean - std_dev);
        double err_sum  = int_res - aprx_sum;
        double err_sub  = int_res - aprx_sub;

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> dur = duration_cast<duration<double>>(t2 - t1);

        printf("Tempo sequencial em segundos: %lf \n"
               "Erro no calculo com a soma: %lf \n"
               "Erro no calculo com a subtracao: %lf \n\n",
               dur, err_sum, err_sub);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
