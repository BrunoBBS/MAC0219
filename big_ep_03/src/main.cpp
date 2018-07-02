#include "cpu.hpp"
#include "gpu.hpp"
#include "util.hpp"

#include <iostream>
#include <mpi.h>
#include <string>

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

    read(N, N_str, "N");
    read(k, k_str, "k");
    read(M, M_str, "M");

    // TODO: CODE HERE
    std::cout << cpu_probing(10000, M, k) << std::endl;
    std::cout << gpu_probing(10000, M, k) << std::endl;

    // TODO: Balance
    uint64_t gpu_ops = floor(N * 1.0);
    uint64_t cpu_ops = floor(N * 0.0);
    cpu_ops += (N % 2 > 0);

    printf("gpu: %lld cpu:%lld\n", gpu_ops, cpu_ops);
    // Calculation
    std::vector<double> gpu_sums = gpu_integration(gpu_ops, M, k);
    std::vector<double> cpu_sums = cpu_integration(cpu_ops, M, k, 'm');

    // Calculation of mean and standard deviation
    double mean, std_dev, mean_sq;
    mean    = (gpu_sums[0] + cpu_sums[0]) / N;
    mean_sq = (gpu_sums[1] + cpu_sums[1]) / N;
    std_dev = sqrt((mean_sq - mean * mean) / N);

    printf("sum : %lf, mean: %lf\n", cpu_sums[0] + gpu_sums[0], mean);
    double aprx_sum, aprx_sub;
    aprx_sum = 2 * 0.5 * (mean + std_dev);
    aprx_sub = 2 * 0.5 * (mean - std_dev);

    printf("Approximate integral value with sum = %lf with sub = %lf \n",
           aprx_sum, aprx_sub);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
