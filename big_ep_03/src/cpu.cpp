#include "cpu.hpp"
#include "util.hpp"
#include <chrono>
#include <cmath>
#include <omp.h>
#include <random>

using namespace std::chrono;

double cpu_probing(uint64_t n_ops, int64_t M, int64_t k)
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    cpu_integration(n_ops, M, k);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> dur = duration_cast<duration<double>>(t2 - t1);
    double flops         = n_ops / dur.count();
    return flops;
}

void cpu_calc(std::mt19937_64 &gen,
              std::uniform_real_distribution<double> &dist, double &sum,
              double &sum_2, uint64_t n_ops, int64_t M, int64_t k)
{
#pragma omp parallel for
    for (uint64_t i = 0; i < n_ops; i++)
    {
        double x   = dist(gen);
        double res = cpu_f(M, k, x);
        sum += res;
        sum_2 += res * res;
    }
}

std::vector<double> cpu_integration(uint64_t n_ops, int64_t M, int64_t k)
{
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> dist(1e-320, 0.5 + 1e-320);
    double sum = 0, sum_2 = 0;
    cpu_calc(gen, dist, sum, sum_2, n_ops, M, k);
    std::vector<double> result(2);
    result[0] = 0 /*Put integral formula here*/;
    result[1] = 0 /*Put integral formula here*/;
    return result;
}