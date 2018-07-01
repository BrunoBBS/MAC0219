#ifndef __CPU_HPP__
#define __CPU_HPP__

#include "util.hpp"

double cpu_probing(uint64_t n_ops, int64_t M, int64_t k);

inline double cpu_f(int64_t M, int64_t k, double x)
{
    return (sin((2 * M + 1) * M_PI * x) * cos(2 * M_PI * k * x))/sin(M_PI * x);
}

void cpu_calc(std::mt19937_64 &gen,
              std::uniform_real_distribution<double> &dist, double &sum,
              double &sum_2, uint64_t n_ops);

void cpu_calc_serial(std::mt19937_64 &gen,
              std::uniform_real_distribution<double> &dist, double &sum,
              double &sum_2, uint64_t n_ops);

std::vector<double> cpu_integration(uint64_t n_ops, int64_t M, int64_t k, char type);

#endif