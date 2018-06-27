#include "cpu.hpp"
#include <chrono>
#include <cmath>

using namespace std::chrono;

double cpu_flops(uint64_t n_ops)
{
    float a = M_PI;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

#pragma omp parallel for
    for (uint64_t i = 0; i < n_ops; i++)
        a = a * a;

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> dur = duration_cast<duration<double>>(t2 - t1);
    double flops = n_ops / dur.count();
    return flops;
}