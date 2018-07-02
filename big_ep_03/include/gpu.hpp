#ifndef __GPU_HPP__
#define __GPU_HPP__

#include <vector>
#include "util.hpp"

double gpu_probing(uint64_t n_ops, int64_t M, int64_t k);

std::vector<double> gpu_integration(uint64_t n_ops, int64_t M,
                                             int64_t k);
#endif