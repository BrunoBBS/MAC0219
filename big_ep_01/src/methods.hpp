#ifndef __METHODS_HPP__
#define __METHODS_HPP__

#include "typedef.hpp"
#include "util.hpp"
#include <fstream>

void run_pthreads();
void run_openmp(std::ifstream &file_A, mat &M, mat &C, uint64_t p);

#endif