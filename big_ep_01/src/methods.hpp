#ifndef __METHODS_HPP__
#define __METHODS_HPP__

#include <fstream>

void generate_c_pthreads();
void generate_c_openmp(std::ifstream &file_A, mat &M, mat &C, uint64_t p);

#endif