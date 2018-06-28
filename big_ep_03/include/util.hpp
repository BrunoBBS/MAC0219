#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <string>
#include <fstream>

#include <stdarg.h>
#include <cmath>
#include <random>

// Converts strings to integers
void read(uint64_t &v, std::string v_str, std::string name);
void read(int64_t  &v, std::string v_str, std::string name);

// Returns printf-like formatted std::string
std::string format(std::string pattern, ...);

// Prints error message and exits
void error(std::string message, int retval = -1);

#endif