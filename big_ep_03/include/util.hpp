#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <string>
#include <fstream>

#include <stdarg.h>
#include <math.h>


inline double f(uint64_t M, uint64_t k, double x)
{
    return (sin((2 * M + 1) * M_PI * x) * cos(2 * M_PI * k * x))/sin(M_PI * x);
}

// Returns printf-like formatted std::string
std::string format(std::string pattern, ...);

// Prints error message and exits
void error(std::string message, int retval = -1);

#endif