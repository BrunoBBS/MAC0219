#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <string>
#include <fstream>

#include <stdarg.h>

// Returns printf-like formatted std::string
std::string format(std::string pattern, ...);

// Prints error message and exits
void error(std::string message, int retval = -1);

/********************************************************************
 * This function reads a marix line from a given file and returns it.
 ********************************************************************/
double *loadRow(std::ifstream &file_M, uint64_t line_len, uint64_t line_no, double *existing_row = nullptr);

#endif