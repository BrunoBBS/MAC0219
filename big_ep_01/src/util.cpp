#include <iostream>

#include "util.hpp"

// Returns printf-like formatted std::string
std::string format(std::string pattern, ...)
{
    va_list argptr;
    va_start(argptr, pattern);

    char *c_text;
    vasprintf(&c_text, pattern.c_str(), argptr);
    auto ret = std::string(c_text);
    free(c_text);
    return ret;
}

// Prints error message and exits
void error(std::string message, int retval)
{
    std::cerr << "Error: " << message << std::endl;
    exit(retval);
}

/********************************************************************
 * This function reads a marix line from a given file and returns it.
 ********************************************************************/
double *loadRow(std::ifstream &file_M, uint64_t line_len, uint64_t line_no,
                double *existing_row)
{
    static uint64_t i, j;
    static double val;
    static bool last = false;

    double *M_line = existing_row ? existing_row : new double[line_len];

    for (uint64_t i = 0; i < line_len; i++)
        M_line[i] = 0;

    if (last && line_no == i) M_line[j] = val;

    while (file_M >> i >> j >> val)
    {
        i--;
        j--;

        if (i < 0 || i >= line_len)
            error(format("Invalid coordinates (%lld, %lld) in matrix", i, j));

        if (i != line_no)
        {
            last = true;
            break;
        }

        M_line[j] = val;
    }
    return M_line;
}