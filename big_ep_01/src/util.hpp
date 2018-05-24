#ifndef __UTIL_HPP__
#include <string>
#include <iostream>

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
void error(std::string message, int retval = -1)
{
    std::cerr << "Error: " << message << std::endl;
    exit(retval);
}


/********************************************************************
 * This function reads a marix line from a given file and returns it.
 ********************************************************************/
double *loadRow(ifstream &file_M, uint64_t line_len, uint64_t line_no)
{
    static uint64_t i, j;
    static double val;
    static bool last = false;

    double *M_line = new double[line_len];

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

#endif