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
