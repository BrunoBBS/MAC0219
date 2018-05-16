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

#endif