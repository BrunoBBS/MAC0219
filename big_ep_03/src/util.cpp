#include <iostream>

#include "util.hpp"

// Parses numbers from string
void read(uint64_t &v, std::string v_str, std::string name)
{
    try
    {
        v = std::stoull(v_str);
    }
    catch (std::invalid_argument e)
    {
        error(format("%s (%s) is not a valid uint64_t!",
            v_str.c_str(), name.c_str()));
    }
    catch (std::out_of_range e)
    {
        error(format("%s (%s) is too big for an uint64_t!",
            v_str.c_str(), name.c_str()));
    }
}

void read(int64_t &v, std::string v_str, std::string name)
{
    try
    {
        v = std::stoll(v_str);
    }
    catch (std::invalid_argument e)
    {
        error(format("%s (%s) is not a valid int64_t!",
            v_str.c_str(), name.c_str()));
    }
    catch (std::out_of_range e)
    {
        error(format("%s (%s) is too big for an int64_t!",
            v_str.c_str(), name.c_str()));
    }
}

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