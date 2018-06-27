#include "util.hpp"

#include <functional>
#include <iostream>
#include <string>

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

// Main functions of the program
int main(int argc, char *argv[])
{
    // Argument parsing
    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <N> <k> <M>" << std::endl;
        return 0;
    }

    std::string N_str = argv[1];
    std::string k_str = argv[2];
    std::string M_str = argv[3];

    uint64_t N;
    int64_t k, M;

    read(N, N_str, "N");
    read(k, k_str, "k");
    read(M, M_str, "M");

    // CODE HERE

    return 0;
}
