#include "util.hpp"
#include "cpu.hpp"

#include <iostream>
#include <string>

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

    // TODO: CODE HERE
    std::cout << cpu_flops(1000000000) << std::endl;


    return 0;
}
