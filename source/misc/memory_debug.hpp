#ifndef MEMORY_DEBUG_HPP
#define MEMORY_DEBUG_HPP

#ifdef MEMORY_DEBUG

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace memory_debug {
inline void print_memory(const std::string& label) {
    double rss_gb = 0.0;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos)
                rss_gb = std::stod(line.substr(pos)) / 1048576.0;
            break;
        }
    }
    double max_rss = rss_gb, sum_rss = rss_gb;
    int rank = 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_rss, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_rss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (rank == 0) {
        std::cerr << "[MEMORY] " << label
                  << ": max_rank=" << std::fixed << std::setprecision(3) << max_rss
                  << " GB, sum_all=" << sum_rss << " GB" << std::endl;
    }
}
} // namespace memory_debug

#define MEMORY_DEBUG_PRINT(label) memory_debug::print_memory(label)

#else
#define MEMORY_DEBUG_PRINT(label) ((void)0)
#endif // MEMORY_DEBUG

#endif // MEMORY_DEBUG_HPP
