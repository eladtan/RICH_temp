#ifndef VORONOI_SUBDOMAINS_HPP
#define VORONOI_SUBDOMAINS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

// The number of thread subdomains asked for with --voronoi-subdomains: a number (0 builds
// each rank as one domain), or "auto" for one per OpenMP thread of the rank, the threads
// MadVoro's Kokkos host execution space runs on.
inline std::size_t ParseVoronoiSubdomains(const std::string &value)
{
    if(value == "auto")
    {
#ifdef _OPENMP
        return static_cast<std::size_t>(omp_get_max_threads());
#else
        return 1;
#endif
    }
    std::size_t parsed = 0;
    const unsigned long count = std::stoul(value, &parsed);
    if(parsed != value.size())
        throw std::invalid_argument("--voronoi-subdomains expects a number or 'auto', not '" + value + "'");
    return static_cast<std::size_t>(count);
}

#endif // VORONOI_SUBDOMAINS_HPP
