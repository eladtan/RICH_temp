#ifndef MC_RESULTS_DIR_HPP
#define MC_RESULTS_DIR_HPP

#include <filesystem>
#include <string>

#include "utils/arguments/EnvironmentParser.hpp"

// The directory a benchmark writes its results to: RICH_OUTPUT_DIR, which the run must set.
// There is no default, so results never go to a place nobody chose; submit/rich-submit
// reads the declaration below and asks for the variable.
inline std::string McResultsDirectory(const char *benchmarkName)
{
    EnvironmentParser environment(benchmarkName);
    environment.addRequired<std::string>("RICH_OUTPUT_DIR", "directory the benchmark writes its results to");
    environment.parse();
    return environment.get<std::string>("RICH_OUTPUT_DIR");
}

inline void EnsureDirectory(const std::string &dir, int rank)
{
    if(rank == 0 && !dir.empty())
    {
        std::filesystem::create_directories(dir);
    }
}

inline void EnsureParentDirectory(const std::string &path, int rank)
{
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if(rank == 0 && !parent.empty())
    {
        std::filesystem::create_directories(parent);
    }
}

#endif
