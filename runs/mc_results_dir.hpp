#ifndef MC_RESULTS_DIR_HPP
#define MC_RESULTS_DIR_HPP

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>

inline std::string McResultsDirectory(const char *benchmarkName)
{
    if(const char *env = std::getenv("RICH_OUTPUT_DIR"))
    {
        if(env[0] != '\0')
        {
            return env;
        }
    }
    std::time_t now = std::time(nullptr);
    std::tm localTime{};
    localtime_r(&now, &localTime);
    char dateBuf[16];
    std::strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d", &localTime);
    return std::string("/data/shared/maorm/MC_results/") + benchmarkName + "/" + dateBuf;
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
