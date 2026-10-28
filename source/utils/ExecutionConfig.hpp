#ifndef EXECUTION_CONFIG_HPP
#define EXECUTION_CONFIG_HPP

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// How a run asked each parallel feature to execute. The submission tool (submit/)
// writes one line per feature to a file and names it in RICH_EXECUTION_CONFIG:
//
//   # feature backend threads gpus
//   storm.transport gpu 0 1
//   madvoro.build.subdomains openmp 7 0
//
// Feature ids are declared in submit/features/*.json; `submit/rich-submit features --check`
// lists the ids code reads here and the ids no code reads yet. Typical use:
//
//   #include "utils/ExecutionConfig.hpp"
//   slopes.RunOnDevice(rich::execution::OnDevice("rich.hydro.slopes"));
//   const int threads = rich::execution::Threads("madvoro.build.subdomains", HostThreads());
//
// Without RICH_EXECUTION_CONFIG (a run not started by the tool), or for a feature the
// file does not name, every query returns its fallback, so code keeps its own default.
// A named file that cannot be read or parsed throws: a run must not silently ignore
// the execution it was submitted with.
namespace rich
{
namespace execution
{
enum class Backend
{
    Serial,
    OpenMP,
    GPU
};

struct Execution
{
    Backend backend = Backend::Serial;
    int threads = 1; // host threads (OpenMP); 1 for Serial, 0 for GPU
    int gpus = 0;    // GPUs per rank (GPU)
};

inline Backend ParseBackend(const std::string &name)
{
    if(name == "serial")
    {
        return Backend::Serial;
    }
    if(name == "openmp")
    {
        return Backend::OpenMP;
    }
    if(name == "gpu")
    {
        return Backend::GPU;
    }
    throw std::invalid_argument("unknown execution backend '" + name + "' (expected serial, openmp or gpu)");
}

// Parses the config text; `source` names it in error messages.
inline std::unordered_map<std::string, Execution> ParseConfig(std::istream &in, const std::string &source)
{
    std::unordered_map<std::string, Execution> features;
    std::string line;
    for(std::size_t number = 1; std::getline(in, line); ++number)
    {
        const std::size_t comment = line.find('#');
        std::istringstream fields(line.substr(0, comment));
        std::string feature, backend;
        if(!(fields >> feature))
        {
            continue;
        }
        Execution execution;
        std::string extra;
        if(!(fields >> backend >> execution.threads >> execution.gpus) || (fields >> extra))
        {
            throw std::runtime_error(source + ":" + std::to_string(number) +
                                     ": expected 'feature backend threads gpus', got '" + line + "'");
        }
        try
        {
            execution.backend = ParseBackend(backend);
        }
        catch(const std::invalid_argument &error)
        {
            throw std::runtime_error(source + ":" + std::to_string(number) + ": " + error.what());
        }
        features[feature] = execution;
    }
    return features;
}

// The features of RICH_EXECUTION_CONFIG, read once. A function-local static of an
// inline function is one object for the whole program, and its initialization is
// thread safe.
inline const std::unordered_map<std::string, Execution> &Features()
{
    static const std::unordered_map<std::string, Execution> features = []
    {
        const char *path = std::getenv("RICH_EXECUTION_CONFIG");
        if(path == nullptr || *path == '\0')
        {
            return std::unordered_map<std::string, Execution>();
        }
        std::ifstream file(path);
        if(!file)
        {
            throw std::runtime_error(std::string("cannot read RICH_EXECUTION_CONFIG file ") + path);
        }
        return ParseConfig(file, path);
    }();
    return features;
}

// True when the run named the feature.
inline bool Configured(const std::string &feature)
{
    return Features().count(feature) != 0;
}

inline Execution Get(const std::string &feature, const Execution &fallback = Execution())
{
    const auto &features = Features();
    const auto found = features.find(feature);
    return found == features.end() ? fallback : found->second;
}

inline Backend BackendOf(const std::string &feature, Backend fallback = Backend::Serial)
{
    const auto &features = Features();
    const auto found = features.find(feature);
    return found == features.end() ? fallback : found->second.backend;
}

inline bool OnDevice(const std::string &feature, bool fallback = false)
{
    return BackendOf(feature, fallback ? Backend::GPU : Backend::Serial) == Backend::GPU;
}

// Host threads the feature should use: 1 when serial, its count when OpenMP.
inline int Threads(const std::string &feature, int fallback)
{
    const auto &features = Features();
    const auto found = features.find(feature);
    if(found == features.end())
    {
        return fallback;
    }
    return found->second.backend == Backend::OpenMP ? found->second.threads : 1;
}
} // namespace execution
} // namespace rich

#endif // EXECUTION_CONFIG_HPP
