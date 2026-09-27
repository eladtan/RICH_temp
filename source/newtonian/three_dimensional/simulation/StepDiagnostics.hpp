#ifndef STEP_DIAGNOSTICS_HPP
#define STEP_DIAGNOSTICS_HPP

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>

struct SourceStepTiming
{
    double first_seconds = 0;
    double second_seconds = 0;
    std::size_t calls = 0;

    double totalSeconds(void) const
    {
        return first_seconds + second_seconds;
    }
};

struct MeshBuildTiming
{
    double seconds = 0;
    std::size_t builds = 0;
    // Full (whole-mesh) Voronoi builds among `builds`; the individual
    // scheme's partial builds are the remainder.
    std::size_t full_builds = 0;
};

class MeshBuildTimer
{
public:
    explicit MeshBuildTimer(MeshBuildTiming& timing) :
        timing_(timing), start_(std::chrono::steady_clock::now())
    {
        ++timing_.builds;
    }

    ~MeshBuildTimer()
    {
        timing_.seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_).count();
    }

private:
    MeshBuildTiming& timing_;
    std::chrono::steady_clock::time_point const start_;
};

struct StepRetryRecord
{
    double attempted_dt_min = 0;
    double attempted_dt_max = 0;
    double elapsed_seconds = 0;
    std::string reason;
    std::string diagnostics;
    std::size_t representative_cell = std::numeric_limits<std::size_t>::max();
};

#endif // STEP_DIAGNOSTICS_HPP
