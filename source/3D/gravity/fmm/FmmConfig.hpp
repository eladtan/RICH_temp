#ifndef FMM_CONFIG_HPP
#define FMM_CONFIG_HPP

#include <cstddef>
#include <cstdint>

static constexpr int FMM_MAX_ORDER = 8;
static constexpr int FMM_MAX_TREE_DEPTH = 21;

struct FmmGravityOptions
{
    int expansionOrder = 4;
    double thetaCritical = 0.5;
    std::size_t leafCapacity = 32;
    int maxDepth = FMM_MAX_TREE_DEPTH;
    bool computePotential = false;
    bool validateFinite = true;
};

inline std::size_t fmmCoefficientCount(int expansionOrder)
{
    return static_cast<std::size_t>(expansionOrder + 1) *
           static_cast<std::size_t>(expansionOrder + 1);
}

inline std::size_t fmmTaylorCoefficientCount(int expansionOrder)
{
    const std::size_t p = static_cast<std::size_t>(expansionOrder);
    return (p + 1) * (p + 2) * (p + 3) / 6;
}

#endif // FMM_CONFIG_HPP
