#ifndef IMC_POSTPROCESS_TDE_FLUX_SOURCE_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_FLUX_SOURCE_CALCULATION_HPP

#include <cstdint>

#include "postprocess_config.hpp"
#include "postprocess_runtime.hpp"

namespace imc_postprocess_tde {

struct FluxSourcePolarizationSummary
{
    double luminosityWeightedDegree = 0.0;
    uint64_t observerCount = 0;
};

void InitializeFluxSourceSurface(
    Config const& cfg,
    PostprocessRuntime& runtime);

void ConfigureFluxSourceForCurrentDecomposition(
    Config const& cfg,
    PostprocessRuntime& runtime,
    RadiationIMC& physics);

FluxSourcePolarizationSummary ComputeFluxSourcePolarizationSummary(
    SphericalObserver::ObserverQualitySnapshot const& snapshot);

} // namespace imc_postprocess_tde
#endif
