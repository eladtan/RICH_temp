#ifndef IMC_POSTPROCESS_TDE_FLUX_SOURCE_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_FLUX_SOURCE_CALCULATION_HPP

#include <cstdint>

#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

struct FluxSourcePolarizationSummary
{
    double luminosityWeightedDegree = 0.0;
    uint64_t observerCount = 0;
};

void InitializeFluxSourceSurface(
    Config const& cfg,
    PostprocessRuntime& runtime);

// Adds the emission surface along each observer direction (nearest
// construction ray) to the observer's photosphere output.
void AttachEmissionSurfaceToObserver(
    Config const& cfg,
    PostprocessRuntime& runtime);

// Per tessellation point (owned and ghost): 1 at or beyond the flux-source
// surface, 0 inside it. Enclosed outside regions are not removed here.
std::vector<unsigned char> BuildOutsideFluxSourceMask(
    Config const& cfg,
    PostprocessRuntime const& runtime);

// Installs the face sources of the current decomposition on `physics`, and,
// when volume emission is enabled, the mask of outside cells whose thermal
// emission (Planck-mean of `emissionOpacity`) passes the cutoff.
void ConfigureFluxSourceForCurrentDecomposition(
    Config const& cfg,
    PostprocessRuntime& runtime,
    RadiationIMC& physics,
    OpacityCalculator const& emissionOpacity,
    bool multigroupPass);

FluxSourcePolarizationSummary ComputeFluxSourcePolarizationSummary(
    SphericalObserver::ObserverQualitySnapshot const& snapshot);

} // namespace imc_postprocess_tde
#endif
