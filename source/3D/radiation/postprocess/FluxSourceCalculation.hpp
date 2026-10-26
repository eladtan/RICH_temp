#ifndef IMC_POSTPROCESS_TDE_FLUX_SOURCE_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_FLUX_SOURCE_CALCULATION_HPP

#include <cstdint>

#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

// Luminosity-weighted mean of the observers' noise-debiased polarization
// degrees, sum_i I_i sqrt(max(0, P_i^2 - sigma_P,i^2)) / sum_i I_i, with the
// packet-level sigma_P^2 = sigma_q^2 + sigma_u^2 (the noise floor of P^2).
// Q and U of different observers are in different sky bases, so they are not
// summed. An observer without a packet-level uncertainty (fewer than
// polarization_statistics::kMinEffectivePackets effective packets) keeps its
// raw degree and is counted in undebiasedObserverCount.
struct FluxSourcePolarizationSummary
{
    double luminosityWeightedDegree = 0.0;
    uint64_t observerCount = 0;
    uint64_t undebiasedObserverCount = 0;
};

// Definition string written next to the summary degrees in the result file.
extern char const* const kLuminosityWeightedDegreeDefinition;

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
