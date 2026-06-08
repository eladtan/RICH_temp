#include "PostProcessIMCHelpers.hpp"
#include "SphericalObserver.hpp"
#include "misc/universal_error.hpp"
#include <cmath>
#include <iostream>
#include <sstream>

namespace PostProcessIMC
{

void NormalizeAndValidateConfig(RadiationIMCPostProcessConfig &config,
                    bool withCompton,
                    bool withMultigroupOpacity,
                    bool withRandomWalk,
                    bool withDDMC)
{
    if(config.peelOff.enabled && !config.enabled)
        throw UniversalError("PostProcess peel-off requires postProcess.enabled");
    if(config.polarization.enabled && !config.enabled)
        throw UniversalError("PostProcess polarization requires postProcess.enabled");
    if (!config.enabled)
        return;
    if(config.peelOff.enabled)
    {
        if(config.peelOff.maxTau <= 0.0)
            throw UniversalError("PostProcess peel-off: maxTau must be positive");
        if(config.peelOff.rayNudgeFraction <= 0.0 || config.peelOff.rayNudgeFraction >= 1.0)
            throw UniversalError("PostProcess peel-off: rayNudgeFraction must be in (0, 1)");
        if(config.peelOff.maxRayCells == 0)
            throw UniversalError("PostProcess peel-off: maxRayCells must be positive");

        if(config.peelOff.resolvedEvents)
            throw UniversalError("PostProcess peel-off: resolvedEvents is deprecated. "
                "Use resolvedElasticScattering and resolvedEffectiveScattering explicitly");
        if(config.peelOff.acceleratedBoundaryEvents)
            throw UniversalError("PostProcess peel-off: acceleratedBoundaryEvents is deprecated. "
                "Use ddmcLeakEvents, ddmcUpscatterEvents, randomWalkUpscatterEvents explicitly");

        bool const anyEventPeelOff =
            config.peelOff.resolvedElasticScattering ||
            config.peelOff.resolvedEffectiveScattering ||
            config.peelOff.randomWalkClosureEvents ||
            config.peelOff.randomWalkUpscatterEvents ||
            config.peelOff.ddmcLeakEvents ||
            config.peelOff.ddmcUpscatterEvents;

        if(withCompton && anyEventPeelOff)
            throw UniversalError("PostProcess peel-off: event peel-off beyond source emission does not support Compton yet");
        if(withCompton && config.peelOff.sourceEmission)
        {
            // Source-emission-only peel-off without Compton events is guarded:
            // Compton changes opacity semantics. Disallow until validated.
            throw UniversalError("PostProcess peel-off does not support Compton yet");
        }

        if(config.useCellVelocities && anyEventPeelOff)
            throw UniversalError("PostProcess peel-off: resolved/accelerated event peel-off with moving media (useCellVelocities=true) is not yet implemented; disable event peel-off or set useCellVelocities=false");

        if(config.peelOff.randomWalkClosureEvents)
            throw UniversalError("PostProcess peel-off: RW closure peel-off requires boundary-source geometry that is not yet implemented");
        if(!withRandomWalk && config.peelOff.randomWalkUpscatterEvents)
            throw UniversalError("PostProcess peel-off: RW upscatter peel-off requires withRandomWalk=true");
        if(!withDDMC && (config.peelOff.ddmcLeakEvents || config.peelOff.ddmcUpscatterEvents))
            throw UniversalError("PostProcess peel-off: DDMC peel-off flags require withDDMC=true");

        using MpiPolicy = RadiationIMCPostProcessConfig::PeelOffConfig::MpiRayPolicy;
#ifdef RICH_MPI
        if(config.peelOff.mpiRayPolicy != MpiPolicy::DistributedExact &&
           !config.peelOff.allowApproximateMpiPeelOff)
            throw UniversalError("PostProcess peel-off: MPI builds require mpiRayPolicy=DistributedExact "
                "for correct peel-off. Set allowApproximateMpiPeelOff=true to use "
                "StrictAbort or LocalConservativeVacuum as debug/fallback modes");
#endif
        if(config.peelOff.maxDistributedExchangeRounds == 0)
            throw UniversalError("PostProcess peel-off: maxDistributedExchangeRounds must be positive");

    }
    if (config.sourceDt <= 0.0)
        throw UniversalError("PostProcess: sourceDt must be positive");
    if (config.transportTime <= 0.0)
        throw UniversalError("PostProcess: transportTime must be positive");
    if (withCompton && !withMultigroupOpacity)
        throw UniversalError("PostProcess: Compton requires multigroup opacity");
#if ENERGY_GROUPS_NUM <= 1
    if (withCompton)
        throw UniversalError("PostProcess: Compton requires ENERGY_GROUPS_NUM > 1");
#endif
    if (withCompton && withRandomWalk)
        throw UniversalError("PostProcess: Compton and random walk cannot both be enabled");
    if(config.polarization.enabled)
    {
#ifndef MONTECARLO_POLARIZATION
        throw UniversalError("PostProcess polarization requested, but binary was compiled without MONTECARLO_POLARIZATION");
#else
        if(withCompton)
            throw UniversalError("PostProcess polarization does not support Compton yet");
        if(config.polarization.manualScatteringsAfterAcceleration < 0)
            throw UniversalError("PostProcess polarization manualScatteringsAfterAcceleration must be non-negative");
        if(config.polarization.manualScatteringsAfterAcceleration > 128)
            throw UniversalError("PostProcess polarization manualScatteringsAfterAcceleration is too large");
        if(!(config.polarization.depolarizationScatterings > 0.0) ||
           !std::isfinite(config.polarization.depolarizationScatterings))
            throw UniversalError("PostProcess polarization depolarizationScatterings must be finite and positive");
        if(config.polarization.acceleratedClosure != "damped_last_scatterings")
            throw UniversalError("Unsupported postprocess polarization acceleratedClosure");
#endif
    }
}

Vector3D ObserverNudgeCandidate(const SphericalObserver &observer,
                                const Vector3D &location)
{
    Vector3D radial = location - observer.getCenter();
    double norm = abs(radial);
    if (norm <= 0.0)
        return location;
    double nudge = 1e-10 * observer.getRadius();
    return location + radial * (nudge / norm);
}

} // namespace PostProcessIMC
