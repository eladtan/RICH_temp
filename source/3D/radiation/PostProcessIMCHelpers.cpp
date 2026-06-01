#include "PostProcessIMCHelpers.hpp"
#include "SphericalObserver.hpp"
#include "misc/universal_error.hpp"
#include <cmath>
#include <sstream>

namespace PostProcessIMC
{

void ValidateConfig(const RadiationIMCPostProcessConfig &config,
                    bool withCompton,
                    bool withMultigroupOpacity,
                    bool withRandomWalk)
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
        if(config.peelOff.resolvedEvents)
            throw UniversalError("PostProcess peel-off: resolvedEvents is not implemented in phase 1");
        if(config.peelOff.acceleratedBoundaryEvents)
            throw UniversalError("PostProcess peel-off: acceleratedBoundaryEvents is not implemented in phase 1");
        if(withCompton)
            throw UniversalError("PostProcess peel-off does not support Compton yet");
#ifdef RICH_MPI
        // Peel-off rays terminate at domain boundaries (ghost cells treated as
        // vacuum escape). This underestimates tau for cross-domain rays but is
        // safe — contributions are conservative (over-estimated flux).
#endif
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
        if(config.polarization.manualScatteringsAfterAcceleration > 64)
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
