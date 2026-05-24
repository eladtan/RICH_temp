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
    if (!config.enabled)
        return;

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
