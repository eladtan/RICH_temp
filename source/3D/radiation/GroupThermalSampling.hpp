#ifndef GROUP_THERMAL_SAMPLING_HPP
#define GROUP_THERMAL_SAMPLING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>

// Thermal frequency sampling from the cumulative of alpha kappa_g b_g over the
// energy groups (cumulative[0] = 0, cumulative[groups] = total), which is
// linear inside each group. The cumulative holds opacities in cm^-1, which can
// be far below STORM::LinearInterpolation's closeness tolerance, so it is
// inverted here directly: a draw always lies inside its group, strictly below
// the group's upper boundary (the next group's lower edge). Shared by
// MultigroupOpacity and the STORM opacity adapter in RadiationIMC.hpp.
namespace group_thermal_sampling
{

// h nu at `fraction` of the way through `group` (fraction clamped to [0, 1],
// non-finite taken as 0.5), strictly below the group's upper boundary.
template <typename Boundaries>
double SampleLinearInGroup(Boundaries const& boundaries, std::size_t group, double fraction)
{
    double const left = boundaries[group];
    double const right = boundaries[group + 1];
    double const f = std::isfinite(fraction) ? std::clamp(fraction, 0.0, 1.0) : 0.5;
    double const frequency = left + f * (right - left);
    return (frequency < right) ? std::max(frequency, left) : std::nextafter(right, left);
}

// h nu of a thermal draw for unitRandom in [0, 1): the group whose cumulative
// bracket holds unitRandom * total (zero-width groups are skipped, and the
// last group with positive width is taken when rounding puts the target at
// the total), then linear inside that group.
template <typename Cumulative, typename Boundaries>
double SampleThermalEnergy(Cumulative const& cumulative, Boundaries const& boundaries,
                           std::size_t groups, double unitRandom)
{
    auto const first = std::begin(cumulative) + 1;
    double const target = unitRandom * cumulative[groups];
    std::size_t group = static_cast<std::size_t>(
        std::upper_bound(first, first + static_cast<std::ptrdiff_t>(groups), target) - first);
    if(group >= groups)
    {
        group = groups - 1;
        while(group > 0 && !(cumulative[group + 1] > cumulative[group]))
            --group;
    }
    double const width = cumulative[group + 1] - cumulative[group];
    double const fraction = (width > 0.0) ? (target - cumulative[group]) / width : 0.5;
    return SampleLinearInGroup(boundaries, group, fraction);
}

} // namespace group_thermal_sampling

#endif // GROUP_THERMAL_SAMPLING_HPP
