#ifndef POLARIZATION_STATISTICS_HPP
#define POLARIZATION_STATISTICS_HPP

#include <algorithm>
#include <cmath>

namespace polarization_statistics {

struct Quality
{
    bool intensityValid = false;
    bool uncertaintyValid = false;
    double q = 0.0;
    double u = 0.0;
    double degree = 0.0;
    double angle = 0.0;
    double effectivePackets = 0.0;
    double sigmaQ = 0.0;
    double sigmaU = 0.0;
    double sigmaP = 0.0;
    double snr = 0.0;
};

inline Quality ComputeQuality(
    double intensity,
    double stokesQ,
    double stokesU,
    double weightSq,
    double sumWQ2,
    double sumWU2)
{
    Quality result;
    if (!(intensity > 0.0) || !std::isfinite(intensity) ||
        !std::isfinite(stokesQ) || !std::isfinite(stokesU))
        return result;

    result.q = stokesQ / intensity;
    result.u = stokesU / intensity;
    if (!std::isfinite(result.q) || !std::isfinite(result.u))
        return Quality{};

    result.intensityValid = true;
    result.degree = std::sqrt(result.q * result.q + result.u * result.u);
    result.angle = 0.5 * std::atan2(stokesU, stokesQ);

    if (!(weightSq > 0.0) || !std::isfinite(weightSq) ||
        !std::isfinite(sumWQ2) || !std::isfinite(sumWU2))
        return result;

    result.effectivePackets = intensity * intensity / weightSq;
    if (!(result.effectivePackets > 0.0) ||
        !std::isfinite(result.effectivePackets)) {
        result.effectivePackets = 0.0;
        return result;
    }

    double const secondQ = sumWQ2 / intensity;
    double const secondU = sumWU2 / intensity;
    if (!std::isfinite(secondQ) || !std::isfinite(secondU))
        return result;

    double const varQ = std::max(0.0, secondQ - result.q * result.q);
    double const varU = std::max(0.0, secondU - result.u * result.u);
    result.sigmaQ = std::sqrt(varQ / result.effectivePackets);
    result.sigmaU = std::sqrt(varU / result.effectivePackets);
    result.sigmaP = std::sqrt(
        result.sigmaQ * result.sigmaQ + result.sigmaU * result.sigmaU);

    if (!(result.sigmaP > 0.0) || !std::isfinite(result.sigmaP))
        return result;

    result.snr = result.degree / result.sigmaP;
    if (!std::isfinite(result.snr)) {
        result.snr = 0.0;
        return result;
    }

    result.uncertaintyValid = true;
    return result;
}

} // namespace polarization_statistics

#endif // POLARIZATION_STATISTICS_HPP
