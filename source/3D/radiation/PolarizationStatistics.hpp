#ifndef POLARIZATION_STATISTICS_HPP
#define POLARIZATION_STATISTICS_HPP

#include <algorithm>
#include <cmath>

namespace polarization_statistics {

// Fewest effective packets (I^2 / kw2) for which ComputeQuality reports an
// uncertainty and an SNR.
constexpr double kMinEffectivePackets = 10.0;

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

// Fractional Stokes parameters, degree and angle of one observer (or
// observer/group) bin, with the packet-level error of q = Q/I and u = U/I.
// Each crossing of weight w, fractional Stokes parameters q_i, u_i in the
// observer basis and weight-window split factor k adds
//   kw2 += k w^2,  kw2q += k w^2 q_i,  kw2qq += k w^2 q_i^2  (u alike).
// q = sum w q_i / sum w is a ratio estimator; its variance is
//   var(q) = sum k w^2 (q_i - q)^2 / I^2 = (kw2qq - 2 q kw2q + q^2 kw2) / I^2.
// k w^2 is the weight-squared of the unsplit packet, so the copies of a split
// packet count once: exact when they leave together (same direction and
// polarization), conservative when their paths diverge. The effective packet
// count is I^2 / kw2.
inline Quality ComputeQuality(
    double intensity,
    double stokesQ,
    double stokesU,
    double kw2,
    double kw2q,
    double kw2qq,
    double kw2u,
    double kw2uu)
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

    if (!(kw2 > 0.0) || !std::isfinite(kw2) ||
        !std::isfinite(kw2q) || !std::isfinite(kw2qq) ||
        !std::isfinite(kw2u) || !std::isfinite(kw2uu))
        return result;

    result.effectivePackets = intensity * intensity / kw2;
    if (!(result.effectivePackets > 0.0) ||
        !std::isfinite(result.effectivePackets)) {
        result.effectivePackets = 0.0;
        return result;
    }
    // Below kMinEffectivePackets the packet variance is no estimate: one
    // packet gives exactly zero (rounding leaves ~1e-20, an SNR of ~1e9), a
    // few dominant ones far too little. The bin keeps its degree and angle
    // but has no uncertainty: sigma and SNR 0, uncertaintyValid false.
    if (result.effectivePackets < kMinEffectivePackets)
        return result;

    double const invI2 = 1.0 / (intensity * intensity);
    double const varQ = std::max(
        0.0, (kw2qq - 2.0 * result.q * kw2q + result.q * result.q * kw2) * invI2);
    double const varU = std::max(
        0.0, (kw2uu - 2.0 * result.u * kw2u + result.u * result.u * kw2) * invI2);
    if (!std::isfinite(varQ) || !std::isfinite(varU))
        return result;
    result.sigmaQ = std::sqrt(varQ);
    result.sigmaU = std::sqrt(varU);
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
