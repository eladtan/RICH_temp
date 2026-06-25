#ifndef REVERSE_POLARIZATION_MUELLER_HPP
#define REVERSE_POLARIZATION_MUELLER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include "3D/elementary/Vector3D.hpp"
#include "ReversePacket.hpp"
#include "ReverseEstimatorConfig.hpp"

namespace ReverseMueller
{

inline constexpr double POL_EPS = 1e-14;
inline constexpr double POL_PI = 3.141592653589793238462643383279502884;

inline Vector3D safeNormalize(Vector3D const &v, Vector3D const &fallback)
{
    double n = abs(v);
    if (n > POL_EPS && std::isfinite(n))
        return v * (1.0 / n);
    return fallback;
}

inline Vector3D choosePerpendicularBasis(Vector3D const &direction)
{
    Vector3D k = safeNormalize(direction, Vector3D(0.0, 0.0, 1.0));
    Vector3D helper = (std::abs(k.z) < 0.9) ? Vector3D(0.0, 0.0, 1.0)
                                             : Vector3D(0.0, 1.0, 0.0);
    Vector3D e = helper - ScalarProd(helper, k) * k;
    return safeNormalize(e, Vector3D(1.0, 0.0, 0.0));
}

template <class Uniform01>
inline Vector3D sampleIsotropicDirection(Uniform01 &&u01)
{
    double costheta = 2.0 * u01() - 1.0;
    double sintheta = std::sqrt(std::max(0.0, 1.0 - costheta * costheta));
    double phi = 2.0 * POL_PI * u01();
    return Vector3D(sintheta * std::cos(phi), sintheta * std::sin(phi), costheta);
}

inline Vector3D projectBasisToDirection(Vector3D const &basis,
                                        Vector3D const &newDirection)
{
    Vector3D k = safeNormalize(newDirection, Vector3D(0.0, 0.0, 1.0));
    Vector3D e = basis - ScalarProd(basis, k) * k;
    if (abs(e) <= POL_EPS || !std::isfinite(abs(e)))
        return choosePerpendicularBasis(k);
    return normalize(e);
}

inline Vector3D observerBasis1(Vector3D const &observerDirection)
{
    Vector3D n = safeNormalize(observerDirection, Vector3D(0.0, 0.0, 1.0));
    Vector3D up = (std::abs(n.z) < 0.9) ? Vector3D(0.0, 0.0, 1.0)
                                         : Vector3D(0.0, 1.0, 0.0);
    Vector3D e = up - ScalarProd(up, n) * n;
    return safeNormalize(e, choosePerpendicularBasis(n));
}

// Stokes rotation B(psi) for angle psi between two polarization bases
inline MuellerResponse3 basisRotation(double cosPsi, double sinPsi)
{
    double cos2 = cosPsi * cosPsi - sinPsi * sinPsi;
    double sin2 = 2.0 * cosPsi * sinPsi;
    MuellerResponse3 B;
    B.m[0] = {1.0, 0.0, 0.0};
    B.m[1] = {0.0, cos2, sin2};
    B.m[2] = {0.0, -sin2, cos2};
    return B;
}

// Compute the rotation from oldBasis to newBasis around direction k
MuellerResponse3 basisRotationBetween(Vector3D const &kDir,
                                      Vector3D const &oldBasis,
                                      Vector3D const &newBasis);

// Thomson scattering matrix T(mu) for cosine of scattering angle
inline MuellerResponse3 thomsonMatrix(double mu)
{
    double mu2 = mu * mu;
    MuellerResponse3 T;
    T.m[0] = {1.0 + mu2, 1.0 - mu2, 0.0};
    T.m[1] = {1.0 - mu2, 1.0 + mu2, 0.0};
    T.m[2] = {0.0, 0.0, 2.0 * mu};
    return T;
}

// Full reverse Thomson update: rotates from source basis into scattering
// plane, applies Thomson matrix, then rotates to observer-side basis.
// The proposal correction 1/q is applied as a scalar weight externally.
MuellerResponse3 reverseThomsonUpdate(
    Vector3D const &kIn, Vector3D const &kOut,
    Vector3D const &basisIn, Vector3D const &basisOut);

// Build e1/e2 around a direction
inline void buildBasisAroundDirection(Vector3D const &k,
                                      Vector3D &e1, Vector3D &e2)
{
    Vector3D kk = safeNormalize(k, Vector3D(0.0, 0.0, 1.0));
    e1 = choosePerpendicularBasis(kk);
    e2 = CrossProduct(kk, e1);
    e2 = safeNormalize(e2, Vector3D(0.0, 1.0, 0.0));
}

// Sample Thomson mu via rejection
template <class Uniform01>
inline double sampleThomsonMu(Uniform01 &&u01)
{
    for (int tries = 0; tries < 10000; ++tries)
    {
        double mu = 2.0 * u01() - 1.0;
        double accept = 0.5 * (1.0 + mu * mu);
        if (u01() <= accept)
            return mu;
    }
    return 2.0 * u01() - 1.0;
}

// Sample an isotropic Thomson direction from given forward direction
template <class Uniform01>
inline Vector3D sampleIsotropicThomsonDirection(Vector3D const &kCurrent,
                                                Uniform01 &&u01)
{
    Vector3D k = safeNormalize(kCurrent, Vector3D(0.0, 0.0, 1.0));
    Vector3D e1, e2;
    buildBasisAroundDirection(k, e1, e2);

    double mu = sampleThomsonMu(u01);
    double phi = 2.0 * POL_PI * u01();
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

    Vector3D dir = mu * k + sinTheta * std::cos(phi) * e1
                   + sinTheta * std::sin(phi) * e2;
    return safeNormalize(dir, k);
}

// Apply a full reverse scatter event to the packet's Mueller response.
// kInForward and kOutForward are the represented forward photon directions.
template <class Uniform01>
inline void applyReverseThomsonScatter(
    ReverseAdjointPacket &pkt,
    Vector3D const &kInForward, Vector3D const &kOutForward,
    Uniform01 &&u01)
{
    Vector3D basisIn = pkt.basisInitialized
        ? projectBasisToDirection(pkt.sourceBasisLab, kInForward)
        : choosePerpendicularBasis(kInForward);

    // basisOut must match M_obs_from_src's current source-side basis
    // (not the observer sky basis, which is only for final projection).
    Vector3D basisOut = pkt.basisInitialized
        ? projectBasisToDirection(pkt.sourceBasisLab, kOutForward)
        : choosePerpendicularBasis(kOutForward);

    MuellerResponse3 update = reverseThomsonUpdate(kInForward, kOutForward,
                                                   basisIn, basisOut);
    pkt.M_obs_from_src = pkt.M_obs_from_src * update;
    pkt.sourceBasisLab = basisIn;
    pkt.basisInitialized = true;
    ++pkt.scatterCountExplicit;
}

// Apply synthetic scatterings for DDMC closure.
// kBeforeForward = pre-residence observer-side direction (exit direction).
// kAfterForward = post-leak source-side direction (entry direction).
// Bridge builds forward photon chain: kAfterForward → ... → kBeforeForward.
template <class Uniform01>
inline void applySyntheticScatterings(
    ReverseAdjointPacket &pkt,
    Vector3D const &kBeforeForward, Vector3D const &kAfterForward,
    uint64_t K, Uniform01 &&u01)
{
    if (K == 0)
    {
        pkt.sourceBasisLab = projectBasisToDirection(
            pkt.basisInitialized ? pkt.sourceBasisLab
                                 : choosePerpendicularBasis(kBeforeForward),
            kAfterForward);
        pkt.basisInitialized = true;
        return;
    }

    // Iterate from observer side (kBeforeForward) toward source side (kAfterForward).
    // Each update T(kPrev, kCurrent) = forward scattering from kPrev to kCurrent.
    // Applied left-to-right: M = M * T_first * ... * T_last
    // Gives forward chain: kAfterForward → ... → kBeforeForward (source to observer).
    Vector3D kCurrent = kBeforeForward;
    for (uint64_t j = 0; j < K; ++j)
    {
        Vector3D kPrev;
        if (j == K - 1)
            kPrev = kAfterForward;
        else
            kPrev = sampleIsotropicThomsonDirection(kCurrent, u01);

        Vector3D basisCurrent = (j == 0 && pkt.basisInitialized)
            ? projectBasisToDirection(pkt.sourceBasisLab, kCurrent)
            : choosePerpendicularBasis(kCurrent);

        Vector3D basisPrev = choosePerpendicularBasis(kPrev);

        MuellerResponse3 update = reverseThomsonUpdate(kPrev, kCurrent,
                                                       basisPrev, basisCurrent);
        pkt.M_obs_from_src = pkt.M_obs_from_src * update;
        kCurrent = kPrev;
        ++pkt.scatterCountSynthetic;
    }

    pkt.sourceBasisLab = choosePerpendicularBasis(kAfterForward);
    pkt.basisInitialized = true;
}

} // namespace ReverseMueller

#endif // REVERSE_POLARIZATION_MUELLER_HPP
