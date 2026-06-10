#ifndef IMC_POLARIZATION_HPP
#define IMC_POLARIZATION_HPP

#include "3D/elementary/Vector3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>

namespace IMCPolarization
{

#ifdef MONTECARLO_POLARIZATION

inline constexpr double POL_EPS = 1e-14;
inline constexpr double POL_PI = 3.141592653589793238462643383279502884;

inline Vector3D SafeNormalize(Vector3D const &v, Vector3D const &fallback)
{
    double const n = abs(v);
    if(n > POL_EPS && std::isfinite(n))
        return v * (1.0 / n);
    return fallback;
}

inline Vector3D ChoosePerpendicularBasis(Vector3D const &direction)
{
    Vector3D const k = SafeNormalize(direction, Vector3D(0.0, 0.0, 1.0));
    Vector3D helper = (std::abs(k.z) < 0.9) ? Vector3D(0.0, 0.0, 1.0)
                                            : Vector3D(0.0, 1.0, 0.0);
    Vector3D e = helper - ScalarProd(helper, k) * k;
    return SafeNormalize(e, Vector3D(1.0, 0.0, 0.0));
}

inline Vector3D ProjectBasisToDirection(Vector3D const &basis,
                                         Vector3D const &newDirection)
{
    Vector3D const k = SafeNormalize(newDirection, Vector3D(0.0, 0.0, 1.0));
    Vector3D e = basis - ScalarProd(basis, k) * k;
    if(abs(e) <= POL_EPS || !std::isfinite(abs(e)))
        return ChoosePerpendicularBasis(k);
    return normalize(e);
}

inline void ClampLinearPolarization(double &q, double &u)
{
    if(!std::isfinite(q))
        q = 0.0;
    if(!std::isfinite(u))
        u = 0.0;

    double const p2 = q*q + u*u;
    if(p2 > 1.0)
    {
        double const invP = 1.0 / std::sqrt(p2);
        q *= invP;
        u *= invP;
    }
}

template<class Particle>
inline void InitializeIfNeeded(Particle &p)
{
    if(p.polarizationInitialized)
        return;

    p.stokesQ = 0.0;
    p.stokesU = 0.0;
    p.polarizationBasis = ChoosePerpendicularBasis(p.velocity);
    p.polarizationInitialized = true;
}

template<class Particle>
inline void ResetUnpolarized(Particle &p)
{
    p.stokesQ = 0.0;
    p.stokesU = 0.0;
    p.polarizationBasis = ChoosePerpendicularBasis(p.velocity);
    p.polarizationInitialized = true;
}

template<class Particle>
inline void RotateStokesToBasis(Particle &p, Vector3D const &newBasis)
{
    InitializeIfNeeded(p);

    Vector3D const k = SafeNormalize(p.velocity, Vector3D(0.0, 0.0, 1.0));
    Vector3D const eOld = ProjectBasisToDirection(p.polarizationBasis, k);
    Vector3D const eNew = ProjectBasisToDirection(newBasis, k);

    double const c = std::clamp(ScalarProd(eOld, eNew), -1.0, 1.0);
    double const s = ScalarProd(k, CrossProduct(eOld, eNew));
    double const cos2 = c*c - s*s;
    double const sin2 = 2.0*c*s;

    double const qOld = p.stokesQ;
    double const uOld = p.stokesU;
    p.stokesQ = qOld * cos2 + uOld * sin2;
    p.stokesU = -qOld * sin2 + uOld * cos2;
    p.polarizationBasis = eNew;

    ClampLinearPolarization(p.stokesQ, p.stokesU);
}

template<class Particle>
inline void ApplyThomsonScatter(Particle &p,
                                Vector3D const &oldVelocity,
                                Vector3D const &newVelocity)
{
    InitializeIfNeeded(p);

    Vector3D const kIn = SafeNormalize(oldVelocity, Vector3D(0.0, 0.0, 1.0));
    Vector3D const kOut = SafeNormalize(newVelocity, kIn);
    Vector3D planeNormal = CrossProduct(kIn, kOut);
    double const planeNorm = abs(planeNormal);

    if(planeNorm <= 1e-10)
    {
        p.velocity = newVelocity;
        p.polarizationBasis = ProjectBasisToDirection(p.polarizationBasis, kOut);
        ClampLinearPolarization(p.stokesQ, p.stokesU);
        return;
    }

    planeNormal = planeNormal * (1.0 / planeNorm);
    Vector3D const basisIn = planeNormal;
    Vector3D const basisOut = planeNormal;

    double const oldSpeed = std::max(abs(oldVelocity), POL_EPS);
    p.velocity = kIn * oldSpeed;
    RotateStokesToBasis(p, basisIn);

    double const q = p.stokesQ;
    double const u = p.stokesU;
    double const mu = std::clamp(ScalarProd(kIn, kOut), -1.0, 1.0);
    double const mu2 = mu * mu;
    double const Iprime = (1.0 + mu2) + (1.0 - mu2) * q;
    double const Qprime = (1.0 - mu2) + (1.0 + mu2) * q;
    double const Uprime = 2.0 * mu * u;

    p.velocity = newVelocity;

    if(std::abs(Iprime) <= POL_EPS || !std::isfinite(Iprime))
    {
        p.stokesQ = 0.0;
        p.stokesU = 0.0;
    }
    else
    {
        p.stokesQ = Qprime / Iprime;
        p.stokesU = Uprime / Iprime;
    }

    p.polarizationBasis = ProjectBasisToDirection(basisOut, kOut);
    p.polarizationInitialized = true;
    ClampLinearPolarization(p.stokesQ, p.stokesU);
}

inline void BuildBasisAroundDirection(Vector3D const &k,
                                      Vector3D &e1,
                                      Vector3D &e2)
{
    Vector3D const kk = SafeNormalize(k, Vector3D(0.0, 0.0, 1.0));
    e1 = ChoosePerpendicularBasis(kk);
    e2 = CrossProduct(kk, e1);
    e2 = SafeNormalize(e2, Vector3D(0.0, 1.0, 0.0));
}

template<class Uniform01>
inline double SampleThomsonMu(Uniform01 &&u01)
{
    for(int tries = 0; tries < 10000; ++tries)
    {
        double const mu = 2.0 * u01() - 1.0;
        double const accept = 0.5 * (1.0 + mu * mu);
        if(u01() <= accept)
            return mu;
    }
    return 2.0 * u01() - 1.0;
}

template<class Uniform01>
inline Vector3D SampleSyntheticThomsonDirection(Vector3D const &oldVelocity,
                                                Uniform01 &&u01)
{
    Vector3D const k = SafeNormalize(oldVelocity, Vector3D(0.0, 0.0, 1.0));
    double const speed = std::max(abs(oldVelocity), POL_EPS);

    Vector3D e1, e2;
    BuildBasisAroundDirection(k, e1, e2);

    double const mu = SampleThomsonMu(u01);
    double const phi = 2.0 * POL_PI * u01();
    double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

    Vector3D const dir = mu * k + sinTheta * std::cos(phi) * e1 + sinTheta * std::sin(phi) * e2;
    return SafeNormalize(dir, k) * speed;
}

inline Vector3D ObserverBasis1(Vector3D const &observerDirection)
{
    Vector3D const n = SafeNormalize(observerDirection, Vector3D(0.0, 0.0, 1.0));
    Vector3D up = (std::abs(n.z) < 0.9) ? Vector3D(0.0, 0.0, 1.0)
                                        : Vector3D(0.0, 1.0, 0.0);
    Vector3D e = up - ScalarProd(up, n) * n;
    return SafeNormalize(e, ChoosePerpendicularBasis(n));
}

template<class Particle>
inline std::pair<double, double> ProjectToBasis(Particle const &p,
                                                Vector3D const &direction,
                                                Vector3D const &basis)
{
    if(!p.polarizationInitialized)
        return {0.0, 0.0};

    Vector3D const k = SafeNormalize(direction, Vector3D(0.0, 0.0, 1.0));
    Vector3D const eOld = ProjectBasisToDirection(p.polarizationBasis, k);
    Vector3D const eNew = ProjectBasisToDirection(basis, k);

    double const c = std::clamp(ScalarProd(eOld, eNew), -1.0, 1.0);
    double const s = ScalarProd(k, CrossProduct(eOld, eNew));
    double const cos2 = c*c - s*s;
    double const sin2 = 2.0*c*s;

    double q = p.stokesQ * cos2 + p.stokesU * sin2;
    double u = -p.stokesQ * sin2 + p.stokesU * cos2;
    ClampLinearPolarization(q, u);
    return {q, u};
}

template<class Particle, class Uniform01>
inline Vector3D SamplePolarizedThomsonDirection(Particle const &p,
                                                Vector3D const &oldVelocity,
                                                Uniform01 &&u01)
{
    Vector3D const kIn = SafeNormalize(oldVelocity, Vector3D(0.0, 0.0, 1.0));
    for(int tries = 0; tries < 10000; ++tries)
    {
        Vector3D const candidate = SampleSyntheticThomsonDirection(oldVelocity, u01);
        Vector3D const kOut = SafeNormalize(candidate, kIn);
        Vector3D planeNormal = CrossProduct(kIn, kOut);
        if(abs(planeNormal) <= 1e-10)
            return candidate;

        planeNormal = normalize(planeNormal);
        auto const quPlane = ProjectToBasis(p, oldVelocity, planeNormal);
        double const mu = std::clamp(ScalarProd(kIn, kOut), -1.0, 1.0);
        double const mu2 = mu * mu;
        double const ratio = 1.0 + ((1.0 - mu2) / (1.0 + mu2)) * quPlane.first;
        if(u01() <= std::clamp(0.5 * ratio, 0.0, 1.0))
            return candidate;
    }
    return SampleSyntheticThomsonDirection(oldVelocity, u01);
}

template<class RandomEngine, class UniformDist>
inline std::uint64_t DrawPoissonForPolarization(double mean,
                                                std::uint64_t manualK,
                                                double depolN,
                                                RandomEngine &engine,
                                                UniformDist &)
{
    if(!(mean > 0.0) || !std::isfinite(mean))
        return 0;

    // For very large means, exact N is unnecessary. Once N-K is much larger
    // than depolN, damping is effectively zero and nManual is capped at K.
    double const zeroDampThreshold = static_cast<double>(manualK) + 80.0 * depolN;
    if(mean > zeroDampThreshold + 10.0 * std::sqrt(std::max(1.0, mean)))
        return static_cast<std::uint64_t>(std::ceil(zeroDampThreshold + 1.0));

    std::poisson_distribution<unsigned long long> pois(mean);
    return static_cast<std::uint64_t>(pois(engine));
}

template<class Uniform01>
inline double SampleAgeSinceLastReset(double resetRate, double dt, Uniform01 &&u01,
                                      bool &resetOccurred)
{
    resetOccurred = false;

    if(!(resetRate > 0.0) || !(dt > 0.0) || !std::isfinite(resetRate) || !std::isfinite(dt))
        return dt;

    double const pReset = -std::expm1(-resetRate * dt);
    if(u01() >= pReset)
        return dt;

    resetOccurred = true;
    double const xi = std::clamp(u01(), 0.0, 1.0 - std::numeric_limits<double>::epsilon());
    double const y = xi * pReset;
    double const age = -std::log1p(-y) / resetRate;
    return std::min(age, dt);
}

template<class Particle, class Uniform01>
inline void ApplyManualSyntheticScatterings(Particle &p,
                                            int nManual,
                                            Vector3D const &finalVelocity,
                                            Uniform01 &&u01)
{
    if(nManual <= 0)
    {
        p.velocity = finalVelocity;
        p.polarizationBasis = ProjectBasisToDirection(p.polarizationBasis, finalVelocity);
        return;
    }

    Vector3D currentVelocity = p.velocity;
    for(int i = 0; i < nManual; ++i)
    {
        Vector3D const nextVelocity = (i == nManual - 1)
            ? finalVelocity
            : SamplePolarizedThomsonDirection(p, currentVelocity, u01);
        ApplyThomsonScatter(p, currentVelocity, nextVelocity);
        currentVelocity = nextVelocity;
    }
}

template<class Particle, class RandomEngine, class UniformDist>
inline void ApplyAcceleratedPolarizationHistory(
    Particle &p,
    double dtCo,
    double sigmaScattering,
    double sigmaEffectiveReset,
    Vector3D const &finalVelocityCo,
    int manualScatteringsAfterAcceleration,
    double depolarizationScatterings,
    RandomEngine &engine,
    UniformDist &uniformDist)
{
    InitializeIfNeeded(p);

    auto u01 = [&]() -> double {
        return std::clamp(uniformDist(engine),
                          std::numeric_limits<double>::min(),
                          1.0 - std::numeric_limits<double>::epsilon());
    };

    int const K = std::max(0, manualScatteringsAfterAcceleration);
    double const Npol = depolarizationScatterings;

    if(!(dtCo > 0.0) || !std::isfinite(dtCo))
    {
        p.velocity = finalVelocityCo;
        p.polarizationBasis = ProjectBasisToDirection(p.polarizationBasis, p.velocity);
        return;
    }

    double const resetRate = units::clight * std::max(0.0, sigmaEffectiveReset);
    double const scatRate = units::clight * std::max(0.0, sigmaScattering);

    bool resetOccurred = false;
    double const ageSinceReset = SampleAgeSinceLastReset(resetRate, dtCo, u01, resetOccurred);

    if(resetOccurred)
        ResetUnpolarized(p);

    double const meanScat = scatRate * ageSinceReset;
    std::uint64_t const N = DrawPoissonForPolarization(meanScat,
                                                       static_cast<std::uint64_t>(K),
                                                       Npol,
                                                       engine,
                                                       uniformDist);

    int const nManual = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(K), N));
    std::uint64_t const nDamped = (N > static_cast<std::uint64_t>(K)) ?
                                  (N - static_cast<std::uint64_t>(K)) : 0ULL;

    double damping = 1.0;
    if(nDamped > 0)
    {
        double const exponent = -static_cast<double>(nDamped) / Npol;
        damping = (exponent < -745.0) ? 0.0 : std::exp(exponent);
    }

    p.stokesQ *= damping;
    p.stokesU *= damping;
    ClampLinearPolarization(p.stokesQ, p.stokesU);

    ApplyManualSyntheticScatterings(p, nManual, finalVelocityCo, u01);

    p.velocity = finalVelocityCo;
    p.polarizationBasis = ProjectBasisToDirection(p.polarizationBasis, p.velocity);
    ClampLinearPolarization(p.stokesQ, p.stokesU);
}

#endif // MONTECARLO_POLARIZATION

} // namespace IMCPolarization

#endif // IMC_POLARIZATION_HPP
