#include "LorentzTransformation.hpp"
#include <cmath>

#ifdef MONTECARLO_POLARIZATION
namespace
{
Vector3D ChooseTransverseBasis(Vector3D const &direction)
{
    Vector3D const n = normalize(direction);
    Vector3D helper = std::abs(n.z) < 0.9
        ? Vector3D(0.0, 0.0, 1.0)
        : Vector3D(0.0, 1.0, 0.0);
    Vector3D basis = helper - ScalarProd(helper, n) * n;
    double const norm = abs(basis);
    if(!(norm > 0.0) || !std::isfinite(norm))
        return Vector3D(1.0, 0.0, 0.0);
    return basis / norm;
}

Vector3D TransformPolarizationBasis(Vector3D const &oldVelocity,
                                    Vector3D const &newVelocity,
                                    Vector3D const &oldBasis,
                                    Vector3D const &boostVelocity,
                                    double gamma)
{
    Vector3D const nOld = normalize(oldVelocity);
    Vector3D const nNew = normalize(newVelocity);
    Vector3D eOld = oldBasis - ScalarProd(oldBasis, nOld) * nOld;
    double const oldNorm = abs(eOld);
    if(!(oldNorm > 0.0) || !std::isfinite(oldNorm))
        eOld = ChooseTransverseBasis(nOld);
    else
        eOld = eOld / oldNorm;

    // Transform the electric-field basis of a plane wave.  With
    // B_scaled = n x E, the standard boost is
    // E' = gamma(E + beta x B_scaled)
    //      - gamma^2/(gamma+1) beta(beta.E).
    Vector3D const beta = boostVelocity / units::clight;
    Vector3D const bScaled = CrossProduct(nOld, eOld);
    Vector3D eNew = gamma * (eOld + CrossProduct(beta, bScaled))
        - (gamma * gamma / (gamma + 1.0))
          * ScalarProd(beta, eOld) * beta;
    eNew = eNew - ScalarProd(eNew, nNew) * nNew;
    double const newNorm = abs(eNew);
    if(!(newNorm > 0.0) || !std::isfinite(newNorm))
        return ChooseTransverseBasis(nNew);
    return eNew / newNorm;
}
}
#endif

void LorentzTransformation(Particle3D &particle, const Vector3D &velocity)
{
    double v2 = ScalarProd(velocity, velocity);
    if(v2 < 1e-30)
        return;
#ifdef MONTECARLO_POLARIZATION
    bool const transformPolarization = particle.polarizationInitialized;
    Vector3D const oldPhotonVelocity = particle.velocity;
    Vector3D const oldPolarizationBasis = particle.polarizationBasis;
#endif
    double gamma = 1.0 / std::sqrt(1 - units::inv_clight2 * v2);
    double dopplerShift = DopplerShift(particle, velocity);
    particle.frequency *= dopplerShift;
    particle.weight *= dopplerShift;
    particle.velocity = particle.velocity + velocity * ((gamma - 1) * ScalarProd(particle.velocity, velocity) / v2 - gamma);
    particle.velocity *= units::clight / abs(particle.velocity); // normalize to speed of light
#ifdef MONTECARLO_POLARIZATION
    if(transformPolarization)
    {
        particle.polarizationBasis = TransformPolarizationBasis(
            oldPhotonVelocity, particle.velocity, oldPolarizationBasis,
            velocity, gamma);
    }
#endif
}

void LabToComovingPacket(Particle3D &particle, const Vector3D &cellVelocity)
{
    LorentzTransformation(particle, cellVelocity);
}

void ComovingToLabPacket(Particle3D &particle, const Vector3D &cellVelocity)
{
    LorentzTransformation(particle, -1.0 * cellVelocity);
}

double DopplerShift(const Particle3D &particle, const Vector3D &velocity)
{
    double v2 = ScalarProd(velocity, velocity);
    double gamma = 1.0 / std::sqrt(1 - units::inv_clight2 * v2);
    return gamma * (1 - ScalarProd(velocity, particle.velocity) * units::inv_clight2);
}
