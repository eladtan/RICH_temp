#ifndef POST_PROCESS_IMC_HELPERS_HPP
#define POST_PROCESS_IMC_HELPERS_HPP

#include <cmath>
#include <string>
#include "3D/elementary/Vector3D.hpp"

class SphericalObserver;
class Tessellation3D;

struct RadiationIMCPostProcessConfig
{
    bool enabled = false;
    double sourceDt = 0.0;
    double transportTime = 0.0;
    bool forceGreyFleckOne = true;
    // In post-process mode, hydro feedback is disabled, but snapshot cell
    // velocities may still be used for Doppler shifts, comoving opacities,
    // frequency-group decisions, and lab/comoving transforms.
    bool useCellVelocities = true;

    struct PolarizationConfig
    {
        bool enabled = false;
        int manualScatteringsAfterAcceleration = 4;
        double depolarizationScatterings = 2.0;
        std::string acceleratedClosure = "damped_last_scatterings";
        Vector3D referenceAxis = Vector3D(0.0, 0.0, 1.0);
        Vector3D fallbackAxis = Vector3D(1.0, 0.0, 0.0);
        double poleTolerance = 0.999999;
        double warnMismatchAngle = 0.01;
        double failMismatchAngle = -1.0;
    } polarization;
};

namespace PostProcessIMC
{
    void ValidateConfig(const RadiationIMCPostProcessConfig &config,
                        bool withCompton,
                        bool withMultigroupOpacity,
                        bool withRandomWalk);

    template<class ParticleContainer>
    double PrepareGeneratedParticles(ParticleContainer &particles,
                                     double transportTime)
    {
        double emitted = 0.0;
        for (auto &p : particles) {
            p.timeLeft = transportTime;
            p.initialWeight = std::abs(p.weight);
#ifdef MONTECARLO_POLARIZATION
            p.stokesQ = 0.0;
            p.stokesU = 0.0;
            p.polarizationInitialized = false;
#endif
            emitted += p.weight;
        }
        return emitted;
    }

    Vector3D ObserverNudgeCandidate(const SphericalObserver &observer,
                                    const Vector3D &location);
}

#endif // POST_PROCESS_IMC_HELPERS_HPP
