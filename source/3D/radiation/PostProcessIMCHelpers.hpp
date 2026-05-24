#ifndef POST_PROCESS_IMC_HELPERS_HPP
#define POST_PROCESS_IMC_HELPERS_HPP

#include <cmath>
#include "3D/elementary/Vector3D.hpp"

class SphericalObserver;
class Tessellation3D;

struct RadiationIMCPostProcessConfig
{
    bool enabled = false;
    double sourceDt = 0.0;
    double transportTime = 0.0;
    bool forceGreyFleckOne = true;
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
            emitted += p.weight;
        }
        return emitted;
    }

    Vector3D ObserverNudgeCandidate(const SphericalObserver &observer,
                                    const Vector3D &location);
}

#endif // POST_PROCESS_IMC_HELPERS_HPP
