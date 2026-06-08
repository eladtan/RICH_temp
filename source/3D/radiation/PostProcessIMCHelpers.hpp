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

    struct PeelOffConfig
    {
        bool enabled = false;

        // [exact] Source emission peel-off (static media only for event modes).
        bool sourceEmission = true;
        // [exact, static-only] Elastic scatter peel-off.
        // Uses polarized Thomson phase functions when polarization is enabled;
        // otherwise matches the current isotropic elastic transport sampler.
        bool resolvedElasticScattering = false;
        // [exact, static-only] Effective (absorption+reemission) scatter peel-off.
        bool resolvedEffectiveScattering = false;

        // [unsupported] RW closure peel-off — rejected in validation.
        // Requires boundary-source geometry not yet implemented.
        bool randomWalkClosureEvents = false;
        // [exact, static-only] RW upscatter peel-off.
        bool randomWalkUpscatterEvents = false;
        // [approximate geometry] DDMC interface leak peel-off. Uses face-center
        // cosine-law (Lambertian) source; face-area variation of optical depth
        // to the observer is neglected. Under MPI with DistributedExact (the
        // production default), ray propagation across ranks is exact.
        // StrictAbort and LocalConservativeVacuum are explicit debug/fallback
        // modes requiring allowApproximateMpiPeelOff=true.
        bool ddmcLeakEvents = false;
        // [exact, static-only] DDMC upscatter peel-off.
        bool ddmcUpscatterEvents = false;

        // [deprecated] Use explicit flags instead. Throws in validation.
        bool resolvedEvents = false;
        // [deprecated] Use explicit flags instead. Throws in validation.
        bool acceleratedBoundaryEvents = false;

        double maxTau = 700.0;
        double rayNudgeFraction = 1e-10;
        size_t maxRayCells = 100000;

        // MPI ray policy for peel-off rays that exit the local rank domain.
        // Exact distributed MPI requires collective queue processing; do not
        // send MPI messages from inside maybeRecordPeelOff or event handlers.
        //
        // StrictAbort [debug/fallback]: reject all rays leaving local real
        //   cells (conservative lower bound, not exact).
        // LocalConservativeVacuum [debug/fallback]: keep tau accumulated to
        //   the local MPI boundary; assume zero additional optical depth
        //   beyond that boundary (approximate upper bound, not exact).
        // DistributedExact [production MPI]: continue rays across ranks via
        //   collective queue drained at known synchronization points.
        enum class MpiRayPolicy { StrictAbort, LocalConservativeVacuum, DistributedExact };
        MpiRayPolicy mpiRayPolicy = MpiRayPolicy::DistributedExact;

        // Allow StrictAbort/LocalConservativeVacuum in MPI builds. If false,
        // validation requires DistributedExact when RICH_MPI is defined.
        bool allowApproximateMpiPeelOff = false;

        // Maximum number of MPI exchange rounds in the distributed ray
        // queue before marking remaining rays as failed.
        size_t maxDistributedExchangeRounds = 64;

        static const char* mpiRayPolicyName(MpiRayPolicy p)
        {
            switch (p)
            {
                case MpiRayPolicy::StrictAbort:              return "StrictAbort";
                case MpiRayPolicy::LocalConservativeVacuum:  return "LocalConservativeVacuum";
                case MpiRayPolicy::DistributedExact:         return "DistributedExact";
                default:                                     return "Unknown";
            }
        }

        bool writePerKindTallies = true;
    } peelOff;

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
    void NormalizeAndValidateConfig(RadiationIMCPostProcessConfig &config,
                        bool withCompton,
                        bool withMultigroupOpacity,
                        bool withRandomWalk,
                        bool withDDMC);

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
