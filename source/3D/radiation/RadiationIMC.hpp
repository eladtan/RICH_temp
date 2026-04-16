#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <memory>
#include "MonteCarloPhysics3D.hpp"
#include "MultigroupOpacity.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "RandomWalk.hpp"

struct RadiationIMCParameters
{
    size_t newPhotonsPerCell;
    bool withHydro = false;
    bool diffusionPressureGradient = false;
    bool MMC = false;
    bool withMultigroupOpacity = false;
    bool withRandomWalk = false;
    double rwMinCellOpticalDepth = 25.0;
    double rwMinParticleOpticalDepth = 5.0;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;

    friend std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters &parameters);
};

class RadiationIMC : public MonteCarloRadiationPhysics3D
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters);

    std::vector<Particle> preStep(double fullDt) override;

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override;

    void postStep(const std::vector<Particle> &particles, double fullDt) override;

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override;

    std::vector<Particle> generateInitialParticles(size_t particlesPerCell) override;

    void adjustExistingParticles(std::vector<Particle> &particles, double fullDt) override;

    inline const std::vector<double> &getFactorFleck(void) const{return this->factorFleck;}

    inline const std::vector<double> &getPlanckOpacities(void) const {return this->planckOpacities;}

    size_t lastGenSlab = 0;
    size_t lastGenVacuum = 0;

private:    
    std::vector<Particle> generateParticles(double fullDt);

    std::vector<double> factorFleck;
    std::vector<double> planckOpacities;
    std::shared_ptr<MultigroupOpacity> multigroupOpacity;

    bool withHydro;
    bool diffusionPressureGradient;
    bool MMC;
    size_t newPhotonsPerCell;
    bool withRandomWalk;
    double rwMinCellOpticalDepth;
    double rwMinParticleOpticalDepth;
    bool noHydroFeedback;
    bool withEgTimeAvg;

    std::unique_ptr<RandomWalk> randomWalk;
    std::vector<bool> rwCellEligible;
    std::vector<double> rwCellTotalOpacity;
    std::vector<PGRWCellData> rwCellData;
    size_t rwStepCount = 0;

    bool tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void precomputeRandomWalkData();
};

#endif // RADIATION_IMC_HPP
