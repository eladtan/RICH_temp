#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include "MonteCarloPhysics3D.hpp"

class RadiationIMC : public MonteCarloPhysics3D
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, const EquationOfState &eos, const RadiationOpacity &opacity, size_t newPhotonsPerCell, bool withHydro = false);

    std::vector<Particle> preStep(double fullDt) override;

    Functionality step(Particle &particle) override;

    void postStep(const std::vector<MCParticle> &particles, double fullDt) override;

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override;

    const std::vector<double> &getFactorFleck(void) const { return this->factorFleck; }

    const std::vector<double> &getPlanckOpacities(void) const { return this->planckOpacities; }

private:    
    std::vector<MCParticle> generateParticles(double fullDt);

    std::vector<double> factorFleck;
    std::vector<double> planckOpacities;

    bool withHydro;
    size_t newPhotonsPerCell;
};

#endif // RADIATION_IMC_HPP