#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <boost/math/special_functions/pow.hpp>
#include "monte/physics/MonteCarloPhysics.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/tesselation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/tesselation/utils/RandomInCell.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "RadiationOpacity.hpp"

class RadiationIMC : public MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, const EquationOfState &eos, const RadiationOpacity &opacity, size_t newPhotonsPerCell);

    std::vector<Particle> preStep(double fullDt) override;

    Functionality step(Particle &particle) override;

    void postStep(const std::vector<MCParticle> &particles) override;

private:    
    std::vector<MCParticle> generateParticles(double fullDt);

    std::vector<ComputationalCell3D> &cells;
    std::vector<Conserved3D> &conserved;
    const EquationOfState &eos;
    const RadiationOpacity &opacity;
    std::vector<double> factorFleck;
    std::vector<double> planckOpacities;
    std::uniform_real_distribution<double> dist;
    std::mt19937_64 re;
    size_t newPhotonsPerCell;
};

#endif // RADIATION_IMC_HPP