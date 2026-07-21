#ifndef MC_PHYSICS_3D_HPP
#define MC_PHYSICS_3D_HPP

#include <boost/math/special_functions/pow.hpp>
#include "monte/physics/MonteCarloPhysics.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/tessellation/utils/RandomInCell.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "RadiationOpacity.hpp"
#include "LorentzTransformation.hpp"

class MonteCarloPhysics3D : public MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    MonteCarloPhysics3D(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, const EquationOfState &eos, const RadiationOpacity &opacity);
    
    virtual MCParticle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const = 0;

protected:    
    std::vector<MCParticle> generateParticles(double fullDt);

    std::vector<ComputationalCell3D> &cells;
    std::vector<Conserved3D> &conserved;
    const EquationOfState &eos;
    const RadiationOpacity &opacity;
    std::uniform_real_distribution<double> dist;
    std::mt19937_64 re;
};

#endif // MC_PHYSICS_3D_HPP