#ifndef MC_PHYSICS_3D_HPP
#define MC_PHYSICS_3D_HPP

#include <array>
#include <boost/math/special_functions/pow.hpp>
#include "monte/physics/MonteCarloPhysics.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "CMMC/src/units/units.hpp"
#include "monte/utils/RandomInCell.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "RadiationOpacity.hpp"
#include "LorentzTransformation.hpp"

class MonteCarloRadiationPhysics3D : public STORM::MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = STORM::Particle<Vector3D, Tessellation3D>;
    using Functionality = STORM::StepResult<Vector3D, Tessellation3D>;
    using BoundaryCond = STORM::BoundaryCondition<Vector3D, Tessellation3D>;

    MonteCarloRadiationPhysics3D(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity);
    
    virtual Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const = 0;

    virtual std::vector<Particle> generateInitialParticles(size_t /*particlesPerCell*/) { return {}; }

    virtual void adjustExistingParticles(std::vector<Particle> &/*particles*/, double /*fullDt*/) {}

    virtual const std::vector<double> &getEradTimeAvg(void) const = 0;
    
    virtual std::vector<double> &getEradTimeAvg(void) = 0;

    virtual const std::vector<std::array<double, ENERGY_GROUPS_NUM>> &getEgTimeAvg(void) const = 0;

    virtual std::vector<std::array<double, ENERGY_GROUPS_NUM>> &getEgTimeAvg(void) = 0;

    void reseedRNG(uint64_t seed)
    {
        this->re.seed(seed);
        this->opacity->rng_.seed(seed + 1);
        ReseedRandomInCell(seed + 2);
    }

    const OpacityCalculator* getOpacity() const { return opacity.get(); }
    
protected:    
    std::vector<Particle> generateParticles(double fullDt);

    std::vector<ComputationalCell3D> &cells;
    std::vector<Conserved3D> &conserved;
    std::shared_ptr<EquationOfState> eos;
    std::shared_ptr<OpacityCalculator> opacity;
    std::uniform_real_distribution<double> dist;
    std::mt19937_64 re;
};

#endif // MC_PHYSICS_3D_HPP
