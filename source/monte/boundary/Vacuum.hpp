#ifndef VACUUM_BOUNDARY_CONDITION_HPP
#define VACUUM_BOUNDARY_CONDITION_HPP

#include "BoundaryCondition.hpp"

template<typename T, typename Grid>
class VacuumBoundaryCondition : public BoundaryCondition<T, Grid>
{
public:
    using MCParticle = MonteCarloParticle<T, Grid>;

    explicit VacuumBoundaryCondition(const Grid &grid)
        : BoundaryCondition<T, Grid>(grid)
    {}

    MonteCarloParticleStatus apply(MCParticle &particle) override
    {
        escapedEnergy_ += particle.weight;
        return MonteCarloParticleStatus::REMOVE;
    }

    std::vector<MCParticle> generateNewBoundaryParticles(double) override
    {
        return {};
    }

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const override
    {
        (void)faceIdx;
        (void)insideCellIndex;
        (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::Unsupported;
    }

    double getEscapedEnergy() const
    {
        return escapedEnergy_;
    }

    void resetEscapedEnergy()
    {
        escapedEnergy_ = 0.0;
    }

private:
    double escapedEnergy_ = 0.0;
};

#endif // VACUUM_BOUNDARY_CONDITION_HPP
