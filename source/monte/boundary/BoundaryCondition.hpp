#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include "monte/MonteCarloParticle.hpp"
#include "monte/MonteCarloParticleStatus.hpp"
#include <cmath>

// DDMCBoundaryFaceBehavior describes how DDMC should treat an outside-box
// face adjacent to a candidate DDMC cell.
//
// Unsupported:
//   DDMC does not know how to model this boundary face. The cell must be
//   excluded from DDMC acceleration.
//
// ReflectingRigid:
//   The face is a rigid reflecting wall. In DDMC this is a zero-normal-current
//   boundary. No leakage event should be added through this face, but the cell
//   should not be excluded solely because of this face.
enum class DDMCBoundaryFaceBehavior {
    Unsupported,
    ReflectingRigid
};

template<typename T, typename Grid>
class BoundaryCondition
{
public:
    BoundaryCondition(const Grid &grid);

    virtual ~BoundaryCondition() = default;

    virtual MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) = 0;

    virtual std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) = 0;

    virtual DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const
    {
        (void)faceIdx;
        (void)insideCellIndex;
        (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::Unsupported;
    }

protected:
    const Grid &grid;

    bool getDDMCOrientedOutwardNormal(
        size_t /*faceIdx*/,
        size_t insideCellIndex,
        size_t outsidePointIndex,
        T &nOut) const
    {
        nOut = this->grid.GetMeshPoint(outsidePointIndex) -
               this->grid.GetMeshPoint(insideCellIndex);

        double const nNorm = abs(nOut);
        if (!(nNorm > 0.0) || !std::isfinite(nNorm))
            return false;

        nOut *= 1.0 / nNorm;
        return true;
    }
};

template<typename T, typename Grid>
BoundaryCondition<T, Grid>::BoundaryCondition(const Grid &grid)
    : grid(grid)
{}

#endif // BOUNDARY_CONDITION_HPP