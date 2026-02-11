#ifndef OPACITY_HPP
#define OPACITY_HPP

#include "3D/tessellation/Tessellation3D.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "monte/MonteCarloParticle.hpp"

class RadiationOpacity
{
public:
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

    RadiationOpacity() = default;

    virtual ~RadiationOpacity() = default;

    virtual double getPlanckOpacity(const ComputationalCell3D &cell) const = 0;

    virtual double getScatteringOpacity(const ComputationalCell3D &cell) const = 0;

    virtual Vector3D getRandomVelocity(const ComputationalCell3D &cell) const = 0;

    virtual Vector3D getNewScatterVelocity(const ComputationalCell3D &cell, const MCParticle &particle) const = 0;
};

#endif // OPACITY_HPP