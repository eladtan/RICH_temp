#ifndef OPACITY_CALCULATOR_HPP
#define OPACITY_CALCULATOR_HPP

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "3D/tessellation/Tessellation3D.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "misc/universal_error.hpp"
#include "CMMC/src/units/units.hpp"
#ifdef RICH_MPI
#include <mpi.h>
#endif

class OpacityCalculator
{
public:
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

    OpacityCalculator()
    {
        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        rng_ = std::mt19937_64(static_cast<uint64_t>(rank) * 3 + 1);
    }
    virtual ~OpacityCalculator() = default;

    virtual double CalcPlanckOpacity(ComputationalCell3D const& cell) const
    { throw UniversalError("CalcPlanckOpacity not implemented"); }

    virtual double CalcScatteringOpacity(ComputationalCell3D const& cell) const
    { return 0.0; }

    virtual double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const
    { throw UniversalError("CalcDiffusionCoefficient not implemented"); }

    virtual double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const
    { throw UniversalError("CalcAbsorptionOpacity not implemented"); }

    virtual double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const
    { throw UniversalError("CalcScatteringOpacity(energy) not implemented"); }

    virtual bool ComptonIncludedInTransport() const { return false; }

    virtual double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const
    { throw UniversalError("CalcDiffusionCoefficient(energy) not implemented"); }

    virtual Vector3D getRandomVelocity(ComputationalCell3D const& /*cell*/) const
    {
        // Sampling three Cartesian components uniformly and normalizing maps a
        // cube onto the sphere, overpopulating the body diagonals. Uniform
        // solid angle requires uniform mu = cos(theta) and azimuth.
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        double const mu = 1.0 - 2.0 * unit(rng_);
        double const phi = 2.0 * std::acos(-1.0) * unit(rng_);
        double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        return Vector3D(sinTheta * std::cos(phi),
                        sinTheta * std::sin(phi),
                        mu) * units::clight;
    }

    virtual Vector3D getNewScatterVelocity(ComputationalCell3D const& cell, MCParticle& particle) const
    {
        return getRandomVelocity(cell);
    }

    std::size_t findGroup(double energy) const
    {
        if (energy_groups_boundary.empty())
            return 0;
        auto it = std::upper_bound(energy_groups_boundary.begin(), energy_groups_boundary.end(), energy);
        if (it == energy_groups_boundary.begin())
            return 0;
        std::size_t idx = static_cast<std::size_t>(std::distance(energy_groups_boundary.begin(), it)) - 1;
        if (idx >= energy_groups_center.size())
            idx = energy_groups_center.size() - 1;
        return idx;
    }

    std::vector<double> energy_groups_center;
    std::vector<double> energy_groups_boundary;
    mutable std::mt19937_64 rng_;
};

#endif // OPACITY_CALCULATOR_HPP
