#ifndef COST_CALCULATOR3D_HPP
#define COST_CALCULATOR3D_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

class CostCalculator3D
{
public:
    CostCalculator3D() = default;

    virtual ~CostCalculator3D() = default;

    virtual std::vector<double> CalculateCost(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const;

    virtual void Dump(size_t /*cycle*/) const {}
};

#endif // RICH_MPI

#endif // COST_CALCULATOR3D_HPP