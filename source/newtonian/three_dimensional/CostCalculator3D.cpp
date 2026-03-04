#ifdef RICH_MPI

#include "CostCalculator3D.hpp"

std::vector<double> CostCalculator3D::CalculateCost(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const
{
    return std::vector<double>(tess.GetPointNo(), 1.0);
}

#endif // RICH_MPI