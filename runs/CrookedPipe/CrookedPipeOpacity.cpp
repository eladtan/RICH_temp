#include "CrookedPipeOpacity.hpp"

CrookedPipeOpacity::CrookedPipeOpacity() = default;

double CrookedPipeOpacity::CalcPlanckOpacity(const ComputationalCell3D &cell) const
{
    if(cell.tracers[1] > 0.5)
    {
        return 0.2;
    }
    else
    {
        return 2000;
    }
}

double CrookedPipeOpacity::CalcScatteringOpacity(const ComputationalCell3D &cell) const
{
    return 0;
}
