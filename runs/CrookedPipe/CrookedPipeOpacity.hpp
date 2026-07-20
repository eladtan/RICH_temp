#ifndef CROOKED_PIPE_OPACITY_HPP
#define CROOKED_PIPE_OPACITY_HPP

#include <random>
#include "3D/radiation/RadiationOpacity.hpp"
#include "CMMC/src/units/units.hpp"
#include "3D/radiation/LorentzTransformation.hpp"

class CrookedPipeOpacity : public OpacityCalculator
{
public:
    CrookedPipeOpacity();

    inline ~CrookedPipeOpacity() override = default;

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override;

    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override;
};

#endif // CROOKED_PIPE_OPACITY_HPP
