#ifndef HOHLRAUM_OPACITY_HPP
#define HOHLRAUM_OPACITY_HPP

#include <random>
#include "3D/radiation/RadiationOpacity.hpp"
#include "CMMC/src/units/units.hpp"
#include "3D/radiation/LorentzTransformation.hpp"

class HohlraumOpacity : public OpacityCalculator
{
public:
    HohlraumOpacity();

    inline ~HohlraumOpacity() override = default;

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override;

    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override;

    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energyGroup) const override;
};

#endif // HOHLRAUM_OPACITY_HPP
