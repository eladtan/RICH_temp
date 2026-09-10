#ifndef RMTV_OPACITY_HPP
#define RMTV_OPACITY_HPP
#include "Radiation/OpacityCalculator.hpp"
#include "RMTVReference.hpp"

// Macroscopic cm^-1, not a mass opacity. Pure grey absorption.
class RMTVOpacity final : public OpacityCalculator
{
    double coefficient_, floor_, cap_;

  public:
    RMTVOpacity(const rmtv::Scales &s, double floor, double cap)
        : coefficient_(4 * units::arad * units::clight / (3 * s.chi0())), floor_(floor), cap_(cap)
    {
    }
    double raw(double rho, double T) const
    {
        if(!(rho > 0) || !(T >= 0) || !std::isfinite(rho) || !std::isfinite(T))
        {
            throw std::runtime_error("RMTV opacity received invalid material state");
        }
        return coefficient_ * rho * rho * std::pow(std::max(T, floor_), -3.5);
    }
    double CalcPlanckOpacity(const ComputationalCell3D &c) const override
    {
        return std::min(cap_, raw(c.density, c.temperature));
    }
    double CalcAbsorptionOpacity(const ComputationalCell3D &c, double) const override
    {
        return CalcPlanckOpacity(c);
    }
    double CalcPlanckOpacityAtTemperature(const ComputationalCell3D &c, double T) const override
    {
        return std::min(cap_, raw(c.density, T));
    }
    double CalcAbsorptionOpacityAtTemperature(const ComputationalCell3D &c, double,
                                              double T) const override
    {
        return CalcPlanckOpacityAtTemperature(c, T);
    }
    double CalcDiffusionCoefficient(const ComputationalCell3D &c) const override
    {
        return units::clight / (3 * CalcPlanckOpacity(c));
    }
};
#endif
