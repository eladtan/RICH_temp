#ifndef MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP
#define MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP

#include "conj_grad_solve.hpp"
#include <functional>

class MultigroupDiffusionCoefficientCalculator {
public:

    MultigroupDiffusionCoefficientCalculator(std::vector<double> const& energy_groups_center_,
                                             std::vector<double> const& energy_groups_boundary_);

    virtual ~MultigroupDiffusionCoefficientCalculator() = default;

    virtual double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    virtual double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    virtual double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    std::vector<double> const energy_groups_center;
    std::vector<double> const energy_groups_boundary;
};

class GraySTAopacity : public MultigroupDiffusionCoefficientCalculator {
    private:
        std::vector<double> rho_, T_;
        std::vector<std::vector<double>> rossland_, planck_, scatter_;
    
    public:
        GraySTAopacity(std::string const file_directory);

        double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;
};

double interpolateTable(double const T, double const d, 
                        std::vector<double> const& T_, 
                        std::vector<double> const& rho_, 
                        std::vector<std::vector<double>> const& data,
		                double const T_high_slope = 0);


class AnalyticOpacity : public MultigroupDiffusionCoefficientCalculator {
    public:
        AnalyticOpacity(std::function<double(ComputationalCell3D const&, std::size_t const)> diffusion_coefficient_groups_function_,
                        std::function<double(ComputationalCell3D const&, std::size_t const)> sigma_absorption_groups_function_,
                        std::function<double(ComputationalCell3D const&, std::size_t const)> sigma_scattering_groups_function_,
                        std::vector<double> const& energy_groups_center_,
                        std::vector<double> const& energy_groups_boundary_);

        double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        std::function<double(ComputationalCell3D const&, std::size_t const)> const diffusion_coefficient_groups_function;
        std::function<double(ComputationalCell3D const&, std::size_t const)> const sigma_absorption_groups_function;
        std::function<double(ComputationalCell3D const&, std::size_t const)> const sigma_scattering_groups_function;
};

//! D=D0*rho^alpha*T^beta, sigma_planck=sigma_planck0*rho^alpha_planck*T^beta_planck
class GrayPowerLawOpacity : public MultigroupDiffusionCoefficientCalculator {
    private:
        double const D0_, alpha_, beta_, planck0_, alpha_planck_, beta_planck_;

    public:
        GrayPowerLawOpacity(double const D0, 
                            double const alpha, 
                            double const beta, 
                            double const planck0, 
                            double const alpha_planck, 
                            double const beta_planck);
        
        double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

};

#endif // MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP