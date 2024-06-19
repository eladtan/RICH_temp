#ifndef MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP
#define MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP

#include "conj_grad_solve.hpp"

class MultigroupDiffusionCoefficientCalculator {
public:

    MultigroupDiffusionCoefficientCalculator(std::vector<double> const& energy_groups_center_,
                                             std::vector<double> const& energy_groups_boundary_);

    virtual ~MultigroupDiffusionCoefficientCalculator() = default;

    virtual double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const =0;

    virtual double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    virtual double CalcScatteringCoefficientGroup(ComputationalCell3D const&, std::size_t const group) const = 0;

    std::vector<double> const energy_groups_center;
    std::vector<double> const energy_groups_boundary;
};

class GraySTAopacity : public MultigroupDiffusionCoefficientCalculator {
    private:
        std::vector<double> rho_, T_;
        std::vector<std::vector<double>> rossland_, planck_, scatter_;
    
    public:
        GraySTAopacity(std::string const file_directory);

        double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const;

        double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const;

        double CalcScatteringCoefficientGroup(ComputationalCell3D const&, std::size_t const group) const;
};

double interpolateTable(double const T, double const d, 
                        std::vector<double> const& T_, 
                        std::vector<double> const& rho_, 
                        std::vector<std::vector<double>> const& data,
		                double const T_high_slope = 0);

// class PowerLawOpacity : public MultigroupDiffusionCoefficientCalculator {
    
// };

#endif // MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP