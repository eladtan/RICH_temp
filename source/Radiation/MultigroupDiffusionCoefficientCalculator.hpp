#ifndef MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP
#define MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP

#include "conj_grad_solve.hpp"
#include <functional>

#include <boost/math/special_functions/pow.hpp>
// TODO: make a units namespace used by all the program 
#include "tau_matrix_calculator/src/units.hpp"
class MultigroupDiffusionCoefficientCalculator {
public:

    MultigroupDiffusionCoefficientCalculator(std::vector<double> const& energy_groups_center_,
                                             std::vector<double> const& energy_groups_boundary_);

    MultigroupDiffusionCoefficientCalculator() = default;

    virtual ~MultigroupDiffusionCoefficientCalculator() = default;

    virtual double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    virtual double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    virtual double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    std::vector<double> energy_groups_center;
    std::vector<double> energy_groups_boundary;
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

using boost::math::pow;
class FreeFreeAbsorptionOpacityMultigroup : public MultigroupDiffusionCoefficientCalculator {
    private:
        std::size_t const Z;
        
        static double constexpr m_e = CG::electron_mass;
        static double constexpr c = CG::speed_of_light;
        static double constexpr c2 = pow<2>(c);
        static double constexpr h = units::planck_constant;
        static double constexpr h_bar = h / (2.0*M_PI);
        static double constexpr electric_constant = 8.8541878188e-12 / 1e6 / 1e3;
        static double constexpr q_e = 4.8032e-10;
        static double constexpr pi = M_PI;
        static double constexpr kB = CG::boltzmann_constant;

        double const coefficient = 8.0 * pow<6>(q_e)/(3.0*std::sqrt(2.*pi) * std::pow(m_e, 1.5)*c*h);
    
    public:
        FreeFreeAbsorptionOpacityMultigroup(std::size_t const Z_,
                                            std::vector<double> const& energy_groups_center_,
                                            std::vector<double> const& energy_groups_boundary_);

        double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;

        double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const override;
};

#endif // MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP