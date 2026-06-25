#ifndef MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP
#define MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP

#include "OpacityCalculator.hpp"
#include "conj_grad_solve.hpp"
#include <functional>

#include <boost/math/special_functions/pow.hpp>
#include "CMMC/src/units/units.hpp"


class GraySTAopacity : public OpacityCalculator {
private:
    std::vector<double> rho_, T_;
    std::vector<std::vector<double>> rossland_, planck_, scatter_;

public:
    GraySTAopacity(std::string const file_directory);

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override;

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override;

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override;
};

double interpolateTable(double const T, double const d,
                        std::vector<double> const& T_,
                        std::vector<double> const& rho_,
                        std::vector<std::vector<double>> const& data,
                        double const T_high_slope = 0);


class AnalyticOpacity : public OpacityCalculator {
public:
    AnalyticOpacity(std::function<double(ComputationalCell3D const&, double)> diffusion_coefficient_groups_function_,
                    std::function<double(ComputationalCell3D const&, double)> sigma_absorption_groups_function_,
                    std::function<double(ComputationalCell3D const&, double)> sigma_scattering_groups_function_,
                    std::vector<double> const& energy_groups_center_,
                    std::vector<double> const& energy_groups_boundary_);

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override;

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override;

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override;

    std::function<double(ComputationalCell3D const&, double)> const diffusion_coefficient_groups_function;
    std::function<double(ComputationalCell3D const&, double)> const sigma_absorption_groups_function;
    std::function<double(ComputationalCell3D const&, double)> const sigma_scattering_groups_function;
};

//! D=D0*rho^alpha*T^beta, sigma_planck=sigma_planck0*rho^alpha_planck*T^beta_planck
class GrayPowerLawOpacity : public OpacityCalculator {
private:
    double const D0_, alpha_, beta_, planck0_, alpha_planck_, beta_planck_;

public:
    GrayPowerLawOpacity(double const D0,
                        double const alpha,
                        double const beta,
                        double const planck0,
                        double const alpha_planck,
                        double const beta_planck);

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override;

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override;

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override;

};

using boost::math::pow;
class FreeFreeAbsorptionOpacityMultigroup : public OpacityCalculator {
private:
    double const Z;
    bool const include_plasma_cutoff_;
    bool const use_free_free_cgs_formula_;

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
    FreeFreeAbsorptionOpacityMultigroup(double const Z_,
                                        std::vector<double> const& energy_groups_center_,
                                        std::vector<double> const& energy_groups_boundary_,
                                        bool include_plasma_cutoff = false,
                                        bool use_free_free_cgs_formula = false);

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override;

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override;

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override;
};

class ZeroAbsorptionZeroDiffusionMultigroup : public OpacityCalculator {
public:
    ZeroAbsorptionZeroDiffusionMultigroup(
        std::vector<double> const& energy_groups_center_,
        std::vector<double> const& energy_groups_boundary_
    );

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override { return std::sqrt(std::numeric_limits<double>::min()*1e50); }

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override { return std::numeric_limits<double>::min()*1e50; }

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override { return std::numeric_limits<double>::min()*1e50; }
};

using MultigroupDiffusionCoefficientCalculator = OpacityCalculator;

#endif // MULTI_GROUP_DIFFUSION_COEFFICIENT_CALCULATOR_HPP
