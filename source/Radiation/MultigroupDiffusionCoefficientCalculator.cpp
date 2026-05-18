#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "misc/simple_io.hpp"


double interpolateTable(double const T,
                        double const d,
                        std::vector<double> const& T_,
                        std::vector<double> const& rho_,
                        std::vector<std::vector<double>> const& data,
                        double const T_high_slope) {

    constexpr std::size_t slope_length = 7;

    if (T < T_[0]) {
        if (d < rho_[0]) {
            double const d_slope = (data[0][slope_length - 1] -data[0][0]) / (rho_[slope_length - 1] - rho_[0]);
            double const T_slope = (data[slope_length - 1][0] -data[0][0]) / (T_[slope_length - 1] - T_[0]);

            return std::exp(data[0][0] + d_slope * (d - rho_[0]) + T_slope * (T - T_[0]));
        } else {
            double const data_T0 = BiLinearInterpolation(T_, rho_, data, T_[0] * 1.00001, d);
            double const T_slope = (BiLinearInterpolation(T_, rho_, data, T_[slope_length - 1], d) - data_T0) / (T_[slope_length - 1] - T_[0]);

            return std::exp(BiLinearInterpolation(T_, rho_, data, T_[0] * 1.00001, d) + T_slope * (T - T_[0]));
        }
    } if (T > T_.back()) {
        if (d < rho_[0]) {
            double const d_slope = (data[T_.size() - 1][slope_length - 1] -data[T_.size() - 1][0]) / (rho_[slope_length - 1] - rho_[0]);

            return std::exp(data[T_.size() - 1][0] + d_slope * (d - rho_[0]) + T_high_slope * (T - T_.back()));
        } else {
            return std::exp(BiLinearInterpolation(T_, rho_, data, T_.back() * 0.99999, d) + T_high_slope * (T - T_.back()));
        }
    }

    if (d < rho_[0]) {
        double const data_d0 = BiLinearInterpolation(T_, rho_, data, T, rho_[0] * 0.9999);
        double const d_slope =(BiLinearInterpolation(T_, rho_, data, T, rho_[slope_length - 1]) - data_d0) / (rho_[slope_length - 1] - rho_[0]);

        return std::exp(BiLinearInterpolation(T_, rho_, data, T, rho_[0] * 0.9999) + d_slope * (d - rho_[0]));
    }

    return std::exp(BiLinearInterpolation(T_, rho_, data, T, d));
}

GraySTAopacity::GraySTAopacity(std::string const file_directory) :
    rho_(),
    T_(),
    rossland_(),
    planck_(),
    scatter_() {

    constexpr std::size_t Nmatrix = 128;

    std::vector<double> temp = read_vector(file_directory + "ross.txt");
    rossland_.resize(Nmatrix);
    for (std::size_t i = 0; i < Nmatrix; ++i) {
        rossland_[i].resize(Nmatrix);
        for (std::size_t j=0; j < Nmatrix; ++j) {
            rossland_[i][j] = temp[i * Nmatrix + j];
        }
    }

    temp = read_vector(file_directory + "planck.txt");
    planck_.resize(Nmatrix);
    for (std::size_t i=0; i < Nmatrix; ++i) {
        planck_[i].resize(Nmatrix);
        for (std::size_t j=0; j < Nmatrix; ++j) {
            planck_[i][j] = temp[i*Nmatrix + j];
        }
    }

    temp = read_vector(file_directory + "scatter.txt");
    scatter_.resize(Nmatrix);
    for (std::size_t i=0; i < Nmatrix; ++i) {
        scatter_[i].resize(Nmatrix);
        for (std::size_t j=0; j < Nmatrix; ++j) {
            scatter_[i][j] = temp[i*Nmatrix + j];
        }
    }

    T_ = read_vector(file_directory + "T.txt");
    rho_ = read_vector(file_directory + "rho.txt");
}

double GraySTAopacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const {
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);

    return CG::speed_of_light / (3.0 * interpolateTable(T, d, T_, rho_, rossland_));
}

double GraySTAopacity::CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const {
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);

    return interpolateTable(T, d, T_, rho_, planck_, -3.5);
}

double GraySTAopacity::CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const {
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);
    return interpolateTable(T, d, T_, rho_, scatter_);
}

AnalyticOpacity::AnalyticOpacity(std::function<double(ComputationalCell3D const& cell, double)> diffusion_coefficient_groups_function_,
                                 std::function<double(ComputationalCell3D const& cell, double)> sigma_absorption_groups_function_,
                                 std::function<double(ComputationalCell3D const& cell, double)> sigma_scattering_groups_function_,
                                 std::vector<double> const& energy_groups_center_,
                                 std::vector<double> const& energy_groups_boundary_) :
    diffusion_coefficient_groups_function(diffusion_coefficient_groups_function_),
    sigma_absorption_groups_function(sigma_absorption_groups_function_),
    sigma_scattering_groups_function(sigma_scattering_groups_function_) {
    energy_groups_center = energy_groups_center_;
    energy_groups_boundary = energy_groups_boundary_;
}

double AnalyticOpacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell,
                                                      double energy) const {
    return diffusion_coefficient_groups_function(cell, energy);
}

double AnalyticOpacity::CalcAbsorptionOpacity(ComputationalCell3D const& cell,
                                                       double energy) const {
    return sigma_absorption_groups_function(cell, energy);
}

double AnalyticOpacity::CalcScatteringOpacity(ComputationalCell3D const& cell,
                                                       double energy) const {
    return sigma_scattering_groups_function(cell, energy);
}

GrayPowerLawOpacity::GrayPowerLawOpacity(double const D0,
                                         double const alpha,
                                         double const beta,
                                         double const planck0,
                                         double const alpha_planck,
                                         double const beta_planck) :
    D0_(D0),
    alpha_(alpha),
    beta_(beta),
    planck0_(planck0),
    alpha_planck_(alpha_planck),
    beta_planck_(beta_planck) {
}

double GrayPowerLawOpacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const {
    return D0_ * std::pow(cell.density, alpha_) * std::pow(cell.temperature, beta_);
}

double GrayPowerLawOpacity::CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const {
    return planck0_ * std::pow(cell.density, alpha_planck_) * std::pow(cell.temperature, beta_planck_);
}

double GrayPowerLawOpacity::CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const {
    return 0.0;
}


FreeFreeAbsorptionOpacityMultigroup::FreeFreeAbsorptionOpacityMultigroup(double const Z_,
                                                                         std::vector<double> const& energy_groups_center_,
                                                                         std::vector<double> const& energy_groups_boundary_) :
    Z(Z_) {
    energy_groups_center = energy_groups_center_;
    energy_groups_boundary = energy_groups_boundary_;
}

double FreeFreeAbsorptionOpacityMultigroup::CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const {
    return CG::speed_of_light / (3.0 * (CalcAbsorptionOpacity(cell, energy) + CalcScatteringOpacity(cell, energy)));
}

double FreeFreeAbsorptionOpacityMultigroup::CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const {
    double const nu_g = energy/h;
    double const e = energy;
    double const kT = kB*cell.temperature;

    double const m_p = 1.6726231e-24;
    double const n_i = cell.density/m_p;
    double const n_e = n_i;

    double g_ff = std::max(1.0, std::log(std::exp(5.960) * std::pow(cell.temperature, 1.5) / (nu_g * Z)));

    return g_ff * 3.7e8*Z*Z*Z*std::pow(cell.density*6.02214076e23, 2)/std::sqrt(cell.temperature)*(1.-std::exp(-e/(CG::boltzmann_constant*cell.temperature)))*std::pow(e/h, -3);
}

double FreeFreeAbsorptionOpacityMultigroup::CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const {
    return 0.0;
}

ZeroAbsorptionZeroDiffusionMultigroup::ZeroAbsorptionZeroDiffusionMultigroup(
    std::vector<double> const& energy_groups_center_,
    std::vector<double> const& energy_groups_boundary_
) {
    energy_groups_center = energy_groups_center_;
    energy_groups_boundary = energy_groups_boundary_;
}
