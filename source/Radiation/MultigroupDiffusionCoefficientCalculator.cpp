#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "misc/simple_io.hpp"

MultigroupDiffusionCoefficientCalculator
::MultigroupDiffusionCoefficientCalculator(std::vector<double> const& energy_groups_center_,
                                           std::vector<double> const& energy_groups_boundary_) : energy_groups_center(energy_groups_center_),
                                           energy_groups_boundary(energy_groups_boundary_) {}


double interpolateTable(double const T, 
                        double const d, 
                        std::vector<double> const& T_, 
                        std::vector<double> const& rho_, 
                        std::vector<std::vector<double>> const& data,
		                double const T_high_slope){

    constexpr std::size_t slope_length = 7;

    if(T < T_[0]){
        if(d < rho_[0]){
            double const d_slope = (data[0][slope_length - 1] -data[0][0]) / (rho_[slope_length - 1] - rho_[0]);
            double const T_slope = (data[slope_length - 1][0] -data[0][0]) / (T_[slope_length - 1] - T_[0]);
				
                return std::exp(data[0][0] + d_slope * (d - rho_[0]) + T_slope * (T - T_[0]));        
        } else {
            double const data_T0 = BiLinearInterpolation(T_, rho_, data, T_[0] * 1.00001, d);
            double const T_slope = (BiLinearInterpolation(T_, rho_, data, T_[slope_length - 1], d) - data_T0) / (T_[slope_length - 1] - T_[0]);
            
            return std::exp(BiLinearInterpolation(T_, rho_, data, T_[0] * 1.00001, d) + T_slope * (T - T_[0]));
        }
    } if(T > T_.back()){
        if(d < rho_[0]){
            double const d_slope = (data[T_.size() - 1][slope_length - 1] -data[T_.size() - 1][0]) / (rho_[slope_length - 1] - rho_[0]);
            
            return std::exp(data[T_.size() - 1][0] + d_slope * (d - rho_[0]) + T_high_slope * (T - T_.back())); 
        } else {
            return std::exp(BiLinearInterpolation(T_, rho_, data, T_.back() * 0.99999, d) + T_high_slope * (T - T_.back()));
        }
    }
    
    if(d < rho_[0]){
        double const data_d0 = BiLinearInterpolation(T_, rho_, data, T, rho_[0] * 0.9999);
        double const d_slope =(BiLinearInterpolation(T_, rho_, data, T, rho_[slope_length - 1]) - data_d0) / (rho_[slope_length - 1] - rho_[0]);
        
        return std::exp(BiLinearInterpolation(T_, rho_, data, T, rho_[0] * 0.9999) + d_slope * (d - rho_[0]));
    }

    return std::exp(BiLinearInterpolation(T_, rho_, data, T, d));
}

GraySTAopacity::GraySTAopacity(std::string const file_directory) : T_(), 
                                                                   rho_(), 
                                                                   rossland_(), 
                                                                   planck_(), 
                                                                   scatter_(),
                                                                   MultigroupDiffusionCoefficientCalculator(std::vector<double>(), std::vector<double>()){
    constexpr std::size_t Nmatrix = 128;
    
    std::vector<double> temp = read_vector(file_directory + "ross.txt");
    rossland_.resize(Nmatrix);
    for(std::size_t i = 0; i < Nmatrix; ++i){
        rossland_[i].resize(Nmatrix);
        for(std::size_t j=0; j < Nmatrix; ++j){
            rossland_[i][j] = temp[i * Nmatrix + j];
        }
    }

    temp = read_vector(file_directory + "planck.txt");
    planck_.resize(Nmatrix);
    for(std::size_t i=0; i < Nmatrix; ++i){
        planck_[i].resize(Nmatrix);
        for(std::size_t j=0; j < Nmatrix; ++j){
            planck_[i][j] = temp[i*Nmatrix + j];
        }
    }

    temp = read_vector(file_directory + "scatter.txt");
    scatter_.resize(Nmatrix);
    for(std::size_t i=0; i < Nmatrix; ++i){
        scatter_[i].resize(Nmatrix);
        for(std::size_t j=0; j < Nmatrix; ++j){
            scatter_[i][j] = temp[i*Nmatrix + j]; 
        }
    }

    T_ = read_vector(file_directory + "T.txt");
    rho_ = read_vector(file_directory + "rho.txt");
}

double GraySTAopacity::CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const {
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);

    return CG::speed_of_light / (3.0 * interpolateTable(T, d, T_, rho_, rossland_));
}

double GraySTAopacity::CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const {
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);

    return interpolateTable(T, d, T_, rho_, planck_, -3.5);
}

double GraySTAopacity::CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const {
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);
    return interpolateTable(T, d, T_, rho_, scatter_);
}