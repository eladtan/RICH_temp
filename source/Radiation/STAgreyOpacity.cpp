#include "STAgreyOpacity.hpp"

STAgreyOpacity::STAgreyOpacity(std::string file_directory)
{
    size_t const Nmatrix = 128;
    std::vector<double> temp = read_vector(file_directory + "ross.txt");
    rossland_.resize(Nmatrix);
    for(size_t i = 0; i < Nmatrix; ++i)
    {
        rossland_[i].resize(Nmatrix);
        for(size_t j = 0; j < Nmatrix; ++j)
            rossland_[i][j] = temp[i * Nmatrix + j];
    }
    temp = read_vector(file_directory +"planck.txt");
    planck_.resize(Nmatrix);
    for(size_t i = 0; i < Nmatrix; ++i)
    {
        planck_[i].resize(Nmatrix);
        for(size_t j = 0; j < Nmatrix; ++j)
            planck_[i][j] = temp[i * Nmatrix + j];
    }
    temp = read_vector(file_directory +"scatter.txt");
    scatter_.resize(Nmatrix);
    for(size_t i = 0; i < Nmatrix; ++i)
    {
        scatter_[i].resize(Nmatrix);
        for(size_t j = 0; j < Nmatrix; ++j)
            scatter_[i][j] = temp[i * Nmatrix + j];
    }
    T_ = read_vector(file_directory +"T.txt");
    rho_ = read_vector(file_directory +"rho.txt");
}

double STAgreyOpacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell) const 
{
    double const T = std::log(cell.temperature);
    double d = std::log(cell.density);
    double d_ratio = 1;
    if(d < rho_[0])
    {
        double const scattering = CalcScatteringOpacity(cell);
        double sigma_rossland = Interpolate2DTable(T, d, T_, rho_, rossland_);
        return CG::speed_of_light / (3 * std::max(sigma_rossland, scattering));
    }
    if(d > rho_.back())
    {
        d_ratio = cell.density / std::exp(rho_.back());
        d = rho_.back();
    }
    return CG::speed_of_light / (3 * Interpolate2DTable(T, d, T_, rho_, rossland_) * d_ratio);
}

double STAgreyOpacity::CalcPlanckOpacity(ComputationalCell3D const& cell) const 
{
    double const T = std::log(cell.temperature);
    double d = std::log(cell.density);
    double d_ratio = 1;
    double d_slope = 2;
    if(d < rho_[0])
    {
        if(T > T_[0] && T < T_.back())
        {
            auto it = std::lower_bound(T_.begin(), T_.end(), T);
            auto idx = std::distance(T_.begin(), it);
            d_slope = (planck_[idx][10] - planck_[idx][0]) / (rho_[10] - rho_[0]);
            // d_slope = (planck_[10][idx] - planck_[0][idx]) / (rho_[10] - rho_[0]);
            if(d_slope < 0.35 || d_slope > 2.75)
            {
                std::cout<<"planck slope "<<d_slope<<" T "<<T<<" idx "<<idx<<std::endl;
                throw UniversalError("Planck opacity interpolation failed");
            }
        }
        d_ratio = cell.density / std::exp(rho_[0]);
        d = rho_[0];
    }
    if(d > rho_.back())
    {
        d_ratio = cell.density / std::exp(rho_.back());
        d = rho_.back();
    }
    return Interpolate2DTable(T, d, T_, rho_, planck_, -3.5) * std::pow(d_ratio, d_slope);
}

double STAgreyOpacity::CalcScatteringOpacity(ComputationalCell3D const& cell) const 
{
    double const T = std::log(cell.temperature);
    double d = std::log(cell.density);
    double d_ratio = 1;
    if(d < rho_[0])
    {
        d_ratio = cell.density / std::exp(rho_[0]);
        d = rho_[0];
    }
    if(d > rho_.back())
    {
        d_ratio = cell.density / std::exp(rho_.back());
        d = rho_.back();
    }
    return Interpolate2DTable(T, d, T_, rho_, scatter_) * d_ratio;
}