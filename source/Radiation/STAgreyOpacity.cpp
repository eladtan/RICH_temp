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
    double const d = std::log(cell.density);
    return CG::speed_of_light / (3 * Interpolate2DTable(T, d, T_, rho_, rossland_));
}

double STAgreyOpacity::CalcPlanckOpacity(ComputationalCell3D const& cell) const 
{
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);
    return Interpolate2DTable(T, d, T_, rho_, planck_, -3.5);
}

double STAgreyOpacity::CalcScatteringOpacity(ComputationalCell3D const& cell) const 
{
    double const T = std::log(cell.temperature);
    double const d = std::log(cell.density);
    return Interpolate2DTable(T, d, T_, rho_, scatter_);
}