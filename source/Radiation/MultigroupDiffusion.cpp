#include "MultigroupDiffusion.hpp"

MultigroupDiffusion::MultigroupDiffusion(MultigroupDiffusionCoefficientCalculator const& D_coefficient_calc, 
                                         EquationOfState const& eos,
                                         std::vector<std::string> const zero_cells,
                                         bool const flux_limiter,
                                         bool const hydro_on,
                                         bool const compton_on):
                                                                D_coefficient_calculator(D_coefficient_calc),
                                                                current_group(0),
                                                                gray(false),
                                                                cells_temp(),
                                                                extensives_temp(),
                                                                sigma_planck_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                D_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                R2_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                cell_flux_limiter_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                new_Eg(),
                                                                new_Eg_full(),
                                                                new_Er(),
                                                                new_Er_full(),
                                                                RadiationDriver(eos,
                                                                                zero_cells,
                                                                                flux_limiter,
                                                                                hydro_on,
                                                                                compton_on) {}

MultigroupDiffusion::~MultigroupDiffusion() {}

bool MultigroupDiffusion::prestep(Tessellation3D const& tess) const {
    auto const N = tess.GetPointNo();

    sigma_planck_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    D_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    R2_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    cell_flux_limiter_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    
    new_Eg.resize(N, 0.0);
    new_Eg_full.resize(N, 0.0);
    new_Er.resize(N, 0.0);
    new_Er_full.resize(N, 0.0);

    return true;
}

bool MultigroupDiffusion::poststep() const {
    return true;
}

bool MultigroupDiffusion::step(double const tolerance, 
                               int& total_iters, 
                               Tessellation3D const& tess, 
                               std::vector<ComputationalCell3D>& cells,
                               std::vector<Conserved3D>& extensives,
                               double const dt,
                               double const time) const {

    return true;
}



void MultigroupDiffusion::BuildMatrix(Tessellation3D const& tess, 
                                      mat& A, 
                                      size_t_mat& A_indeces, 
                                      std::vector<ComputationalCell3D> const& cells, 
                                      double const dt, 
                                      std::vector<double>& b, 
                                      std::vector<double>& x0, 
                                      double const current_time) const {

}

void MultigroupDiffusion::PostCG(Tessellation3D const& tess, 
                                 std::vector<Conserved3D>& extensives, 
                                 double const dt, 
                                 std::vector<ComputationalCell3D>& cells,
                                 std::vector<double>const& CG_result, 
                                 std::vector<double> const&  full_CG_result) const {

}
