#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "MultigroupDiffusionBoundaryCalculator.hpp"

using namespace CG;

class MultigroupDiffusion : public RadiationDriver {
public:
    MultigroupDiffusion(std::vector<double> const& energy_groups_center_, 
                        std::vector<double> const& energy_groups_boundary_,
                        MultigroupDiffusionCoefficientCalculator const& coefficient_calc,
                        MultigroupDiffusionBoundaryCalculator const& boundary_calc, 
                        EquationOfState const& eos,
                        std::vector<std::string> const zero_cells,
                        bool const flux_limiter,
                        bool const hydro_on,
                        bool const compton_on);

    ~MultigroupDiffusion() = default;

    bool prestep(Tessellation3D const& tess) const override;

    bool step(double const tolerance, 
              int& total_iters, 
              Tessellation3D const& tess, 
              std::vector<ComputationalCell3D>& cells,
              std::vector<Conserved3D>& extensives,
              double const dt,
              double const time) const override;

    bool poststep() const override;

    void BuildMatrix(Tessellation3D const& tess, 
                     mat& A, 
                     size_t_mat& A_indeces, 
                     std::vector<ComputationalCell3D> const& cells, 
                     double const dt, 
                     std::vector<double>& b, 
                     std::vector<double>& x0, 
                     double const current_time) const override;

    void PostCG(Tessellation3D const& tess, 
                std::vector<Conserved3D>& extensives, 
                double const dt, 
                std::vector<ComputationalCell3D>& cells,
                std::vector<double>const& CG_result, 
                std::vector<double> const&  full_CG_result) const override;

    MultigroupDiffusionCoefficientCalculator const& coefficient_calculator;
    MultigroupDiffusionBoundaryCalculator const& boundary_calculator;
    
    std::vector<double> const energy_groups_center;
    std::vector<double> const energy_groups_boundary;

    mutable std::size_t current_group;
    mutable bool gray;

    mutable std::vector<ComputationalCell3D> cells_temp;
    mutable std::vector<ComputationalCell3D> cells_cgs;
    mutable std::vector<Conserved3D> extensives_temp;

    mutable std::vector<std::vector<double>> sigma_absorption_group; // [group][cell]
    mutable std::vector<std::vector<double>> sigma_scattering_group; // [group][cell]
    mutable std::vector<std::vector<double>> planck_integal_group; // [group][cell]
    mutable std::vector<std::vector<double>> R2_group; // [group][cell]
    mutable std::vector<std::vector<double>> cell_flux_limiter_group; // [group][cell]

    mutable std::vector<double> sigma_absorption_planck;
    mutable std::vector<double> sigma_absorption_average;
    mutable std::vector<double> sigma_scattering_gray;
    mutable std::vector<double> fleck_factor;
    
    
    mutable std::vector<double> new_Eg; 
    mutable std::vector<double> new_Eg_full; 

    mutable std::vector<double> new_Er;
    mutable std::vector<double> new_Er_full; 

    mutable std::vector<double> max_abs_grad_E;
    mutable std::vector<double> max_neighbor_abs_grad_E;

    mutable std::vector<Vector3D> grad; // gradient ij for i < j
private:
    void BuildMatrixGroup(std::size_t group,
                          Tessellation3D const& tess, 
                          mat& A, 
                          size_t_mat& A_indeces, 
                          std::vector<ComputationalCell3D> const& cells, 
                          double const dt, 
                          std::vector<double>& b, 
                          std::vector<double>& x0, 
                          double const current_time) const;

    void BuildMatrixGray(Tessellation3D const& tess, 
                         mat& A, 
                         size_t_mat& A_indeces, 
                         std::vector<ComputationalCell3D> const& cells, 
                         double const dt, 
                         std::vector<double>& b, 
                         std::vector<double>& x0, 
                         double const current_time) const; 

    void PostCGGroup(std::size_t const group,
                     Tessellation3D const& tess, 
                     std::vector<Conserved3D>& extensives, 
                     double const dt, 
                     std::vector<ComputationalCell3D>& cells,
                     std::vector<double>const& CG_result, 
                     std::vector<double> const&  full_CG_result) const;
    
    void PostCGGray(Tessellation3D const& tess, 
                     std::vector<Conserved3D>& extensives, 
                     double const dt, 
                     std::vector<ComputationalCell3D>& cells,
                     std::vector<double>const& CG_result, 
                     std::vector<double> const&  full_CG_result) const;

    void calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                std::vector<ComputationalCell3D> const& cells) const;

    void calculate_planck_integrals(Tessellation3D const& tess,
                                    std::vector<ComputationalCell3D> const& cells) const;

    void calculate_gray_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                               std::vector<ComputationalCell3D> const& cells) const;
};


#endif