#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include "planck_integral/planck_integral.hpp"

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
                        bool const compton_on,
                        bool const doppler_on,
                        bool const mix_frame_on);

    ~MultigroupDiffusion() = default;

    double GetLengthScale() const override { return length_scale_; }

    bool prestep(Tessellation3D const& tess,
                 std::vector<ComputationalCell3D> const& cells) const override;

    bool step(double const tolerance, 
              int& total_iters, 
              Tessellation3D const& tess, 
              std::vector<ComputationalCell3D>& cells,
              std::vector<Conserved3D>& extensives,
              double const dt,
              double const time) const override;

    bool poststep() const override;

    double calculate_dt(double const dt,
                        Tessellation3D& tess,
                        std::vector<ComputationalCell3D>& cells) const override;

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

    void solve_doppler_shift(Tessellation3D const& tess,
                             std::vector<ComputationalCell3D>& cells,
                             double const dt, std::vector<Conserved3D>& extensives) const;
                             

    MultigroupDiffusionCoefficientCalculator const& coefficient_calculator;
    MultigroupDiffusionBoundaryCalculator const& boundary_calculator;
    
    std::vector<double> const energy_groups_center;
    std::vector<double> const energy_groups_boundary;
    std::vector<double> energy_groups_width;

    mutable std::size_t current_group;
    mutable bool gray;

    mutable std::vector<ComputationalCell3D> cells_temp;
    mutable std::vector<ComputationalCell3D> cells_cgs;
    mutable std::vector<Conserved3D> extensives_temp;

    mutable std::vector<std::vector<double>> sigma_absorption_group; // [group][cell]
    mutable std::vector<std::vector<double>> sigma_scattering_group; // [group][cell]
    mutable std::vector<std::vector<double>> planck_integal_group; // [group][cell]

    mutable std::vector<double> sigma_absorption_planck;
    mutable std::vector<double> sigma_absorption_average;
    mutable std::vector<double> sigma_scattering_gray;
    mutable std::vector<double> fleck_factor;

    mutable std::vector<double> new_Eg; 
    mutable std::vector<double> new_Eg_full; 

    mutable std::vector<std::vector<double>> old_Eg;

    mutable std::vector<double> new_Er;
    mutable std::vector<double> new_Er_full; 

    mutable std::vector<double> old_Er;
    mutable std::vector<double> old_Tm; 

    mutable std::vector<double> max_abs_grad_E;
    mutable std::vector<double> max_neighbor_abs_grad_E;

    mutable std::vector<Vector3D> grad; // gradient ij for i < j
    
    mutable std::vector<std::pair<double, double>> lambda_face_gray;
    mutable std::vector<std::pair<double, double>> sigma_ratio_lambda_face_gray;
    mutable std::vector<double> lambda_cell_gray;
    mutable std::vector<double> sigma_ratio_lambda_cell_gray;


    bool const doppler_on_;
    bool const mix_frame_on_;
    // for doppler step
    mutable std::vector<std::vector<double>> R2;
    mutable std::vector<std::vector<double>> D;

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

    void BuildMatrixGroupFull(Tessellation3D const& tess, 
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

    void PostCGFull(Tessellation3D const& tess, 
                     std::vector<Conserved3D>& extensives, 
                     double const dt, 
                     std::vector<ComputationalCell3D>& cells,
                     std::vector<double>const& CG_result, 
                     std::vector<double> const&  full_CG_result) const;

    void calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                std::vector<ComputationalCell3D> const& cells,
                                                                double const dt) const;

    void calculate_planck_integrals(Tessellation3D const& tess,
                                    std::vector<ComputationalCell3D> const& cells) const;

    void calculate_gray_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                               std::vector<ComputationalCell3D> const& cells) const;

    // helper functions
    void calculate_fleck_factor(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double dt_cgs) const;

    void calculate_lambda_g_and_R2_g(std::size_t const group,
                            Tessellation3D const& tess,
                            std::vector<ComputationalCell3D> const& cells,
                            std::vector<double>& lambda_g,
                            std::vector<double>& R2_g) const; 
};



#endif