#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include "CMMC/src/compton_matrix_mc.hpp"

using namespace CG;
using namespace compton_matrix_mc;

class MultigroupDiffusion : public RadiationDriver {
public:
    /**
     * @brief Constructor for the MultigroupDiffusion class.
     *
     * Initializes the class with the given parameters and sets up the necessary data structures.
     *
     * @param energy_groups_center_ Center energies of the energy groups.
     * @param energy_groups_boundary_ Boundary energies of the energy groups.
     * @param coefficient_calc Reference to the MultigroupDiffusionCoefficientCalculator object.
     * @param boundary_calc Reference to the MultigroupDiffusionBoundaryCalculator object.
     * @param eos Reference to the EquationOfState object.
     * @param zero_cells Vector of strings representing the names of cells with zero radiation.
     * @param flux_limiter Boolean indicating whether to use flux limiter in the diffusion calculation.
     * @param hydro_on Boolean indicating whether hydrodynamic effects are on.
     * @param compton_on Boolean indicating whether Compton scattering is on.
     * @param doppler_on Boolean indicating whether Doppler shift correction is on.
     * @param mix_frame_on Boolean indicating whether to use mixed frame in the diffusion calculation.
     * @param minimum_temperature Minimum temperature for the diffusion calculation. Default value is -1.
     * @param protections_on Enables protections in the diffusion calculation (modifies coupling strength). Default value is true.
     */
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
                        bool const mix_frame_on,
                        double const minimum_temperature = -1,
                        bool const protections_on = true);

    /**
     * @brief Destructor for the MultigroupDiffusion class.
     *
     * Default destructor, does nothing.
     */
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
                std::vector<double> const& full_CG_result) const override;

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
    mutable std::vector<std::vector<double>> planck_integal_group;   // [group][cell]

    mutable std::vector<double> sigma_absorption_planck;
    mutable std::vector<double> sigma_absorption_average;
    mutable std::vector<double> sigma_scattering_gray;
    mutable std::vector<double> fleck_factor;

    mutable std::vector<double> new_Eg;
    mutable std::vector<double> new_Eg_full;

    mutable std::vector<std::vector<double>> old_Eg;

    mutable std::vector<double> old_Er;
    mutable std::vector<double> old_Tm;

    mutable std::vector<double> max_abs_grad_E;
    mutable std::vector<double> max_neighbor_abs_grad_E;

    mutable std::vector<Vector3D> grad; // gradient ij for i < j

    bool const doppler_on_;
    double const minimum_temperature_;
    // for doppler step
    mutable std::vector<std::vector<double>> R2;
    mutable std::vector<std::vector<double>> D;

    mutable ComptonMatrixMC compton_matrix_gen;

    mutable std::vector<std::vector<double>> tau;
    mutable std::vector<std::vector<double>> dtau_dUm;
    mutable std::vector<std::vector<double>> S;
    mutable std::vector<std::vector<double>> dSdUm;

    mutable std::vector<double> n, n_bg; // occupancy number
    mutable std::size_t cell_id_of_compton_matrices;

    mutable std::vector<double> Gammas;

    mutable bool small_rel_diff = 0.0;

    mutable std::vector<double> Q_vector;
    mutable std::vector<double> Upsilon_vector;
    mutable std::vector<double> sum_dSdUm;

private:
    bool const protections_on_;
    mutable bool displayed_warning_;

    void calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                std::vector<ComputationalCell3D> const& cells,
                                                                double const dt) const;

    void calculate_planck_integrals(Tessellation3D const& tess,
                                    std::vector<ComputationalCell3D> const& cells) const;

    void calculate_gray_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                               std::vector<ComputationalCell3D> const& cells) const;

    // helper functions
    void calculate_fleck_factor(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double dt_cgs) const;

    void generate_S_and_dSdUm_matrices(ComputationalCell3D const& cell, std::size_t const cell_index, double const dt_cgs) const;

    double calculate_Upsilon(ComputationalCell3D const& cell) const;

    void calculate_compton_quantities(ComputationalCell3D const& cell, std::size_t const cell_index) const;

    double get_implicit_compton_contribution(Tessellation3D const& tess, ComputationalCell3D const& cell, std::size_t const cell_index, std::size_t const g, std::size_t const gt, double const dt_cgs) const;

    double get_implicit_compton_contribution_to_b(Tessellation3D const& tess, ComputationalCell3D const& cell, std::size_t const cell_index, std::size_t const g, double const dt_cgs) const;

    double GetDopplerSlope(ComputationalCell3D const& cell, size_t const g, bool const expansion) const;
};

#endif