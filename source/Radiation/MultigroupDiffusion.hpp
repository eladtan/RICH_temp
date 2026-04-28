#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include "CMMC/src/compton_matrix_mc.hpp"

using namespace CG;

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
                        double const minimum_temperature = -1,
                        bool const protections_on = true,
                        bool const cooling_time_limiter_on = false);

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
                std::vector<double> const& CG_result,
                std::vector<double> const& full_CG_result) const override;

    MultigroupDiffusionCoefficientCalculator const& coefficient_calculator;
    MultigroupDiffusionBoundaryCalculator    const& boundary_calculator;

    std::vector<double> const energy_groups_center;
    std::vector<double> const energy_groups_boundary;
    std::vector<double> const energy_groups_width;

    mutable std::vector<ComputationalCell3D> cells_cgs; // cells vector s.t. the fields used in the radiation module (i.e. density, velocity, internal_energy, Erad, Eg) are in cgs.

    mutable std::vector<std::vector<double>> sigma_absorption_group; // absorption opacity per group per cell (1/cm) [group][cell]
    mutable std::vector<std::vector<double>> sigma_scattering_group; // scattering opacity per group per cell (1/cm) [group][cell]
    mutable std::vector<std::vector<double>> planck_integal_group;   // the integral of the Planck distribution on each group per cell [group][cell]

    mutable std::vector<double> sigma_absorption_planck; // the Planck weighted absorption opacity per cell (1/cm)
    mutable std::vector<double> fleck_factor;

    mutable std::vector<double> new_Eg; // energy groups per cell at the end of the time step [cell*ENERGY_GROUPS_NUM + group]
    mutable std::vector<double> new_Eg_full; // energy groups per cell at the end of the time step (plus the residue/volume for algebraic energy conservation) [cell*ENERGY_GROUPS_NUM + group]

    mutable std::vector<std::vector<double>> old_Eg; // energy groups per cell at the beggining of the time step [cell][group] 

    mutable std::vector<double> old_Er; // total radiation energy per cell at the beggining of the time step [cell]
    mutable std::vector<double> old_Tm; // temperature per cell at the beggining of the time step [cell]

    mutable std::vector<Vector3D> grad; // numerical gradient operator bewteen cells ij for i < j [face]

    bool const doppler_on_; // flag to indicate whether to add the doppler terms to the matrix
    double const minimum_temperature_; // enforce a minimal temperature 

    mutable ComptonMatrixMC compton_matrix_gen; // generator for Compton cross sections matrices

    // used for Compton 
    mutable std::vector<std::vector<double>> tau;
    mutable std::vector<std::vector<double>> dtau_dUm;
    mutable std::vector<std::vector<double>> S;
    mutable std::vector<std::vector<double>> dSdUm;

    mutable std::vector<double> n; // occupancy number
    mutable std::size_t cell_id_of_compton_matrices; // for debugging make sure that the compton values are generated for the correct cell 

    mutable std::vector<double> Gammas; // fleck_factor = 1.0 / (1.0 + c*dt*beta*Gamma)
    mutable std::vector<bool> use_n_zero; // flag for cells that need to use n=0 due to negative fleck factor
    mutable std::vector<double> compton_limiter_scale_; // per-cell Compton coupling reduction factor from cooling-time limiter (1.0 = no reduction)
    mutable CG::BiCGSTABWorkspace cg_workspace_;

private:
    bool const protections_on_; // flag whether to use Elad's protections on the amount of change allowed per time step (should not be on when running tests...)
    bool const cooling_time_limiter_on_;
    mutable bool displayed_warning_; // flag whether to display the planck sum warning

    void calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                std::vector<ComputationalCell3D> const& cells,
                                                                double const dt) const;

    void calculate_planck_integrals(Tessellation3D const& tess,
                                    std::vector<ComputationalCell3D> const& cells) const;

    void calculate_planck_absorption_coefficient(Tessellation3D const& tess,
                                                 std::vector<ComputationalCell3D> const& cells) const;

    // helper functions
    void calculate_fleck_factor(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double dt_cgs) const;

    void generate_S_and_dSdUm_matrices(ComputationalCell3D const& cell, std::size_t const cell_index, double const dt_cgs, bool const calculate_n = true) const;

    double calculate_Upsilon(ComputationalCell3D const& cell) const;


    /**
     * @brief Calculates the implicit compton scheme's contribution terms to the linear system left side.
     *
     * @param tess
     * @param cell
     * @param cell_index, g, gt indicates the equations and the group to which the term is for, the term for group `gt` of cell `cell_index` in the equation for energy group `g` of cell `cell_index`
     * @param dt_cgs the time step in cgs units
     */
    double get_implicit_compton_contribution(Tessellation3D const& tess,
                                             ComputationalCell3D const& cell,
                                             std::size_t const cell_index,
                                             std::size_t const g,
                                             std::size_t const gt,
                                             double const dt_cgs) const;

    /**
     * @brief Calculates the implicit compton scheme's contribution to the right side of the linear system.
     *
     * @param tess
     * @param cell
     * @param cell_index, g indicates the equations to which the term is for, equation for energy group `g` of cell `cell_index`
     * @param dt_cgs the time step in cgs units
     */
    double get_implicit_compton_contribution_to_b(Tessellation3D const& tess,
                                                  ComputationalCell3D const& cell,
                                                  std::size_t const cell_index,
                                                  std::size_t const g,
                                                  double const dt_cgs) const;

    double get_doppler_slope(ComputationalCell3D const& cell, size_t const g, bool const expansion) const;
};

#endif
