#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include <optional>
#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include <memory>
#include "radiation/compton/ComptonTableBackend.hpp"
#include "radiation/compton/MACEComptonBackend.hpp"

using namespace CG;

class MultigroupDiffusion : public RadiationDriver {
public:
    enum class ComptonOccupationMode
    {
        Off,
        Zero,
        RadiationField,
        PlanckFunction
    };

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
                        bool const cooling_time_limiter_on = false,
                        std::size_t const compton_matrix_samples = 100000);

    /**
     * @brief Destructor for the MultigroupDiffusion class.
     *
     * Default destructor, does nothing.
     */
    ~MultigroupDiffusion() = default;

    void SetComptonInGroupTransport(bool const on) { compton_in_group_transport_ = on; }

    // Whether the opacity's diffusion coefficient (its Rosseland mean) already contains electron scattering. With
    // Compton on, true adds nothing to the transport opacity and false adds the scattering opacity, so electron
    // scattering is counted once. Unless set, the opacity's ComptonIncludedInTransport() decides (false by default).
    void SetScatteringInRosseland(bool const included) { scattering_in_rosseland_ = included; }

    // Compton kernel consistency with the STORM IMC (plan v10, section 6). The last group's in-group energy-only
    // term (CMMC up minus down scatter) is off unless enabled. The occupation of group g is capped at
    // max(100, factor * nP_g(max(Te, Trad))), nP_g the group-integrated Planck occupation; an optional fixed cap
    // nMax > 0 applies on top.
    void SetComptonTopGroupTerm(bool const on);

    // Compton table source, mirroring STORM's comptonTableSource: MACE number-weighted effective-rate tables (the
    // default), read from the cache file in cache_dir ("" = $STORM_COMPTON_TABLE_DIR, else the working directory) or
    // built with all MPI ranks and cached there; or CMMC (mace = false).  Call before the first step; the tables are
    // created at the first prestep, which every rank runs.
    void SetComptonTableSource(bool const mace, std::string const& cache_dir = "", std::size_t const angle_bins = 32,
                               bool const exact_solver = true, int const threads = 0);
    void SetComptonOccupationCapFactor(double const factor) { compton_occupation_cap_factor_ = factor; }
    void SetComptonOccupationMax(double const occupation_max) { compton_occupation_max_ = occupation_max; }
    // Symmetric pair throttle, mirroring STORM's comptonPairThrottleOccupation (default 100, <= 0 = off): both
    // directions of a pair scaled by min(1, (1 + occupation) / (1 + max(n_i, n_j))); see
    // radiation/compton/ComptonPairThrottle.hpp.
    void SetComptonPairThrottleOccupation(double const occupation) { compton_pair_throttle_occupation_ = occupation; }
    double GetComptonPairThrottleOccupation() const { return compton_pair_throttle_occupation_; }
    // Try Planck-function occupations when the radiation-field Upsilon is not the larger one (default on); off keeps
    // the radiation-field occupations (with the zero-occupation fallback for Upsilon < 0), like STORM's
    // ComptonInducedMode::RadiationField.
    void SetComptonPlanckFallback(bool const on) { compton_planck_fallback_ = on; }

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
    // Include scatterings that stay in the group (tau_gg) in the Compton transport opacity of the diffusion
    // coefficient. Every scattering deflects the photon, so the physical transport opacity is the total
    // Compton rate; false reproduces the old out-of-group-only coefficient. Only used when the opacity has no
    // scattering of its own and its diffusion coefficient excludes electron scattering (see SetScatteringInRosseland).
    bool compton_in_group_transport_ = true;
    bool compton_top_group_term_ = false;
    double compton_occupation_cap_factor_ = 10.0;
    double compton_occupation_max_ = 0.0;
    double compton_pair_throttle_occupation_ = 100.0;
    bool compton_planck_fallback_ = true;
    std::optional<bool> scattering_in_rosseland_;
    mutable bool scattering_in_rosseland_reported_ = false;
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

    std::size_t const compton_matrix_samples_; // CMMC samples per table temperature
    mutable std::unique_ptr<STORM::ComptonTableBackendCore> compton_tables_; // Compton cross-section tables
    bool compton_mace_tables_ = true; // table source chosen by SetComptonTableSource
    STORM::MACEComptonTableOptions compton_table_options_;
    // The tables of the chosen source, created on first use (the first prestep).
    STORM::ComptonTableBackendCore const& compton_tables() const;
    // The temperature passed to the tables: physical for a self-clamping backend, else inside the table.
    double compton_table_temperature(double T) const;

    // used for Compton 
    mutable std::vector<std::vector<double>> tau;
    mutable std::vector<std::vector<double>> dtau_dUm;
    mutable std::vector<std::vector<double>> S;
    mutable std::vector<std::vector<double>> dSdUm;

    mutable std::vector<double> n; // occupancy number
    mutable std::size_t cell_id_of_compton_matrices; // for debugging make sure that the compton values are generated for the correct cell 

    mutable std::vector<double> Gammas; // fleck_factor = 1.0 / (1.0 + c*dt*beta*Gamma)
    mutable std::vector<double> upsilon_;
    mutable std::vector<double> upsilon_erad_;
    mutable std::vector<double> upsilon_lte_;
    mutable std::vector<double> upsilon_n0_;
    mutable std::vector<ComptonOccupationMode> compton_occupation_mode_;
    mutable std::vector<bool> use_n_zero; // flag for cells that need to use n=0 due to negative fleck factor
    mutable std::vector<bool> compton_jacobian_frozen_; // zero dSdUm when upsilon remains negative after occupation selection
    mutable std::vector<double> compton_limiter_scale_; // per-cell Compton coupling reduction factor from cooling-time limiter (1.0 = no reduction)
    // Cells whose Compton source must be deferred to the local operator split.
    mutable std::vector<bool> compton_deferred_;
    mutable std::vector<bool> split_compton_cells_;
    mutable bool matrix_unrecoverable_ = false;
    mutable bool postcg_unrecoverable_ = false;
    mutable std::size_t split_subcycle_count_ = 0;
    mutable double split_suppressed_energy_ = 0.0;
    mutable double split_injected_energy_ = 0.0;
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

    // Occupation cap per group (see SetComptonOccupationCapFactor); T_e is the gas temperature in K.
    std::vector<double> compton_occupation_cap(ComputationalCell3D const& cell, double const T_e) const;

    void generate_S_and_dSdUm_matrices(ComputationalCell3D const& cell,
                                       std::size_t const cell_index,
                                       double const dt_cgs,
                                       ComptonOccupationMode occupation_mode = ComptonOccupationMode::RadiationField) const;

    static char const* comptonOccupationModeLabel(ComptonOccupationMode mode);

    double compute_lte_temperature(ComputationalCell3D const& cell) const;

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

    double calcEffectiveDiffusionCoefficient(ComputationalCell3D const& cell,
                                             std::size_t cell_index,
                                             std::size_t group) const;

    void apply_operator_split_compton(Tessellation3D const& tess,
                                      std::vector<ComputationalCell3D>& cells,
                                      std::vector<Conserved3D>& extensives,
                                      double dt) const;

    bool solve_local_compton_substep(Tessellation3D const& tess,
                                     std::size_t cell_index,
                                     ComputationalCell3D& cell,
                                     Conserved3D& extensive,
                                     double dt) const;

    void fillComptonScatteringRates(std::size_t cell_index,
                                    std::vector<std::vector<double>> const& tau_mat,
                                    std::vector<double> const& occ) const;
};

#endif
