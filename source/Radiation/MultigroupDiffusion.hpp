#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include <cmath>
#include <limits>
#include <set>
#include <string>

#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include "CMMC/src/compton_matrix_mc.hpp"

using namespace CG;

class MultigroupDiffusion : public RadiationDriver {
public:
    struct CoefficientDiagnostics
    {
        double minimum_fleck_factor = std::numeric_limits<double>::max();
        double maximum_fleck_factor = 0;
        unsigned long long fleck_samples = 0;
        double maximum_transport_scattering = 0;
    };

    size_t GetUnknownsPerCell() const override
    {
        return energy_groups_center.size();
    }

    CG::PreconditionerKind GetPreconditionerKind() const override
    {
        return preconditioner_kind_;
    }

    bool HistoricalMGComptonFallbackAvailable(
        std::size_t CellId) const override;

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
                        CG::PreconditionerKind const preconditioner_kind =
                            CG::PreconditionerKind::CellBlockJacobi);

    /**
     * @brief Destructor for the MultigroupDiffusion class.
     *
     * Default destructor, does nothing.
     */
    ~MultigroupDiffusion() = default;

    double GetLengthScale() const override { return length_scale_; }

    bool supportsIndividualTimeSteps() const override { return true; }

    bool supportsAllActiveIndividualGlobalStep() const override { return true; }

    std::size_t individualUnknownsPerCell() const override
    {
        return energy_groups_center.size();
    }

    bool prestep(Tessellation3D const& tess,
                 std::vector<ComputationalCell3D> const& cells) const override;

    bool prestepIndividual(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        IndividualStepContext const& context) const override;

    bool step(double const tolerance,
              int& total_iters,
              Tessellation3D const& tess,
              std::vector<ComputationalCell3D>& cells,
              std::vector<Conserved3D>& extensives,
              double const dt,
              double const time) const override;

    bool stepIndividual(
        double tolerance,
        int& total_iters,
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>& extensives,
        IndividualStepContext const& context,
        double interval_fraction,
        double time,
        std::vector<ComputationalCell3D> const* canonical_cells = nullptr,
        std::vector<Conserved3D>* canonical_extensives = nullptr,
        std::vector<std::size_t> const* owned_to_canonical = nullptr) const override;

    bool poststep() const override;

    double calculate_dt(double const dt,
                        Tessellation3D& tess,
                        std::vector<ComputationalCell3D>& cells) const override;

    void calculateIndividualTimeSteps(
        IndividualStepContext const& context,
        Tessellation3D& tess,
        std::vector<ComputationalCell3D>& cells,
        std::vector<double>& time_step_limits,
        std::vector<ComputationalCell3D> const* canonical_owned_cells,
        std::vector<std::size_t> const* local_to_global) const override;

    void BuildMatrix(Tessellation3D const& tess,
                     mat& A,
                     size_t_mat& A_indeces,
                     std::vector<ComputationalCell3D> const& cells,
                     double const dt,
                     std::vector<double>& b,
                     std::vector<double>& x0,
                     double const current_time) const override;

    bool SupportsDirectCSR() const noexcept override {return true;}

    bool SupportsFixed16BlockStencil() const noexcept override
    {
        return ENERGY_GROUPS_NUM == Fixed16BlockStencilMatrix::BlockSize;
    }

    bool Fixed16BlockStencilEligible(
        Tessellation3D const& tess) const noexcept override
    {
        if(ENERGY_GROUPS_NUM != Fixed16BlockStencilMatrix::BlockSize)
            return false;
        for(std::size_t cell = 0; cell < tess.GetPointNo(); ++cell)
            if(!individualCellActive(cell))
                return false;
        return true;
    }

    void BuildMatrixCSR(Tessellation3D const& tess,
                        std::vector<size_t>& row_offsets,
                        std::vector<size_t>& column_indices,
                        std::vector<double>& values,
                        std::vector<ComputationalCell3D> const& cells,
                        double const dt,
                        std::vector<double>& b,
                        std::vector<double>& x0,
                        double const current_time) const override;

    void BuildMatrixCSRFixed16BlockStencil(
        Tessellation3D const& tess,
        std::vector<size_t>& row_offsets,
        std::vector<size_t>& column_indices,
        std::vector<double>& values,
        std::vector<ComputationalCell3D> const& cells,
        double const dt,
        std::vector<double>& b,
        std::vector<double>& x0,
        double const current_time,
        bool const build_csr_shadow,
        Fixed16BlockStencilMatrix& fixed16_block_stencil) const override;

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
    CG::PreconditionerKind const preconditioner_kind_;

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

    // Preserved across fractional candidates for the event-level timestep
    // estimate. old_* above are refreshed from the latest accepted candidate.
    mutable std::vector<std::vector<double>> event_old_Eg;
    mutable std::vector<double> event_old_Er;
    mutable std::vector<double> event_old_Tm;

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
    mutable std::vector<double> radiation_force_time_step_limits_;
    mutable bool matrix_unrecoverable_ = false;
    mutable bool postcg_unrecoverable_ = false;
    mutable std::size_t split_subcycle_count_ = 0;
    mutable double split_suppressed_energy_ = 0.0;
    mutable double split_injected_energy_ = 0.0;
    mutable SpectralRepairEvent pending_split_spectral_repair_event_;
    mutable std::set<std::size_t> individual_compton_force_deferred_ids_;
    mutable std::vector<std::size_t> current_cell_ids_;
    mutable bool compton_solution_rebuild_used_ = false;
    mutable CG::BiCGSTABWorkspace cg_workspace_;

    CoefficientDiagnostics coefficientDiagnostics() const noexcept;
    void releaseIndividualTopologyStorage() const noexcept override;

private:
    class MatrixRows;
    struct ImplicitComptonCellCoefficients;

    mutable CoefficientDiagnostics released_coefficient_diagnostics_;

    void ensureComptonBulkRuntimeOptions() const;

    void fillImplicitComptonCellCoefficients(
        Tessellation3D const& tess,
        ComputationalCell3D const& cell,
        std::size_t cell_index,
        double dt_cgs,
        ImplicitComptonCellCoefficients& coefficients) const;

    bool fillLegacyImplicitComptonCellCoefficientsAndCompare(
        Tessellation3D const& tess,
        ComputationalCell3D const& cell,
        std::size_t cell_index,
        double dt_cgs,
        ImplicitComptonCellCoefficients const& bulk,
        ImplicitComptonCellCoefficients& legacy,
        char const* phase) const;

    mutable bool compton_bulk_runtime_options_initialized_ = false;
    mutable bool compton_bulk_coefficients_enabled_ = false;
    mutable bool compton_bulk_shadow_enabled_ = false;
    mutable bool skip_reverse_faces_requested_ = false;
    mutable bool hoist_face_mean_temperature_requested_ = false;
    mutable bool freefree_pair16_requested_ = false;
    mutable bool freefree_pair16_shadow_requested_ = false;
    mutable bool freefree_pair16_failed_closed_ = false;
    mutable bool postcg_face_geometry_requested_ = false;
    mutable bool postcg_face_geometry_shadow_requested_ = false;
    mutable bool postcg_face_geometry_failed_closed_ = false;

    void buildDirectStructureCacheKey(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells) const;

    void invalidateDirectStructureCache(bool release_storage) const;

    mutable bool direct_structure_cache_valid_ = false;
    mutable std::vector<std::size_t> direct_structure_cache_key_;
    mutable std::vector<std::size_t> direct_structure_cache_probe_;
    mutable std::size_t direct_structure_cache_rows_ = 0;
    mutable std::size_t direct_structure_cache_nnz_ = 0;
    mutable unsigned long long direct_structure_cache_solver_epoch_ = 0;
    mutable unsigned long long direct_structure_cache_bound_epoch_ = 0;
    mutable unsigned long long direct_structure_cache_hits_ = 0;
    mutable unsigned long long direct_structure_cache_misses_ = 0;

    void BuildMatrixImpl(Tessellation3D const& tess,
                         MatrixRows& matrix_rows,
                         std::vector<ComputationalCell3D> const& cells,
                         double const dt,
                         std::vector<double>& b,
                         std::vector<double>& x0,
                         double const current_time) const;

    void prepareIndividualCandidate(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells) const override;

    void ReleaseDormantGlobalSolverStorage() const override;

    bool validateIndividualCoefficients(
        IndividualStepContext const& context,
        std::vector<ComputationalCell3D> const& cells) const override
    {
        for(std::size_t i : context.active_indices)
            if(!std::isfinite(fleck_factor.at(i)) || fleck_factor.at(i) <= 0) {
                setCellLocalStepFailure(
                    "non-positive multigroup Fleck factor", cells.at(i).ID);
                return false;
            }
        return !matrix_unrecoverable_;
    }

    bool requestIndividualSolutionRetry(
        std::vector<double> const& pre_correction_solution,
        std::vector<double> const& physical_solution,
        std::vector<std::size_t> const& local_to_global,
        std::vector<ComputationalCell3D> const& cells,
        char const* scope) const override;

    bool applyIndividualPostSolvePhysics(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>& extensives,
        double fallback_dt,
        double global_maximum_cell_radiation_extent) const override;

    void appendPendingSpectralRepairEvent(
        SpectralRepairEvent& event) const override;

    bool deferComptonForNonphysicalSolution(
        std::vector<double> const& pre_correction_solution,
        std::vector<double> const& physical_solution,
        std::vector<std::size_t> const* local_to_global,
        std::vector<ComputationalCell3D> const& cells,
        char const* scope,
        bool persist_for_individual_retry) const;

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

    double applyComptonTransportCorrection(
        double diffusion_coefficient,
        std::size_t cell_index,
        std::size_t group) const;

    bool apply_operator_split_compton(Tessellation3D const& tess,
                                      std::vector<ComputationalCell3D>& cells,
                                      std::vector<Conserved3D>& extensives,
                                      double dt,
                                      double global_maximum_cell_radiation_extent) const;

    bool repairMultigroupSpectraAfterStage(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>& extensives,
        double global_maximum_cell_radiation_extent,
        char const* stage,
        SpectralRepairEvent& event) const;

    struct SplitComptonDiagnosticSummary
    {
        unsigned long long upsilon_fallback_events = 0;
        unsigned long long representative_cell_id = 0;
        unsigned long long representative_failed_group = 0;
        double representative_dt = 0;
        double representative_original_upsilon = 0;
        double representative_original_fleck = 0;
        double representative_failed_value = 0;
        std::string latest_failure_reason;
        unsigned long long latest_failure_cell_id = 0;
        unsigned long long latest_failure_group =
            std::numeric_limits<unsigned long long>::max();
        double latest_failure_dt = 0;
        double latest_failure_extent = 0;
        double latest_negative_extent = 0;
        double latest_positive_extent = 0;
        double latest_negative_to_global_max_ratio =
            std::numeric_limits<double>::quiet_NaN();
        double latest_total_to_global_max_ratio =
            std::numeric_limits<double>::quiet_NaN();
        double latest_beta = 0;
        double latest_gamma = 0;
        double latest_kappa_planck = 0;
        double latest_fleck = 0;
        ComptonOccupationMode latest_occupation =
            ComptonOccupationMode::Off;
        bool latest_repair_attempted = false;
        bool latest_repair_injected = false;
        bool latest_upsilon_fallback = false;
    };

    bool solve_local_compton_substep(Tessellation3D const& tess,
                                     std::size_t cell_index,
                                     ComputationalCell3D& cell,
                                     Conserved3D& extensive,
                                     double dt,
                                     double global_maximum_cell_radiation_extent,
                                     bool emit_diagnostics,
                                     SplitComptonDiagnosticSummary& diagnostics) const;

    bool prepare_compton_only_fleck(ComputationalCell3D const& cell,
                                    std::size_t cell_index,
                                    double dt_cgs,
                                    double& inverse_cv_bar,
                                    double& compton_fleck,
                                    double& compton_upsilon,
                                    ComptonOccupationMode& occupation_mode) const;

    void fillComptonScatteringRates(std::size_t cell_index,
                                    std::vector<std::vector<double>> const& tau_mat,
                                    std::vector<double> const& occ) const;
};

#endif
