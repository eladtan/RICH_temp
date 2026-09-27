#include "Diffusion.hpp" // for CalcSingleFluxLimiter and FleckFactor
#include "MultigroupDiffusion.hpp"
#include "SpectralPositivity.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
#include "newtonian/three_dimensional/simulation/RuntimeLog.hpp"
// TODO: make a units namespace used by all the program 
#include "CMMC/src/units/units.hpp"
#include "CMMC/src/planck_integral/planck_integral.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

using boost::math::pow;

namespace
{
bool mgRuntimeFlagEnabled(char const* const name)
{
	if(std::strcmp(name, "RICH_INDIVIDUAL_PERF_TRACE") == 0 &&
	   RuntimeLogDetailed())
		return true;
    char const* const value = std::getenv(name);
    std::string const setting = value == nullptr ? "" : std::string(value);
    return !setting.empty() && setting != "0" && setting != "false" &&
        setting != "off" && setting != "no";
}

struct MgStrictRuntimeFlag
{
    bool valid;
    bool enabled;
};

MgStrictRuntimeFlag mgStrictRuntimeFlag(char const* const name)
{
    char const* const value = std::getenv(name);
    std::string const setting = value == nullptr ? "" : std::string(value);
    if(setting.empty() || setting == "0" || setting == "false" ||
       setting == "off" || setting == "no")
        return {true, false};
    if(setting == "1" || setting == "true" || setting == "on" ||
       setting == "yes")
        return {true, true};
    return {false, false};
}

using MultigroupClock = std::chrono::steady_clock;

double mgElapsedSeconds(MultigroupClock::time_point const start)
{
    return std::chrono::duration<double>(
        MultigroupClock::now() - start).count();
}

ComputationalCell3D multigroupRadiationCellInCgs(
    ComputationalCell3D const& cell,
    double const length_scale,
    double const time_scale,
    double const mass_scale)
{
    ComputationalCell3D result(cell);
    result.density *= mass_scale / pow<3>(length_scale);
    double const specific_energy_scale =
        pow<2>(length_scale) / pow<2>(time_scale);
    result.internal_energy *= specific_energy_scale;
    result.Erad *= specific_energy_scale;
    result.velocity *= length_scale / time_scale;
    for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
        result.Eg[group] *= specific_energy_scale;
    return result;
}

std::vector<double> comptonTransportScatteringExtinction(
    std::vector<std::vector<double>> const& tau_matrix,
    std::vector<double> const& occupation)
{
    if(tau_matrix.size() != occupation.size())
        throw std::logic_error("inconsistent Compton transport group count");
    std::vector<double> extinction(occupation.size(), 0.0);
    for(std::size_t group = 0; group < occupation.size(); ++group) {
        if(tau_matrix[group].size() != occupation.size())
            throw std::logic_error("inconsistent Compton transport matrix");
        double outgoing_rate = 0;
        for(std::size_t target_group = 0;
            target_group < occupation.size(); ++target_group)
            if(target_group != group)
                outgoing_rate += tau_matrix[group][target_group] *
                    (1 + occupation[target_group]);
        extinction[group] = std::max(0.0, outgoing_rate);
    }
    return extinction;
}

class MultigroupPhaseTimer
{
public:
    MultigroupPhaseTimer(bool const enabled, double& elapsed)
        : enabled_(enabled), elapsed_(&elapsed), start_(enabled ?
              MultigroupClock::now() : MultigroupClock::time_point{})
    {}

    ~MultigroupPhaseTimer()
    {
        if(enabled_)
            *elapsed_ += mgElapsedSeconds(start_);
    }

private:
    bool enabled_;
    double* elapsed_;
    MultigroupClock::time_point start_;
};

template<class Sequence>
void appendDirectStructureSequence(std::vector<std::size_t>& key,
                                   Sequence const& sequence)
{
    key.push_back(sequence.size());
    for(auto const value : sequence)
        key.push_back(static_cast<std::size_t>(value));
}

template<class NestedSequence>
void appendDirectStructureNestedSequence(
    std::vector<std::size_t>& key,
    NestedSequence const& sequences)
{
    key.push_back(sequences.size());
    for(auto const& sequence : sequences)
        appendDirectStructureSequence(key, sequence);
}

std::uint64_t mgDoubleBits(double const value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value),
                  "double and uint64_t must have equal size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t mgUlpDistance(double const left, double const right)
{
    if(!std::isfinite(left) || !std::isfinite(right))
        return std::numeric_limits<std::uint64_t>::max();
    auto const ordered = [](std::uint64_t const bits) {
        std::uint64_t const mask = (bits >> 63) != 0 ?
            std::numeric_limits<std::uint64_t>::max() :
            (std::uint64_t(1) << 63);
        return bits ^ mask;
    };
    std::uint64_t const ordered_left = ordered(mgDoubleBits(left));
    std::uint64_t const ordered_right = ordered(mgDoubleBits(right));
    return ordered_left >= ordered_right ?
        ordered_left - ordered_right : ordered_right - ordered_left;
}
}

char const* MultigroupDiffusion::comptonOccupationModeLabel(ComptonOccupationMode const mode)
{
    switch (mode) {
    case ComptonOccupationMode::Off:
        return "off";
    case ComptonOccupationMode::Zero:
        return "n=0";
    case ComptonOccupationMode::RadiationField:
        return "n=Erad";
    case ComptonOccupationMode::PlanckFunction:
        return "n=planck";
    }
    return "unknown";
}

namespace {

bool solve_dense_system(std::vector<double> matrix,
                        std::vector<double>& rhs,
                        std::size_t const n)
{
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        double pivot_abs = std::abs(matrix[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            double const candidate = std::abs(matrix[i * n + k]);
            if (candidate > pivot_abs) {
                pivot = i;
                pivot_abs = candidate;
            }
        }
        if (!std::isfinite(pivot_abs) || pivot_abs <= std::numeric_limits<double>::min())
            return false;
        if (pivot != k) {
            for (std::size_t j = k; j < n; ++j)
                std::swap(matrix[k * n + j], matrix[pivot * n + j]);
            std::swap(rhs[k], rhs[pivot]);
        }
        double const diagonal = matrix[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            double const factor = matrix[i * n + k] / diagonal;
            if (!std::isfinite(factor))
                return false;
            for (std::size_t j = k + 1; j < n; ++j)
                matrix[i * n + j] -= factor * matrix[k * n + j];
            rhs[i] -= factor * rhs[k];
        }
    }
    for (std::size_t ii = n; ii-- > 0;) {
        double value = rhs[ii];
        for (std::size_t j = ii + 1; j < n; ++j)
            value -= matrix[ii * n + j] * rhs[j];
        rhs[ii] = value / matrix[ii * n + ii];
        if (!std::isfinite(rhs[ii]))
            return false;
    }
    return true;
}

void log_postcg_crash_precursor(int const rank,
                                char const* reason,
                                ComputationalCell3D const& cell,
                                Vector3D const& loc,
                                double const old_e_therm_ext,
                                double const dE_absorption_emission,
                                double const dE_compton,
                                double const internal_energy_specific)
{
    std::clog << std::scientific << std::setprecision(6)
              << "MG PostCG crash-precursor rank " << rank
              << " cell ID " << cell.ID
              << " " << reason
              << " loc=" << loc
              << "\n  T=" << cell.temperature
              << " rho=" << cell.density
              << " e_int=" << internal_energy_specific
              << " old_e_therm_ext=" << old_e_therm_ext
              << " dE_abs=" << dE_absorption_emission
              << " dE_compton=" << dE_compton
              << " Erad=" << (cell.Erad * cell.density)
              << std::endl;
}

struct MGTimeStepReferenceScales
{
    double maximum_radiation_energy_density;
    double maximum_density_temperature;
};

constexpr double mg_timestep_change_fraction = 0.15;
// Global timesteps are continuous, so retain the historical 40% damping cap.
constexpr double mg_global_timestep_growth_cap = 1.4;
// Individual bins are powers of two; this permits exactly one aligned bin
// increase, which the scheduler independently enforces.
constexpr double mg_individual_timestep_growth_cap = 2.0;

MGTimeStepReferenceScales reduce_mg_timestep_reference_scales(
    double local_maximum_radiation_energy_density,
    double local_maximum_density_temperature)
{
    double maxima[2] = {
        local_maximum_radiation_energy_density,
        local_maximum_density_temperature};
#ifdef RICH_MPI
    // Every rank enters this reduction, including ranks with no active cells.
    MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    MGTimeStepReferenceScales result = {maxima[0], maxima[1]};
    return result;
}

void accumulate_mg_timestep_reference_scales(
    MGTimeStepReferenceScales& scales,
    double const radiation_energy_density,
    ComputationalCell3D const& current_cell)
{
    scales.maximum_radiation_energy_density = std::max(
        scales.maximum_radiation_energy_density,
        radiation_energy_density);
    scales.maximum_density_temperature = std::max(
        scales.maximum_density_temperature,
        current_cell.density * current_cell.temperature);
}

struct MGTimeStepChange
{
    double difference;
    double equilibrium_factor;
    bool radiation_equilibrium;
    int component;
};

MGTimeStepChange calculate_mg_timestep_change(
    ComputationalCell3D const& cell,
    double const old_radiation_energy_density,
    double const old_temperature,
    std::vector<double> const& old_group_energy_densities,
    double const fleck_factor,
    MGTimeStepReferenceScales const& scales)
{
    if(cell.Eg.size() < ENERGY_GROUPS_NUM ||
       old_group_energy_densities.size() < ENERGY_GROUPS_NUM)
        throw std::runtime_error(
            "multigroup timestep baseline has the wrong group count");

    double const tiny = std::numeric_limits<double>::min();
    double const new_radiation_energy_density = cell.Erad * cell.density;
    double const equilibrium_factor =
        std::abs(cell.temperature -
                 std::pow(std::max(new_radiation_energy_density, 0.0) /
                              CG::radiation_constant,
                          0.25)) <
                0.05 * cell.temperature
            ? 0.5
            : 1.0;
    bool const radiation_equilibrium =
        std::abs(new_radiation_energy_density -
                 old_radiation_energy_density) <
        0.075 * old_radiation_energy_density;

    MGTimeStepChange result = {
        equilibrium_factor *
            std::abs(new_radiation_energy_density -
                     old_radiation_energy_density) /
            (new_radiation_energy_density +
             0.02 * scales.maximum_radiation_energy_density + tiny),
        equilibrium_factor,
        radiation_equilibrium,
        0};

    double temperature_difference = cell.density * equilibrium_factor *
        std::abs(cell.temperature - old_temperature) /
        (cell.density * cell.temperature +
         1e-3 * scales.maximum_density_temperature + tiny);
    if(radiation_equilibrium)
        temperature_difference *= 0.25;
    if(fleck_factor < 0.9)
        temperature_difference *= std::pow(0.1 + fleck_factor, 4.0);
    if(temperature_difference > result.difference) {
        result.difference = temperature_difference;
        result.component = 1;
    }

    for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
        double group_difference = 0.2 *
            (7 * equilibrium_factor / 6 - 1.0 / 6.0) *
            std::abs(cell.Eg[group] * cell.density -
                     old_group_energy_densities[group]) /
            (cell.Eg[group] * cell.density +
             0.01 * scales.maximum_radiation_energy_density /
                 ENERGY_GROUPS_NUM +
             new_radiation_energy_density / ENERGY_GROUPS_NUM + tiny);
        if(radiation_equilibrium)
            group_difference *= 0.25;
        if(group_difference > result.difference) {
            result.difference = group_difference;
            result.component = static_cast<int>(group) + 2;
        }
    }
    return result;
}

} // namespace

void fill_zero(std::vector<double>& vec) {
    std::fill(vec.begin(), vec.end(), 0.0);
}

void fill_zero(std::vector<std::vector<double>>& mat) {
    for (std::vector<double>& row : mat) {
        std::fill(row.begin(), row.end(), 0.0);
    }
}

void resize_group_matrix(std::vector<std::vector<double>>& mat, std::size_t cells) {
    mat.resize(ENERGY_GROUPS_NUM);
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        mat[g].resize(cells);
        std::fill(mat[g].begin(), mat[g].end(), 0.0);
    }
}

std::vector<double> compton_temperatures() {
    std::vector<double> tmp_grid = linspace(-2, 4, 128);

    for (size_t i = 0; i < tmp_grid.size(); ++i) {
        tmp_grid[i] = std::pow(10.0, tmp_grid[i]);
    }

    tmp_grid.insert(tmp_grid.begin(), 0.005);
    tmp_grid.insert(tmp_grid.begin(), 0.001);
    tmp_grid.insert(tmp_grid.begin(), 0.0001);
    // tmp_grid = {1e-2, 0.1, 0.2, 0.3, 0.8, 1.5, 3.0, 4.0, 5.0, 7.5, 10.0, 13.0, 18.0, 20.0, 21.};
    for (auto& temp : tmp_grid) {
        temp *= units::kev_kelvin;
    }

    return tmp_grid;
}

namespace {

std::size_t compton_matrix_sample_count(bool const compton_on) {
    constexpr std::size_t production_sample_count = 100000;
    constexpr std::size_t disabled_sample_count = 10;
    if (!compton_on) {
        return disabled_sample_count;
    }

    char const* const value = std::getenv("RICH_TEST_COMPTON_MATRIX_SAMPLES");
    if (value == nullptr || value[0] == '\0') {
        return production_sample_count;
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long const parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || value[0] == '-' || end == value || *end != '\0' ||
        parsed < 4 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "RICH_TEST_COMPTON_MATRIX_SAMPLES must be an integer of at least 4");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

std::vector<double> get_energy_groups_width(std::vector<double> const& energy_groups_boundary) {
    std::vector<double> energy_groups_width(energy_groups_boundary.size()-1, std::numeric_limits<double>::signaling_NaN());

    for (std::size_t g=0; g<energy_groups_boundary.size()-1; ++g) {
        energy_groups_width[g] = energy_groups_boundary[g+1] - energy_groups_boundary[g];
    }

    return energy_groups_width;
}

MultigroupDiffusion::MultigroupDiffusion(std::vector<double> const& energy_groups_center_,
                                         std::vector<double> const& energy_groups_boundary_,
                                         MultigroupDiffusionCoefficientCalculator const& coefficient_calc,
                                         MultigroupDiffusionBoundaryCalculator const& boundary_calc,
                                         EquationOfState const& eos,
                                         std::vector<std::string> const zero_cells,
                                         bool const flux_limiter,
                                         bool const hydro_on,
                                         bool const compton_on,
                                         bool const doppler_on,
                                         double const minimum_temperature,
                                         bool const protections_on,
                                         bool const cooling_time_limiter_on,
                                         CG::PreconditionerKind const preconditioner_kind) :
    RadiationDriver(eos,
        zero_cells,
        flux_limiter,
        hydro_on,
        compton_on),
    coefficient_calculator(coefficient_calc),
    boundary_calculator(boundary_calc),
    energy_groups_center(energy_groups_center_),
    energy_groups_boundary(energy_groups_boundary_),
    energy_groups_width(get_energy_groups_width(energy_groups_boundary)),
    preconditioner_kind_(preconditioner_kind),
    cells_cgs(),
    sigma_absorption_group(ENERGY_GROUPS_NUM, std::vector<double>()),
    sigma_scattering_group(ENERGY_GROUPS_NUM, std::vector<double>()),
    planck_integal_group(ENERGY_GROUPS_NUM, std::vector<double>()),
    sigma_absorption_planck(),
    fleck_factor(),
    new_Eg(),
    new_Eg_full(),
    old_Eg(ENERGY_GROUPS_NUM, std::vector<double>()),
    old_Er(),
    old_Tm(),
    grad(),
    doppler_on_(doppler_on),
    minimum_temperature_(minimum_temperature),
    displayed_warning_(false),
    compton_matrix_gen(
        energy_groups_center_,
        energy_groups_boundary_,
        compton_matrix_sample_count(compton_on),
        true, // num of samples
        1),
    tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    dtau_dUm(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    S(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    dSdUm(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    n(ENERGY_GROUPS_NUM, 0.0),
    cell_id_of_compton_matrices(std::numeric_limits<std::size_t>::max()),
    Gammas(),
    upsilon_(),
    upsilon_erad_(),
    upsilon_lte_(),
    upsilon_n0_(),
    compton_occupation_mode_(),
    use_n_zero(),
    compton_jacobian_frozen_(),
    compton_deferred_(),
    protections_on_(protections_on),
    cooling_time_limiter_on_(cooling_time_limiter_on) {

    if (energy_groups_center.size() != ENERGY_GROUPS_NUM) {
        std::clog << "bad energy_groups_center.size()" << std::endl;
        exit(1);
    }

    if (energy_groups_boundary.size() != ENERGY_GROUPS_NUM + 1) {
        std::clog << "bad energy_groups_boundary.size()" << std::endl;
        exit(1);
    }

    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        if (energy_groups_boundary[g] >= energy_groups_boundary[g+1]) {
            std::clog << "bad energy_groups_boundary" << std::endl;
            exit(1);
        }

        if (energy_groups_boundary[g] >= energy_groups_center[g] or energy_groups_boundary[g+1] <= energy_groups_center[g]) {
            std::clog << "bad energy_groups_boundary and energy_groups_center" << std::endl;
            exit(1);
        }
    }
    if(compton_on) {
        compton_matrix_gen.set_tables(compton_temperatures());
        compton_matrix_gen.set_precomputed_linear_tables(
            mgRuntimeFlagEnabled("RICH_MG_PRECOMPUTED_COMPTON_EXP"));
    }
}

bool MultigroupDiffusion::HistoricalMGComptonFallbackAvailable(
    std::size_t const CellId) const
{
    int available = 0;
    if(compton_on_ && !compton_solution_rebuild_used_) {
        std::size_t const count = std::min(
            current_cell_ids_.size(), std::min(
                compton_deferred_.size(), compton_occupation_mode_.size()));
        for(std::size_t cell = 0; cell < count; ++cell)
            if(current_cell_ids_[cell] == CellId &&
               !compton_deferred_[cell] &&
               compton_occupation_mode_[cell] != ComptonOccupationMode::Off) {
                available = 1;
                break;
            }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &available, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    return available != 0;
}

bool MultigroupDiffusion::prestep(Tessellation3D const& tess,
                                  std::vector<ComputationalCell3D> const& cells) const {
    MEMORY_PROFILE_SCOPE("multigroup diffusion prestep");
    auto const N = tess.GetPointNo();
    if(N == 0)
        released_coefficient_diagnostics_ = CoefficientDiagnostics{};
    current_cell_ids_.resize(N);
    for(std::size_t Cell = 0; Cell < N; ++Cell)
        current_cell_ids_[Cell] = cells[Cell].ID;

    resize_group_matrix(sigma_absorption_group, N);
    resize_group_matrix(sigma_scattering_group, N);
    resize_group_matrix(planck_integal_group, N);

    new_Eg.resize(N, 0.0);
    new_Eg_full.resize(N, 0.0);

    sigma_absorption_planck.resize(N, 0.0);
    fleck_factor.resize(N, 0.0);

    old_Er.resize(N, 0.0);
    old_Tm.resize(N, 0.0);

    for (std::size_t i=0; i < N; ++i) {
        old_Er[i] = cells[i].Erad * cells[i].density;
        old_Tm[i] = cells[i].temperature;
    }

    old_Eg.resize(N);
    for (std::size_t i=0; i < N; ++i) {
        old_Eg[i].resize(ENERGY_GROUPS_NUM, 0.0);

        for (std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g) {
            old_Eg[i][g] = cells[i].Eg[g] * cells[i].density;
        }
    }
    // Keep the start-of-event state for timestep feedback.  Fractional
    // candidates refresh old_* below from the latest accepted state.
    event_old_Eg = old_Eg;
    event_old_Er = old_Er;
    event_old_Tm = old_Tm;

    auto const Nfaces = tess.GetTotalFacesNumber();
    grad.resize(Nfaces);

    // temporary vectors
    std::vector<std::size_t> neighbors;
    face_vec faces;

    // create gradient per face
    for (std::size_t i=0; i < N; ++i) {
        tess.GetNeighbors(i, neighbors);
        faces = tess.GetCellFaces(i);

        Vector3D CM_i = tess.GetCellCM(i);

        auto const Nneighbors = neighbors.size();
        for (std::size_t j=0; j < Nneighbors; ++j) {
            std::size_t const neighbor_j = neighbors[j];

            if (!tess.IsPointOutsideBox(neighbor_j)) {
                if (i < neighbor_j) {
                    Vector3D const CM_ij = CM_i - tess.GetCellCM(neighbor_j);
                    grad[faces[j]] = CM_ij * (1.0 / (length_scale_*ScalarProd(CM_ij, CM_ij)));
                }
            }
        }
    }

    Gammas.resize(N, 0.0);
    upsilon_.resize(N, 0.0);
    upsilon_erad_.assign(N, std::numeric_limits<double>::quiet_NaN());
    upsilon_lte_.assign(N, std::numeric_limits<double>::quiet_NaN());
    upsilon_n0_.assign(N, std::numeric_limits<double>::quiet_NaN());
    compton_occupation_mode_.assign(N, ComptonOccupationMode::Off);
    use_n_zero.resize(N, false);
    compton_jacobian_frozen_.assign(N, false);
    compton_limiter_scale_.assign(N, 1.0);
    compton_deferred_.assign(N, false);
    split_compton_cells_.assign(N, false);
    radiation_force_time_step_limits_.assign(
        N, std::numeric_limits<double>::max());
    matrix_unrecoverable_ = false;
    postcg_unrecoverable_ = false;
    split_subcycle_count_ = 0;
    split_suppressed_energy_ = 0.0;
    split_injected_energy_ = 0.0;
    pending_split_spectral_repair_event_ = SpectralRepairEvent{};

    return true;
}

bool MultigroupDiffusion::prestepIndividual(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    IndividualStepContext const&) const
{
    return prestep(tess, cells);
}

void MultigroupDiffusion::ReleaseDormantGlobalSolverStorage() const
{
    bool const had_storage =
        direct_structure_cache_key_.capacity() != 0 ||
        direct_structure_cache_probe_.capacity() != 0 ||
        cg_workspace_.HasAllocatedStorage() ||
        new_Eg.capacity() != 0 ||
        new_Eg_full.capacity() != 0;
    invalidateDirectStructureCache(true);
    if(cg_workspace_.HasAllocatedStorage())
        cg_workspace_.Release();
    release_container_memory(new_Eg);
    release_container_memory(new_Eg_full);
    if(had_storage)
        rich_trim_after_rare_spike();
}

MultigroupDiffusion::CoefficientDiagnostics
MultigroupDiffusion::coefficientDiagnostics() const noexcept
{
    if(fleck_factor.empty())
        return released_coefficient_diagnostics_;

    CoefficientDiagnostics result;
    result.fleck_samples = static_cast<unsigned long long>(
        fleck_factor.size());
    for(double const value : fleck_factor)
    {
        result.minimum_fleck_factor = std::min(
            result.minimum_fleck_factor, value);
        result.maximum_fleck_factor = std::max(
            result.maximum_fleck_factor, value);
    }
    for(std::vector<double> const& group_values : sigma_scattering_group)
        for(double const value : group_values)
            result.maximum_transport_scattering = std::max(
                result.maximum_transport_scattering, value);
    return result;
}

void MultigroupDiffusion::releaseIndividualTopologyStorage() const noexcept
{
    // All of these arrays are reconstructed by prestep() from the committed
    // cells and current tessellation.  Retaining their old capacities across
    // AMR or migration pins the former heavy rank's allocation and can make
    // the next all-active matrix setup overlap two ownership generations.
    // Preserve the latest coefficient-field diagnostics without retaining the
    // large topology-dependent arrays. A repeated release after AMR and then
    // migration must not replace a valid snapshot with an empty one.
    if(!fleck_factor.empty())
        released_coefficient_diagnostics_ = coefficientDiagnostics();
    release_container_memory(cells_cgs);
    release_container_memory(sigma_absorption_group);
    release_container_memory(sigma_scattering_group);
    release_container_memory(planck_integal_group);
    release_container_memory(sigma_absorption_planck);
    release_container_memory(fleck_factor);
    release_container_memory(old_Eg);
    release_container_memory(old_Er);
    release_container_memory(old_Tm);
    release_container_memory(event_old_Eg);
    release_container_memory(event_old_Er);
    release_container_memory(event_old_Tm);
    release_container_memory(grad);
    release_container_memory(Gammas);
    release_container_memory(upsilon_);
    release_container_memory(upsilon_erad_);
    release_container_memory(upsilon_lte_);
    release_container_memory(upsilon_n0_);
    release_container_memory(use_n_zero);
    release_container_memory(compton_occupation_mode_);
    release_container_memory(compton_jacobian_frozen_);
    release_container_memory(compton_limiter_scale_);
    release_container_memory(compton_deferred_);
    release_container_memory(split_compton_cells_);
    release_container_memory(radiation_force_time_step_limits_);
    release_container_memory(current_cell_ids_);
    RadiationDriver::releaseIndividualTopologyStorage();
}

bool MultigroupDiffusion::stepIndividual(
    double const tolerance,
    int& total_iters,
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    IndividualStepContext const& context,
    double const interval_fraction,
    double const time,
    std::vector<ComputationalCell3D> const* canonical_cells,
    std::vector<Conserved3D>* canonical_extensives,
    std::vector<std::size_t> const* owned_to_canonical) const
{
    individual_compton_force_deferred_ids_.clear();
    compton_solution_rebuild_used_ = false;
    struct ClearForcedComptonFallback
    {
        std::set<std::size_t>& ids;
        ~ClearForcedComptonFallback() { ids.clear(); }
    } clear_on_return{individual_compton_force_deferred_ids_};
    return RadiationDriver::stepIndividual(
        tolerance, total_iters, tess, cells, extensives, context,
        interval_fraction, time, canonical_cells, canonical_extensives,
        owned_to_canonical);
}

bool MultigroupDiffusion::poststep() const {
    cells_cgs.clear();

    return true;
}

double MultigroupDiffusion::calculate_dt(double const dt,
                                         Tessellation3D& tess,
                                         std::vector<ComputationalCell3D>& cells) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    auto const N = tess.GetPointNo();
    std::vector<std::vector<double>> const& event_baseline_Eg =
        event_old_Eg.size() == N ? event_old_Eg : old_Eg;
    std::vector<double> const& event_baseline_Er =
        event_old_Er.size() == N ? event_old_Er : old_Er;
    std::vector<double> const& event_baseline_Tm =
        event_old_Tm.size() == N ? event_old_Tm : old_Tm;
    MGTimeStepReferenceScales local_reference_scales = {
        std::numeric_limits<double>::min(), 0};
    for(size_t i = 0; i < N; ++i)
        accumulate_mg_timestep_reference_scales(
            local_reference_scales,
            cells.at(i).Erad * cells.at(i).density, cells.at(i));
    MGTimeStepReferenceScales const reference_scales =
        reduce_mg_timestep_reference_scales(
            local_reference_scales.maximum_radiation_energy_density,
            local_reference_scales.maximum_density_temperature);
    double const max_Er =
        reference_scales.maximum_radiation_energy_density;
    double const max_rhoT = reference_scales.maximum_density_temperature;

    size_t const Nzero = zero_cells_.size();
    std::vector<size_t> zero_indeces;
    for (size_t i = 0; i < Nzero; ++i) {
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
    }

    double max_diff = std::numeric_limits<double>::min() * 100;
    int max_which = 0;
    int max_loc = 0;
    double equlibrium_factor_final = 0, final_Erad_eq = 0;
    for (size_t i = 0; i < N; ++i)
    {
        bool to_calc = true;
        for (size_t j = 0; j < Nzero; ++j) {
            if (cells[i].stickers[zero_indeces[j]]) {
                to_calc = false;
            }
        }

        if (not to_calc)
            continue;

        MGTimeStepChange const change = calculate_mg_timestep_change(
            cells[i], event_baseline_Er.at(i), event_baseline_Tm.at(i),
            event_baseline_Eg.at(i),
            fleck_factor.at(i), reference_scales);
        int const which_one = change.component;
        double const diff = change.difference;

        if (diff > max_diff) {
            max_which = which_one;
            max_diff = diff;
            max_loc = i;
            equlibrium_factor_final = change.equilibrium_factor;
            final_Erad_eq = change.radiation_equilibrium;
        }
    }

    struct {
        double val;
        int mpi_id;
    } max_data;

    max_data.mpi_id = rank;
    max_data.val = max_diff;

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    max_diff = max_data.val;
    MPI_exchange_data(tess, cells, true);
#endif
    double const spectral_suggested_dt = std::min(
        dt * mg_timestep_change_fraction / max_diff,
        dt * mg_global_timestep_growth_cap);

    double local_force_limit = std::numeric_limits<double>::max();
    std::size_t local_force_cell = max_size_t;
    for(std::size_t i = 0;
        i < N && i < radiation_force_time_step_limits_.size(); ++i) {
        if(radiation_force_time_step_limits_[i] < local_force_limit) {
            local_force_limit = radiation_force_time_step_limits_[i];
            local_force_cell = i;
        }
    }
    struct {
        double val;
        int mpi_id;
    } force_data = {local_force_limit, rank};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &force_data, 1, MPI_DOUBLE_INT,
                  MPI_MINLOC, MPI_COMM_WORLD);
#endif
    double const suggested_dt =
        std::min(spectral_suggested_dt, force_data.val);
    if (rank == max_data.mpi_id) {
        std::clog << "MG_TIMESTEP_LIMIT mode=global"
                  << " cell_id=" << cells[max_loc].ID
                  << " rank=" << rank
                  << " current_dt=" << dt
                  << " suggested_dt=" << spectral_suggested_dt
                  << " difference=" << max_diff
                  << " max_Er=" << max_Er
                  << " max_rhoT=" << max_rhoT
                  << " growth_cap=" << mg_global_timestep_growth_cap
                  << " reference_scope=canonical_owned_global"
                  << " source=";
        if(max_which == 0)
            std::clog << "radiation_energy";
        else if(max_which == 1)
            std::clog << "material_temperature";
        else
            std::clog << "group group=" << (max_which - 2);
        std::clog << std::endl;
        // max_loc /= ENERGY_GROUPS_NUM;
        std::clog<<"Radiation time step ID "<<cells[max_loc].ID<<" old Er "<<event_baseline_Er[max_loc]<<" new Er "<<cells[max_loc].Erad * cells[max_loc].density<<
            " diff "<<max_diff<<" Tgas "<<cells[max_loc].temperature<<" Trad "<<std::pow(cells[max_loc].density * cells[max_loc].Erad * mass_scale_ / (length_scale_ * pow<2>(time_scale_) * CG::radiation_constant), 0.25)<<" max_Er "<<max_Er<<" max_rhoT "<<max_rhoT<<" rank "<<rank<<" density "<<cells[max_loc].density<<
            " width "<<tess.GetWidth(max_loc)<<" Tgas_old "<<event_baseline_Tm[max_loc]<<" loc="<<tess.GetMeshPoint(max_loc)<<std::endl;
        std::clog<<"kp="<<sigma_absorption_planck[max_loc]<<" fleck factor "<<fleck_factor[max_loc]<<" which one "<<max_which<<" equlibrium_factor "<<equlibrium_factor_final<<" final_Erad_eq "<<final_Erad_eq<<" upsilon "<<upsilon_[max_loc]<<" upsilon_erad "<<upsilon_erad_[max_loc]<<" upsilon_lte "<<upsilon_lte_[max_loc]<<" upsilon_n0 "<<upsilon_n0_[max_loc]<<" occupation "<<comptonOccupationModeLabel(compton_occupation_mode_[max_loc])<<" jacobian_frozen "<<(compton_jacobian_frozen_[max_loc] ? 1 : 0)<<std::endl;

        if (max_which >= 2) std::clog << "Group number "<<max_which - 2<<" New_Eg="<<cells[max_loc].Eg[max_which - 2]*cells[max_loc].density<<" old_Eg="<<event_baseline_Eg[max_loc][max_which - 2]<<std::endl;
#ifdef DEBUG
        for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j) {
            std::clog<<"Eg["<<j<<"]="<<cells[max_loc].Eg[j]*cells[max_loc].density*mass_scale_ / (length_scale_ * pow<2>(time_scale_)) <<" old Eg["<<j<<"]="<<event_baseline_Eg[max_loc][j]*mass_scale_ / (length_scale_ * pow<2>(time_scale_))<<" energy(keV) "<<energy_groups_center[j] / units::kev<<" bg="<<
            planck_integral::planck_energy_density_group_integral(energy_groups_boundary[j], energy_groups_boundary[j+1], cells[max_loc].temperature)<<
            " bg_old="<<
            planck_integral::planck_energy_density_group_integral(energy_groups_boundary[j], energy_groups_boundary[j+1], event_baseline_Tm[max_loc])<< ", sigma[g]=" << sigma_absorption_group[max_loc][j] << ", cdt*sigma_g=" <<sigma_absorption_group[max_loc][j]*CG::speed_of_light*dt*time_scale_<< std::endl;
        }
#endif
    }

    if(force_data.val < spectral_suggested_dt &&
       rank == force_data.mpi_id && local_force_cell != max_size_t) {
        std::clog << "MG_RADIATION_FORCE_TIMESTEP_LIMIT mode=global"
                  << " cell_id=" << cells[local_force_cell].ID
                  << " rank=" << rank
                  << " current_dt=" << dt
                  << " suggested_dt=" << force_data.val
                  << " spectral_suggested_dt=" << spectral_suggested_dt
                  << std::endl;
    }

    return suggested_dt;
}


void MultigroupDiffusion::calculateIndividualTimeSteps(
    IndividualStepContext const& context,
    Tessellation3D& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<double>& time_step_limits,
    std::vector<ComputationalCell3D> const* canonical_owned_cells,
    std::vector<std::size_t> const* local_to_global) const
{
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    unsigned long long active_cell_count =
        static_cast<unsigned long long>(context.active_indices.size());
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &active_cell_count, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

    std::vector<std::vector<double>> const& event_baseline_Eg =
        event_old_Eg.empty() ? old_Eg : event_old_Eg;
    std::vector<double> const& event_baseline_Er =
        event_old_Er.empty() ? old_Er : event_old_Er;
    std::vector<double> const& event_baseline_Tm =
        event_old_Tm.empty() ? old_Tm : event_old_Tm;

    int canonical_mapping_valid =
        (canonical_owned_cells == nullptr) == (local_to_global == nullptr) ?
        1 : 0;
    bool const canonical_reference = canonical_owned_cells != nullptr &&
        local_to_global != nullptr;
    std::vector<std::size_t> canonical_active_local;
    if(canonical_reference) {
        canonical_active_local.assign(
            canonical_owned_cells->size(),
            std::numeric_limits<std::size_t>::max());
        if(local_to_global->size() != tess.GetPointNo())
            canonical_mapping_valid = 0;
        for(std::size_t local : context.active_indices) {
            if(local >= cells.size() || local >= local_to_global->size()) {
                canonical_mapping_valid = 0;
                continue;
            }
            std::size_t const global = local_to_global->at(local);
            if(global >= canonical_owned_cells->size() ||
               canonical_active_local[global] !=
                   std::numeric_limits<std::size_t>::max() ||
               canonical_owned_cells->at(global).ID != cells[local].ID) {
                canonical_mapping_valid = 0;
                continue;
            }
            canonical_active_local[global] = local;
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &canonical_mapping_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
#endif
    if(canonical_mapping_valid == 0)
        throw std::runtime_error(
            "multigroup individual canonical-owned mapping is invalid");

    MGTimeStepReferenceScales local_reference_scales = {
        std::numeric_limits<double>::min(), 0};
    if(canonical_reference) {
        for(std::size_t global = 0;
            global < canonical_owned_cells->size(); ++global) {
            ComputationalCell3D const& old_cell =
                canonical_owned_cells->at(global);
            std::size_t const local = canonical_active_local[global];
            ComputationalCell3D const& current_cell =
                local == std::numeric_limits<std::size_t>::max() ?
                old_cell : cells.at(local);
            // Match calculate_dt exactly: both reference scales use the
            // accepted current state.  Inactive canonical cells remain at
            // their own primitive timestamp and are never refreshed here.
            accumulate_mg_timestep_reference_scales(
                local_reference_scales,
                current_cell.Erad * current_cell.density, current_cell);
        }
    }
    else {
        for(std::size_t local = 0; local < cells.size(); ++local)
            accumulate_mg_timestep_reference_scales(
                local_reference_scales,
                cells.at(local).Erad * cells.at(local).density,
                cells.at(local));
    }
    MGTimeStepReferenceScales const reference_scales =
        reduce_mg_timestep_reference_scales(
            local_reference_scales.maximum_radiation_energy_density,
            local_reference_scales.maximum_density_temperature);
    double const max_Er =
        reference_scales.maximum_radiation_energy_density;
    double const max_rhoT = reference_scales.maximum_density_temperature;

    std::vector<std::size_t> zero_indices;
    zero_indices.reserve(zero_cells_.size());
    for(std::string const& name : zero_cells_)
        zero_indices.push_back(binary_index_find(ComputationalCell3D::stickerNames, name));

    double minimum_limit = std::numeric_limits<double>::max();
    double limiting_current_dt = 0;
    double limiting_difference = 0;
    double limiting_equilibrium_factor = 0;
    bool limiting_radiation_equilibrium = false;
    int limiting_component = -1;
    std::size_t limiting_cell = max_size_t;
    double local_force_limit = std::numeric_limits<double>::max();
    std::size_t local_force_cell = max_size_t;

    for(std::size_t i : context.active_indices) {
        bool calculate = true;
        for(std::size_t sticker : zero_indices)
            if(cells[i].stickers[sticker])
                calculate = false;
        if(!calculate)
            continue;

        if(i >= event_baseline_Er.size() || i >= event_baseline_Tm.size() ||
           i >= event_baseline_Eg.size() ||
           event_baseline_Eg[i].size() < ENERGY_GROUPS_NUM)
            throw std::runtime_error("multigroup individual event baseline is incomplete");
        MGTimeStepChange const change = calculate_mg_timestep_change(
            cells[i], event_baseline_Er[i], event_baseline_Tm[i],
            event_baseline_Eg[i],
            fleck_factor.at(i), reference_scales);
        double const difference = change.difference;
        int const component = change.component;

        double const current_dt = context.cellTimeStep(i);
        double const nominal_dt = context.nominalCellTimeStep(i);
        double const limit = std::min(
            current_dt * mg_timestep_change_fraction /
                std::max(difference, std::numeric_limits<double>::min()),
            nominal_dt * mg_individual_timestep_growth_cap);
        time_step_limits.at(i) = std::min(time_step_limits.at(i), limit);
        if(i < radiation_force_time_step_limits_.size()) {
            double const force_limit = radiation_force_time_step_limits_[i];
            time_step_limits.at(i) =
                std::min(time_step_limits.at(i), force_limit);
            if(force_limit < local_force_limit) {
                local_force_limit = force_limit;
                local_force_cell = i;
            }
        }
        if(limit < minimum_limit) {
            minimum_limit = limit;
            limiting_current_dt = current_dt;
            limiting_difference = difference;
            limiting_equilibrium_factor = change.equilibrium_factor;
            limiting_radiation_equilibrium = change.radiation_equilibrium;
            limiting_component = component;
            limiting_cell = i;
        }
    }

    struct {
        double val;
        int mpi_id;
    } minimum_data = {minimum_limit, rank};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &minimum_data, 1, MPI_DOUBLE_INT,
                  MPI_MINLOC, MPI_COMM_WORLD);
#endif
    if(rank == minimum_data.mpi_id && limiting_cell != max_size_t &&
       minimum_data.val < std::numeric_limits<double>::max()) {
        std::clog << "MG_TIMESTEP_LIMIT mode=individual"
                  << " event_time=" << context.event_time
                  << " active_cells=" << active_cell_count
                  << " cell_id=" << cells[limiting_cell].ID
                  << " rank=" << rank
                  << " current_dt=" << limiting_current_dt
                  << " suggested_dt=" << minimum_data.val
                  << " difference=" << limiting_difference
                  << " max_Er=" << max_Er
                  << " max_rhoT=" << max_rhoT
                  << " growth_cap="
                  << mg_individual_timestep_growth_cap
                  << " reference_scope="
                  << (canonical_reference ? "canonical_owned_global" :
                                             "local_mesh_global_fallback")
                  << " source=";
        if(limiting_component == 0)
            std::clog << "radiation_energy";
        else if(limiting_component == 1)
            std::clog << "material_temperature";
        else
            std::clog << "group group=" << (limiting_component - 2);
        std::clog << std::endl;

        double const new_Er_cell =
            cells[limiting_cell].Erad * cells[limiting_cell].density;
        double const radiation_temperature = std::pow(
            std::max(new_Er_cell, 0.0) * mass_scale_ /
                (length_scale_ * pow<2>(time_scale_) *
                 CG::radiation_constant),
            0.25);
        std::clog << "Radiation time step ID " << cells[limiting_cell].ID
                  << " mode individual"
                  << " old Er " << event_baseline_Er[limiting_cell]
                  << " new Er " << new_Er_cell
                  << " diff " << limiting_difference
                  << " Tgas " << cells[limiting_cell].temperature
                  << " Trad " << radiation_temperature
                  << " max_Er " << max_Er
                  << " max_rhoT " << max_rhoT
                  << " rank " << rank
                  << " density " << cells[limiting_cell].density
                  << " width " << tess.GetWidth(limiting_cell)
                  << " Tgas_old " << event_baseline_Tm[limiting_cell]
                  << " loc=" << tess.GetMeshPoint(limiting_cell)
                  << std::endl;
        std::clog << "kp=" << sigma_absorption_planck[limiting_cell]
                  << " fleck factor " << fleck_factor[limiting_cell]
                  << " which one " << limiting_component
                  << " equlibrium_factor " << limiting_equilibrium_factor
                  << " final_Erad_eq " << limiting_radiation_equilibrium
                  << " upsilon " << upsilon_[limiting_cell]
                  << " upsilon_erad " << upsilon_erad_[limiting_cell]
                  << " upsilon_lte " << upsilon_lte_[limiting_cell]
                  << " upsilon_n0 " << upsilon_n0_[limiting_cell]
                  << " occupation "
                  << comptonOccupationModeLabel(
                         compton_occupation_mode_[limiting_cell])
                  << " jacobian_frozen "
                  << (compton_jacobian_frozen_[limiting_cell] ? 1 : 0)
                  << std::endl;
        if(limiting_component >= 2) {
            std::size_t const group =
                static_cast<std::size_t>(limiting_component - 2);
            std::clog << "Group number " << group
                      << " New_Eg="
                      << cells[limiting_cell].Eg[group] *
                             cells[limiting_cell].density
                      << " old_Eg=" << event_baseline_Eg[limiting_cell][group]
                  << std::endl;
        }
    }

    struct {
        double val;
        int mpi_id;
    } force_data = {local_force_limit, rank};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &force_data, 1, MPI_DOUBLE_INT,
                  MPI_MINLOC, MPI_COMM_WORLD);
#endif
    if(rank == force_data.mpi_id && local_force_cell != max_size_t &&
       force_data.val < std::numeric_limits<double>::max()) {
        std::clog << "MG_RADIATION_FORCE_TIMESTEP_LIMIT mode=individual"
                  << " event_time=" << context.event_time
                  << " active_cells=" << active_cell_count
                  << " cell_id=" << cells[local_force_cell].ID
                  << " rank=" << rank
                  << " current_dt=" << context.cellTimeStep(local_force_cell)
                  << " suggested_dt=" << force_data.val
                  << std::endl;
    }
}

void MultigroupDiffusion::prepareIndividualCandidate(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells) const
{
    std::size_t const cell_count = tess.GetPointNo();
    current_cell_ids_.resize(cell_count);
    for(std::size_t Cell = 0; Cell < cell_count; ++Cell)
        current_cell_ids_[Cell] = cells[Cell].ID;

    char const* const elide_value =
        std::getenv("RICH_MG_ELIDE_ALL_ACTIVE_PREPARE");
    std::string const elide_setting = elide_value == nullptr ? "" :
        std::string(elide_value);
    bool const elide_option_valid = elide_setting.empty() ||
        elide_setting == "0" || elide_setting == "false" ||
        elide_setting == "off" || elide_setting == "no" ||
        elide_setting == "1" || elide_setting == "true" ||
        elide_setting == "on" || elide_setting == "yes";
    bool const elide_requested = elide_option_valid &&
        (elide_setting == "1" || elide_setting == "true" ||
         elide_setting == "on" || elide_setting == "yes");
    char const* const reset_limiter_value =
        std::getenv("RICH_MG_RESET_COMPTON_LIMITER_CANDIDATE");
    std::string const reset_limiter_setting =
        reset_limiter_value == nullptr ? "" :
        std::string(reset_limiter_value);
    bool const reset_limiter_valid = reset_limiter_setting.empty() ||
        reset_limiter_setting == "0" ||
        reset_limiter_setting == "false" ||
        reset_limiter_setting == "off" ||
        reset_limiter_setting == "no" ||
        reset_limiter_setting == "1" ||
        reset_limiter_setting == "true" ||
        reset_limiter_setting == "on" ||
        reset_limiter_setting == "yes";
    bool const reset_limiter_requested = reset_limiter_valid &&
        (reset_limiter_setting == "1" ||
         reset_limiter_setting == "true" ||
         reset_limiter_setting == "on" ||
         reset_limiter_setting == "yes");
    bool const all_active_global_route = individual_context_ == nullptr;
    int collectively_elide = elide_requested && reset_limiter_requested &&
        all_active_global_route &&
        individual_compton_force_deferred_ids_.empty() ? 1 : 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &collectively_elide, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
#endif

    bool const trace_prepare =
        mgRuntimeFlagEnabled("RICH_INDIVIDUAL_PERF_TRACE");
    int prepare_rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &prepare_rank);
#endif
    if(trace_prepare && prepare_rank == 0)
        RuntimeTraceStream() << "MG_ALL_ACTIVE_PREPARE"
                  << " scope=" << (all_active_global_route ?
                        "all_active_global" : "reduced_active")
                  << " requested=" << (elide_requested ? 1 : 0)
                  << " valid=" << (elide_option_valid ? 1 : 0)
                  << " reset=" << (reset_limiter_requested ? 1 : 0)
                  << " reset_valid=" << (reset_limiter_valid ? 1 : 0)
                  << " forced_ids="
                  << individual_compton_force_deferred_ids_.size()
                  << " elided=" << collectively_elide
                  << std::endl;
    if(collectively_elide != 0)
        return;

    old_Er.resize(cell_count);
    old_Tm.resize(cell_count);
    old_Eg.resize(cell_count);
    for(std::size_t i = 0; i < cell_count; ++i) {
        old_Er[i] = cells[i].Erad * cells[i].density;
        old_Tm[i] = cells[i].temperature;
        old_Eg[i].resize(ENERGY_GROUPS_NUM);
        for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            old_Eg[i][group] = cells[i].Eg[group] * cells[i].density;
    }
    split_compton_cells_.assign(cell_count, false);
    split_subcycle_count_ = 0;
    split_suppressed_energy_ = 0.0;
    split_injected_energy_ = 0.0;
    pending_split_spectral_repair_event_ = SpectralRepairEvent{};
    matrix_unrecoverable_ = false;
    postcg_unrecoverable_ = false;
    compton_deferred_.assign(cell_count, false);
    compton_occupation_mode_.assign(cell_count, ComptonOccupationMode::Off);
    compton_jacobian_frozen_.assign(cell_count, false);
    for(std::size_t i = 0; i < cell_count; ++i)
        if(individual_compton_force_deferred_ids_.count(cells[i].ID) != 0) {
            compton_deferred_[i] = true;
            split_compton_cells_[i] = true;
        }

    cells_cgs = cells;
    for(std::size_t i = 0; i < cells_cgs.size(); ++i)
        cells_cgs[i] = multigroupRadiationCellInCgs(
            cells[i], length_scale_, time_scale_, mass_scale_);

    calculate_group_absorption_and_scattering_coefficients(tess, cells_cgs, 0);
    calculate_planck_integrals(tess, cells_cgs);
    calculate_planck_absorption_coefficient(tess, cells);
    calculate_fleck_factor(tess, cells, 0);
}

bool MultigroupDiffusion::deferComptonForNonphysicalSolution(
    std::vector<double> const& pre_correction_solution,
    std::vector<double> const& physical_solution,
    std::vector<std::size_t> const* local_to_global,
    std::vector<ComputationalCell3D> const& cells,
    char const* const scope,
    bool const persist_for_individual_retry) const
{
    if(!compton_on_ || ENERGY_GROUPS_NUM == 0 ||
       compton_solution_rebuild_used_)
        return false;

    std::set<std::size_t> candidate_cells;
    if(local_to_global == nullptr) {
        std::size_t const count = std::min(
            compton_deferred_.size(),
            std::min(pre_correction_solution.size(), physical_solution.size()) /
                ENERGY_GROUPS_NUM);
        for(std::size_t cell = 0; cell < count; ++cell)
            candidate_cells.insert(cell);
    }
    else
        for(std::size_t const global_unknown : *local_to_global)
            candidate_cells.insert(global_unknown / ENERGY_GROUPS_NUM);

    // Use one immutable comparison scale for the original solve and its
    // Compton-free rebuild.  Recomputing this scale from the rejected candidate
    // makes the cell mask solver-path dependent: a large unrelated post-solve
    // group can move the 1e-10 threshold enough to include or exclude another
    // cell.  old_Eg is refreshed from the accepted pre-solve spectrum for both
    // global and individual solves.
    double global_maximum_absolute_Eg = 0;
    bool fixed_scale_valid = old_Eg.size() >= compton_deferred_.size();
    for(std::size_t cell = 0;
        fixed_scale_valid && cell < compton_deferred_.size(); ++cell) {
        fixed_scale_valid = old_Eg[cell].size() >= ENERGY_GROUPS_NUM;
        for(std::size_t group = 0;
            fixed_scale_valid && group < ENERGY_GROUPS_NUM; ++group) {
            double const energy = old_Eg[cell][group];
            fixed_scale_valid = std::isfinite(energy);
            if(fixed_scale_valid)
                global_maximum_absolute_Eg = std::max(
                    global_maximum_absolute_Eg, std::abs(energy));
        }
    }
#ifdef RICH_MPI
    int globally_valid = fixed_scale_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &globally_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    fixed_scale_valid = globally_valid != 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_maximum_absolute_Eg, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    // Defensive compatibility for an incompletely prepared diagnostic call.
    // Normal production paths always have a finite old_Eg snapshot.
    if(!fixed_scale_valid) {
        global_maximum_absolute_Eg = 0;
        for(double const energy : physical_solution)
            if(std::isfinite(energy))
                global_maximum_absolute_Eg = std::max(
                    global_maximum_absolute_Eg, std::abs(energy));
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &global_maximum_absolute_Eg, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    bool const emit_fallback_details =
        std::getenv("RICH_TEST_MG_COMPTON_FALLBACK_DETAILS") != nullptr;

    std::set<std::size_t> offending_cells;
    unsigned long long local_cells = 0;
    unsigned long long local_groups = 0;
    double local_maximum_deficit = 0;
    std::size_t representative_cell_id =
        std::numeric_limits<std::size_t>::max();
    std::size_t representative_group =
        std::numeric_limits<std::size_t>::max();
    double representative_group_energy = 0;
    double representative_positive_energy = 0;
    struct SpectrumNegativity
    {
        bool finite = true;
        bool significant = false;
        long double positive = 0;
        long double negative = 0;
        std::size_t negative_groups = 0;
        std::size_t most_negative_group = 0;
        double most_negative_energy = 0;
    };
    auto const analyze_spectrum = [global_maximum_absolute_Eg](
        std::vector<double> const& solution, std::size_t const cell)
    {
        SpectrumNegativity result;
        for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
            double const energy =
                solution[cell * ENERGY_GROUPS_NUM + group];
            if(!std::isfinite(energy)) {
                result.finite = false;
                return result;
            }
            if(energy >= 0)
                result.positive += static_cast<long double>(energy);
            else {
                result.negative -= static_cast<long double>(energy);
                ++result.negative_groups;
                if(energy < result.most_negative_energy) {
                    result.most_negative_energy = energy;
                    result.most_negative_group = group;
                }
            }
        }
        result.significant =
            CG::HistoricalMGNegativeValueRequiresComptonFallback(
                result.most_negative_energy, global_maximum_absolute_Eg);
        return result;
    };
    for(std::size_t const cell : candidate_cells) {
        if(cell >= cells.size() || cell >= compton_deferred_.size() ||
           cell >= compton_occupation_mode_.size() ||
           compton_deferred_[cell] ||
           compton_occupation_mode_[cell] == ComptonOccupationMode::Off ||
           (cell + 1) * ENERGY_GROUPS_NUM >
               pre_correction_solution.size() ||
           (cell + 1) * ENERGY_GROUPS_NUM > physical_solution.size())
            continue;

        SpectrumNegativity const pre =
            analyze_spectrum(pre_correction_solution, cell);
        SpectrumNegativity const post =
            analyze_spectrum(physical_solution, cell);
        bool const selected = pre.finite && post.finite &&
            CG::ShouldDeferComptonForResidualCorrectionCausality(
                pre.significant, post.significant);
        if(emit_fallback_details && pre.finite && post.finite &&
           (pre.significant || post.significant))
            std::clog << std::setprecision(17)
                      << "MG_COMPTON_FALLBACK_CELL_PROBE"
                      << " rank=" << rank
                      << " cell_id=" << cells[cell].ID
                      << " pre_group=" << pre.most_negative_group
                      << " pre_Eg=" << pre.most_negative_energy
                      << " pre_significant=" << pre.significant
                      << " post_group=" << post.most_negative_group
                      << " post_Eg=" << post.most_negative_energy
                      << " post_significant=" << post.significant
                      << " global_maximum_absolute_Eg="
                      << global_maximum_absolute_Eg
                      << " comparison_scale_source="
                      << (fixed_scale_valid ? "fixed_pre_solve" :
                                                "post_candidate_fallback")
                      << " threshold="
                      << CG::historical_mg_positivity_continuation_trigger_fraction *
                             global_maximum_absolute_Eg
                      << " selected=" << selected << std::endl;
        if(!selected)
            continue;

        double const relative_deficit = post.positive > 0
            ? static_cast<double>(post.negative / post.positive)
            : std::numeric_limits<double>::max();
        ++local_cells;
        offending_cells.insert(cell);
        local_groups +=
            static_cast<unsigned long long>(post.negative_groups);
        if(relative_deficit >= local_maximum_deficit) {
            local_maximum_deficit = relative_deficit;
            representative_cell_id = cells[cell].ID;
            representative_group = post.most_negative_group;
            representative_group_energy = post.most_negative_energy;
            representative_positive_energy =
                static_cast<double>(post.positive);
        }
    }

    unsigned long long global_counts[2] = {local_cells, local_groups};
    double global_maximum_deficit = local_maximum_deficit;
    int representative_rank = 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, global_counts, 2, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    struct DoubleRank
    {
        double value;
        int rank;
    } local_pick{local_maximum_deficit, rank}, global_pick{0, 0};
    MPI_Allreduce(&local_pick, &global_pick, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);
    global_maximum_deficit = global_pick.value;
    representative_rank = global_pick.rank;
    unsigned long long identity[2] = {
        static_cast<unsigned long long>(representative_cell_id),
        static_cast<unsigned long long>(representative_group)};
    double details[2] = {representative_group_energy,
                         representative_positive_energy};
    MPI_Bcast(identity, 2, MPI_UNSIGNED_LONG_LONG, representative_rank,
              MPI_COMM_WORLD);
    MPI_Bcast(details, 2, MPI_DOUBLE, representative_rank, MPI_COMM_WORLD);
    representative_cell_id = static_cast<std::size_t>(identity[0]);
    representative_group = static_cast<std::size_t>(identity[1]);
    representative_group_energy = details[0];
    representative_positive_energy = details[1];
#endif

    // Rebuild once with implicit Compton disabled for every cell that remains
    // significantly negative after the complete Krylov rescue budget.  The
    // mask is assembled in one batch; a newly offending unmasked cell in the
    // mixed rebuild is handled by the normal floor/rejection policy rather
    // than opening a second rebuild.
    unsigned long long deferred_cells = 0;
    if(global_counts[0] > 0) {
        compton_solution_rebuild_used_ = true;
        for(std::size_t const cell : offending_cells) {
            if(cell >= cells.size() || cell >= compton_deferred_.size() ||
               cell >= split_compton_cells_.size() ||
               cell >= compton_occupation_mode_.size() ||
               compton_occupation_mode_[cell] == ComptonOccupationMode::Off)
                continue;
            compton_deferred_[cell] = true;
            split_compton_cells_[cell] = true;
            if(persist_for_individual_retry)
                individual_compton_force_deferred_ids_.insert(cells[cell].ID);
            ++deferred_cells;
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &deferred_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(global_counts[0] > 0 && rank == 0)
        std::clog << std::setprecision(17)
                  << "MG_COMPTON_POSITIVITY_FALLBACK"
                  << " scope=" << scope
                  << " cells=" << global_counts[0]
                  << " negative_groups=" << global_counts[1]
                  << " deferred_cells=" << deferred_cells
                  << " max_relative_deficit=" << global_maximum_deficit
                  << " representative_rank=" << representative_rank
                  << " representative_cell_id=" << representative_cell_id
                  << " representative_group=" << representative_group
                  << " representative_group_energy="
                  << representative_group_energy
                  << " representative_positive_energy="
                  << representative_positive_energy
                  << " action=rebuild_without_local_implicit_compton"
                  << " defer_scope=offending_cells_batch"
                  << " rebuild_index=1"
                  << " rebuild_limit=1"
                  << std::endl;
    return global_counts[0] > 0;
}

bool MultigroupDiffusion::requestIndividualSolutionRetry(
    std::vector<double> const& pre_correction_solution,
    std::vector<double> const& physical_solution,
    std::vector<std::size_t> const& local_to_global,
    std::vector<ComputationalCell3D> const& cells,
    char const* const scope) const
{
    return deferComptonForNonphysicalSolution(
        pre_correction_solution, physical_solution, &local_to_global, cells,
        scope, true);
}

bool MultigroupDiffusion::applyIndividualPostSolvePhysics(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    double const fallback_dt,
    double const global_maximum_cell_radiation_extent) const
{
    return apply_operator_split_compton(
        tess, cells, extensives, fallback_dt,
        global_maximum_cell_radiation_extent);
}

void MultigroupDiffusion::appendPendingSpectralRepairEvent(
    SpectralRepairEvent& event) const
{
    SpectralRepairEvent const& pending =
        pending_split_spectral_repair_event_;
    event.repaired_cells += pending.repaired_cells;
    event.repaired_groups += pending.repaired_groups;
    event.injected_energy += pending.injected_energy;
    if(pending.maximum_relative_deficit >
       event.maximum_relative_deficit) {
        event.maximum_relative_deficit =
            pending.maximum_relative_deficit;
        event.representative_cell_id = pending.representative_cell_id;
        event.representative_group = pending.representative_group;
        event.representative_original_extent =
            pending.representative_original_extent;
        event.representative_floor_extent =
            pending.representative_floor_extent;
        event.representative_injected_extent =
            pending.representative_injected_extent;
    }
}

bool MultigroupDiffusion::repairMultigroupSpectraAfterStage(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    double const global_maximum_cell_radiation_extent,
    char const* const stage,
    SpectralRepairEvent& event) const
{
    std::size_t const owned_cells = std::min(
        tess.GetPointNo(), std::min(cells.size(), extensives.size()));
    bool local_valid = owned_cells == tess.GetPointNo();
    if(!local_valid) {
        std::ostringstream reason;
        reason << "multigroup radiation spectral validation has incomplete owned arrays"
               << " stage=" << stage
               << " owned_cells=" << tess.GetPointNo()
               << " primitive_cells=" << cells.size()
               << " extensive_cells=" << extensives.size();
        setStepFailure(reason.str());
    }
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    for(std::size_t cell = 0; cell < owned_cells; ++cell) {
        double const original_total_extent = extensives[cell].Erad;
        auto const controlled =
            RadiationPositivity::RepairControlledNegativeGroupExtents(
                extensives[cell].Eg, extensives[cell].Erad,
                RadiationPositivity::spectral_repair_relative_limit,
                global_maximum_cell_radiation_extent);
        RadiationPositivity::SpectralRepairResult const& repair =
            controlled.repair;
        if(!repair.valid) {
            std::ostringstream reason;
            reason << std::setprecision(17)
                   << "multigroup radiation spectral validation failed"
                   << " stage=" << stage
                   << " failure="
                   << RadiationPositivity::SpectralRepairFailureLabel(
                          repair.failure)
                   << " group=";
            if(repair.failure_group ==
               std::numeric_limits<std::size_t>::max())
                reason << repair.most_negative_group;
            else
                reason << repair.failure_group;
            reason << " extent="
                   << (repair.failure_group ==
                           std::numeric_limits<std::size_t>::max()
                       ? repair.most_negative_extent
                       : repair.failure_extent)
                   << " negative_extent=" << repair.negative_extent
                   << " positive_extent=" << repair.positive_extent
                   << " relative_deficit=" << repair.relative_deficit
                   << " local_tolerance="
                   << RadiationPositivity::spectral_repair_relative_limit
                   << " negative_extent_over_global_E_max="
                   << controlled.global_negative.
                          negative_extent_to_global_max_ratio
                   << " global_negative_tolerance="
                   << RadiationPositivity::
                          spectral_globally_negligible_negative_fraction
                   << " diagnostic_E_cell_over_E_max="
                   << controlled.diagnostic_total_to_global_max_ratio
                   << " global_E_max="
                   << global_maximum_cell_radiation_extent;
            setCellLocalStepFailure(reason.str(), cells[cell].ID);
            local_valid = false;
            continue;
        }

        if(!std::isfinite(extensives[cell].mass) ||
           extensives[cell].mass <= 0) {
            std::ostringstream reason;
            reason << std::setprecision(17)
                   << "multigroup radiation spectral repair has invalid cell mass"
                   << " stage=" << stage
                   << " mass=" << extensives[cell].mass;
            setCellLocalStepFailure(reason.str(), cells[cell].ID);
            local_valid = false;
            continue;
        }
        if(repair.repaired || controlled.aggregate_sync_correction != 0) {
            for(std::size_t group = 0;
                group < extensives[cell].Eg.size(); ++group)
                cells[cell].Eg[group] =
                    extensives[cell].Eg[group] / extensives[cell].mass;
            cells[cell].Erad =
                extensives[cell].Erad / extensives[cell].mass;
        }
        if(controlled.used_global_negative_exception)
            std::clog << std::setprecision(17)
                      << "MG_SPECTRAL_POSITIVITY_REPAIR"
                      << " scope=" << stage
                      << " rank=" << rank
                      << " cell_id=" << cells[cell].ID
                      << " E_cell=" << original_total_extent
                      << " global_E_max="
                      << global_maximum_cell_radiation_extent
                      << " diagnostic_E_cell_over_E_max="
                      << controlled.diagnostic_total_to_global_max_ratio
                      << " group=" << repair.most_negative_group
                      << " signed_group_extent="
                      << repair.most_negative_extent
                      << " negative_extent=" << repair.negative_extent
                      << " positive_extent=" << repair.positive_extent
                      << " negative_extent_over_global_E_max="
                      << controlled.global_negative.
                             negative_extent_to_global_max_ratio
                      << " global_negative_tolerance="
                      << RadiationPositivity::
                             spectral_globally_negligible_negative_fraction
                      << " relative_deficit=" << repair.relative_deficit
                      << " affected_groups=" << repair.repaired_groups
                      << " floor_extent=" << repair.floor_extent
                      << " injected_extent=" << repair.injected_extent
                      << " aggregate_sync_correction="
                      << controlled.aggregate_sync_correction
                      << " action=globally_negligible_negative_nonconservative_floor"
                      << std::endl;
        if(!repair.repaired)
            continue;
        ++event.repaired_cells;
        event.repaired_groups += repair.repaired_groups;
        event.injected_energy += repair.injected_extent;
        if(repair.relative_deficit > event.maximum_relative_deficit) {
            event.maximum_relative_deficit = repair.relative_deficit;
            event.representative_cell_id = cells[cell].ID;
            event.representative_group = repair.most_negative_group;
            event.representative_original_extent =
                repair.most_negative_extent;
            event.representative_floor_extent = repair.floor_extent;
            event.representative_injected_extent = repair.injected_extent;
        }
    }

    int collective_valid = local_valid ? 1 : 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    if(collective_valid == 0 && local_valid &&
       getLastStepFailureReason().empty()) {
        std::ostringstream reason;
        reason << "multigroup radiation spectral validation failed on another rank"
               << " stage=" << stage;
        setStepFailure(reason.str());
        markStepFailureRemote();
    }
    return collective_valid != 0;
}

bool MultigroupDiffusion::step(double const tolerance,
                               int& total_iters,
                               Tessellation3D const& tess,
                               std::vector<ComputationalCell3D>& cells,
                               std::vector<Conserved3D>& extensives,
                               double const dt,
                               double const time) const {

    auto const N = tess.GetPointNo();
    bool const outer_transaction_covers_state =
        outerAllActiveTransactionCovers(cells, extensives);
    std::vector<ComputationalCell3D> const base_cells =
        outer_transaction_covers_state ?
        std::vector<ComputationalCell3D>() : cells;
    std::vector<Conserved3D> const base_extensives =
        outer_transaction_covers_state ? std::vector<Conserved3D>() :
        extensives;
    auto rollback = [&cells, &extensives, &base_cells, &base_extensives,
                     outer_transaction_covers_state]() {
        if(!outer_transaction_covers_state) {
            cells = base_cells;
            extensives = base_extensives;
        }
    };

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    bool const trace_step_phases =
        mgRuntimeFlagEnabled("RICH_INDIVIDUAL_PERF_TRACE");
    MultigroupClock::time_point const step_phase_total_start =
        trace_step_phases ? MultigroupClock::now() :
        MultigroupClock::time_point{};
    double group_opacity_seconds = 0.0;
    double planck_integral_seconds = 0.0;
    double planck_absorption_seconds = 0.0;
    double fleck_seconds = 0.0;
    double bicgstab_seconds = 0.0;
    double postcg_seconds = 0.0;
    double split_compton_repair_seconds = 0.0;

    total_iters = 0;
    compton_solution_rebuild_used_ = false;
    split_compton_cells_.assign(N, false);
    split_subcycle_count_ = 0;
    split_suppressed_energy_ = 0.0;
    split_injected_energy_ = 0.0;
    pending_split_spectral_repair_event_ = SpectralRepairEvent{};
    clearStepFailure();
    {
        if(!outer_transaction_covers_state) {
            cells = base_cells;
            extensives = base_extensives;
        }
        matrix_unrecoverable_ = false;
        postcg_unrecoverable_ = false;
        compton_deferred_.assign(N, false);
        compton_occupation_mode_.assign(N, ComptonOccupationMode::Off);
        compton_jacobian_frozen_.assign(N, false);

        // Each fractional retry/substep is a new backward-Euler problem.  Its
        // thermodynamic and radiation baseline must be the latest accepted
        // state, not the beginning of the enclosing scheduled step.
        old_Er.resize(N);
        old_Tm.resize(N);
        old_Eg.resize(N);
        for(std::size_t i = 0; i < N; ++i) {
            old_Er[i] = cells[i].Erad * cells[i].density;
            old_Tm[i] = cells[i].temperature;
            old_Eg[i].resize(ENERGY_GROUPS_NUM);
            for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                old_Eg[i][group] = cells[i].Eg[group] * cells[i].density;
        }

        cells_cgs = cells;
        for(std::size_t i = 0; i < N; ++i)
            cells_cgs[i] = multigroupRadiationCellInCgs(
                cells[i], length_scale_, time_scale_, mass_scale_);

#ifdef RICH_MPI
        MPI_exchange_data(tess, cells_cgs, true);
#endif
        char const* const reset_limiter_value =
            std::getenv("RICH_MG_RESET_COMPTON_LIMITER_CANDIDATE");
        std::string const reset_limiter_setting =
            reset_limiter_value == nullptr ? "" :
            std::string(reset_limiter_value);
        bool const reset_limiter_valid = reset_limiter_setting.empty() ||
            reset_limiter_setting == "0" ||
            reset_limiter_setting == "false" ||
            reset_limiter_setting == "off" ||
            reset_limiter_setting == "no" ||
            reset_limiter_setting == "1" ||
            reset_limiter_setting == "true" ||
            reset_limiter_setting == "on" ||
            reset_limiter_setting == "yes";
        int collectively_reset_limiter = reset_limiter_valid &&
            (reset_limiter_setting == "1" ||
             reset_limiter_setting == "true" ||
             reset_limiter_setting == "on" ||
             reset_limiter_setting == "yes") ? 1 : 0;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &collectively_reset_limiter, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
#endif
        if(collectively_reset_limiter != 0)
            compton_limiter_scale_.assign(N, 1.0);
        {
            MultigroupPhaseTimer timer(
                trace_step_phases, group_opacity_seconds);
            calculate_group_absorption_and_scattering_coefficients(
                tess, cells_cgs, dt * time_scale_);
        }
        {
            MultigroupPhaseTimer timer(
                trace_step_phases, planck_integral_seconds);
            calculate_planck_integrals(tess, cells_cgs);
        }
        {
            MultigroupPhaseTimer timer(
                trace_step_phases, planck_absorption_seconds);
            calculate_planck_absorption_coefficient(tess, cells);
        }
        {
            MultigroupPhaseTimer timer(trace_step_phases, fleck_seconds);
            calculate_fleck_factor(tess, cells, dt * time_scale_);
        }

        bool good_end = false;
        for(;;) {
            try {
                int iteration_count = 0;
                {
                    MultigroupPhaseTimer timer(
                        trace_step_phases, bicgstab_seconds);
                    new_Eg = CG::BiCGSTAB(tolerance, iteration_count, tess,
                        cells, dt, *this, time, new_Eg_full, good_end,
                        cg_workspace_);
                }
                total_iters += iteration_count;
            } catch (UniversalError const&) {
                if (matrix_unrecoverable_) {
                    if (getLastStepFailureReason().empty())
                        setStepFailure("radiation matrix diagonal below 0.25 cell volume after Compton removal");
                    rollback();
                    return false;
                }
                throw;
            }
            MEMORY_DEBUG_PRINT("multigroup: after BiCGSTAB");
            if (!good_end) {
                CG::HistoricalMGResidualCorrectionDiagnostics const&
                    diagnostic = cg_workspace_.historical_correction;
                if(!diagnostic.failure_reason.empty()) {
                    std::ostringstream reason;
                    CG::AppendHistoricalMGResidualCorrectionFailureDiagnostics(
                        reason, diagnostic,
                        static_cast<std::size_t>(std::max(total_iters, 0)));
                    setCellLocalStepFailure(
                        reason.str(), diagnostic.failure_cell_id);
                }
                else
                    setStepFailure("BiCGSTAB did not converge");
                rollback();
                return false;
            }
            std::vector<double> const& pre_correction_solution =
                cg_workspace_.historical_correction.available &&
                cg_workspace_.historical_correction.pre_correction_solution.size() ==
                    new_Eg_full.size()
                ? cg_workspace_.historical_correction.pre_correction_solution
                : new_Eg_full;
            if(!deferComptonForNonphysicalSolution(
                   pre_correction_solution, new_Eg_full, nullptr, cells,
                   "global", false))
                break;
            // Removing the implicit Compton block also changes Gamma and the
            // Fleck factor for every participating Compton-enabled cell.
            // Rebuild that thermodynamic coupling before the single
            // replacement solve, matching the individual-step retry.
            {
                MultigroupPhaseTimer timer(trace_step_phases, fleck_seconds);
                calculate_fleck_factor(tess, cells, dt * time_scale_);
            }
            // The replacement matrix has different cell-local Compton blocks.
            // Drop the first solve's retained global CSR and Krylov storage
            // before rebuilding it; otherwise both complete systems overlap.
            invalidateDirectStructureCache(false);
            cg_workspace_.Release();
            rich_trim_after_rare_spike();
            good_end = false;
        }

        {
            MultigroupPhaseTimer timer(trace_step_phases, postcg_seconds);
            PostCG(tess, extensives, dt, cells, new_Eg, new_Eg_full);
        }
        MEMORY_DEBUG_PRINT("multigroup: after PostCG");
        if (postcg_unrecoverable_) {
            if (getLastStepFailureReason().empty())
                setStepFailure("PostCG rejected one or more cells");
            rollback();
            return false;
        }

        MultigroupClock::time_point const split_repair_start =
            trace_step_phases ? MultigroupClock::now() :
            MultigroupClock::time_point{};
        SpectralRepairEvent spectral_repair_event;
        double global_maximum_post_absorption_cell_radiation_extent = 0;
        for(std::size_t i = 0; i < N; ++i)
            if(std::isfinite(extensives[i].Erad))
                global_maximum_post_absorption_cell_radiation_extent =
                    std::max(
                    global_maximum_post_absorption_cell_radiation_extent,
                    extensives[i].Erad);
#ifdef RICH_MPI
        MPI_Allreduce(
            MPI_IN_PLACE,
            &global_maximum_post_absorption_cell_radiation_extent, 1,
            MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(!repairMultigroupSpectraAfterStage(
               tess, cells, extensives,
               global_maximum_post_absorption_cell_radiation_extent,
               "post_absorption_diffusion", spectral_repair_event)) {
            rollback();
            return false;
        }

        if(!apply_operator_split_compton(
               tess, cells, extensives, dt,
               global_maximum_post_absorption_cell_radiation_extent)) {
            if(getLastStepFailureReason().empty())
                setStepFailure("operator-split Compton solve failed");
            rollback();
            return false;
        }

        double global_maximum_post_compton_cell_radiation_extent = 0;
        for(std::size_t i = 0; i < N; ++i)
            if(std::isfinite(extensives[i].Erad))
                global_maximum_post_compton_cell_radiation_extent = std::max(
                    global_maximum_post_compton_cell_radiation_extent,
                    extensives[i].Erad);
#ifdef RICH_MPI
        MPI_Allreduce(
            MPI_IN_PLACE,
            &global_maximum_post_compton_cell_radiation_extent, 1,
            MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(!repairMultigroupSpectraAfterStage(
               tess, cells, extensives,
               global_maximum_post_compton_cell_radiation_extent,
               "post_compton", spectral_repair_event)) {
            rollback();
            return false;
        }

        int local_valid = 1;
        size_t first_invalid_cell_id = std::numeric_limits<size_t>::max();
        for(std::size_t i = 0; i < N; ++i)
            spectral_repair_event.owned_radiation_energy +=
                extensives[i].Erad;
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(cells[i].internal_energy) || cells[i].internal_energy < 0.0) {
                local_valid = 0;
                if (first_invalid_cell_id == std::numeric_limits<size_t>::max())
                    first_invalid_cell_id = cells[i].ID;
            }
            for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                if (!std::isfinite(cells[i].Eg[g]) || cells[i].Eg[g] < 0.0) {
                    local_valid = 0;
                    if (first_invalid_cell_id == std::numeric_limits<size_t>::max())
                        first_invalid_cell_id = cells[i].ID;
                }
            }
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &local_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        if (local_valid == 0) {
            if(getLastStepFailureReason().empty())
                setStepFailure("non-finite or negative gas/radiation state after radiation step", first_invalid_cell_id);
            rollback();
            return false;
        }

        appendPendingSpectralRepairEvent(spectral_repair_event);
        commitResidualCorrectionAccounting(
            cg_workspace_.historical_correction);
        commitSpectralRepairAccounting(
            spectral_repair_event,
            individual_context_ != nullptr
                ? "individual_all_active_global"
#ifdef RICH_MPI
                : "distributed_global");
#else
                : "serial_global");
#endif

        if(trace_step_phases)
            split_compton_repair_seconds +=
                mgElapsedSeconds(split_repair_start);
        int split_count = static_cast<int>(std::count(split_compton_cells_.begin(), split_compton_cells_.end(), true));
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &split_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &split_suppressed_energy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &split_injected_energy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        if (RuntimeLogDetailed() && rank == 0)
            std::cout << "Total iterations: " << total_iters
                      << " split Compton cells " << split_count
                      << " split substeps " << split_subcycle_count_
                      << " suppressed energy " << split_suppressed_energy_
                      << " injected energy " << split_injected_energy_ << std::endl;
#ifdef RICH_MPI
        MPI_exchange_data(tess, cells, true);
#endif
        if(trace_step_phases && rank == 0) {
            double const total_seconds =
                mgElapsedSeconds(step_phase_total_start);
            double const classified_seconds =
                group_opacity_seconds + planck_integral_seconds +
                planck_absorption_seconds + fleck_seconds +
                bicgstab_seconds + postcg_seconds +
                split_compton_repair_seconds;
            RuntimeTraceStream()
                      << "MG_GLOBAL_STEP_PHASE_TIMING scope=rank_local"
                      << " rank=" << rank
                      << " group_opacity_seconds=" << group_opacity_seconds
                      << " planck_integral_seconds=" << planck_integral_seconds
                      << " planck_absorption_seconds="
                      << planck_absorption_seconds
                      << " fleck_seconds=" << fleck_seconds
                      << " bicgstab_inclusive_seconds=" << bicgstab_seconds
                      << " postcg_seconds=" << postcg_seconds
                      << " split_compton_repair_seconds="
                      << split_compton_repair_seconds
                      << " unclassified_seconds="
                      << std::max(0.0, total_seconds - classified_seconds)
                      << " total_seconds=" << total_seconds
                      << std::endl;
        }
        return true;
    }

    return false;
}

bool MultigroupDiffusion::prepare_compton_only_fleck(
    ComputationalCell3D const& cell,
    std::size_t const cell_index,
    double const dt_cgs,
    double& inverse_cv_bar,
    double& compton_fleck,
    double& compton_upsilon,
    ComptonOccupationMode& occupation_mode) const
{
    if(!compton_on_ || cell_index >= old_Tm.size() ||
       !std::isfinite(dt_cgs) || dt_cgs <= 0.0)
        return false;

    double const T = old_Tm[cell_index];
    if(!std::isfinite(T) || T <= 0.0)
        return false;

    double cv = 0.0;
    try {
        cv = eos_.dT2cv(cell.density, T, cell.tracers,
                        ComputationalCell3D::tracerNames);
    }
    catch(UniversalError const&) {
        return false;
    }
    double const material_energy_density =
        cell.internal_energy * cell.density;
    double const energy_ratio =
        std::isfinite(material_energy_density) &&
        material_energy_density > 0.0 && std::isfinite(cv) && cv > 0.0
            ? cv * T / material_energy_density
            : 1.0;
    double const beta_scale = std::max(1.0, 0.5 * energy_ratio);
    cv *= mass_scale_ / (pow<2>(time_scale_) * length_scale_);
    double const radiation_cv = get_radiation_cv(T);
    if(!std::isfinite(cv) || cv <= 0.0 ||
       !std::isfinite(radiation_cv) || radiation_cv <= 0.0)
        return false;
    inverse_cv_bar = beta_scale * radiation_cv / cv;

    compton_jacobian_frozen_[cell_index] = false;
    auto evaluate = [&](ComptonOccupationMode const mode) {
        generate_S_and_dSdUm_matrices(cell, cell_index, dt_cgs, mode);
        return calculate_Upsilon(cell);
    };

    occupation_mode = ComptonOccupationMode::RadiationField;
    double best_upsilon = evaluate(occupation_mode);
    upsilon_erad_[cell_index] = best_upsilon;
    upsilon_lte_[cell_index] = std::numeric_limits<double>::quiet_NaN();
    upsilon_n0_[cell_index] = std::numeric_limits<double>::quiet_NaN();

    double const coupling = std::abs(best_upsilon) * inverse_cv_bar *
        CG::speed_of_light * dt_cgs;
    if(std::isfinite(coupling) && coupling > 0.1) {
        double const lte_upsilon = evaluate(
            ComptonOccupationMode::PlanckFunction);
        upsilon_lte_[cell_index] = lte_upsilon;
        if(std::isfinite(lte_upsilon) &&
           (!std::isfinite(best_upsilon) || lte_upsilon > best_upsilon)) {
            best_upsilon = lte_upsilon;
            occupation_mode = ComptonOccupationMode::PlanckFunction;
        }
    }
    if(!std::isfinite(best_upsilon) || best_upsilon < 0.0) {
        double const zero_upsilon = evaluate(ComptonOccupationMode::Zero);
        upsilon_n0_[cell_index] = zero_upsilon;
        if(std::isfinite(zero_upsilon) &&
           (!std::isfinite(best_upsilon) || zero_upsilon > best_upsilon)) {
            best_upsilon = zero_upsilon;
            occupation_mode = ComptonOccupationMode::Zero;
        }
    }
    // Regenerate the operator for the selected occupation.  The split Fleck
    // factor contains only the Compton material Jacobian Upsilon: absorption
    // was already advanced by the preceding coupled retry.
    generate_S_and_dSdUm_matrices(cell, cell_index, dt_cgs,
                                  occupation_mode);
    compton_upsilon = calculate_Upsilon(cell);
    if(!std::isfinite(compton_upsilon) || compton_upsilon < 0.0) {
        // A bad material derivative must not force an immediate timestep
        // reduction.  Keep the finite redistribution operator S, freeze its
        // material derivative, and first try the f_C=1 Compton block.
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt)
                if(!std::isfinite(S[g][gt]))
                    return false;
        fill_zero(dSdUm);
        compton_jacobian_frozen_[cell_index] = true;
        compton_upsilon = 0.0;
        compton_fleck = 1.0;
    }
    else
        compton_fleck = CG::FleckFactor(dt_cgs, inverse_cv_bar,
                                        compton_upsilon);
    if(!std::isfinite(compton_fleck) || compton_fleck <= 0.0 ||
       compton_fleck > 1.0)
        return false;

    use_n_zero[cell_index] = occupation_mode == ComptonOccupationMode::Zero;
    compton_occupation_mode_[cell_index] = occupation_mode;
    upsilon_[cell_index] = compton_upsilon;
    return true;
}

bool MultigroupDiffusion::solve_local_compton_substep(
    Tessellation3D const& tess,
    std::size_t const cell_index,
    ComputationalCell3D& cell,
    Conserved3D& extensive,
    double const dt,
    double const global_maximum_cell_radiation_extent,
    bool const emit_diagnostics,
    SplitComptonDiagnosticSummary& diagnostics) const
{
    std::size_t const no_failure_group =
        std::numeric_limits<std::size_t>::max();
    double const no_failure_extent =
        std::numeric_limits<double>::quiet_NaN();
    auto record_basic_failure = [&](char const* const failure,
                                    std::size_t const group,
                                    double const extent) {
        diagnostics.latest_failure_reason = failure;
        diagnostics.latest_failure_cell_id =
            static_cast<unsigned long long>(cell.ID);
        diagnostics.latest_failure_group =
            static_cast<unsigned long long>(group);
        diagnostics.latest_failure_dt = dt;
        diagnostics.latest_failure_extent = extent;
        diagnostics.latest_negative_extent = 0;
        diagnostics.latest_positive_extent = 0;
        diagnostics.latest_negative_to_global_max_ratio =
            std::numeric_limits<double>::quiet_NaN();
        diagnostics.latest_total_to_global_max_ratio =
            std::isfinite(extensive.Erad) &&
            std::isfinite(global_maximum_cell_radiation_extent) &&
            global_maximum_cell_radiation_extent > 0
                ? extensive.Erad /
                      global_maximum_cell_radiation_extent
                : std::numeric_limits<double>::quiet_NaN();
        diagnostics.latest_beta = 0;
        diagnostics.latest_gamma = 0;
        diagnostics.latest_kappa_planck =
            cell_index < sigma_absorption_planck.size()
                ? sigma_absorption_planck[cell_index] : 0;
        diagnostics.latest_fleck = 0;
        diagnostics.latest_occupation = ComptonOccupationMode::Off;
        diagnostics.latest_repair_attempted = false;
        diagnostics.latest_repair_injected = false;
        diagnostics.latest_upsilon_fallback = false;
    };
    record_basic_failure("none", no_failure_group, no_failure_extent);
    if (extensive.mass <= 0.0 || !std::isfinite(extensive.mass)
        || !std::isfinite(cell.temperature) || cell.temperature <= 0.0
        || !std::isfinite(cell.density) || cell.density <= 0.0) {
        record_basic_failure(
            "invalid_cell_thermodynamic_state",
            no_failure_group, no_failure_extent);
        return false;
    }

    ComputationalCell3D const saved_cell = cell;
    Conserved3D const saved_extensive = extensive;
    double const saved_old_temperature = old_Tm[cell_index];
    double const saved_split_suppressed_energy = split_suppressed_energy_;
    double const saved_split_injected_energy = split_injected_energy_;
    SpectralRepairEvent const saved_pending_repair_event =
        pending_split_spectral_repair_event_;
    auto rollback_substep_accounting = [&]() {
        split_suppressed_energy_ = saved_split_suppressed_energy;
        split_injected_energy_ = saved_split_injected_energy;
        pending_split_spectral_repair_event_ = saved_pending_repair_event;
    };
    try {
    old_Tm[cell_index] = cell.temperature;

    double const density_factor = mass_scale_ / (length_scale_ * pow<2>(time_scale_));
    double const volume = tess.GetVolume(cell_index) * pow<3>(length_scale_);
    double const extensive_factor = volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
    if(!std::isfinite(extensive_factor) || extensive_factor <= 0) {
        record_basic_failure(
            "invalid_extensive_conversion",
            no_failure_group, no_failure_extent);
        return false;
    }
    std::vector<double> old_group_cgs(ENERGY_GROUPS_NUM, 0.0);
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        old_group_cgs[g] = cell.Eg[g] * cell.density * density_factor;
        if (!std::isfinite(old_group_cgs[g]) || old_group_cgs[g] < 0.0) {
            record_basic_failure(
                "invalid_pre_compton_group_state", g, old_group_cgs[g]);
            old_Tm[cell_index] = saved_old_temperature;
            return false;
        }
    }

    double const dt_cgs = dt * time_scale_;
    double inverse_cv_bar = 0.0;
    double compton_fleck = 0.0;
    double compton_upsilon = 0.0;
    ComptonOccupationMode occupation_mode =
        ComptonOccupationMode::RadiationField;
    if(!prepare_compton_only_fleck(cell, cell_index, dt_cgs,
                                   inverse_cv_bar, compton_fleck,
                                   compton_upsilon, occupation_mode)) {
        record_basic_failure(
            "invalid_compton_only_fleck_state",
            no_failure_group, no_failure_extent);
        diagnostics.latest_beta = inverse_cv_bar;
        diagnostics.latest_gamma = compton_upsilon;
        diagnostics.latest_fleck = compton_fleck;
        diagnostics.latest_occupation = occupation_mode;
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }

    double const cdt = CG::speed_of_light * dt_cgs;
    auto solve_candidate = [&](double const candidate_fleck,
                               bool const allow_controlled_repair,
                               std::vector<double>& candidate,
                               std::size_t& bad_group,
                               double& bad_value,
                               RadiationPositivity::
                                   ControlledSpectralRepairResult&
                                       repair_diagnostic) {
        std::vector<double> matrix(
            ENERGY_GROUPS_NUM * ENERGY_GROUPS_NUM, 0.0);
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            long double dsource_dmaterial_times_energy = 0.0;
            for(std::size_t q = 0; q < ENERGY_GROUPS_NUM; ++q)
                dsource_dmaterial_times_energy +=
                    static_cast<long double>(dSdUm[q][g]) *
                    static_cast<long double>(old_group_cgs[q]);
            double const material_feedback = cdt * cdt * inverse_cv_bar *
                candidate_fleck *
                static_cast<double>(dsource_dmaterial_times_energy);
            for (std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt) {
                long double source_sum = 0.0;
                for(std::size_t q = 0; q < ENERGY_GROUPS_NUM; ++q)
                    source_sum += static_cast<long double>(S[gt][q]);
                matrix[g * ENERGY_GROUPS_NUM + gt] =
                    (g == gt ? 1.0 : 0.0) - cdt * S[gt][g] +
                    material_feedback * static_cast<double>(source_sum);
            }
        }
        candidate = old_group_cgs;
        if(!solve_dense_system(matrix, candidate, ENERGY_GROUPS_NUM)) {
            bad_group = std::numeric_limits<std::size_t>::max();
            bad_value = std::numeric_limits<double>::quiet_NaN();
            return false;
        }
        bool has_negative_group = false;
        long double candidate_sum = 0.0;
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            if(!std::isfinite(candidate[g])) {
                bad_group = g;
                bad_value = candidate[g];
                return false;
            }
            candidate_sum += static_cast<long double>(candidate[g]);
            if(candidate[g] < 0.0) {
                if(!has_negative_group) {
                    bad_group = g;
                    bad_value = candidate[g];
                }
                has_negative_group = true;
            }
        }
        if(has_negative_group) {
            if(!allow_controlled_repair)
                return false;
            std::vector<double> repaired_candidate_extensive = candidate;
            for(double& group_extent : repaired_candidate_extensive)
                group_extent *= extensive_factor;
            double candidate_total_extent =
                static_cast<double>(candidate_sum) * extensive_factor;
            repair_diagnostic =
                RadiationPositivity::RepairControlledNegativeGroupExtents(
                    repaired_candidate_extensive,
                    candidate_total_extent,
                    RadiationPositivity::spectral_repair_relative_limit,
                    global_maximum_cell_radiation_extent);
            if(!repair_diagnostic.repair.valid) {
                bad_group = repair_diagnostic.repair.most_negative_group;
                bad_value = repair_diagnostic.repair.most_negative_extent;
                return false;
            }
        }
        return true;
    };

    int split_rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &split_rank);
#endif
    double const original_compton_upsilon = compton_upsilon;
    double const original_compton_fleck = compton_fleck;
    bool used_frozen_jacobian_retry = false;
    std::size_t bad_group = std::numeric_limits<std::size_t>::max();
    double bad_value = std::numeric_limits<double>::quiet_NaN();
    RadiationPositivity::ControlledSpectralRepairResult
        candidate_repair_diagnostic;
    std::vector<double> new_group_cgs;
    bool candidate_valid = solve_candidate(
        compton_fleck, compton_jacobian_frozen_[cell_index],
        new_group_cgs, bad_group, bad_value,
        candidate_repair_diagnostic);
    if(!candidate_valid &&
       !compton_jacobian_frozen_[cell_index]) {
        // A finite Upsilon can still make the material-Jacobian correction
        // non-monotone.  Retry this same substep with dS/dUm=0 before asking
        // the event controller to halve the timestep.
        used_frozen_jacobian_retry = true;
        compton_jacobian_frozen_[cell_index] = true;
        generate_S_and_dSdUm_matrices(cell, cell_index, dt_cgs,
                                      occupation_mode);
        fill_zero(dSdUm);
        compton_upsilon = 0.0;
        compton_fleck = 1.0;
        upsilon_[cell_index] = 0.0;
        if(diagnostics.upsilon_fallback_events == 0) {
            diagnostics.representative_cell_id =
                static_cast<unsigned long long>(cell.ID);
            diagnostics.representative_failed_group =
                static_cast<unsigned long long>(bad_group);
            diagnostics.representative_dt = dt;
            diagnostics.representative_original_upsilon =
                original_compton_upsilon;
            diagnostics.representative_original_fleck =
                original_compton_fleck;
            diagnostics.representative_failed_value = bad_value;
        }
        ++diagnostics.upsilon_fallback_events;
        candidate_repair_diagnostic =
            RadiationPositivity::ControlledSpectralRepairResult{};
        candidate_valid = solve_candidate(
            compton_fleck, true, new_group_cgs, bad_group, bad_value,
            candidate_repair_diagnostic);
    }
    if(!candidate_valid) {
        diagnostics.latest_failure_reason =
            candidate_repair_diagnostic.repair.valid
                ? "compton_candidate_linear_solve_nonphysical"
                : RadiationPositivity::SpectralRepairFailureLabel(
                      candidate_repair_diagnostic.repair.failure);
        diagnostics.latest_failure_cell_id =
            static_cast<unsigned long long>(cell.ID);
        diagnostics.latest_failure_group =
            static_cast<unsigned long long>(bad_group);
        diagnostics.latest_failure_dt = dt;
        diagnostics.latest_failure_extent = bad_value;
        diagnostics.latest_negative_extent =
            candidate_repair_diagnostic.repair.negative_extent;
        diagnostics.latest_positive_extent =
            candidate_repair_diagnostic.repair.positive_extent;
        diagnostics.latest_negative_to_global_max_ratio =
            candidate_repair_diagnostic.global_negative.
                negative_extent_to_global_max_ratio;
        diagnostics.latest_total_to_global_max_ratio =
            candidate_repair_diagnostic.
                diagnostic_total_to_global_max_ratio;
        diagnostics.latest_beta = inverse_cv_bar;
        diagnostics.latest_gamma = compton_upsilon;
        diagnostics.latest_kappa_planck =
            cell_index < sigma_absorption_planck.size()
                ? sigma_absorption_planck[cell_index] : 0;
        diagnostics.latest_fleck = compton_fleck;
        diagnostics.latest_occupation = occupation_mode;
        diagnostics.latest_repair_attempted =
            candidate_repair_diagnostic.repair.repaired_groups > 0;
        diagnostics.latest_repair_injected = false;
        diagnostics.latest_upsilon_fallback =
            used_frozen_jacobian_retry;
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }
    if(emit_diagnostics)
        std::clog << std::setprecision(17)
                  << "MG_COMPTON_ONLY_FLECK"
                  << " scope=representative"
                  << " rank=" << split_rank
                  << " cell_id=" << cell.ID
                  << " dt=" << dt
                  << " beta=" << inverse_cv_bar
                  << " upsilon=" << compton_upsilon
                  << " gamma=" << compton_upsilon
                  << " kappa_planck="
                  << (cell_index < sigma_absorption_planck.size()
                      ? sigma_absorption_planck[cell_index] : 0)
                  << " fleck=" << compton_fleck
                  << " occupation="
                  << comptonOccupationModeLabel(occupation_mode)
                  << " jacobian_frozen="
                  << (compton_jacobian_frozen_[cell_index] ? 1 : 0)
                  << " upsilon_fallback="
                  << (used_frozen_jacobian_retry ? 1 : 0)
                  << " compton_in_fleck=1"
                  << " absorption_in_fleck=0"
                  << std::endl;

    std::vector<double> old_group_ext(ENERGY_GROUPS_NUM, 0.0);
    std::vector<double> new_group_ext(ENERGY_GROUPS_NUM, 0.0);
    double old_radiation_ext = 0.0;
    double new_radiation_ext = 0.0;
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        old_group_ext[g] = extensive.Eg[g];
        new_group_ext[g] = new_group_cgs[g] * extensive_factor;
        old_radiation_ext += old_group_ext[g];
        new_radiation_ext += new_group_ext[g];
        if (!std::isfinite(new_group_ext[g])) {
            record_basic_failure(
                "nonfinite_post_compton_group_extent", g,
                new_group_ext[g]);
            diagnostics.latest_beta = inverse_cv_bar;
            diagnostics.latest_gamma = compton_upsilon;
            diagnostics.latest_fleck = compton_fleck;
            diagnostics.latest_occupation = occupation_mode;
            cell = saved_cell;
            extensive = saved_extensive;
            old_Tm[cell_index] = saved_old_temperature;
            return false;
        }
    }

    double const radiation_delta = new_radiation_ext - old_radiation_ext;
    double const old_internal_ext = extensive.internal_energy;
    double new_internal_ext = old_internal_ext - radiation_delta;
    double const floor_specific = minimum_temperature_ > 0.0
        ? eos_.dT2e(cell.density, minimum_temperature_, cell.tracers, ComputationalCell3D::tracerNames)
        : 0.0;
    double const floor_ext = floor_specific * extensive.mass;
    if (!std::isfinite(new_internal_ext)) {
        record_basic_failure(
            "nonfinite_post_compton_material_extent",
            no_failure_group, no_failure_extent);
        diagnostics.latest_beta = inverse_cv_bar;
        diagnostics.latest_gamma = compton_upsilon;
        diagnostics.latest_fleck = compton_fleck;
        diagnostics.latest_occupation = occupation_mode;
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }

    if (new_internal_ext < floor_ext) {
        double const conserved_total = old_internal_ext + old_radiation_ext;
        double desired_radiation = 0.0;
        if (conserved_total >= floor_ext) {
            desired_radiation = conserved_total - floor_ext;
        } else {
            // The cell does not contain enough energy to reach the EOS floor.
            // Inject only the deficit and keep the nonnegative radiation
            // spectrum produced by the local solve.
            desired_radiation = new_radiation_ext;
            double const injected = floor_ext + desired_radiation - conserved_total;
            if (injected > 0.0) {
                split_injected_energy_ += injected;
            }
        }

        std::vector<double> weights(ENERGY_GROUPS_NUM, 0.0);
        double weight_sum = 0.0;
        double const lte_temperature = minimum_temperature_ > 0.0
            ? minimum_temperature_ : std::max(cell.temperature, 1e-200);
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            double const a = energy_groups_boundary[g] / (CG::boltzmann_constant * lte_temperature);
            double const b = energy_groups_boundary[g + 1] / (CG::boltzmann_constant * lte_temperature);
            weights[g] = std::max(0.0, planck_integral::planck_integral(a, b));
            weight_sum += weights[g];
        }
        if (weight_sum <= 0.0) {
            std::fill(weights.begin(), weights.end(), 1.0);
            weight_sum = static_cast<double>(ENERGY_GROUPS_NUM);
        }
        split_suppressed_energy_ += std::max(0.0, new_radiation_ext - desired_radiation);
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            new_group_ext[g] = desired_radiation * weights[g] / weight_sum;
        new_radiation_ext = desired_radiation;
        new_internal_ext = floor_ext;
    }
    auto const controlled_repair =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            new_group_ext, new_radiation_ext,
            RadiationPositivity::spectral_repair_relative_limit,
            global_maximum_cell_radiation_extent);
    RadiationPositivity::SpectralRepairResult const& repair =
        controlled_repair.repair;
    if(!repair.valid) {
        diagnostics.latest_failure_reason =
            RadiationPositivity::SpectralRepairFailureLabel(repair.failure);
        diagnostics.latest_failure_cell_id =
            static_cast<unsigned long long>(cell.ID);
        diagnostics.latest_failure_group =
            static_cast<unsigned long long>(repair.most_negative_group);
        diagnostics.latest_failure_dt = dt;
        diagnostics.latest_failure_extent = repair.most_negative_extent;
        diagnostics.latest_negative_extent = repair.negative_extent;
        diagnostics.latest_positive_extent = repair.positive_extent;
        diagnostics.latest_negative_to_global_max_ratio =
            controlled_repair.global_negative.
                negative_extent_to_global_max_ratio;
        diagnostics.latest_total_to_global_max_ratio =
            controlled_repair.diagnostic_total_to_global_max_ratio;
        diagnostics.latest_beta = inverse_cv_bar;
        diagnostics.latest_gamma = compton_upsilon;
        diagnostics.latest_kappa_planck =
            cell_index < sigma_absorption_planck.size()
                ? sigma_absorption_planck[cell_index] : 0;
        diagnostics.latest_fleck = compton_fleck;
        diagnostics.latest_occupation = occupation_mode;
        diagnostics.latest_repair_attempted = repair.repaired_groups > 0;
        diagnostics.latest_repair_injected = false;
        diagnostics.latest_upsilon_fallback =
            used_frozen_jacobian_retry;
        rollback_substep_accounting();
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }
    if(repair.repaired) {
        split_injected_energy_ += repair.injected_extent;
        ++pending_split_spectral_repair_event_.repaired_cells;
        pending_split_spectral_repair_event_.repaired_groups +=
            repair.repaired_groups;
        pending_split_spectral_repair_event_.injected_energy +=
            repair.injected_extent;
        if(repair.relative_deficit >
           pending_split_spectral_repair_event_.maximum_relative_deficit) {
            pending_split_spectral_repair_event_.maximum_relative_deficit =
                repair.relative_deficit;
            pending_split_spectral_repair_event_.representative_cell_id =
                cell.ID;
            pending_split_spectral_repair_event_.representative_group =
                repair.most_negative_group;
            pending_split_spectral_repair_event_.representative_original_extent =
                repair.most_negative_extent;
            pending_split_spectral_repair_event_.representative_floor_extent =
                repair.floor_extent;
            pending_split_spectral_repair_event_.representative_injected_extent =
                repair.injected_extent;
        }
    }
    if(controlled_repair.used_global_negative_exception)
        std::clog << std::setprecision(17)
                  << "MG_SPECTRAL_POSITIVITY_REPAIR"
                  << " scope=post_compton_substep"
                  << " rank=" << split_rank
                  << " cell_id=" << cell.ID
                  << " dt=" << dt
                  << " group=" << repair.most_negative_group
                  << " signed_group_extent="
                  << repair.most_negative_extent
                  << " negative_extent=" << repair.negative_extent
                  << " positive_extent=" << repair.positive_extent
                  << " negative_extent_over_global_E_max="
                  << controlled_repair.global_negative.
                         negative_extent_to_global_max_ratio
                  << " global_negative_tolerance="
                  << RadiationPositivity::
                         spectral_globally_negligible_negative_fraction
                  << " diagnostic_E_cell_over_E_max="
                  << controlled_repair.diagnostic_total_to_global_max_ratio
                  << " relative_deficit=" << repair.relative_deficit
                  << " floor_extent=" << repair.floor_extent
                  << " injected_extent=" << repair.injected_extent
                  << " aggregate_sync_correction="
                  << controlled_repair.aggregate_sync_correction
                  << " occupation="
                  << comptonOccupationModeLabel(occupation_mode)
                  << " action=globally_negligible_negative_nonconservative_floor"
                  << std::endl;

    double const internal_delta = new_internal_ext - old_internal_ext;
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        extensive.Eg[g] = new_group_ext[g];
    extensive.Erad = new_radiation_ext;
    extensive.internal_energy = new_internal_ext;
    extensive.energy += internal_delta;
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        cell.Eg[g] = extensive.Eg[g] / extensive.mass;
    cell.Erad = extensive.Erad / extensive.mass;
    cell.internal_energy = extensive.internal_energy / extensive.mass;
    cell.temperature = eos_.de2T(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
    cell.pressure = eos_.de2p(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
    old_Tm[cell_index] = saved_old_temperature;
    return true;
    }
    catch (UniversalError const&) {
        record_basic_failure(
            "compton_substep_universal_error",
            no_failure_group, no_failure_extent);
        rollback_substep_accounting();
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }
    catch (std::exception const&) {
        record_basic_failure(
            "compton_substep_standard_exception",
            no_failure_group, no_failure_extent);
        rollback_substep_accounting();
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }
    catch (...) {
        record_basic_failure(
            "compton_substep_unknown_exception",
            no_failure_group, no_failure_extent);
        rollback_substep_accounting();
        cell = saved_cell;
        extensive = saved_extensive;
        old_Tm[cell_index] = saved_old_temperature;
        return false;
    }
}

bool MultigroupDiffusion::apply_operator_split_compton(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    double const dt,
    double const global_maximum_cell_radiation_extent) const
{
    bool valid = true;
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    std::size_t diagnostic_cell = std::numeric_limits<std::size_t>::max();
    for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
        if(split_compton_cells_[i] && compton_on_) {
            diagnostic_cell = i;
            break;
        }
    int diagnostic_rank =
        diagnostic_cell != std::numeric_limits<std::size_t>::max()
        ? rank : std::numeric_limits<int>::max();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &diagnostic_rank, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    SplitComptonDiagnosticSummary split_diagnostics;
    std::size_t const saved_split_subcycle_count = split_subcycle_count_;
    double const saved_split_suppressed_energy = split_suppressed_energy_;
    double const saved_split_injected_energy = split_injected_energy_;
    SpectralRepairEvent const saved_pending_repair_event =
        pending_split_spectral_repair_event_;
    auto rollback_split_accounting = [&]() {
        split_subcycle_count_ = saved_split_subcycle_count;
        split_suppressed_energy_ = saved_split_suppressed_energy;
        split_injected_energy_ = saved_split_injected_energy;
        pending_split_spectral_repair_event_ = saved_pending_repair_event;
    };
    unsigned long long local_split_cells = 0;
    std::size_t const substeps_before = split_subcycle_count_;
    long double local_radiation_before = 0;
    long double local_radiation_after = 0;
    long double local_material_before = 0;
    long double local_material_after = 0;
    auto repair_split_input = [&](std::size_t const cell_index) {
        double const cell_radiation_extent = extensives[cell_index].Erad;
        auto const controlled =
            RadiationPositivity::RepairControlledNegativeGroupExtents(
                extensives[cell_index].Eg,
                extensives[cell_index].Erad,
                RadiationPositivity::spectral_repair_relative_limit,
                global_maximum_cell_radiation_extent);
        RadiationPositivity::SpectralRepairResult const& repair =
            controlled.repair;
        if(!repair.valid) {
            std::ostringstream reason;
            reason << std::setprecision(17)
                   << "operator-split Compton input spectral validation failed"
                   << " stage=pre_compton"
                   << " failure="
                   << RadiationPositivity::SpectralRepairFailureLabel(
                          repair.failure)
                   << " group=" << repair.most_negative_group
                   << " signed_group_extent="
                   << repair.most_negative_extent
                   << " negative_extent=" << repair.negative_extent
                   << " positive_extent=" << repair.positive_extent
                   << " relative_deficit=" << repair.relative_deficit
                   << " local_tolerance="
                   << RadiationPositivity::spectral_repair_relative_limit
                   << " negative_extent_over_global_E_max="
                   << controlled.global_negative.
                          negative_extent_to_global_max_ratio
                   << " global_negative_tolerance="
                   << RadiationPositivity::
                          spectral_globally_negligible_negative_fraction
                   << " diagnostic_E_cell_over_E_max="
                   << controlled.diagnostic_total_to_global_max_ratio
                   << " global_E_max="
                   << global_maximum_cell_radiation_extent;
            setCellLocalStepFailure(reason.str(), cells[cell_index].ID);
            return false;
        }
        if(!repair.repaired &&
           controlled.aggregate_sync_correction == 0)
            return true;

        if(!std::isfinite(extensives[cell_index].mass) ||
           extensives[cell_index].mass <= 0) {
            setCellLocalStepFailure(
                "operator-split Compton input has invalid cell mass",
                cells[cell_index].ID);
            return false;
        }
        for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            cells[cell_index].Eg[group] =
                extensives[cell_index].Eg[group] /
                extensives[cell_index].mass;
        cells[cell_index].Erad =
            extensives[cell_index].Erad / extensives[cell_index].mass;
        if(repair.repaired) {
            split_injected_energy_ += repair.injected_extent;
            ++pending_split_spectral_repair_event_.repaired_cells;
            pending_split_spectral_repair_event_.repaired_groups +=
                repair.repaired_groups;
            pending_split_spectral_repair_event_.injected_energy +=
                repair.injected_extent;
            if(repair.relative_deficit >
               pending_split_spectral_repair_event_.maximum_relative_deficit) {
                pending_split_spectral_repair_event_.maximum_relative_deficit =
                    repair.relative_deficit;
                pending_split_spectral_repair_event_.representative_cell_id =
                    cells[cell_index].ID;
                pending_split_spectral_repair_event_.representative_group =
                    repair.most_negative_group;
                pending_split_spectral_repair_event_.representative_original_extent =
                    repair.most_negative_extent;
                pending_split_spectral_repair_event_.representative_floor_extent =
                    repair.floor_extent;
                pending_split_spectral_repair_event_.representative_injected_extent =
                    repair.injected_extent;
            }
        }
        if(controlled.used_global_negative_exception)
            std::clog << std::setprecision(17)
                      << "MG_SPECTRAL_POSITIVITY_REPAIR"
                      << " scope=pre_compton"
                      << " rank=" << rank
                      << " cell_id=" << cells[cell_index].ID
                      << " E_cell=" << cell_radiation_extent
                      << " global_E_max="
                      << global_maximum_cell_radiation_extent
                      << " diagnostic_E_cell_over_E_max="
                      << controlled.diagnostic_total_to_global_max_ratio
                      << " group=" << repair.most_negative_group
                      << " signed_group_extent="
                      << repair.most_negative_extent
                      << " negative_extent=" << repair.negative_extent
                      << " positive_extent=" << repair.positive_extent
                      << " negative_extent_over_global_E_max="
                      << controlled.global_negative.
                             negative_extent_to_global_max_ratio
                      << " global_negative_tolerance="
                      << RadiationPositivity::
                             spectral_globally_negligible_negative_fraction
                      << " relative_deficit=" << repair.relative_deficit
                      << " affected_groups=" << repair.repaired_groups
                      << " floor_extent=" << repair.floor_extent
                      << " injected_extent=" << repair.injected_extent
                      << " aggregate_sync_correction="
                      << controlled.aggregate_sync_correction
                      << " action=globally_negligible_negative_nonconservative_floor"
                      << std::endl;
        return true;
    };
    // Validate every participating spectrum collectively before doing any
    // cell-local Compton work.  A single unrecoverable input rejects the
    // transaction, so solving up to millions of other cells first cannot
    // affect the result and makes fractional retries unnecessarily costly.
    for (std::size_t i = 0; i < tess.GetPointNo(); ++i) {
        if (!split_compton_cells_[i] || !compton_on_)
            continue;
        ++local_split_cells;
        local_radiation_before += extensives[i].Erad;
        local_material_before += extensives[i].internal_energy;
        if(!repair_split_input(i))
            valid = false;
    }
#ifdef RICH_MPI
    bool const local_input_valid = valid;
    int collective_input_valid = valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &collective_input_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    valid = collective_input_valid != 0;
    if(!valid && local_input_valid && getLastStepFailureReason().empty()) {
        setStepFailure(
            "operator-split Compton input validation failed on another rank");
        markStepFailureRemote();
    }
#endif
    if(!valid) {
        unsigned long long global_split_cells = local_split_cells;
        double totals[4] = {
            static_cast<double>(local_radiation_before),
            static_cast<double>(local_radiation_before),
            static_cast<double>(local_material_before),
            static_cast<double>(local_material_before)};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &global_split_cells, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, totals, 4, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
#endif
        if(rank == 0)
            std::clog << std::setprecision(17)
                      << "MG_COMPTON_OPERATOR_SPLIT"
                      << " scope=" << (individual_context_ == nullptr
                          ? "global" : "individual")
                      << " cells=" << global_split_cells
                      << " substeps=0"
                      << " radiation_before=" << totals[0]
                      << " radiation_after=" << totals[1]
                      << " material_before=" << totals[2]
                      << " material_after=" << totals[3]
                      << " combined_before=" << totals[0] + totals[2]
                      << " combined_after=" << totals[1] + totals[3]
                      << " outcome=rejected"
                      << " phase=input_precheck"
                      << std::endl;
        rollback_split_accounting();
        return false;
    }

    for (std::size_t i = 0; i < tess.GetPointNo(); ++i) {
        if (!split_compton_cells_[i] || !compton_on_)
            continue;
        double const cell_dt = individual_context_ == nullptr
            ? dt : individualCellTimeStep(i, dt);
        if(!std::isfinite(cell_dt) || cell_dt <= 0)
            continue;
        double const min_dt = std::max(
            64.0 * std::numeric_limits<double>::epsilon() * cell_dt,
            std::ldexp(cell_dt, -30));
        double remaining = cell_dt;
        double local_dt = cell_dt;
        while (remaining > min_dt) {
            local_dt = std::min(local_dt, remaining);
            ++split_subcycle_count_;
            bool const emit_diagnostics = rank == diagnostic_rank &&
                i == diagnostic_cell;
            if (solve_local_compton_substep(
                    tess, i, cells[i], extensives[i], local_dt,
                    global_maximum_cell_radiation_extent,
                    emit_diagnostics, split_diagnostics)) {
                remaining -= local_dt;
                local_dt = std::min(2.0 * local_dt, remaining);
            } else {
                local_dt *= 0.5;
                if (local_dt < min_dt) {
                    if(getLastStepFailureReason().empty()) {
                        std::ostringstream reason;
                        reason << std::setprecision(17)
                               << "local operator-split Compton solve could not cover the candidate interval"
                               << " stage=post_compton_substep"
                               << " failure="
                               << (split_diagnostics.latest_failure_reason.empty()
                                   ? "unclassified_local_substep_failure"
                                   : split_diagnostics.latest_failure_reason)
                               << " group=";
                        if(split_diagnostics.latest_failure_group ==
                           std::numeric_limits<unsigned long long>::max())
                            reason << "none";
                        else
                            reason << split_diagnostics.latest_failure_group;
                        reason << " signed_group_extent="
                               << split_diagnostics.latest_failure_extent
                               << " negative_extent="
                               << split_diagnostics.latest_negative_extent
                               << " positive_extent="
                               << split_diagnostics.latest_positive_extent
                               << " negative_extent_over_global_E_max="
                               << split_diagnostics.
                                      latest_negative_to_global_max_ratio
                               << " global_negative_tolerance="
                               << RadiationPositivity::
                                      spectral_globally_negligible_negative_fraction
                               << " diagnostic_E_cell_over_E_max="
                               << split_diagnostics.
                                      latest_total_to_global_max_ratio
                               << " global_E_max="
                               << global_maximum_cell_radiation_extent
                               << " attempted_dt="
                               << split_diagnostics.latest_failure_dt
                               << " minimum_dt=" << min_dt
                               << " beta=" << split_diagnostics.latest_beta
                               << " gamma=" << split_diagnostics.latest_gamma
                               << " kappa_planck="
                               << split_diagnostics.latest_kappa_planck
                               << " fleck=" << split_diagnostics.latest_fleck
                               << " compton_in_fleck=1"
                               << " absorption_in_fleck=0"
                               << " occupation="
                               << comptonOccupationModeLabel(
                                      split_diagnostics.latest_occupation)
                               << " solver=local_dense"
                               << " solver_iterations=0"
                               << " repair_attempted="
                               << (split_diagnostics.latest_repair_attempted
                                   ? 1 : 0)
                               << " repair_injected="
                               << (split_diagnostics.latest_repair_injected
                                   ? 1 : 0)
                               << " upsilon_fallback="
                               << (split_diagnostics.latest_upsilon_fallback
                                   ? 1 : 0);
                        setCellLocalStepFailure(reason.str(), cells[i].ID);
                    }
                    valid = false;
                    break;
                }
            }
        }
        local_radiation_after += extensives[i].Erad;
        local_material_after += extensives[i].internal_energy;
    }
    unsigned long long global_split_cells = local_split_cells;
    unsigned long long global_substeps = static_cast<unsigned long long>(
        split_subcycle_count_ - substeps_before);
    unsigned long long global_upsilon_fallback_events =
        split_diagnostics.upsilon_fallback_events;
    int upsilon_representative_rank =
        split_diagnostics.upsilon_fallback_events > 0
        ? rank : std::numeric_limits<int>::max();
    unsigned long long upsilon_representative_ids[2] = {
        split_diagnostics.representative_cell_id,
        split_diagnostics.representative_failed_group};
    double upsilon_representative_values[4] = {
        split_diagnostics.representative_dt,
        split_diagnostics.representative_original_upsilon,
        split_diagnostics.representative_original_fleck,
        split_diagnostics.representative_failed_value};
    double totals[4] = {
        static_cast<double>(local_radiation_before),
        static_cast<double>(local_radiation_after),
        static_cast<double>(local_material_before),
        static_cast<double>(local_material_after)};
#ifdef RICH_MPI
    bool const local_valid = valid;
    int collective_valid = valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    valid = collective_valid != 0;
    if(!valid && local_valid && getLastStepFailureReason().empty()) {
        setStepFailure(
            "local operator-split Compton solve failed on another rank");
        markStepFailureRemote();
    }
    MPI_Allreduce(MPI_IN_PLACE, &global_split_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_substeps, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_upsilon_fallback_events, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &upsilon_representative_rank, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(global_upsilon_fallback_events > 0) {
        MPI_Bcast(upsilon_representative_ids, 2, MPI_UNSIGNED_LONG_LONG,
                  upsilon_representative_rank, MPI_COMM_WORLD);
        MPI_Bcast(upsilon_representative_values, 4, MPI_DOUBLE,
                  upsilon_representative_rank, MPI_COMM_WORLD);
    }
    MPI_Allreduce(MPI_IN_PLACE, totals, 4, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if(global_upsilon_fallback_events > 0 && rank == 0) {
        std::clog << std::setprecision(17)
                  << "MG_COMPTON_UPSILON_FALLBACK"
                  << " scope=aggregate"
                  << " events=" << global_upsilon_fallback_events
                  << " rank=" << upsilon_representative_rank
                  << " cell_id=" << upsilon_representative_ids[0]
                  << " dt=" << upsilon_representative_values[0]
                  << " original_upsilon="
                  << upsilon_representative_values[1]
                  << " original_fleck=" << upsilon_representative_values[2]
                  << " failed_group=";
        if(upsilon_representative_ids[1] ==
           std::numeric_limits<unsigned long long>::max())
            std::clog << "none";
        else
            std::clog << upsilon_representative_ids[1];
        std::clog << " failed_value=" << upsilon_representative_values[3]
                  << " action=freeze_material_jacobian"
                  << std::endl;
    }
    if(global_split_cells > 0 && rank == 0)
        std::clog << std::setprecision(17)
                  << "MG_COMPTON_OPERATOR_SPLIT"
                  << " scope=" << (individual_context_ == nullptr
                      ? "global" : "individual")
                  << " cells=" << global_split_cells
                  << " substeps=" << global_substeps
                  << " radiation_before=" << totals[0]
                  << " radiation_after=" << totals[1]
                  << " material_before=" << totals[2]
                  << " material_after=" << totals[3]
                  << " combined_before=" << totals[0] + totals[2]
                  << " combined_after=" << totals[1] + totals[3]
                  << " outcome=" << (valid ? "accepted" : "rejected")
                  << std::endl;
    if(!valid)
        rollback_split_accounting();
    return valid;
}

double  MultigroupDiffusion::get_doppler_slope(ComputationalCell3D const& cell, size_t const g, bool const expansion) const
{
    MEMORY_PROFILE_SCOPE("multigroup diffusion step");
    if (g == 0 or (g + 1) == ENERGY_GROUPS_NUM) {
        return 0.0;
    }

    double const dw_left = expansion ? energy_groups_width[g] : energy_groups_width[g - 1];
    double const dw_right = expansion ? energy_groups_width[g + 1] : energy_groups_width[g];

    double const slope_left = (cell.Eg[g] * cell.density / energy_groups_width[g] - cell.Eg[g - 1] * cell.density / energy_groups_width[g - 1]) / dw_left;
    double const slope_right = (cell.Eg[g + 1] * cell.density / energy_groups_width[g + 1] - cell.Eg[g] * cell.density / energy_groups_width[g]) / dw_right;

    double const r = slope_left / (slope_right + std::max({ slope_right, slope_left, std::numeric_limits<double>::min() * 1e50 })*1e-16);

    double const slope = std::max(std::max(0.0, std::min(2 * r, 1.0)), std::min(r, 2.0));

    return slope;
}

struct MultigroupDiffusion::ImplicitComptonCellCoefficients
{
    std::array<double, ENERGY_GROUPS_NUM * ENERGY_GROUPS_NUM> delta_A;
    std::array<double, ENERGY_GROUPS_NUM> delta_b;
};

void MultigroupDiffusion::ensureComptonBulkRuntimeOptions() const
{
    if(compton_bulk_runtime_options_initialized_)
        return;

    bool const local_bulk =
        mgRuntimeFlagEnabled("RICH_MG_COMPTON_BULK_COEFFICIENTS");
    bool const local_shadow =
        mgRuntimeFlagEnabled("RICH_MG_COMPTON_BULK_SHADOW");
    MgStrictRuntimeFlag const local_skip_reverse_faces =
        mgStrictRuntimeFlag("RICH_MG_SKIP_REVERSE_FACES");
    MgStrictRuntimeFlag const local_hoist_face_mean_temperature =
        mgStrictRuntimeFlag("RICH_MG_HOIST_FACE_MEAN_TEMPERATURE");
    MgStrictRuntimeFlag const local_freefree_pair16 =
        mgStrictRuntimeFlag("RICH_MG_FREEFREE_PAIR16");
    MgStrictRuntimeFlag const local_freefree_pair16_shadow =
        mgStrictRuntimeFlag("RICH_MG_FREEFREE_PAIR16_SHADOW");
    MgStrictRuntimeFlag const local_postcg_face_geometry =
        mgStrictRuntimeFlag("RICH_MG_POSTCG_FACE_GEOMETRY");
    MgStrictRuntimeFlag const local_postcg_face_geometry_shadow =
        mgStrictRuntimeFlag("RICH_MG_POSTCG_FACE_GEOMETRY_SHADOW");
    unsigned int option_state =
        (local_bulk ? 2u : 1u) | (local_shadow ? 8u : 4u) |
        (!local_skip_reverse_faces.valid ? 64u :
         (local_skip_reverse_faces.enabled ? 32u : 16u)) |
        (!local_hoist_face_mean_temperature.valid ? 512u :
         (local_hoist_face_mean_temperature.enabled ? 256u : 128u)) |
        (!local_freefree_pair16.valid ? 4096u :
         (local_freefree_pair16.enabled ? 2048u : 1024u)) |
        (!local_freefree_pair16_shadow.valid ? 32768u :
         (local_freefree_pair16_shadow.enabled ? 16384u : 8192u)) |
        (!local_postcg_face_geometry.valid ? 262144u :
         (local_postcg_face_geometry.enabled ? 131072u : 65536u)) |
        (!local_postcg_face_geometry_shadow.valid ? 2097152u :
         (local_postcg_face_geometry_shadow.enabled ? 1048576u : 524288u));
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &option_state, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if((option_state & 3u) == 3u)
        throw UniversalError(
            "RICH_MG_COMPTON_BULK_COEFFICIENTS differs between MPI ranks");
    if((option_state & 12u) == 12u)
        throw UniversalError(
            "RICH_MG_COMPTON_BULK_SHADOW differs between MPI ranks");
#endif
    if((option_state & 64u) != 0u)
        throw UniversalError(
            "RICH_MG_SKIP_REVERSE_FACES must be unset or one of "
            "0, 1, false, true, off, on, no, yes");
    if((option_state & 512u) != 0u)
        throw UniversalError(
            "RICH_MG_HOIST_FACE_MEAN_TEMPERATURE must be unset or one of "
            "0, 1, false, true, off, on, no, yes");
    if((option_state & 4096u) != 0u)
        throw UniversalError(
            "RICH_MG_FREEFREE_PAIR16 must be unset or one of "
            "0, 1, false, true, off, on, no, yes");
    if((option_state & 32768u) != 0u)
        throw UniversalError(
            "RICH_MG_FREEFREE_PAIR16_SHADOW must be unset or one of "
            "0, 1, false, true, off, on, no, yes");
    if((option_state & 262144u) != 0u)
        throw UniversalError(
            "RICH_MG_POSTCG_FACE_GEOMETRY must be unset or one of "
            "0, 1, false, true, off, on, no, yes");
    if((option_state & 2097152u) != 0u)
        throw UniversalError(
            "RICH_MG_POSTCG_FACE_GEOMETRY_SHADOW must be unset or one of "
            "0, 1, false, true, off, on, no, yes");
    if((option_state & 48u) == 48u)
        throw UniversalError(
            "RICH_MG_SKIP_REVERSE_FACES differs between MPI ranks");
    if((option_state & 384u) == 384u)
        throw UniversalError(
            "RICH_MG_HOIST_FACE_MEAN_TEMPERATURE differs between MPI ranks");
    if((option_state & 3072u) == 3072u)
        throw UniversalError(
            "RICH_MG_FREEFREE_PAIR16 differs between MPI ranks");
    if((option_state & 24576u) == 24576u)
        throw UniversalError(
            "RICH_MG_FREEFREE_PAIR16_SHADOW differs between MPI ranks");
    if((option_state & 196608u) == 196608u)
        throw UniversalError(
            "RICH_MG_POSTCG_FACE_GEOMETRY differs between MPI ranks");
    if((option_state & 1572864u) == 1572864u)
        throw UniversalError(
            "RICH_MG_POSTCG_FACE_GEOMETRY_SHADOW differs between MPI ranks");
    compton_bulk_coefficients_enabled_ = (option_state & 2u) != 0;
    compton_bulk_shadow_enabled_ = (option_state & 8u) != 0;
    skip_reverse_faces_requested_ = (option_state & 32u) != 0;
    hoist_face_mean_temperature_requested_ = (option_state & 256u) != 0;
    freefree_pair16_requested_ = (option_state & 2048u) != 0;
    freefree_pair16_shadow_requested_ = (option_state & 16384u) != 0;
    postcg_face_geometry_requested_ = (option_state & 131072u) != 0;
    postcg_face_geometry_shadow_requested_ =
        (option_state & 1048576u) != 0;
    compton_bulk_runtime_options_initialized_ = true;

    if(compton_bulk_coefficients_enabled_ || compton_bulk_shadow_enabled_ ||
       mgRuntimeFlagEnabled("RICH_INDIVIDUAL_PERF_TRACE")) {
        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        if(rank == 0)
            RuntimeTraceStream()
                      << "MG_COMPTON_BULK_COEFFICIENTS enabled="
                      << (compton_bulk_coefficients_enabled_ ? 1 : 0)
                      << " shadow="
                      << (compton_bulk_shadow_enabled_ ? 1 : 0)
                      << std::endl;
    }
}

void MultigroupDiffusion::invalidateDirectStructureCache(
    bool const release_storage) const
{
    direct_structure_cache_valid_ = false;
    direct_structure_cache_rows_ = 0;
    direct_structure_cache_nnz_ = 0;
    direct_structure_cache_bound_epoch_ = 0;
    ++direct_structure_cache_solver_epoch_;
    if(release_storage) {
        release_container_memory(direct_structure_cache_key_);
        release_container_memory(direct_structure_cache_probe_);
    }
}

void MultigroupDiffusion::buildDirectStructureCacheKey(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells) const
{
    std::vector<std::size_t>& key = direct_structure_cache_probe_;
    key.clear();

    // This is an exact, collision-free description of every interface that
    // can change direct-CSR column placement. Tessellation3D does not expose a
    // generic topology generation, so counts alone are not a safe cache key.
    key.push_back(1); // key format version
    key.push_back(ENERGY_GROUPS_NUM);
    key.push_back(tess.GetPointNo());
    key.push_back(tess.GetAllPointsNo());
    key.push_back(tess.GetTotalFacesNumber());
    key.push_back(cells.size());
    key.push_back(individual_context_ == nullptr ? 1 : 0);
    for(auto const& cell : cells)
        key.push_back(static_cast<std::size_t>(cell.ID));

    std::vector<std::size_t> neighbors;
    for(std::size_t cell = 0; cell < tess.GetPointNo(); ++cell) {
        key.push_back(individualCellActive(cell) ? 1 : 0);
        appendDirectStructureSequence(key, tess.GetCellFaces(cell));
        tess.GetNeighbors(cell, neighbors);
        key.push_back(neighbors.size());
        for(std::size_t const neighbor : neighbors) {
            key.push_back(neighbor);
            key.push_back(tess.IsPointOutsideBox(neighbor) ? 1 : 0);
        }
    }

    Tessellation3D::AllPointsMap const& local_to_all =
        tess.GetIndicesInAllPoints();
    key.push_back(local_to_all.size());
    for(auto const& mapping : local_to_all) {
        key.push_back(mapping.first);
        key.push_back(mapping.second);
    }

#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    key.push_back(static_cast<std::size_t>(rank));
    key.push_back(static_cast<std::size_t>(rank_count));
    appendDirectStructureSequence(key, tess.GetDuplicatedProcs());
    appendDirectStructureNestedSequence(key, tess.GetDuplicatedPoints());
    appendDirectStructureSequence(key, tess.GetSentProcs());
    appendDirectStructureNestedSequence(key, tess.GetSentPoints());
    appendDirectStructureSequence(key, tess.GetSelfIndex());
    appendDirectStructureNestedSequence(key, tess.GetGhostIndeces());
#else
    key.push_back(0);
    key.push_back(1);
#endif
}

class MultigroupDiffusion::MatrixRows
{
public:
    MatrixRows(mat& values, size_t_mat& columns)
        : legacy_values_(&values), legacy_columns_(&columns)
    {}

    MatrixRows(std::vector<size_t>& row_offsets,
               std::vector<size_t>& columns,
               std::vector<double>& values,
               bool const reuse_direct_structure = false)
        : row_offsets_(&row_offsets), columns_(&columns), values_(&values),
          reuse_direct_structure_(reuse_direct_structure)
    {}

    explicit MatrixRows(Fixed16BlockStencilMatrix& fixed16_block_stencil)
        : fixed16_block_stencil_(&fixed16_block_stencil)
    {}

    bool Direct() const noexcept
    {
        return row_offsets_ != nullptr;
    }

    bool ReusingDirectStructure() const noexcept
    {
        return reuse_direct_structure_;
    }

    bool Fixed16BlockStencil() const noexcept
    {
        return fixed16_block_stencil_ != nullptr;
    }

    bool Fixed16BlockStencilValid() const noexcept
    {
        return !Fixed16BlockStencil() || fixed16_block_stencil_valid_;
    }

    void InvalidateFixed16BlockStencil() noexcept
    {
        fixed16_block_stencil_valid_ = false;
    }

    void InitializeFixed16BlockStencil(Tessellation3D const& tess)
    {
        if(!Fixed16BlockStencil())
            throw UniversalError(
                "Cannot initialize fixed-16 storage on another matrix sink");
        Fixed16BlockStencilMatrix& matrix = *fixed16_block_stencil_;
        matrix.LocalCellCount = tess.GetPointNo();
        matrix.VectorCellCount = matrix.LocalCellCount;

        std::size_t local_block_count = 0;
        if(matrix.LocalCellCount >
           std::numeric_limits<std::size_t>::max() /
               (Fixed16BlockStencilMatrix::BlockSize *
                Fixed16BlockStencilMatrix::BlockSize))
            throw UniversalError("Fixed-16 local block size overflow");
        local_block_count = matrix.LocalCellCount *
            Fixed16BlockStencilMatrix::BlockSize *
            Fixed16BlockStencilMatrix::BlockSize;
        matrix.LocalBlockValues.resize(local_block_count);

        matrix.NeighborOffsets.resize(matrix.LocalCellCount + 1);
        matrix.NeighborOffsets[0] = 0;
        std::vector<std::size_t> neighbors;
        for(std::size_t cell = 0; cell < matrix.LocalCellCount; ++cell) {
            tess.GetNeighbors(cell, neighbors);
            std::size_t neighbor_count = 0;
            for(std::size_t const neighbor : neighbors)
                if(!tess.IsPointOutsideBox(neighbor)) {
                    ++neighbor_count;
                    if(neighbor == std::numeric_limits<std::size_t>::max())
                        throw UniversalError(
                            "Fixed-16 neighbor index overflow");
                    matrix.VectorCellCount = std::max(
                        matrix.VectorCellCount, neighbor + 1);
                }
            if(neighbor_count > std::numeric_limits<std::size_t>::max() -
                                    matrix.NeighborOffsets[cell])
                throw UniversalError("Fixed-16 neighbor count overflow");
            matrix.NeighborOffsets[cell + 1] =
                matrix.NeighborOffsets[cell] + neighbor_count;
        }
        if(matrix.VectorCellCount >
           std::numeric_limits<std::size_t>::max() /
               Fixed16BlockStencilMatrix::BlockSize)
            throw UniversalError("Fixed-16 vector row count overflow");
        matrix.NeighborCells.resize(matrix.NeighborOffsets.back());
        if(matrix.NeighborOffsets.back() >
           std::numeric_limits<std::size_t>::max() /
               Fixed16BlockStencilMatrix::BlockSize)
            throw UniversalError("Fixed-16 neighbor value size overflow");
        matrix.NeighborValues.resize(
            matrix.NeighborOffsets.back() *
                Fixed16BlockStencilMatrix::BlockSize);
        cursors_.assign(
            matrix.LocalCellCount * Fixed16BlockStencilMatrix::BlockSize,
            0);
    }

    void InitializeLegacy(std::size_t const row_count)
    {
        if(Direct() || Fixed16BlockStencil())
            throw UniversalError("Cannot initialize direct CSR as legacy rows");
        legacy_values_->resize(row_count);
        legacy_columns_->resize(row_count);
        for(std::size_t row = 0; row < row_count; ++row) {
            (*legacy_values_)[row].clear();
            (*legacy_columns_)[row].clear();
        }
    }

    void ReserveLegacy(std::size_t const row, std::size_t const capacity)
    {
        if(Direct() || Fixed16BlockStencil())
            throw UniversalError("Cannot reserve a direct CSR matrix row");
        (*legacy_values_)[row].reserve(capacity);
        (*legacy_columns_)[row].reserve(capacity);
    }

    void InitializeDirect(std::vector<size_t> const& row_sizes)
    {
        if(!Direct())
            throw UniversalError("Cannot initialize legacy rows as direct CSR");
        row_offsets_->assign(row_sizes.size() + 1, 0);
        for(std::size_t row = 0; row < row_sizes.size(); ++row) {
            if(row_sizes[row] >
               std::numeric_limits<std::size_t>::max() - (*row_offsets_)[row])
                throw UniversalError("Direct CSR matrix size overflow");
            (*row_offsets_)[row + 1] = (*row_offsets_)[row] + row_sizes[row];
        }
        columns_->assign(row_offsets_->back(), max_size_t);
        values_->assign(row_offsets_->back(), 0.0);
        cursors_.assign(row_sizes.size(), 0);
        for(std::size_t row = 0; row < row_sizes.size(); ++row)
            cursors_[row] = (*row_offsets_)[row];
    }

    void InitializeDirectFromCache(std::size_t const row_count)
    {
        if(!Direct() || !ReusingDirectStructure())
            throw UniversalError("Cannot initialize an uncached CSR structure");
        if(row_offsets_->size() != row_count + 1 || row_offsets_->front() != 0 ||
           row_offsets_->back() != columns_->size())
            throw UniversalError("Cached direct CSR structure has invalid dimensions");
        values_->assign(columns_->size(), 0.0);
        cursors_.resize(row_count);
        for(std::size_t row = 0; row < row_count; ++row)
            cursors_[row] = (*row_offsets_)[row];
    }

    void Append(std::size_t const row,
                std::size_t const column,
                double const value)
    {
        if(Fixed16BlockStencil()) {
            Fixed16BlockStencilMatrix& matrix = *fixed16_block_stencil_;
            std::size_t const block_size =
                Fixed16BlockStencilMatrix::BlockSize;
            if(row >= cursors_.size()) {
                fixed16_block_stencil_valid_ = false;
                return;
            }
            std::size_t const cell = row / block_size;
            std::size_t const row_group = row % block_size;
            std::size_t const column_cell = column / block_size;
            std::size_t const column_group = column % block_size;
            if(column_cell >= matrix.VectorCellCount)
                fixed16_block_stencil_valid_ = false;

            std::size_t& cursor = cursors_[row];
            if(cursor < block_size) {
                std::size_t const expected_column_group = cursor == 0 ?
                    row_group : (cursor <= row_group ? cursor - 1 : cursor);
                if(column_cell != cell ||
                   column_group != expected_column_group)
                    fixed16_block_stencil_valid_ = false;
                matrix.LocalBlockValues[row * block_size + cursor] = value;
            }
            else {
                if(column_cell == cell || column_group != row_group)
                    fixed16_block_stencil_valid_ = false;
                std::size_t const neighbor_ordinal = cursor - block_size;
                std::size_t const neighbor_slot =
                    matrix.NeighborOffsets[cell] + neighbor_ordinal;
                if(neighbor_slot >= matrix.NeighborOffsets[cell + 1]) {
                    fixed16_block_stencil_valid_ = false;
                    return;
                }
                matrix_index_t const neighbor_index =
                    static_cast<matrix_index_t>(column_cell);
                if(row_group == 0)
                    matrix.NeighborCells[neighbor_slot] = neighbor_index;
                else if(matrix.NeighborCells[neighbor_slot] != neighbor_index)
                    fixed16_block_stencil_valid_ = false;
                matrix.NeighborValues[
                    neighbor_slot * block_size + row_group] = value;
            }
            ++cursor;
            return;
        }
        if(!Direct()) {
            (*legacy_values_)[row].push_back(value);
            (*legacy_columns_)[row].push_back(column);
            return;
        }
        std::size_t& cursor = cursors_[row];
        if(cursor >= (*row_offsets_)[row + 1])
            throw UniversalError("Direct CSR matrix row exceeded its exact size");
        if(ReusingDirectStructure()) {
            if((*columns_)[cursor] != column)
                throw UniversalError(
                    "Cached direct CSR column order changed during replay");
        }
        else
            (*columns_)[cursor] = column;
        (*values_)[cursor] = value;
        ++cursor;
    }

    std::size_t Size(std::size_t const row) const
    {
        if(Fixed16BlockStencil()) {
            if(row >= cursors_.size())
                return 0;
            return cursors_[row];
        }
        if(!Direct())
            return (*legacy_values_)[row].size();
        return cursors_[row] - (*row_offsets_)[row];
    }

    double& Value(std::size_t const row, std::size_t const slot)
    {
        if(Fixed16BlockStencil()) {
            if(row >= cursors_.size() || slot >= Size(row)) {
                fixed16_block_stencil_valid_ = false;
                fixed16_discard_value_ = 0.0;
                return fixed16_discard_value_;
            }
            std::size_t const block_size =
                Fixed16BlockStencilMatrix::BlockSize;
            if(slot < block_size)
                return fixed16_block_stencil_->LocalBlockValues[
                    row * block_size + slot];
            std::size_t const cell = row / block_size;
            std::size_t const group = row % block_size;
            std::size_t const neighbor_slot =
                fixed16_block_stencil_->NeighborOffsets[cell] +
                slot - block_size;
            return fixed16_block_stencil_->NeighborValues[
                neighbor_slot * block_size + group];
        }
        if(!Direct())
            return (*legacy_values_)[row][slot];
        if(slot >= Size(row))
            throw UniversalError("Direct CSR matrix slot is not initialized");
        return (*values_)[(*row_offsets_)[row] + slot];
    }

    std::size_t Column(std::size_t const row, std::size_t const slot) const
    {
        if(Fixed16BlockStencil()) {
            if(row >= cursors_.size() || slot >= Size(row))
                return max_size_t;
            std::size_t const block_size =
                Fixed16BlockStencilMatrix::BlockSize;
            std::size_t const cell = row / block_size;
            std::size_t const group = row % block_size;
            if(slot < block_size) {
                std::size_t const column_group = slot == 0 ?
                    group : (slot <= group ? slot - 1 : slot);
                return cell * block_size + column_group;
            }
            std::size_t const neighbor_slot =
                fixed16_block_stencil_->NeighborOffsets[cell] +
                slot - block_size;
            return static_cast<std::size_t>(
                       fixed16_block_stencil_->NeighborCells[neighbor_slot]) *
                       block_size +
                group;
        }
        if(!Direct())
            return (*legacy_columns_)[row][slot];
        if(slot >= Size(row))
            throw UniversalError("Direct CSR matrix column is not initialized");
        return (*columns_)[(*row_offsets_)[row] + slot];
    }

    std::size_t FindColumn(std::size_t const row,
                           std::size_t const column) const
    {
        for(std::size_t slot = 0; slot < Size(row); ++slot)
            if(Column(row, slot) == column)
                return slot;
        return max_size_t;
    }

    void ResizeLegacy(std::size_t const row, std::size_t const size)
    {
        if(Direct() || Fixed16BlockStencil())
            throw UniversalError("Cannot pad a direct CSR matrix row");
        (*legacy_values_)[row].resize(size, 0.0);
        (*legacy_columns_)[row].resize(size, max_size_t);
    }

    void FinishDirect()
    {
        if(Fixed16BlockStencil()) {
            Fixed16BlockStencilMatrix const& matrix =
                *fixed16_block_stencil_;
            std::size_t const block_size =
                Fixed16BlockStencilMatrix::BlockSize;
            for(std::size_t row = 0; row < cursors_.size(); ++row) {
                std::size_t const cell = row / block_size;
                std::size_t const expected = block_size +
                    matrix.NeighborOffsets[cell + 1] -
                    matrix.NeighborOffsets[cell];
                if(cursors_[row] != expected)
                    fixed16_block_stencil_valid_ = false;
            }
            return;
        }
        if(!Direct())
            return;

        if(ReusingDirectStructure()) {
            for(std::size_t row = 0; row < cursors_.size(); ++row)
                if(cursors_[row] != (*row_offsets_)[row + 1])
                    throw UniversalError(
                        "Cached direct CSR row length changed during replay");
            return;
        }

        // Row capacities are topology-derived upper bounds. Compact the
        // unused boundary/passive-face slots in place after assembly. Since
        // every destination starts at or before its source, forward copying
        // is overlap-safe and needs no second full matrix.
        std::vector<std::size_t> compact_offsets(cursors_.size() + 1, 0);
        for(std::size_t row = 0; row < cursors_.size(); ++row) {
            std::size_t const source_begin = (*row_offsets_)[row];
            std::size_t const used = cursors_[row] - source_begin;
            compact_offsets[row + 1] = compact_offsets[row] + used;
            if(compact_offsets[row] != source_begin)
                for(std::size_t slot = 0; slot < used; ++slot) {
                    (*columns_)[compact_offsets[row] + slot] =
                        (*columns_)[source_begin + slot];
                    (*values_)[compact_offsets[row] + slot] =
                        (*values_)[source_begin + slot];
                }
        }
        row_offsets_->swap(compact_offsets);
        columns_->resize(row_offsets_->back());
        values_->resize(row_offsets_->back());
        for(std::size_t row = 0; row < cursors_.size(); ++row)
            cursors_[row] = (*row_offsets_)[row + 1];
    }

private:
    mat* legacy_values_ = nullptr;
    size_t_mat* legacy_columns_ = nullptr;
    std::vector<size_t>* row_offsets_ = nullptr;
    std::vector<size_t>* columns_ = nullptr;
    std::vector<double>* values_ = nullptr;
    Fixed16BlockStencilMatrix* fixed16_block_stencil_ = nullptr;
    std::vector<size_t> cursors_;
    bool reuse_direct_structure_ = false;
    bool fixed16_block_stencil_valid_ = true;
    double fixed16_discard_value_ = 0.0;
};

void MultigroupDiffusion::BuildMatrix(
    Tessellation3D const& tess,
    mat& A,
    size_t_mat& A_indeces,
    std::vector<ComputationalCell3D> const& cells,
    double const dt,
    std::vector<double>& b,
    std::vector<double>& x0,
    double const current_time) const
{
    invalidateDirectStructureCache(false);
    MatrixRows matrix_rows(A, A_indeces);
    BuildMatrixImpl(tess, matrix_rows, cells, dt, b, x0, current_time);
}

void MultigroupDiffusion::BuildMatrixCSR(
    Tessellation3D const& tess,
    std::vector<size_t>& row_offsets,
    std::vector<size_t>& column_indices,
    std::vector<double>& values,
    std::vector<ComputationalCell3D> const& cells,
    double const dt,
    std::vector<double>& b,
    std::vector<double>& x0,
    double const current_time) const
{
    bool const trace_cache =
        mgRuntimeFlagEnabled("RICH_INDIVIDUAL_PERF_TRACE");
    MultigroupClock::time_point const lookup_start = trace_cache ?
        MultigroupClock::now() : MultigroupClock::time_point{};
    bool const cache_requested =
        mgRuntimeFlagEnabled("RICH_MG_DIRECT_STRUCTURE_CACHE");
#ifdef RICH_MPI
    unsigned int cache_state = cache_requested ? 2u : 1u;
    MPI_Allreduce(MPI_IN_PLACE, &cache_state, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if(cache_state == 3u)
        throw UniversalError(
            "RICH_MG_DIRECT_STRUCTURE_CACHE differs between MPI ranks");
#endif

    if(ENERGY_GROUPS_NUM != 0 &&
       tess.GetPointNo() >
           std::numeric_limits<std::size_t>::max() / ENERGY_GROUPS_NUM)
        throw UniversalError("Multigroup matrix row count overflow");
    std::size_t const row_count = tess.GetPointNo() * ENERGY_GROUPS_NUM;
    if(row_count == std::numeric_limits<std::size_t>::max())
        throw UniversalError("Multigroup matrix row offset count overflow");

    bool const all_active_route = individual_context_ == nullptr;
    bool const owned_workspace =
        &row_offsets == &cg_workspace_.A_row_ptr &&
        &column_indices == &cg_workspace_.A_col_idx &&
        &values == &cg_workspace_.A_values;
    bool const cache_eligible =
        cache_requested && all_active_route && owned_workspace;
    char const* cache_outcome = "disabled";
    bool cache_hit = false;
    if(cache_eligible) {
        buildDirectStructureCacheKey(tess, cells);
        bool structure_shape_matches =
            row_offsets.size() == row_count + 1 &&
            !row_offsets.empty() && row_offsets.front() == 0 &&
            row_offsets.back() == column_indices.size() &&
            direct_structure_cache_rows_ == row_count &&
            direct_structure_cache_nnz_ == column_indices.size();
        if(structure_shape_matches)
            for(std::size_t row = 0; row < row_count; ++row)
                if(row_offsets[row] > row_offsets[row + 1] ||
                   row_offsets[row + 1] > column_indices.size()) {
                    structure_shape_matches = false;
                    break;
                }

        if(!direct_structure_cache_valid_)
            cache_outcome = "miss_invalid";
        else if(direct_structure_cache_bound_epoch_ !=
                direct_structure_cache_solver_epoch_)
            cache_outcome = "miss_solver_epoch";
        else if(direct_structure_cache_key_ !=
                direct_structure_cache_probe_)
            cache_outcome = "miss_structure_key";
        else if(!structure_shape_matches)
            cache_outcome = "miss_workspace_shape";
        else {
            cache_hit = true;
            cache_outcome = "hit";
        }

        if(cache_hit) {
            ++direct_structure_cache_hits_;
            // A failed replay must not leave a reusable cache entry.
            direct_structure_cache_valid_ = false;
        }
        else {
            ++direct_structure_cache_misses_;
            invalidateDirectStructureCache(false);
        }
    }
    else {
        if(cache_requested)
            cache_outcome = all_active_route ?
                "ineligible_workspace" : "ineligible_active_route";
        invalidateDirectStructureCache(false);
        direct_structure_cache_probe_.clear();
    }

    double const lookup_seconds = trace_cache ?
        mgElapsedSeconds(lookup_start) : 0.0;
    MultigroupClock::time_point const assembly_start = trace_cache ?
        MultigroupClock::now() : MultigroupClock::time_point{};
    MatrixRows matrix_rows(
        row_offsets, column_indices, values, cache_hit);
    try {
        BuildMatrixImpl(tess, matrix_rows, cells, dt, b, x0, current_time);
    }
    catch(...) {
        invalidateDirectStructureCache(false);
        throw;
    }

    if(cache_eligible) {
        if(!cache_hit)
            direct_structure_cache_key_.swap(
                direct_structure_cache_probe_);
        direct_structure_cache_probe_.clear();
        direct_structure_cache_rows_ = row_count;
        direct_structure_cache_nnz_ = column_indices.size();
        direct_structure_cache_bound_epoch_ =
            direct_structure_cache_solver_epoch_;
        direct_structure_cache_valid_ = true;
    }

    if(trace_cache) {
        int cache_rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &cache_rank);
#endif
        if(cache_rank == 0)
            RuntimeTraceStream()
                      << "MG_DIRECT_STRUCTURE_CACHE scope=rank_local"
                      << " rank=" << cache_rank
                      << " requested="
                      << (cache_requested ? 1 : 0)
                      << " eligible=" << (cache_eligible ? 1 : 0)
                      << " outcome=" << cache_outcome
                      << " hits=" << direct_structure_cache_hits_
                      << " misses=" << direct_structure_cache_misses_
                      << " solver_epoch="
                      << direct_structure_cache_solver_epoch_
                      << " key_words=" << direct_structure_cache_key_.size()
                      << " rows=" << row_count
                      << " nnz=" << column_indices.size()
                      << " lookup_seconds=" << lookup_seconds
                      << " assembly_seconds="
                      << mgElapsedSeconds(assembly_start)
                      << std::endl;
    }
}

void MultigroupDiffusion::BuildMatrixCSRFixed16BlockStencil(
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
    Fixed16BlockStencilMatrix& fixed16_block_stencil) const
{
    if(build_csr_shadow) {
        BuildMatrixCSR(
            tess, row_offsets, column_indices, values, cells, dt, b, x0,
            current_time);
        invalidateDirectStructureCache(false);
        std::vector<double> fixed16_b;
        std::vector<double> fixed16_x0;
        MatrixRows matrix_rows(fixed16_block_stencil);
        try {
            BuildMatrixImpl(
                tess, matrix_rows, cells, dt, fixed16_b, fixed16_x0,
                current_time);
        }
        catch(...) {
            fixed16_block_stencil.Release();
            throw;
        }
        if(!matrix_rows.Fixed16BlockStencilValid() || fixed16_b != b ||
           fixed16_x0 != x0)
            fixed16_block_stencil.Release();
        return;
    }

    invalidateDirectStructureCache(false);
    row_offsets.clear();
    column_indices.clear();
    values.clear();
    MatrixRows matrix_rows(fixed16_block_stencil);
    try {
        BuildMatrixImpl(
            tess, matrix_rows, cells, dt, b, x0, current_time);
    }
    catch(...) {
        fixed16_block_stencil.Release();
        throw;
    }
    if(!matrix_rows.Fixed16BlockStencilValid())
        fixed16_block_stencil.Release();
}

void MultigroupDiffusion::BuildMatrixImpl(
    Tessellation3D const& tess,
    MatrixRows& matrix_rows,
    std::vector<ComputationalCell3D> const& cells,
    double const dt,
    std::vector<double>& b,
    std::vector<double>& x0,
    double const current_time) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    ensureComptonBulkRuntimeOptions();

    bool const matrix_loop_option_requested =
        skip_reverse_faces_requested_ ||
        hoist_face_mean_temperature_requested_ ||
        freefree_pair16_requested_ ||
        freefree_pair16_shadow_requested_;
    unsigned int matrix_route_state =
        individual_context_ == nullptr ? 1u : 2u;
#ifdef RICH_MPI
    if(matrix_loop_option_requested)
        MPI_Allreduce(MPI_IN_PLACE, &matrix_route_state, 1, MPI_UNSIGNED,
                      MPI_BOR, MPI_COMM_WORLD);
#endif
    bool const global_matrix_route = matrix_route_state == 1u;
    bool const skip_reverse_faces_enabled =
        skip_reverse_faces_requested_ && global_matrix_route;
    bool const hoist_face_mean_temperature_enabled =
        hoist_face_mean_temperature_requested_ && global_matrix_route;
    char const* const matrix_route = matrix_route_state == 1u ? "global" :
        (matrix_route_state == 2u ? "individual" : "mixed");
    unsigned long long local_face_orientations_considered = 0;
    unsigned long long local_reverse_face_orientations_skipped = 0;
    unsigned long long local_mean_temperature_group_uses = 0;
    unsigned long long local_mean_temperature_evaluations = 0;
    bool const freefree_pair16_option_requested =
        freefree_pair16_requested_ || freefree_pair16_shadow_requested_;
    bool local_freefree_pair16_exact_type = false;
    FreeFreeAbsorptionOpacityMultigroup const* freefree_pair16_calculator =
        nullptr;
    if(freefree_pair16_option_requested &&
       typeid(coefficient_calculator) ==
           typeid(FreeFreeAbsorptionOpacityMultigroup)) {
        local_freefree_pair16_exact_type = true;
        freefree_pair16_calculator =
            static_cast<FreeFreeAbsorptionOpacityMultigroup const*>(
                &coefficient_calculator);
    }
    bool const local_freefree_pair16_groups =
        ENERGY_GROUPS_NUM == 16 && energy_groups_center.size() == 16;
    bool const local_freefree_pair16_options =
        freefree_pair16_calculator != nullptr &&
        freefree_pair16_calculator->SupportsProductionPair16();
    int freefree_pair16_support[4] = {
        local_freefree_pair16_exact_type ? 1 : 0,
        local_freefree_pair16_groups ? 1 : 0,
        local_freefree_pair16_options ? 1 : 0,
        freefree_pair16_failed_closed_ ? 0 : 1};
#ifdef RICH_MPI
    if(freefree_pair16_option_requested)
        MPI_Allreduce(MPI_IN_PLACE, freefree_pair16_support, 4, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
#endif
    if(freefree_pair16_support[3] == 0)
        freefree_pair16_failed_closed_ = true;
    bool const freefree_pair16_collective_supported =
        freefree_pair16_option_requested && global_matrix_route &&
        freefree_pair16_support[0] != 0 &&
        freefree_pair16_support[1] != 0 &&
        freefree_pair16_support[2] != 0;
    bool const freefree_pair16_batch_compute_enabled =
        freefree_pair16_collective_supported &&
        !freefree_pair16_failed_closed_;
    bool const freefree_pair16_apply_enabled =
        freefree_pair16_requested_ &&
        freefree_pair16_batch_compute_enabled;
    bool const freefree_pair16_shadow_enabled =
        freefree_pair16_shadow_requested_ &&
        freefree_pair16_batch_compute_enabled;
    std::array<double, 16> freefree_pair16_energies;
    if(freefree_pair16_batch_compute_enabled)
        std::copy_n(energy_groups_center.begin(), 16,
                    freefree_pair16_energies.begin());
    unsigned long long local_freefree_pair16_batch_attempts = 0;
    unsigned long long local_freefree_pair16_batch_calls = 0;
    unsigned long long local_freefree_pair16_applied_faces = 0;
    unsigned long long local_freefree_pair16_shadow_faces = 0;
    unsigned long long local_freefree_pair16_mismatch_faces = 0;
    unsigned long long local_freefree_pair16_mismatch_coefficients = 0;
    unsigned long long local_freefree_pair16_unsupported_faces = 0;
    unsigned long long local_freefree_pair16_fallback_faces = 0;
    unsigned long long local_freefree_pair16_scalar_calls_avoided = 0;
    double local_freefree_pair16_batch_seconds = 0.0;
    double local_freefree_pair16_shadow_seconds = 0.0;
    bool freefree_pair16_local_failed_closed = false;

    bool const trace_matrix_phases =
        mgRuntimeFlagEnabled("RICH_INDIVIDUAL_PERF_TRACE");
    bool const lazy_compton_recovery_requested =
        mgRuntimeFlagEnabled("RICH_MG_LAZY_COMPTON_RECOVERY");
    bool const lazy_compton_recovery_enabled =
        lazy_compton_recovery_requested &&
        compton_bulk_coefficients_enabled_ &&
        !compton_bulk_shadow_enabled_;
    MultigroupClock::time_point const matrix_total_start =
        trace_matrix_phases ? MultigroupClock::now() :
        MultigroupClock::time_point{};
    MultigroupClock::time_point matrix_phase_start = matrix_total_start;
    double matrix_setup_material_seconds = 0.0;
    double matrix_flux_limiter_seconds = 0.0;
    double matrix_diffusion_seconds = 0.0;
    double matrix_velocity_doppler_seconds = 0.0;
    double matrix_validation_seconds = 0.0;

    std::size_t const Nlocal = tess.GetPointNo();
    unsigned long long local_matrix_active_cells = 0;
    unsigned long long local_matrix_implicit_compton_cells = 0;
    unsigned long long local_matrix_deferred_compton_cells = 0;
    std::size_t const initial_cell_count = individual_context_ == nullptr
        ? Nlocal : cells_cgs.size();
    x0.resize(initial_cell_count * ENERGY_GROUPS_NUM, 0.0);
    b.resize(Nlocal * ENERGY_GROUPS_NUM, 0.0);
    // build the `initial guess` and `b`
    for (std::size_t i=0; i < Nlocal; ++i) {
        double const dt_cell_cgs = individualCellTimeStep(i, dt) * time_scale_;
        double const cdt_cell = CG::speed_of_light * dt_cell_cgs;
        auto const& cell_cgs = cells_cgs[i];
        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);
        double const f = fleck_factor[i];

        auto const Um = get_radiation_energy_density(cell_cgs.temperature);
        double aT4_np1 = f * Um;
        double const kp = sigma_absorption_planck[i];
        if (std::isfinite(kp) && kp > 0.0) {
            for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                aT4_np1 += (1 - f) * sigma_absorption_group[i][group] * cell_cgs.Eg[group] * cell_cgs.density / kp;
            }
        }

        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
            double const Eg_i = cell_cgs.Eg[group] * cell_cgs.density;
            // build the initial guess
            auto const bg = planck_integal_group[i][group];
            double const Eg_guess = (Eg_i + bg * cdt_cell * sigma_absorption_group[i][group] * aT4_np1)
                / (1 + cdt_cell * sigma_absorption_group[i][group]);
            // Keep the initial guess physical without distorting a valid
            // spectrum by arbitrary 0.5x/2x clamps.
            x0[i * ENERGY_GROUPS_NUM + group] =
                (std::isfinite(Eg_guess) && Eg_guess >= 0.0) ? Eg_guess : 0.0;
            // build `b` vector, first term
            b[i * ENERGY_GROUPS_NUM + group] = volume_cgs * old_Eg[i][group] * mass_scale_ / (length_scale_*pow<2>(time_scale_));

            // second term
            auto const cdtkgbgf = f*cdt_cell*sigma_absorption_group[i][group]*bg;
            b[i * ENERGY_GROUPS_NUM + group] += volume_cgs*cdtkgbgf*Um;
        }
    }
    if(individual_context_ != nullptr)
        for(std::size_t i = Nlocal; i < cells_cgs.size(); ++i)
            for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                x0[i * ENERGY_GROUPS_NUM + group] =
                    cells_cgs[i].Eg[group] * cells_cgs[i].density;

    // Initialize Matrix
    if(ENERGY_GROUPS_NUM != 0 &&
       Nlocal > std::numeric_limits<std::size_t>::max() / ENERGY_GROUPS_NUM)
        throw UniversalError("Multigroup matrix row count overflow");
    std::size_t const matrix_row_count = Nlocal * ENERGY_GROUPS_NUM;
    if(matrix_rows.Fixed16BlockStencil()) {
        if(ENERGY_GROUPS_NUM != Fixed16BlockStencilMatrix::BlockSize)
            throw UniversalError(
                "Fixed-16 matrix assembly requires exactly 16 groups");
        for(std::size_t cell = 0; cell < Nlocal; ++cell)
            if(!individualCellActive(cell))
                matrix_rows.InvalidateFixed16BlockStencil();
        matrix_rows.InitializeFixed16BlockStencil(tess);
    }
    else if(matrix_rows.Direct()) {
        if(matrix_rows.ReusingDirectStructure())
            matrix_rows.InitializeDirectFromCache(matrix_row_count);
        else {
            std::vector<std::size_t> row_sizes(matrix_row_count, 1);
            for(std::size_t cell = 0; cell < Nlocal; ++cell) {
                if(!individualCellActive(cell))
                    continue;
                std::size_t const face_count =
                    tess.GetCellFaces(cell).size();
                if(face_count > std::numeric_limits<std::size_t>::max() -
                        ENERGY_GROUPS_NUM)
                    throw UniversalError("Direct CSR row capacity overflow");
                std::size_t const row_capacity =
                    ENERGY_GROUPS_NUM + face_count;
                for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                    row_sizes[cell * ENERGY_GROUPS_NUM + group] =
                        row_capacity;
            }
            matrix_rows.InitializeDirect(row_sizes);
        }
    }
    else {
        // Preserve per-row capacity across candidates. The topology may
        // change, so every row is still rebuilt numerically.
        matrix_rows.InitializeLegacy(matrix_row_count);
        for(std::size_t cell = 0; cell < Nlocal; ++cell) {
            std::size_t const row_capacity = individualCellActive(cell) ?
                ENERGY_GROUPS_NUM + tess.GetCellFaces(cell).size() : 1;
            for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                matrix_rows.ReserveLegacy(
                    cell * ENERGY_GROUPS_NUM + group, row_capacity);
        }
    }
    std::vector<double> compton_delta_b(
        lazy_compton_recovery_enabled ? 0 : Nlocal * ENERGY_GROUPS_NUM,
        0.0);
    std::vector<double> compton_delta_A(
        lazy_compton_recovery_enabled ? 0 :
            Nlocal * ENERGY_GROUPS_NUM * ENERGY_GROUPS_NUM,
        0.0);
    std::vector<bool> has_compton_delta(Nlocal, false);

    ImplicitComptonCellCoefficients bulk_compton_coefficients;
    ImplicitComptonCellCoefficients legacy_compton_coefficients;
    int local_compton_shadow_mismatch = 0;
    std::size_t first_compton_shadow_mismatch_cell_id =
        std::numeric_limits<std::size_t>::max();

    // Add the emission term to the matrix
    for (std::size_t i=0; i < Nlocal; ++i) {
        if(!individualCellActive(i)) {
            double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
            for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                std::size_t const row = i * ENERGY_GROUPS_NUM + group;
                matrix_rows.Append(row, row, volume);
            }
            continue;
        }
        ++local_matrix_active_cells;
        double const dt_cell_cgs = individualCellTimeStep(i, dt) * time_scale_;
        double const cdt_cell = CG::speed_of_light * dt_cell_cgs;
        bool const do_compton = !compton_deferred_[i] && compton_on_ && (sigma_absorption_planck[i] * dt_cell_cgs * CG::speed_of_light < compton_optical_depth_turn_off);
        if(do_compton)
            ++local_matrix_implicit_compton_cells;
        else if(compton_on_ && compton_deferred_[i])
            ++local_matrix_deferred_compton_cells;
        ImplicitComptonCellCoefficients const* selected_compton_coefficients =
            nullptr;
        if (do_compton) {
            ComptonOccupationMode const occupation_mode =
                compton_occupation_mode_[i] != ComptonOccupationMode::Off
                    ? compton_occupation_mode_[i]
                    : (use_n_zero[i] ? ComptonOccupationMode::Zero : ComptonOccupationMode::RadiationField);
            generate_S_and_dSdUm_matrices(cells[i], i, dt_cell_cgs, occupation_mode);

            if(compton_bulk_coefficients_enabled_ ||
               compton_bulk_shadow_enabled_)
                fillImplicitComptonCellCoefficients(
                    tess, cells[i], i, dt_cell_cgs,
                    bulk_compton_coefficients);
            bool shadow_matches = true;
            if(compton_bulk_shadow_enabled_) {
                shadow_matches =
                    fillLegacyImplicitComptonCellCoefficientsAndCompare(
                        tess, cells[i], i, dt_cell_cgs,
                        bulk_compton_coefficients,
                        legacy_compton_coefficients, "BuildMatrixImpl");
                if(!shadow_matches) {
                    local_compton_shadow_mismatch = 1;
                    if(first_compton_shadow_mismatch_cell_id ==
                       std::numeric_limits<std::size_t>::max())
                        first_compton_shadow_mismatch_cell_id = cells[i].ID;
                }
            }
            if(compton_bulk_coefficients_enabled_ && shadow_matches)
                selected_compton_coefficients = &bulk_compton_coefficients;
            else if(compton_bulk_shadow_enabled_)
                selected_compton_coefficients = &legacy_compton_coefficients;
        }

        double const f = fleck_factor[i];
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
            auto const bg = planck_integal_group[i][group];
            double const gamma_safe = (std::isfinite(Gammas[i]) && Gammas[i] > 0.0)
                ? Gammas[i] : std::numeric_limits<double>::min();
            double const Gamma_1 = 1.0 / gamma_safe;

            double const cdtkg = cdt_cell * sigma_absorption_group[i][group];
            double const implicit_self_contribution = -(1 - f) * cdtkg * Gamma_1 * bg * sigma_absorption_group[i][group];

            double implicit_self_compton_contribution = 0.0;

            if (do_compton) {
                double const implicit_compton_contribution_to_b =
                    selected_compton_coefficients != nullptr ?
                        selected_compton_coefficients->delta_b[group] :
                        get_implicit_compton_contribution_to_b(
                            tess, cells[i], i, group, dt_cell_cgs);
                b[i * ENERGY_GROUPS_NUM + group] += implicit_compton_contribution_to_b;
                if(!lazy_compton_recovery_enabled)
                    compton_delta_b[i * ENERGY_GROUPS_NUM + group] =
                        implicit_compton_contribution_to_b;
                has_compton_delta[i] = true;

                implicit_self_compton_contribution =
                    selected_compton_coefficients != nullptr ?
                        selected_compton_coefficients->delta_A[
                            group * ENERGY_GROUPS_NUM + group] :
                        get_implicit_compton_contribution(
                            tess, cells[i], i, group, group, dt_cell_cgs);
                if(!lazy_compton_recovery_enabled)
                    compton_delta_A[
                        (i * ENERGY_GROUPS_NUM + group) *
                            ENERGY_GROUPS_NUM + group] =
                        implicit_self_compton_contribution;
            }

            std::size_t const row = i * ENERGY_GROUPS_NUM + group;
            matrix_rows.Append(
                row, row,
                volume * (1.0 + cdtkg + implicit_self_contribution) +
                    implicit_self_compton_contribution);

            for (size_t group_j=0; group_j<ENERGY_GROUPS_NUM; ++group_j) {
                if (group_j!= group) {
                    double const implicit_conribution_group_j = -volume*bg * (1 - f) * sigma_absorption_group[i][group_j] * sigma_absorption_group[i][group] * cdt_cell * Gamma_1;

                    double implicit_compton_contribution_group_j = 0.0;
                    if (do_compton) {
                        implicit_compton_contribution_group_j =
                            selected_compton_coefficients != nullptr ?
                                selected_compton_coefficients->delta_A[
                                    group * ENERGY_GROUPS_NUM + group_j] :
                                get_implicit_compton_contribution(
                                    tess, cells[i], i, group, group_j,
                                    dt_cell_cgs);
                        if(!lazy_compton_recovery_enabled)
                            compton_delta_A[
                                (i * ENERGY_GROUPS_NUM + group) *
                                    ENERGY_GROUPS_NUM + group_j] =
                                implicit_compton_contribution_group_j;
                    }

                    matrix_rows.Append(
                        row, i * ENERGY_GROUPS_NUM + group_j,
                        implicit_conribution_group_j +
                            implicit_compton_contribution_group_j);
                }
            }
        }
    }

    if(compton_bulk_shadow_enabled_) {
        int global_compton_shadow_mismatch =
            local_compton_shadow_mismatch;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &global_compton_shadow_mismatch, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(global_compton_shadow_mismatch != 0) {
            matrix_unrecoverable_ = true;
            if(local_compton_shadow_mismatch != 0) {
                setCellLocalStepFailure(
                    "Compton bulk coefficient shadow mismatch in BuildMatrixImpl",
                    first_compton_shadow_mismatch_cell_id);
            }
            else if(getLastStepFailureReason().empty()) {
                setStepFailure(
                    "Compton bulk coefficient shadow mismatch on another MPI rank in BuildMatrixImpl");
                markStepFailureRemote();
            }
            throw UniversalError(
                "Compton bulk coefficient shadow mismatch in BuildMatrixImpl");
        }
    }

    if(trace_matrix_phases) {
        matrix_setup_material_seconds = mgElapsedSeconds(matrix_phase_start);
        matrix_phase_start = MultigroupClock::now();
    }
    // calculate R2
    std::vector<std::size_t> neighbors;
    Vector3D dummy_v;
    std::vector<double> R2(matrix_row_count, 1.0 / 3.0);
    std::vector<Vector3D> grad_temp_array(ENERGY_GROUPS_NUM);
    if (flux_limiter_) {
        for (std::size_t i=0; i < Nlocal; ++i) {
            if(!individualCellActive(i))
                continue;
            tess.GetNeighbors(i, neighbors);
            face_vec const& faces = tess.GetCellFaces(i);
            double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);

            auto const Nneighbors = neighbors.size();
            double Er_i = cells_cgs[i].Erad * cells_cgs[i].density;
            Vector3D const r_i = tess.GetMeshPoint(i);
            for (std::size_t j=0; j < Nneighbors; ++j) {
                std::size_t const neighbor_j = neighbors[j];
                Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                if (neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)) {
                    double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density;
                    auto const abs_dE = std::abs(Er_i - Er_j);
                    auto const abs_grad_E = abs_dE * fastabs(grad[faces[j]]);

                }
                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                    if (j == 0) grad_temp_array[g].Set(0, 0, 0);

                    Er_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;
                    if (neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)) {
                        double const Er_j = cells_cgs[neighbor_j].Eg[g] * cells_cgs[neighbor_j].density;
                        grad_temp_array[g] += (tess.GetArea(faces[j]) * pow<2>(length_scale_) * 0.5 * (Er_j + Er_i)) * r_ij;
                    } else {
                        grad_temp_array[g] += (tess.GetArea(faces[j]) * pow<2>(length_scale_) * 0.5 * (Er_i + Er_i)) * r_ij;
                    }
                }
            }
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                double const Dg = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, g);
                Vector3D grad_for_limiter = grad_temp_array[g] / (tess.GetVolume(i) * pow<3>(length_scale_));
                double const Eg_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;
                double const min_grad = std::abs(Eg_i) / (1000.0 * cell_width);
                double const grad_abs = std::abs(fastabs(grad_for_limiter));
                if (grad_abs < min_grad) {
                    if (grad_abs > 0)
                        grad_for_limiter *= min_grad / grad_abs;
                    else
                        grad_for_limiter = Vector3D(min_grad, 0, 0);
                }

                double const lambda  =  CG::CalcSingleFluxLimiter(grad_for_limiter, Dg, Eg_i) / 3;
                double const sigma_t =  CG::speed_of_light / (3 * Dg) + 1e-100;
                double const Erad_i = cells_cgs[i].Erad * cells_cgs[i].density;
                double const Eg_for_Rg = std::max(std::abs(Eg_i), min_grad * cell_width);
                double const Eg_R2_floor = 1e-12 * std::max(std::abs(Erad_i), 1e-200);
                double R_g = 0.0;
                if (abs(grad_temp_array[g]) < 1e-100
                    || Eg_for_Rg < Eg_R2_floor) {
                    R2[i * ENERGY_GROUPS_NUM + g] = 1.0 / 3.0;
                } else {
                    double const grad_mag = fastabs(grad_for_limiter);
                    R_g = grad_mag / (sigma_t * Eg_for_Rg + 1e-200);
                    R2[i * ENERGY_GROUPS_NUM + g] =
                        lambda + lambda * lambda * R_g * R_g;
                    R2[i * ENERGY_GROUPS_NUM + g] =
                        std::min(R2[i * ENERGY_GROUPS_NUM + g], 1.0);
                }

                if (cells[i].ID==-1) {
                    std::clog<<"R2["<<i<<"]["<<g<<"] = "<<R2[i * ENERGY_GROUPS_NUM + g]<<" density "<<cells[i].density<<" T "<<cells[i].temperature<<" location "<<tess.GetMeshPoint(i)<<" lambda "<<lambda<<" R_g "<<R_g<<" sigma_t "<<sigma_t
                        <<" grad_Eg "<<fastabs(grad_for_limiter)<<" Eg "<<cells_cgs[i].Eg[g] * cells_cgs[i].density<<std::endl;
                }
            }
        }
    }

    if(trace_matrix_phases) {
        matrix_flux_limiter_seconds = mgElapsedSeconds(matrix_phase_start);
        matrix_phase_start = MultigroupClock::now();
    }
    // Add the diffusion terms
    for (std::size_t i=0; i < Nlocal; ++i) {
        if(!individualCellActive(i))
            continue;
        face_vec const& faces = tess.GetCellFaces(i);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();

        Vector3D const r_i = tess.GetMeshPoint(i);

        auto& cell_i = cells_cgs[i]; // reference and not const reference is because we change cell_i temperature to calculate the diffusion coefficient 

        for (std::size_t j=0; j < Nneighbors; ++j) {
            std::size_t const neighbor_j = neighbors[j];
            if(skip_reverse_faces_enabled) {
                ++local_face_orientations_considered;
                if(i >= neighbor_j) {
                    ++local_reverse_face_orientations_skipped;
                    continue;
                }
            }
            double const dt_face_cgs =
                individualFaceTimeStep(i, neighbor_j, dt) * time_scale_;

            auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);

            double const abs_r_ij = abs(r_ij);
            r_ij *= 1.0 / abs_r_ij; // normalize the vector perpendicular to the face between cells i and j

            double Eg_j = 0;
            bool const outside_point = tess.IsPointOutsideBox(neighbor_j);
            ComputationalCell3D* cell_j = outside_point ? nullptr : &cells_cgs[neighbor_j];
            double hoisted_face_mean_temperature = 0;
            if(hoist_face_mean_temperature_enabled && !outside_point &&
               i < neighbor_j) {
                double const T_i = cell_i.temperature;
                double const T_j = cell_j->temperature;
                hoisted_face_mean_temperature = std::pow(
                    0.5 * (pow<4>(T_i) + pow<4>(T_j)), 0.25);
                ++local_mean_temperature_evaluations;
            }
            std::array<double, 16> freefree_pair16_first;
            std::array<double, 16> freefree_pair16_second;
            std::array<double, 16> freefree_pair16_legacy_first;
            std::array<double, 16> freefree_pair16_legacy_second;
            bool freefree_pair16_face_coefficients_ready = false;
            bool freefree_pair16_face_uses_batch = false;
            double freefree_pair16_face_mean_temperature = 0;
            bool const freefree_pair16_face_eligible =
                freefree_pair16_batch_compute_enabled && !outside_point &&
                i < neighbor_j;
            if(freefree_pair16_face_eligible &&
               freefree_pair16_local_failed_closed) {
                ++local_freefree_pair16_fallback_faces;
            }
            if(freefree_pair16_face_eligible &&
               !freefree_pair16_local_failed_closed) {
                double const T_i = cell_i.temperature;
                double const T_j = cell_j->temperature;
                freefree_pair16_face_mean_temperature =
                    hoist_face_mean_temperature_enabled ?
                    hoisted_face_mean_temperature :
                    std::pow(
                        0.5 * (pow<4>(T_i) + pow<4>(T_j)), 0.25);
                cell_j->temperature = freefree_pair16_face_mean_temperature;
                cell_i.temperature = freefree_pair16_face_mean_temperature;

                MultigroupClock::time_point const batch_start =
                    MultigroupClock::now();
                ++local_freefree_pair16_batch_attempts;
                bool const face_batch_supported = freefree_pair16_calculator->
                    CalcProductionDiffusionCoefficientPair16(
                        cell_i, *cell_j, freefree_pair16_energies,
                        freefree_pair16_first, freefree_pair16_second);
                if(face_batch_supported) {
                    for(std::size_t group = 0; group < 16; ++group) {
                        freefree_pair16_first[group] =
                            applyComptonTransportCorrection(
                                freefree_pair16_first[group], i, group);
                        freefree_pair16_second[group] =
                            applyComptonTransportCorrection(
                                freefree_pair16_second[group], neighbor_j,
                                group);
                    }
                    ++local_freefree_pair16_batch_calls;
                }
                local_freefree_pair16_batch_seconds +=
                    mgElapsedSeconds(batch_start);

                bool shadow_mismatch = false;
                if(face_batch_supported && freefree_pair16_shadow_enabled) {
                    MultigroupClock::time_point const shadow_start =
                        MultigroupClock::now();
                    for(std::size_t group = 0; group < 16; ++group) {
                        freefree_pair16_legacy_first[group] =
                            calcEffectiveDiffusionCoefficient(
                                cell_i, i, group);
                        freefree_pair16_legacy_second[group] =
                            calcEffectiveDiffusionCoefficient(
                                *cell_j, neighbor_j, group);
                    }
                    local_freefree_pair16_shadow_seconds +=
                        mgElapsedSeconds(shadow_start);
                    ++local_freefree_pair16_shadow_faces;
                    for(std::size_t group = 0; group < 16; ++group) {
                        if(std::memcmp(&freefree_pair16_first[group],
                                       &freefree_pair16_legacy_first[group],
                                       sizeof(double)) != 0) {
                            shadow_mismatch = true;
                            ++local_freefree_pair16_mismatch_coefficients;
                        }
                        if(std::memcmp(&freefree_pair16_second[group],
                                       &freefree_pair16_legacy_second[group],
                                       sizeof(double)) != 0) {
                            shadow_mismatch = true;
                            ++local_freefree_pair16_mismatch_coefficients;
                        }
                    }
                }

                cell_i.temperature = T_i;
                cell_j->temperature = T_j;

                if(!face_batch_supported) {
                    ++local_freefree_pair16_unsupported_faces;
                    ++local_freefree_pair16_fallback_faces;
                    freefree_pair16_local_failed_closed = true;
                }
                else if(shadow_mismatch) {
                    ++local_freefree_pair16_mismatch_faces;
                    ++local_freefree_pair16_fallback_faces;
                    freefree_pair16_local_failed_closed = true;
                    freefree_pair16_face_coefficients_ready = true;
                }
                else if(freefree_pair16_apply_enabled) {
                    freefree_pair16_face_coefficients_ready = true;
                    freefree_pair16_face_uses_batch = true;
                    ++local_freefree_pair16_applied_faces;
                    if(!freefree_pair16_shadow_enabled)
                        local_freefree_pair16_scalar_calls_avoided += 32;
                }
                else if(freefree_pair16_shadow_enabled) {
                    // Shadow-only mode always applies the legacy coefficients.
                    freefree_pair16_face_coefficients_ready = true;
                }
            }
            for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                double const Eg_i = cell_i.Eg[group] * cell_i.density;

                if (!outside_point) {
                    Eg_j = cell_j->Eg[group] * cell_j->density;

                    bool const neighbor_active = individualCellActive(neighbor_j);
                    bool const assemble_face = individual_context_ == nullptr
                        ? i < neighbor_j
                        : (!neighbor_active || i < neighbor_j);
                    // Active-active faces are symmetric. Active-passive faces
                    // contribute only to the active row and fixed passive RHS.
                    if (assemble_face) {
                        auto const& face_j = faces[j];
                        Vector3D const& gradient = grad[face_j];

                        // calculate the diffusion coefficient on the boundary using the maximal temperature of the cells
                        double const T_i = cell_i.temperature;
                        double const T_j = cell_j->temperature;
                        double const max_T =
                            hoist_face_mean_temperature_enabled ?
                            hoisted_face_mean_temperature :
                            (freefree_pair16_face_coefficients_ready ?
                             freefree_pair16_face_mean_temperature :
                             std::pow(
                                 0.5 * (pow<4>(T_i) + pow<4>(T_j)),
                                 0.25));
                        if(hoist_face_mean_temperature_enabled)
                            ++local_mean_temperature_group_uses;

                        cell_j->temperature = max_T;
                        cell_i.temperature = max_T;

                        double const D_i =
                            freefree_pair16_face_coefficients_ready ?
                            (freefree_pair16_face_uses_batch ?
                             freefree_pair16_first[group] :
                             freefree_pair16_legacy_first[group]) :
                            calcEffectiveDiffusionCoefficient(
                                cell_i, i, group);
                        double const D_j =
                            freefree_pair16_face_coefficients_ready ?
                            (freefree_pair16_face_uses_batch ?
                             freefree_pair16_second[group] :
                             freefree_pair16_legacy_second[group]) :
                            calcEffectiveDiffusionCoefficient(
                                *cell_j, neighbor_j, group);

                        cell_i.temperature = T_i;
                        cell_j->temperature = T_j;

                        double const D_ij = 2.0 * D_i * D_j / (D_i + D_j);

                        double lambda = 1.0;
                        if (flux_limiter_) {
                            double const dEg = Eg_i - Eg_j;

                            // double const gradE_magnitude = std::max(std::abs(fastabs(gradient)*dEg), std::numeric_limits<double>::min()*1e40);
                            double const grad_factor = 1;//std::max(0.15 * (max_abs_grad_E[i] + max_abs_grad_E[neighbor_j])/gradE_magnitude, 1.0);
                            Vector3D grad_for_limiter = gradient * dEg * grad_factor;
                            double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
                            double const E_mid = 0.5 * (Eg_i + Eg_j);
                            double const min_grad = std::abs(E_mid) / (1000.0 * cell_width);
                            double const grad_abs = std::abs(fastabs(grad_for_limiter));
                            if (grad_abs < min_grad) {
                                if (grad_abs > 0)
                                    grad_for_limiter *= min_grad / grad_abs;
                                else
                                    grad_for_limiter = Vector3D(min_grad, 0, 0);
                            }

                            lambda = CG::CalcSingleFluxLimiter(grad_for_limiter, D_ij, E_mid);
                        }
                        double const lambdaD = lambda*D_ij;

                        double const A_j = tess.GetArea(face_j) * pow<2>(length_scale_);
                        double const orientation = i < neighbor_j ? 1.0 : -1.0;
                        double const flux = dt_face_cgs * lambdaD * orientation *
                                            ScalarProd(gradient, r_ij) * A_j;

                        recordIndividualFaceCoefficient(i, neighbor_j, group, flux);

                        std::size_t const row =
                            i * ENERGY_GROUPS_NUM + group;
                        matrix_rows.Value(row, 0) += flux;
                        matrix_rows.Append(
                            row, neighbor_j * ENERGY_GROUPS_NUM + group,
                            -flux);

                        if (neighbor_j < Nlocal &&
                            (individual_context_ == nullptr || neighbor_active)) {
                            std::size_t const neighbor_row =
                                neighbor_j * ENERGY_GROUPS_NUM + group;
                            matrix_rows.Value(neighbor_row, 0) += flux;
                            matrix_rows.Append(
                                neighbor_row, i * ENERGY_GROUPS_NUM + group,
                                -flux);
                        }
                    }
                } else { // boundary condition
                    if (individual_context_ != nullptr || i < neighbor_j) {
                        boundary_calculator.setBoundaryValuesGroup(group, tess, i, neighbor_j, dt_face_cgs, cells_cgs, tess.GetArea(faces[j])*pow<2>(length_scale_), matrix_rows.Value(i * ENERGY_GROUPS_NUM + group, 0), b[i * ENERGY_GROUPS_NUM + group], faces[j]);
                    }
                }
            }
        }
    }

    matrix_rows.FinishDirect();

    if(trace_matrix_phases) {
        matrix_diffusion_seconds = mgElapsedSeconds(matrix_phase_start);
        matrix_phase_start = MultigroupClock::now();
    }
    // Add velocity term
    for (std::size_t i=0; i<Nlocal; ++i) {
        if(!individualCellActive(i))
            continue;
        double const dt_cell_cgs = individualCellTimeStep(i, dt) * time_scale_;

        face_vec const& faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        double div_V = 0;
        Vector3D const r_i = tess.GetMeshPoint(i);
        for (std::size_t j=0; j<Nneighbors; ++j) {
            std::size_t const neighbor_j = neighbors[j];

            auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));

            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
            Vector3D velocity_j;
            bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
            if (!is_outside) {
                velocity_j = cells_cgs[neighbor_j].velocity;
            } else {
                double dummyEg_i, dummy_Eg_j;
                boundary_calculator.getOutsideValuesGroup(0, tess, i, neighbor_j, cells_cgs, dummyEg_i, dummy_Eg_j, velocity_j);
            }

            div_V -= 0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij;
            if (hydro_on_ or doppler_on_) {
                for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                    double const velocity_diag = -0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij * dt_cell_cgs * (0.5 - 0.5 * R2[i * ENERGY_GROUPS_NUM + group]);
                    matrix_rows.Value(
                        i * ENERGY_GROUPS_NUM + group, 0) += velocity_diag;
                }
            }
        }

        if (doppler_on_) {
            // double const coeff = -div_V * dt_cell_cgs / 3;
            double const coeff = -div_V * dt_cell_cgs;
            for (std::size_t g=1; g<ENERGY_GROUPS_NUM; ++g) {
                if (div_V < 0) {
                    double const slope_left = get_doppler_slope(cells_cgs[i], g - 1, false);
                    size_t const gm = g - 1;
                    std::size_t const row_g = i * ENERGY_GROUPS_NUM + g;
                    std::size_t const row_gm = i * ENERGY_GROUPS_NUM + gm;
                    std::size_t const gm_index = matrix_rows.FindColumn(
                        row_g, row_gm);
                    if (gm_index == max_size_t) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + g - 1] in A_indeces (1)");
                    }

                    if (matrix_rows.Column(row_g, gm_index) != row_gm) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gm] in A_indeces (2)");
                    }

                    std::size_t const g_index = matrix_rows.FindColumn(
                        row_gm, row_g);
                    if (g_index == max_size_t) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + g] in A_indeces (1)");
                    }

                    double coeff_left = 1 / energy_groups_width[gm] - 0.5 * slope_left  / energy_groups_width[gm];
                    coeff_left *= 0.5 - 0.5*R2[i * ENERGY_GROUPS_NUM + gm];
                    coeff_left *= coeff * energy_groups_boundary[g];

                    double coeff_right = 0.5 * slope_left  / energy_groups_width[g];
                    coeff_right *= 0.5 - 0.5*R2[i * ENERGY_GROUPS_NUM + g];
                    coeff_right *= coeff * energy_groups_boundary[g];

                    matrix_rows.Value(row_gm, 0) += coeff_left;
                    matrix_rows.Value(row_g, 0) -= coeff_right;

                    matrix_rows.Value(row_g, gm_index) -= coeff_left;
                    matrix_rows.Value(row_gm, g_index) += coeff_right;
                } else {
                    double const slope_right = get_doppler_slope(cells_cgs[i], g, true);
                    size_t const gm = g - 1;
                    std::size_t const row_g = i * ENERGY_GROUPS_NUM + g;
                    std::size_t const row_gm = i * ENERGY_GROUPS_NUM + gm;
                    std::size_t const g_index = matrix_rows.FindColumn(
                        row_gm, row_g);
                    if (g_index == max_size_t) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + g] in A_indeces (1)");
                    }

                    double coeff_right = 1 / energy_groups_width[g];
                    double coeff_right_right = 0;
                    if ((g + 1) < ENERGY_GROUPS_NUM) {
                        coeff_right += 0.5 * slope_right  / energy_groups_width[g + 1];
                        coeff_right_right = -0.5 * slope_right * energy_groups_width[g] / (energy_groups_width[g + 1] * energy_groups_width[g + 1]);
                    }
                    coeff_right *= coeff * (0.5 - 0.5*R2[i * ENERGY_GROUPS_NUM + g]) * energy_groups_boundary[g];
                    coeff_right_right *= coeff * energy_groups_boundary[g];

                    coeff_right_right *= (g + 1) < ENERGY_GROUPS_NUM ? 0.5 - 0.5*R2[i * ENERGY_GROUPS_NUM + g + 1] : 0.5 - 0.5*R2[i * ENERGY_GROUPS_NUM + g];

                    matrix_rows.Value(row_g, 0) -= coeff_right;
                    matrix_rows.Value(row_gm, g_index) += coeff_right;
                    if (g + 1 < ENERGY_GROUPS_NUM) {
                        size_t const gp = g + 1;
                        std::size_t const row_gp =
                            i * ENERGY_GROUPS_NUM + gp;
                        std::size_t gp_index = matrix_rows.FindColumn(
                            row_g, row_gp);
                        if (gp_index == max_size_t ||
                            matrix_rows.Column(row_g, gp_index) != row_gp) {
                            throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gp] in A_indeces (2)");
                        }
                        matrix_rows.Value(row_g, gp_index) -=
                            coeff_right_right;

                        gp_index = matrix_rows.FindColumn(row_gm, row_gp);
                        if (gp_index == max_size_t ||
                            matrix_rows.Column(row_gm, gp_index) != row_gp) {
                            throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gp] in A_indeces (2)");
                        }
                        matrix_rows.Value(row_gm, gp_index) +=
                            coeff_right_right;
                    }
                }
            }
        }
    }

    if(trace_matrix_phases) {
        matrix_velocity_doppler_seconds =
            mgElapsedSeconds(matrix_phase_start);
        matrix_phase_start = MultigroupClock::now();
    }
    char const* const ragged_value =
        std::getenv("RICH_MG_RAGGED_MATRIX_ROWS");
    std::string const ragged_setting = ragged_value == nullptr ? "" :
        std::string(ragged_value);
    bool const ragged_matrix_rows = !ragged_setting.empty() &&
        ragged_setting != "0" && ragged_setting != "false" &&
        ragged_setting != "off" && ragged_setting != "no";
    static bool reported_ragged_rows = false;
    if(rank == 0 && !reported_ragged_rows) {
        std::clog << "MG_RAGGED_MATRIX_ROWS enabled="
                  << (ragged_matrix_rows ? 1 : 0) << std::endl;
        reported_ragged_rows = true;
    }
    if(!matrix_rows.Direct() && !ragged_matrix_rows) {
        // Legacy non-CRS matvecs require equal row widths with a sentinel.
        // The optimized mode skips this padding; the active CRS path already
        // accepts variable row lengths and preserves the same entry ordering.
        std::size_t max_neighbors = 0;
        for (std::size_t i=0; i < Nlocal; ++i) {
            for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                max_neighbors = std::max(max_neighbors,
                    matrix_rows.Size(i * ENERGY_GROUPS_NUM + group));
            }
        }
        ++max_neighbors;

        for (std::size_t i=0; i < Nlocal; ++i) {
            for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                matrix_rows.ResizeLegacy(
                    i * ENERGY_GROUPS_NUM + group, max_neighbors);
            }
        }
    }

    int local_existing_bad = 0;
    bool have_example = false;
    std::size_t example_local_index = 0;
    size_t example_bad_cell_id = 0;
    size_t example_bad_group = 0;
    double example_bad_diagonal = 0.0;
    double example_bad_threshold = 0.0;
    auto remember_bad_diagonal = [&](std::size_t const i,
                                     std::size_t const group,
                                     double const diagonal,
                                     double const threshold) {
        if (have_example)
            return;
        have_example = true;
        example_local_index = i;
        example_bad_cell_id = cells[i].ID;
        example_bad_group = group;
        example_bad_diagonal = diagonal;
        example_bad_threshold = threshold;
    };
    ImplicitComptonCellCoefficients lazy_recovery_coefficients;
    auto const refill_lazy_compton_recovery = [&](std::size_t const i)
        -> ImplicitComptonCellCoefficients const& {
        ComptonOccupationMode const occupation_mode =
            compton_occupation_mode_[i] != ComptonOccupationMode::Off ?
                compton_occupation_mode_[i] :
                (use_n_zero[i] ? ComptonOccupationMode::Zero :
                                 ComptonOccupationMode::RadiationField);
        double const dt_cell_cgs =
            individualCellTimeStep(i, dt) * time_scale_;
        generate_S_and_dSdUm_matrices(
            cells[i], i, dt_cell_cgs, occupation_mode);
        fillImplicitComptonCellCoefficients(
            tess, cells[i], i, dt_cell_cgs,
            lazy_recovery_coefficients);
        return lazy_recovery_coefficients;
    };
    for (std::size_t i = 0; i < Nlocal; ++i) {
        double const threshold = 0.25 * tess.GetVolume(i) * pow<3>(length_scale_);
        bool bad_diagonal = false;
        size_t bad_group = 0;
        double bad_diagonal_value = 0.0;
        for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
            double const diagonal = matrix_rows.Value(
                i * ENERGY_GROUPS_NUM + group, 0);
            if (!std::isfinite(diagonal) || diagonal < threshold) {
                bad_diagonal = true;
                bad_group = group;
                bad_diagonal_value = diagonal;
                break;
            }
        }
        if (!bad_diagonal)
            continue;

        if (has_compton_delta[i] && !compton_deferred_[i]) {
            ImplicitComptonCellCoefficients const* lazy_recovery =
                lazy_compton_recovery_enabled ?
                    &refill_lazy_compton_recovery(i) : nullptr;
            compton_deferred_[i] = true;
            for (std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                std::size_t const row = i * ENERGY_GROUPS_NUM + group;
                b[row] -= lazy_recovery != nullptr ?
                    lazy_recovery->delta_b[group] : compton_delta_b[row];
                matrix_rows.Value(row, 0) -=
                    lazy_recovery != nullptr ?
                        lazy_recovery->delta_A[
                            group * ENERGY_GROUPS_NUM + group] :
                        compton_delta_A[
                            row * ENERGY_GROUPS_NUM + group];
                std::size_t slot = 1;
                for (std::size_t target = 0; target < ENERGY_GROUPS_NUM; ++target) {
                    if (target == group)
                        continue;
                    matrix_rows.Value(row, slot) -=
                        lazy_recovery != nullptr ?
                            lazy_recovery->delta_A[
                                group * ENERGY_GROUPS_NUM + target] :
                            compton_delta_A[
                                row * ENERGY_GROUPS_NUM + target];
                    ++slot;
                }
            }
        } else {
            local_existing_bad = 1;
            remember_bad_diagonal(i, bad_group, bad_diagonal_value, threshold);
        }
        if (compton_deferred_[i]) {
            for (std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                double const diagonal = matrix_rows.Value(
                    i * ENERGY_GROUPS_NUM + group, 0);
                if (!std::isfinite(diagonal) || diagonal < threshold) {
                    local_existing_bad = 1;
                    remember_bad_diagonal(i, group, diagonal, threshold);
                    break;
                }
            }
        }
    }

    int global_existing_bad = local_existing_bad;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_existing_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_existing_bad != 0) {
        matrix_unrecoverable_ = true;
        if (have_example) {
            std::size_t const i = example_local_index;
            std::size_t const group = example_bad_group;
            double const dt_cell_cgs = individualCellTimeStep(i, dt) * time_scale_;
            double const cdt_cell = CG::speed_of_light * dt_cell_cgs;
            double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
            double const f = fleck_factor[i];
            double const bg = planck_integal_group[i][group];
            double const gamma_safe = (std::isfinite(Gammas[i]) && Gammas[i] > 0.0)
                ? Gammas[i] : std::numeric_limits<double>::min();
            double const cdtkg = cdt_cell * sigma_absorption_group[i][group];
            double const source_diagonal = volume *
                (1.0 + cdtkg - (1.0 - f) * cdtkg * (1.0 / gamma_safe) *
                 bg * sigma_absorption_group[i][group]);
            std::size_t const row = i * ENERGY_GROUPS_NUM + group;
            double const compton_removed = lazy_compton_recovery_enabled ?
                (has_compton_delta[i] ?
                    refill_lazy_compton_recovery(i).delta_A[
                        group * ENERGY_GROUPS_NUM + group] : 0.0) :
                compton_delta_A[row * ENERGY_GROUPS_NUM + group];
            double const compton_remaining = compton_deferred_[i] ? 0.0 : compton_removed;

            double diffusion_diagonal = 0.0;
            tess.GetNeighbors(i, neighbors);
            face_vec const& faces = tess.GetCellFaces(i);
            Vector3D const r_i = tess.GetMeshPoint(i);
            for (std::size_t j = 0; j < neighbors.size(); ++j) {
                std::size_t const neighbor_j = neighbors[j];
                double const dt_face_cgs =
                    individualFaceTimeStep(i, neighbor_j, dt) * time_scale_;
                if (tess.IsPointOutsideBox(neighbor_j))
                    continue;
                Vector3D r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                double const abs_r_ij = abs(r_ij);
                if (!(abs_r_ij > 0.0) || i == neighbor_j)
                    continue;
                r_ij *= 1.0 / abs_r_ij;
                auto const& face = faces[j];
                Vector3D const& gradient = grad[face];
                auto& cell_i = cells_cgs[i];
                auto& cell_j = cells_cgs[neighbor_j];
                double const T_i = cell_i.temperature;
                double const T_j = cell_j.temperature;
                double const max_T = std::pow(0.5 * (pow<4>(T_i) + pow<4>(T_j)), 0.25);
                cell_i.temperature = max_T;
                cell_j.temperature = max_T;
                double const D_i = calcEffectiveDiffusionCoefficient(cell_i, i, group);
                double const D_j = calcEffectiveDiffusionCoefficient(cell_j, neighbor_j, group);
                cell_i.temperature = T_i;
                cell_j.temperature = T_j;
                double const D_ij = 2.0 * D_i * D_j / (D_i + D_j);
                double lambda = 1.0;
                if (flux_limiter_) {
                    double const Eg_i = cell_i.Eg[group] * cell_i.density;
                    double const Eg_j = cell_j.Eg[group] * cell_j.density;
                    Vector3D grad_for_limiter = gradient * (Eg_i - Eg_j);
                    double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
                    double const E_mid = 0.5 * (Eg_i + Eg_j);
                    double const min_grad = std::abs(E_mid) / (1000.0 * cell_width);
                    double const grad_abs = std::abs(fastabs(grad_for_limiter));
                    if (grad_abs < min_grad) {
                        if (grad_abs > 0.0)
                            grad_for_limiter *= min_grad / grad_abs;
                        else
                            grad_for_limiter = Vector3D(min_grad, 0, 0);
                    }
                    lambda = CG::CalcSingleFluxLimiter(grad_for_limiter, D_ij, E_mid);
                }
                double const flux = dt_face_cgs * lambda * D_ij *
                    (i < neighbor_j ? 1.0 : -1.0) * ScalarProd(gradient, r_ij) *
                    tess.GetArea(face) * pow<2>(length_scale_);
                diffusion_diagonal += flux;
            }

            double velocity_doppler_face = 0.0;
            double velocity_doppler_diagonal = 0.0;
            double div_V = 0.0;
            tess.GetNeighbors(i, neighbors);
            for (std::size_t j = 0; j < neighbors.size(); ++j) {
                std::size_t const neighbor_j = neighbors[j];
                Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                Vector3D velocity_j;
                if (!tess.IsPointOutsideBox(neighbor_j)) {
                    velocity_j = cells_cgs[neighbor_j].velocity;
                } else {
                    double dummy_Eg_i, dummy_Eg_j;
                    boundary_calculator.getOutsideValuesGroup(0, tess, i, neighbor_j,
                        cells_cgs, dummy_Eg_i, dummy_Eg_j, velocity_j);
                }
                double const face_div = -0.5 * ScalarProd(cells_cgs[i].velocity + velocity_j, r_ij) * A_ij;
                div_V += face_div;
                if (hydro_on_ || doppler_on_) {
                    double const face_contrib = face_div * dt_cell_cgs * (0.5 - 0.5 * R2[i * ENERGY_GROUPS_NUM + group]);
                    velocity_doppler_face += face_contrib;
                    velocity_doppler_diagonal += face_contrib;
                }
            }
            double const doppler_coeff = -div_V * dt_cell_cgs;
            double intergroup_contrib_at_group = 0.0;
            if (doppler_on_) {
                for (std::size_t gd = 1; gd < ENERGY_GROUPS_NUM; ++gd) {
                    if (div_V < 0.0) {
                        double const slope_left = get_doppler_slope(cells_cgs[i], gd - 1, false);
                        double coeff_left = (1.0 / energy_groups_width[gd - 1] -
                            0.5 * slope_left / energy_groups_width[gd - 1]);
                        coeff_left *= (0.5 - 0.5 * R2[i * ENERGY_GROUPS_NUM + gd - 1]) * doppler_coeff * energy_groups_boundary[gd];
                        double coeff_right = 0.5 * slope_left / energy_groups_width[gd];
                        coeff_right *= (0.5 - 0.5 * R2[i * ENERGY_GROUPS_NUM + gd]) * doppler_coeff * energy_groups_boundary[gd];
                        if (group == gd - 1) {
                            velocity_doppler_diagonal += coeff_left;
                            intergroup_contrib_at_group += coeff_left;
                        }
                        if (group == gd) {
                            velocity_doppler_diagonal -= coeff_right;
                            intergroup_contrib_at_group -= coeff_right;
                        }
                    } else if (group == gd) {
                        double const slope_right = get_doppler_slope(cells_cgs[i], gd, true);
                        double coeff_right = 1.0 / energy_groups_width[gd];
                        coeff_right += (gd + 1 < ENERGY_GROUPS_NUM)
                            ? 0.5 * slope_right / energy_groups_width[gd + 1] : 0.0;
                        coeff_right *= doppler_coeff * (0.5 - 0.5 * R2[i * ENERGY_GROUPS_NUM + gd]) * energy_groups_boundary[gd];
                        velocity_doppler_diagonal -= coeff_right;
                        intergroup_contrib_at_group -= coeff_right;
                    }
                }
            }
            double const velocity_doppler_intergroup = velocity_doppler_diagonal - velocity_doppler_face;
            double const residual_diagonal = example_bad_diagonal - source_diagonal
                - compton_remaining - diffusion_diagonal - velocity_doppler_diagonal;

            Vector3D const& v_cgs = cells_cgs[i].velocity;
            double const v_mag_cgs = abs(v_cgs);
            Vector3D const v_sim = v_cgs * (time_scale_ / length_scale_);
            double const Eg8_cgs = (ENERGY_GROUPS_NUM > 8)
                ? cells_cgs[i].Eg[8] * cells_cgs[i].density : 0.0;
            double const Eg9_cgs = (ENERGY_GROUPS_NUM > 9)
                ? cells_cgs[i].Eg[9] * cells_cgs[i].density : 0.0;
            double const Eg_fail_cgs = cells_cgs[i].Eg[group] * cells_cgs[i].density;
            double const slope_fail = (doppler_on_ && group > 0)
                ? get_doppler_slope(cells_cgs[i], group, div_V >= 0.0) : 0.0;

            std::clog << std::scientific << std::setprecision(6)
                      << "MG matrix diagonal crash rank " << rank
                      << " cell ID " << example_bad_cell_id
                      << " group " << group
                      << " loc=" << r_i
                      << "\n  state: T=" << cells_cgs[i].temperature
                      << " rho=" << cells_cgs[i].density
                      << " width=" << (tess.GetWidth(i) * length_scale_)
                      << " dt_cgs=" << dt_cell_cgs
                      << " div_V=" << div_V
                      << " (" << (div_V < 0.0 ? "compression" : "expansion") << ")"
                      << "\n  velocity cgs=(" << v_cgs.x << "," << v_cgs.y << "," << v_cgs.z
                      << ") |v|=" << v_mag_cgs
                      << " sim=(" << v_sim.x << "," << v_sim.y << "," << v_sim.z << ")"
                      << "\n  radiation cgs: Eg[" << group << "]=" << Eg_fail_cgs
                      << " Eg8=" << Eg8_cgs << " Eg9=" << Eg9_cgs
                      << " Erad=" << (cells_cgs[i].Erad * cells_cgs[i].density)
                      << " R2[" << group << "]="
                      << R2[i * ENERGY_GROUPS_NUM + group];
            if (ENERGY_GROUPS_NUM > 8)
                std::clog << " R2[8]=" << R2[i * ENERGY_GROUPS_NUM + 8];
            if (ENERGY_GROUPS_NUM > 9)
                std::clog << " R2[9]=" << R2[i * ENERGY_GROUPS_NUM + 9];
            std::clog << "\n  diagonal: A[ii]=" << example_bad_diagonal
                      << " threshold=" << example_bad_threshold
                      << " source=" << source_diagonal
                      << " diffusion=" << diffusion_diagonal
                      << " velocity_face=" << velocity_doppler_face
                      << " velocity_intergroup=" << velocity_doppler_intergroup
                      << " velocity_total=" << velocity_doppler_diagonal
                      << " residual=" << residual_diagonal
                      << "\n  doppler: coeff=-div_V*dt=" << doppler_coeff
                      << " slope@group=" << slope_fail
                      << " intergroup@group=" << intergroup_contrib_at_group
                      << " compton_deferred=" << (compton_deferred_[i] ? 1 : 0)
                      << " fleck=" << f
                      << std::endl;

            std::ostringstream reason;
            reason << "matrix diagonal unsafe, cell ID " << example_bad_cell_id
                   << " group " << group
                   << " (velocity_doppler=" << velocity_doppler_diagonal << ")";
            setStepFailure(reason.str(), example_bad_cell_id);
        } else {
            setStepFailure("matrix diagonal remains unsafe after Compton removal");
        }
        throw UniversalError("matrix diagonal remains unsafe after Compton removal");
    }

    std::array<unsigned long long, 3> matrix_cell_counts = {{
        local_matrix_active_cells,
        local_matrix_implicit_compton_cells,
        local_matrix_deferred_compton_cells}};
    if(trace_matrix_phases) {
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, matrix_cell_counts.data(),
                      static_cast<int>(matrix_cell_counts.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    }

    if(trace_matrix_phases && rank == 0) {
        matrix_validation_seconds = mgElapsedSeconds(matrix_phase_start);
        double const implicit_compton_fraction = matrix_cell_counts[0] == 0 ?
            0.0 : static_cast<double>(matrix_cell_counts[1]) /
                static_cast<double>(matrix_cell_counts[0]);
        RuntimeTraceStream() << "MG_MATRIX_PHASE_TIMING scope=rank_local"
                  << " rank=" << rank
                  << " lazy_compton_recovery_requested="
                  << (lazy_compton_recovery_requested ? 1 : 0)
                  << " lazy_compton_recovery_enabled="
                  << (lazy_compton_recovery_enabled ? 1 : 0)
                  << " active_cells_global=" << matrix_cell_counts[0]
                  << " implicit_compton_cells_global="
                  << matrix_cell_counts[1]
                  << " deferred_compton_cells_global="
                  << matrix_cell_counts[2]
                  << " implicit_compton_fraction="
                  << implicit_compton_fraction
                  << " setup_material_seconds="
                  << matrix_setup_material_seconds
                  << " flux_limiter_seconds="
                  << matrix_flux_limiter_seconds
                  << " diffusion_seconds=" << matrix_diffusion_seconds
                  << " velocity_doppler_seconds="
                  << matrix_velocity_doppler_seconds
                  << " validation_seconds=" << matrix_validation_seconds
                  << " total_seconds="
                  << mgElapsedSeconds(matrix_total_start)
                  << std::endl;
    }

    if(matrix_loop_option_requested) {
        unsigned long long counters[4] = {
            local_face_orientations_considered,
            local_reverse_face_orientations_skipped,
            local_mean_temperature_group_uses,
            local_mean_temperature_evaluations};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, counters, 4, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
#endif
        if(rank == 0) {
            if(skip_reverse_faces_requested_)
                std::clog << "MG_REVERSE_FACE_SKIP scope=aggregate"
                          << " route=" << matrix_route
                          << " requested=1"
                          << " enabled="
                          << (skip_reverse_faces_enabled ? 1 : 0)
                          << " face_orientations_considered=" << counters[0]
                          << " reverse_face_orientations_skipped="
                          << counters[1]
                          << " group_iterations_avoided="
                          << counters[1] *
                                 static_cast<unsigned long long>(
                                     ENERGY_GROUPS_NUM)
                          << std::endl;
            if(hoist_face_mean_temperature_requested_) {
                unsigned long long const evaluations_avoided =
                    counters[2] >= counters[3] ?
                    counters[2] - counters[3] : 0;
                unsigned long long const observed_groups_per_evaluation =
                    counters[3] == 0 ? 0 : counters[2] / counters[3];
                std::clog << "MG_FACE_MEAN_TEMPERATURE scope=aggregate"
                          << " route=" << matrix_route
                          << " requested=1"
                          << " enabled="
                          << (hoist_face_mean_temperature_enabled ? 1 : 0)
                          << " energy_groups=" << ENERGY_GROUPS_NUM
                          << " group_uses=" << counters[2]
                          << " evaluations=" << counters[3]
                          << " evaluations_avoided=" << evaluations_avoided
                          << " observed_groups_per_evaluation="
                          << observed_groups_per_evaluation
                          << std::endl;
            }
        }
    }

    if(freefree_pair16_option_requested) {
        unsigned long long counters[9] = {
            local_freefree_pair16_batch_attempts,
            local_freefree_pair16_batch_calls,
            local_freefree_pair16_applied_faces,
            local_freefree_pair16_shadow_faces,
            local_freefree_pair16_mismatch_faces,
            local_freefree_pair16_mismatch_coefficients,
            local_freefree_pair16_unsupported_faces,
            local_freefree_pair16_fallback_faces,
            local_freefree_pair16_scalar_calls_avoided};
        double timing_sum[2] = {
            local_freefree_pair16_batch_seconds,
            local_freefree_pair16_shadow_seconds};
        double timing_max[2] = {
            local_freefree_pair16_batch_seconds,
            local_freefree_pair16_shadow_seconds};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, counters, 9, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, timing_sum, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, timing_max, 2, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
#endif
        if(counters[4] != 0 || counters[6] != 0)
            freefree_pair16_failed_closed_ = true;
        if(rank == 0) {
            std::clog << "MG_FREEFREE_PAIR16_SUPPORT scope=aggregate"
                      << " route=" << matrix_route
                      << " pair_requested="
                      << (freefree_pair16_requested_ ? 1 : 0)
                      << " shadow_requested="
                      << (freefree_pair16_shadow_requested_ ? 1 : 0)
                      << " exact_type=" << freefree_pair16_support[0]
                      << " groups16=" << freefree_pair16_support[1]
                      << " production_options="
                      << freefree_pair16_support[2]
                      << " start_not_failed="
                      << freefree_pair16_support[3]
                      << " collective_supported="
                      << (freefree_pair16_collective_supported ? 1 : 0)
                      << " batch_compute_enabled="
                      << (freefree_pair16_batch_compute_enabled ? 1 : 0)
                      << " pair_apply_enabled="
                      << (freefree_pair16_apply_enabled ? 1 : 0)
                      << " shadow_enabled="
                      << (freefree_pair16_shadow_enabled ? 1 : 0)
                      << " next_build_enabled="
                      << (freefree_pair16_collective_supported &&
                          !freefree_pair16_failed_closed_ ? 1 : 0)
                      << std::endl;
            std::clog << "MG_FREEFREE_PAIR16_HIT scope=aggregate"
                      << " batch_attempts=" << counters[0]
                      << " batch_pair_calls=" << counters[1]
                      << " batch_applied_faces=" << counters[2]
                      << " shadow_faces=" << counters[3]
                      << " fallback_faces=" << counters[7]
                      << std::endl;
            std::clog << "MG_FREEFREE_PAIR16_MISMATCH scope=aggregate"
                      << " faces=" << counters[4]
                      << " coefficients=" << counters[5]
                      << " unsupported_temperature_faces=" << counters[6]
                      << " failed_closed="
                      << (freefree_pair16_failed_closed_ ? 1 : 0)
                      << std::endl;
            std::clog << "MG_FREEFREE_PAIR16_TIMING scope=aggregate"
                      << " batch_seconds_sum=" << timing_sum[0]
                      << " batch_seconds_max=" << timing_max[0]
                      << " shadow_seconds_sum=" << timing_sum[1]
                      << " shadow_seconds_max=" << timing_max[1]
                      << std::endl;
            std::clog << "MG_FREEFREE_PAIR16_CALL_REDUCTION scope=aggregate"
                      << " legacy_scalar_calls_equivalent="
                      << counters[1] * 32
                      << " batch_pair_calls=" << counters[1]
                      << " shadow_scalar_calls=" << counters[3] * 32
                      << " scalar_calls_avoided=" << counters[8]
                      << " legacy_spectral_evaluations="
                      << counters[1] * 32
                      << " batch_spectral_evaluations="
                      << counters[1] * 16
                      << " shadow_spectral_evaluations="
                      << counters[3] * 32
                      << " spectral_evaluations_reused_within_batch="
                      << counters[1] * 16
                      << " legacy_temperature_state_evaluations="
                      << counters[1] * 32
                      << " batch_temperature_state_evaluations="
                      << counters[1]
                      << " shadow_temperature_state_evaluations="
                      << counters[3] * 32
                      << " temperature_states_reused_within_batch="
                      << counters[1] * 31
                      << std::endl;
        }
    }
}
void MultigroupDiffusion::PostCG(Tessellation3D const& tess,
                                 std::vector<Conserved3D>& extensives,
                                 double const dt,
                                 std::vector<ComputationalCell3D>& cells,
                                 std::vector<double> const& CG_result,
                                 std::vector<double> const& full_CG_result) const {

    auto const N = tess.GetPointNo();
    if(radiation_force_time_step_limits_.size() != N)
        radiation_force_time_step_limits_.assign(
            N, std::numeric_limits<double>::max());
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    ensureComptonBulkRuntimeOptions();

    int local_rejected_count = 0;
    int local_unrecoverable_group = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if(!individualCellActive(i))
            continue;
        bool cell_bad = false;
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            std::size_t const k = i * ENERGY_GROUPS_NUM + g;
            if (k >= CG_result.size() || k >= full_CG_result.size()
                || !std::isfinite(CG_result[k]) || !std::isfinite(full_CG_result[k])) {
                local_unrecoverable_group = 1;
                cell_bad = true;
                break;
            }
        }
        if (cell_bad) {
            ++local_rejected_count;
            if (getLastStepFailureReason().empty())
                setCellLocalStepFailure(
                    "non-finite CG group energy", cells[i].ID);
        }
    }
    int global_unrecoverable_group = local_unrecoverable_group;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_unrecoverable_group, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_unrecoverable_group != 0) {
        postcg_unrecoverable_ = true;
        int global_rejected_count = local_rejected_count;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &global_rejected_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        if (rank == 0 && global_rejected_count > 0)
            std::clog << "PostCG rejected " << global_rejected_count << " cells" << std::endl;
        return;
    }

    std::vector<size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    Vector3D dP;

    double Einit = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        Einit += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Einit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int good_end = 1;

    bool with_entropy = false;
    size_t entropy_index = ComputationalCell3D::tracerNames.size();
    std::vector<std::string>::const_iterator it = binary_find(
        ComputationalCell3D::tracerNames.begin(),
        ComputationalCell3D::tracerNames.end(),
        string("Entropy")
    );

    if (it != ComputationalCell3D::tracerNames.end())
    {
        entropy_index = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
        with_entropy = true;
    }

    double min_T_E_added = 0;
    double d_Ek = 0;
    ImplicitComptonCellCoefficients bulk_compton_coefficients;
    ImplicitComptonCellCoefficients legacy_compton_coefficients;
    int local_postcg_compton_shadow_mismatch = 0;
    std::size_t first_postcg_compton_shadow_mismatch_cell_id =
        std::numeric_limits<std::size_t>::max();
    bool const postcg_face_geometry_option_requested =
        postcg_face_geometry_requested_ ||
        postcg_face_geometry_shadow_requested_;
    unsigned long long local_postcg_face_geometry_cells = 0;
    unsigned long long local_postcg_face_geometry_candidate_cells = 0;
    unsigned long long local_postcg_face_geometry_selected_cells = 0;
    unsigned long long local_postcg_face_geometry_shadow_cells = 0;
    unsigned long long local_postcg_face_geometry_fallback_cells = 0;
    unsigned long long local_postcg_face_geometry_unsupported_cells = 0;
    unsigned long long local_postcg_face_geometry_faces = 0;
    unsigned long long local_postcg_face_geometry_group_face_uses = 0;
    unsigned long long local_postcg_face_geometry_candidate_avoided = 0;
    unsigned long long local_postcg_face_geometry_selected_avoided = 0;
    unsigned long long local_postcg_face_geometry_mismatch_cells = 0;
    unsigned long long local_postcg_face_geometry_mismatch_values = 0;
    double local_postcg_face_geometry_candidate_seconds = 0.0;
    double local_postcg_face_geometry_legacy_shadow_seconds = 0.0;
    bool local_postcg_face_geometry_supported = true;
    std::vector<Vector3D> postcg_face_directions;
    std::vector<double> postcg_face_areas;
    std::vector<unsigned char> postcg_face_is_outside;
    for (std::size_t i=0; i < N; ++i) {
        if(!individualCellActive(i))
            continue;
        double const dt_cell = individualCellTimeStep(i, dt);
        double const dt_cgs = dt_cell * time_scale_;
        double const cdt = CG::speed_of_light * dt_cgs;
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        double Erad_tot = 0;
        double const f = fleck_factor[i];
        double const raw_T = old_Tm[i];
        double const cell_T = (std::isfinite(cells[i].temperature) && cells[i].temperature > 0.0)
            ? cells[i].temperature : 1e-200;
        double const T = (std::isfinite(raw_T) && raw_T > 0.0) ? raw_T : cell_T;
        double const kp_raw = sigma_absorption_planck[i];
        double const kp = (std::isfinite(kp_raw) && kp_raw > 0.0) ? kp_raw : 0.0;
        double const Um = (std::isfinite(raw_T) && raw_T > 0.0)
            ? get_radiation_energy_density(T) : 0.0;

        double dE_absorption_emission = -volume * f * cdt * kp*Um;

        double dE_compton = 0.0;
        if (compton_deferred_[i])
            split_compton_cells_[i] = true;
        bool const do_compton = !compton_deferred_[i] && compton_on_ && (sigma_absorption_planck[i] * dt_cgs * CG::speed_of_light < compton_optical_depth_turn_off);
        if (do_compton) {
            try {
                ComptonOccupationMode const occupation_mode =
                    compton_occupation_mode_[i] != ComptonOccupationMode::Off
                        ? compton_occupation_mode_[i]
                        : (use_n_zero[i] ? ComptonOccupationMode::Zero : ComptonOccupationMode::RadiationField);
                generate_S_and_dSdUm_matrices(cells[i], i, dt_cgs, occupation_mode);

                if(compton_bulk_coefficients_enabled_ ||
                   compton_bulk_shadow_enabled_) {
                    fillImplicitComptonCellCoefficients(
                        tess, cells[i], i, dt_cgs,
                        bulk_compton_coefficients);
                }
                bool shadow_matches = true;
                if(compton_bulk_shadow_enabled_) {
                    shadow_matches =
                        fillLegacyImplicitComptonCellCoefficientsAndCompare(
                            tess, cells[i], i, dt_cgs,
                            bulk_compton_coefficients,
                            legacy_compton_coefficients, "PostCG");
                    if(!shadow_matches) {
                        local_postcg_compton_shadow_mismatch = 1;
                        if(first_postcg_compton_shadow_mismatch_cell_id ==
                           std::numeric_limits<std::size_t>::max())
                            first_postcg_compton_shadow_mismatch_cell_id =
                                cells[i].ID;
                    }
                }
                ImplicitComptonCellCoefficients const*
                    selected_compton_coefficients = nullptr;
                if(compton_bulk_coefficients_enabled_ && shadow_matches)
                    selected_compton_coefficients =
                        &bulk_compton_coefficients;
                else if(compton_bulk_shadow_enabled_)
                    selected_compton_coefficients =
                        &legacy_compton_coefficients;

                for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {

                    dE_compton -=
                        selected_compton_coefficients != nullptr ?
                            selected_compton_coefficients->delta_b[g] :
                            get_implicit_compton_contribution_to_b(
                                tess, cells[i], i, g, dt_cgs);

                    for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                        double const CG_res_i = std::max(
                            CG_result[i * ENERGY_GROUPS_NUM + gt],
                            std::numeric_limits<double>::min()*1e100);

                        double const implicit_compton_contribution =
                            selected_compton_coefficients != nullptr ?
                                selected_compton_coefficients->delta_A[
                                    g * ENERGY_GROUPS_NUM + gt] :
                                get_implicit_compton_contribution(
                                    tess, cells[i], i, g, gt, dt_cgs);
                        dE_compton +=
                            implicit_compton_contribution * CG_res_i;
                    }
                }
            }
            catch (UniversalError const&) {
                postcg_unrecoverable_ = true;
                good_end = 0;
                ++local_rejected_count;
                if (getLastStepFailureReason().empty())
                    setCellLocalStepFailure(
                        "Compton coupling error in PostCG", cells[i].ID);
                break;
            }
            if (!std::isfinite(dE_compton)) {
                postcg_unrecoverable_ = true;
                good_end = 0;
                ++local_rejected_count;
                if (getLastStepFailureReason().empty())
                    setCellLocalStepFailure(
                        "non-finite Compton energy exchange", cells[i].ID);
                break;
            }
        }
        double const gamma_safe = (std::isfinite(Gammas[i]) && Gammas[i] > 0.0)
            ? Gammas[i] : std::numeric_limits<double>::min();
        double const Gamma_1 = 1.0 / gamma_safe;

        for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {

            double const full_CG_res_i = full_CG_result[i * ENERGY_GROUPS_NUM + group];

            extensives[i].Eg[group] = full_CG_res_i * volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

            cells[i].Eg[group] =  extensives[i].Eg[group] / extensives[i].mass;
            Erad_tot += extensives[i].Eg[group];
            // Preserve the historical operator splitting: the unchecked
            // residual correction is committed to the spectrum, while
            // material coupling and force use the accepted Krylov iterate.
            dE_absorption_emission += volume * cdt *
                CG_result[i * ENERGY_GROUPS_NUM + group] *
                sigma_absorption_group[i][group];
            auto const bg = planck_integal_group[i][group];
            for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {

                double const implicit_conribution_group_j = -volume*bg * (1 - f) * sigma_absorption_group[i][gt] * sigma_absorption_group[i][group] * cdt * Gamma_1;

                dE_absorption_emission += implicit_conribution_group_j *
                    CG_result[i * ENERGY_GROUPS_NUM + gt];
            }
        }

        dE_absorption_emission *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
        dE_compton *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

        if(cooling_time_limiter_on_ && std::isfinite(old_e_therm) &&
           old_e_therm > 0.0 && std::isfinite(dE_absorption_emission)) {
            double const minimum_absorption_exchange = -0.5 * old_e_therm;
            if(dE_absorption_emission < minimum_absorption_exchange) {
                double const requested_exchange = dE_absorption_emission;
                double const radiation_adjustment =
                    requested_exchange - minimum_absorption_exchange;
                double const radiation_energy_to_remove =
                    -radiation_adjustment;
                long double positive_radiation = 0.0;
                for(double const group_energy : extensives[i].Eg)
                    positive_radiation +=
                        std::max(group_energy, 0.0);
                double const positive_radiation_double =
                    static_cast<double>(positive_radiation);
                double const availability_guard =
                    1.0 + 64.0 * std::numeric_limits<double>::epsilon();
                if(!std::isfinite(positive_radiation_double) ||
                   positive_radiation_double <= 0.0 ||
                   radiation_energy_to_remove >
                       positive_radiation_double * availability_guard) {
                    postcg_unrecoverable_ = true;
                    good_end = 0;
                    ++local_rejected_count;
                    setCellLocalStepFailure(
                        "absorption cooling limiter cannot return excess radiation energy",
                        cells[i].ID);
                    break;
                }
                double const removal_fraction = std::clamp(
                    radiation_energy_to_remove /
                        positive_radiation_double,
                    0.0, 1.0);
                Erad_tot = 0.0;
                for(std::size_t group = 0;
                    group < ENERGY_GROUPS_NUM; ++group) {
                    if(extensives[i].Eg[group] > 0.0)
                        extensives[i].Eg[group] *=
                            1.0 - removal_fraction;
                    cells[i].Eg[group] =
                        extensives[i].Eg[group] / extensives[i].mass;
                    Erad_tot += extensives[i].Eg[group];
                }
                dE_absorption_emission = minimum_absorption_exchange;
                std::clog << std::setprecision(17)
                          << "MG_ABSORPTION_COOLING_LIMIT"
                          << " rank=" << rank
                          << " cell_id=" << cells[i].ID
                          << " dt=" << dt_cell
                          << " old_material_energy=" << old_e_therm
                          << " requested_exchange=" << requested_exchange
                          << " applied_exchange="
                          << dE_absorption_emission
                          << " radiation_adjustment="
                          << radiation_adjustment
                          << " removal_fraction=" << removal_fraction
                          << std::endl;
            }
        }

        extensives[i].energy += dE_absorption_emission + dE_compton;
        extensives[i].internal_energy += dE_absorption_emission + dE_compton;
        extensives[i].Erad = Erad_tot;
        cells[i].Erad =  extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy =  extensives[i].internal_energy / extensives[i].mass;

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        faces = tess.GetCellFaces(i);
        auto const r_i = tess.GetMeshPoint(i);

        // momentum term
        if (hydro_on_) {
            size_t print_id = -1;
            Vector3D dP;
            if(postcg_face_geometry_option_requested)
                ++local_postcg_face_geometry_cells;
            if(!postcg_face_geometry_option_requested ||
               postcg_face_geometry_failed_closed_) {
                if(postcg_face_geometry_option_requested)
                    ++local_postcg_face_geometry_fallback_cells;
                for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                    Vector3D gradEg, gradEg_new;
                    double Eg_j, Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
                    double Eg_j_new, Eg_i_new = std::max(
                        CG_result[i * ENERGY_GROUPS_NUM + group],
                        std::numeric_limits<double>::min()*1e100);
                    for (size_t j=0; j<Nneighbors; ++j) {
                        size_t const neighbor_j = neighbors[j];
                        Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                        double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                        bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
                        if (!is_outside) {
                            gradEg += A_ij * 0.5 * (Eg_i + cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density) * r_ij;
                            gradEg_new += A_ij * 0.5 *
                                (Eg_i_new + std::max(
                                    CG_result[neighbor_j * ENERGY_GROUPS_NUM + group],
                                    std::numeric_limits<double>::min()*1e100)) * r_ij;
                        } else {
                            Vector3D dummy_v;
                            boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                            gradEg += A_ij * 0.5 * (Eg_i + Eg_j) * r_ij;
                            boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i_new, Eg_j_new, dummy_v);
                            gradEg_new += A_ij * 0.5 * (Eg_i_new + Eg_j_new) * r_ij;
                        }
                    }
                    gradEg *= 1.0 / volume;
                    double const D = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, group);
                    double const flux_limit = CG::CalcSingleFluxLimiter(gradEg, D, Eg_i);
                    if(cells[i].ID == print_id)
                        std::clog<<"Group "<<group<<" flux_limit "<<flux_limit<<" sigma rossland "<<units::clight / (3 * D)<<" Eg_i "<<Eg_i<<" gradEg "<<gradEg<<std::endl;
                    dP += (flux_limit / 3) * gradEg_new;
                }
                dP *= dt_cgs * time_scale_ / (length_scale_ * mass_scale_);
            } else if(faces.size() != Nneighbors) {
                ++local_postcg_face_geometry_fallback_cells;
                ++local_postcg_face_geometry_unsupported_cells;
                local_postcg_face_geometry_supported = false;
                postcg_face_geometry_failed_closed_ = true;
                for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                    Vector3D gradEg, gradEg_new;
                    double Eg_j, Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
                    double Eg_j_new, Eg_i_new = std::max(
                        CG_result[i * ENERGY_GROUPS_NUM + group],
                        std::numeric_limits<double>::min()*1e100);
                    for (size_t j=0; j<Nneighbors; ++j) {
                        size_t const neighbor_j = neighbors[j];
                        Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                        double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                        bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
                        if (!is_outside) {
                            gradEg += A_ij * 0.5 * (Eg_i + cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density) * r_ij;
                            gradEg_new += A_ij * 0.5 *
                                (Eg_i_new + std::max(
                                    CG_result[neighbor_j * ENERGY_GROUPS_NUM + group],
                                    std::numeric_limits<double>::min()*1e100)) * r_ij;
                        } else {
                            Vector3D dummy_v;
                            boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                            gradEg += A_ij * 0.5 * (Eg_i + Eg_j) * r_ij;
                            boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i_new, Eg_j_new, dummy_v);
                            gradEg_new += A_ij * 0.5 * (Eg_i_new + Eg_j_new) * r_ij;
                        }
                    }
                    gradEg *= 1.0 / volume;
                    double const D = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, group);
                    double const flux_limit = CG::CalcSingleFluxLimiter(gradEg, D, Eg_i);
                    if(cells[i].ID == print_id)
                        std::clog<<"Group "<<group<<" flux_limit "<<flux_limit<<" sigma rossland "<<units::clight / (3 * D)<<" Eg_i "<<Eg_i<<" gradEg "<<gradEg<<std::endl;
                    dP += (flux_limit / 3) * gradEg_new;
                }
                dP *= dt_cgs * time_scale_ / (length_scale_ * mass_scale_);
            } else {
                ++local_postcg_face_geometry_candidate_cells;
                unsigned long long const face_count =
                    static_cast<unsigned long long>(Nneighbors);
                unsigned long long const group_count =
                    static_cast<unsigned long long>(ENERGY_GROUPS_NUM);
                local_postcg_face_geometry_faces += face_count;
                local_postcg_face_geometry_group_face_uses +=
                    face_count * group_count;
                local_postcg_face_geometry_candidate_avoided +=
                    group_count > 0 ? face_count * (group_count - 1) : 0;

                auto const candidate_start = MultigroupClock::now();
                postcg_face_directions.resize(Nneighbors);
                postcg_face_areas.resize(Nneighbors);
                postcg_face_is_outside.resize(Nneighbors);
                for(size_t j = 0; j < Nneighbors; ++j) {
                    size_t const neighbor_j = neighbors[j];
                    postcg_face_directions[j] =
                        normalize(r_i - tess.GetMeshPoint(neighbor_j));
                    postcg_face_areas[j] =
                        tess.GetArea(faces[j]) * pow<2>(length_scale_);
                    postcg_face_is_outside[j] =
                        tess.IsPointOutsideBox(neighbor_j) ? 1 : 0;
                }

                Vector3D candidate_dP;
                for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                    Vector3D gradEg, gradEg_new;
                    double Eg_j, Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
                    double Eg_j_new, Eg_i_new = std::max(
                        CG_result[i * ENERGY_GROUPS_NUM + group],
                        std::numeric_limits<double>::min()*1e100);
                    for (size_t j=0; j<Nneighbors; ++j) {
                        size_t const neighbor_j = neighbors[j];
                        Vector3D const& r_ij = postcg_face_directions[j];
                        double const A_ij = postcg_face_areas[j];
                        bool const is_outside =
                            postcg_face_is_outside[j] != 0;
                        if (!is_outside) {
                            gradEg += A_ij * 0.5 * (Eg_i + cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density) * r_ij;
                            gradEg_new += A_ij * 0.5 *
                                (Eg_i_new + std::max(
                                    CG_result[neighbor_j * ENERGY_GROUPS_NUM + group],
                                    std::numeric_limits<double>::min()*1e100)) * r_ij;
                        } else {
                            Vector3D dummy_v;
                            boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                            gradEg += A_ij * 0.5 * (Eg_i + Eg_j) * r_ij;
                            boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i_new, Eg_j_new, dummy_v);
                            gradEg_new += A_ij * 0.5 * (Eg_i_new + Eg_j_new) * r_ij;
                        }
                    }
                    gradEg *= 1.0 / volume;
                    double const D = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, group);
                    double const flux_limit = CG::CalcSingleFluxLimiter(gradEg, D, Eg_i);
                    if(cells[i].ID == print_id)
                        std::clog<<"Group "<<group<<" flux_limit "<<flux_limit<<" sigma rossland "<<units::clight / (3 * D)<<" Eg_i "<<Eg_i<<" gradEg "<<gradEg<<std::endl;
                    candidate_dP += (flux_limit / 3) * gradEg_new;
                }
                candidate_dP *=
                    dt_cgs * time_scale_ / (length_scale_ * mass_scale_);
                local_postcg_face_geometry_candidate_seconds +=
                    mgElapsedSeconds(candidate_start);

                if(postcg_face_geometry_shadow_requested_) {
                    ++local_postcg_face_geometry_shadow_cells;
                    auto const legacy_start = MultigroupClock::now();
                    Vector3D legacy_dP;
                    for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                        Vector3D gradEg, gradEg_new;
                        double Eg_j, Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
                        double Eg_j_new, Eg_i_new = std::max(
                            CG_result[i * ENERGY_GROUPS_NUM + group],
                            std::numeric_limits<double>::min()*1e100);
                        for (size_t j=0; j<Nneighbors; ++j) {
                            size_t const neighbor_j = neighbors[j];
                            Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                            bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
                            if (!is_outside) {
                                gradEg += A_ij * 0.5 * (Eg_i + cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density) * r_ij;
                                gradEg_new += A_ij * 0.5 *
                                    (Eg_i_new + std::max(
                                        CG_result[neighbor_j * ENERGY_GROUPS_NUM + group],
                                        std::numeric_limits<double>::min()*1e100)) * r_ij;
                            } else {
                                Vector3D dummy_v;
                                boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                                gradEg += A_ij * 0.5 * (Eg_i + Eg_j) * r_ij;
                                boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i_new, Eg_j_new, dummy_v);
                                gradEg_new += A_ij * 0.5 * (Eg_i_new + Eg_j_new) * r_ij;
                            }
                        }
                        gradEg *= 1.0 / volume;
                        double const D = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, group);
                        double const flux_limit = CG::CalcSingleFluxLimiter(gradEg, D, Eg_i);
                        if(cells[i].ID == print_id)
                            std::clog<<"Group "<<group<<" flux_limit "<<flux_limit<<" sigma rossland "<<units::clight / (3 * D)<<" Eg_i "<<Eg_i<<" gradEg "<<gradEg<<std::endl;
                        legacy_dP += (flux_limit / 3) * gradEg_new;
                    }
                    legacy_dP *=
                        dt_cgs * time_scale_ / (length_scale_ * mass_scale_);
                    local_postcg_face_geometry_legacy_shadow_seconds +=
                        mgElapsedSeconds(legacy_start);

                    struct ForceLimiterState {
                        double delta_velocity;
                        double acceleration;
                        double safe_force_dt;
                        double limited_force_dt;
                        bool force_step_safe;
                    };
                    double const shadow_mass_i = extensives[i].mass;
                    double const shadow_width_i = tess.GetWidth(i);
                    auto const force_limiter_state =
                        [&](Vector3D const& momentum_change) {
                            double const delta_velocity =
                                std::isfinite(shadow_mass_i) &&
                                shadow_mass_i > 0.0 ?
                                fastabs(momentum_change) / shadow_mass_i :
                                std::numeric_limits<double>::infinity();
                            double const acceleration =
                                std::isfinite(dt_cell) && dt_cell > 0.0 ?
                                delta_velocity / dt_cell :
                                std::numeric_limits<double>::infinity();
                            double safe_force_dt =
                                std::numeric_limits<double>::max();
                            if(acceleration > 0.0 &&
                               std::isfinite(acceleration) &&
                               shadow_width_i > 0.0 &&
                               std::isfinite(shadow_width_i))
                                safe_force_dt =
                                    fastsqrt(shadow_width_i / acceleration);
                            double const limited_force_dt = std::min(
                                radiation_force_time_step_limits_[i],
                                safe_force_dt);
                            double const roundoff_guard = 1.0 + 64.0 *
                                std::numeric_limits<double>::epsilon();
                            bool const force_step_safe =
                                std::isfinite(delta_velocity) &&
                                std::isfinite(safe_force_dt) &&
                                safe_force_dt > 0.0 &&
                                dt_cell <= safe_force_dt * roundoff_guard;
                            return ForceLimiterState{delta_velocity,
                                acceleration, safe_force_dt,
                                limited_force_dt, force_step_safe};
                        };
                    ForceLimiterState const candidate_state =
                        force_limiter_state(candidate_dP);
                    ForceLimiterState const legacy_state =
                        force_limiter_state(legacy_dP);
                    unsigned long long mismatch_values = 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_dP.x) !=
                         mgDoubleBits(legacy_dP.x)) ? 1 : 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_dP.y) !=
                         mgDoubleBits(legacy_dP.y)) ? 1 : 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_dP.z) !=
                         mgDoubleBits(legacy_dP.z)) ? 1 : 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_state.delta_velocity) !=
                         mgDoubleBits(legacy_state.delta_velocity)) ? 1 : 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_state.acceleration) !=
                         mgDoubleBits(legacy_state.acceleration)) ? 1 : 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_state.safe_force_dt) !=
                         mgDoubleBits(legacy_state.safe_force_dt)) ? 1 : 0;
                    mismatch_values +=
                        (mgDoubleBits(candidate_state.limited_force_dt) !=
                         mgDoubleBits(legacy_state.limited_force_dt)) ? 1 : 0;
                    mismatch_values +=
                        (candidate_state.force_step_safe !=
                         legacy_state.force_step_safe) ? 1 : 0;
                    if(mismatch_values != 0) {
                        ++local_postcg_face_geometry_mismatch_cells;
                        local_postcg_face_geometry_mismatch_values +=
                            mismatch_values;
                        ++local_postcg_face_geometry_fallback_cells;
                        postcg_face_geometry_failed_closed_ = true;
                        dP = legacy_dP;
                    } else if(postcg_face_geometry_requested_) {
                        ++local_postcg_face_geometry_selected_cells;
                        dP = candidate_dP;
                    } else {
                        dP = legacy_dP;
                    }
                } else {
                    ++local_postcg_face_geometry_selected_cells;
                    local_postcg_face_geometry_selected_avoided +=
                        group_count > 0 ? face_count * (group_count - 1) : 0;
                    dP = candidate_dP;
                }
            }
            double const mass_i = extensives[i].mass;
            double const width_i = tess.GetWidth(i);
            double const delta_velocity =
                std::isfinite(mass_i) && mass_i > 0.0 ?
                fastabs(dP) / mass_i :
                std::numeric_limits<double>::infinity();
            double const acceleration =
                std::isfinite(dt_cell) && dt_cell > 0.0 ?
                delta_velocity / dt_cell :
                std::numeric_limits<double>::infinity();
            double safe_force_dt = std::numeric_limits<double>::max();
            if(acceleration > 0.0 && std::isfinite(acceleration) &&
               width_i > 0.0 && std::isfinite(width_i))
                safe_force_dt = fastsqrt(width_i / acceleration);
            radiation_force_time_step_limits_[i] = std::min(
                radiation_force_time_step_limits_[i], safe_force_dt);

            double const roundoff_guard =
                1.0 + 64.0 * std::numeric_limits<double>::epsilon();
            bool const force_step_safe =
                std::isfinite(delta_velocity) &&
                std::isfinite(safe_force_dt) &&
                safe_force_dt > 0.0 &&
                dt_cell <= safe_force_dt * roundoff_guard;
            if(!force_step_safe) {
                postcg_unrecoverable_ = true;
                good_end = 0;
                ++local_rejected_count;
                std::ostringstream reason;
                reason << "radiation-force timestep unsafe, cell ID "
                       << cells[i].ID << " candidate_dt " << dt_cell
                       << " safe_dt " << safe_force_dt;
                setCellLocalStepFailure(reason.str(), cells[i].ID);
                std::clog
                    << "MG_RADIATION_FORCE_TIMESTEP mode=candidate"
                    << " rank=" << rank
                    << " cell_id=" << cells[i].ID
                    << " candidate_dt=" << dt_cell
                    << " safe_dt=" << safe_force_dt
                    << " width=" << width_i
                    << " acceleration=" << acceleration
                    << " delta_velocity=" << delta_velocity
                    << " outcome=rejected"
                    << std::endl;
                break;
            }
            double old_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;
            if(cells[i].ID == print_id)
                std::clog<<"dP "<<dP<<" cell momentum "<<extensives[i].momentum<<std::endl;
            extensives[i].momentum += dP;

            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;

            d_Ek += new_Ek - old_Ek;

            extensives[i].energy = extensives[i].internal_energy + new_Ek;
        }
        // EOS
        try {
            if (!std::isfinite(cells[i].internal_energy) || cells[i].internal_energy < 0.0) {
                bool const compton_caused = !compton_deferred_[i]
                    && std::isfinite(dE_compton)
                    && old_e_therm + dE_absorption_emission >= 0.0;
                if (compton_caused) {
                    // Undo the coupled Compton exchange after transport.  The
                    // transport spectrum is retained; only the equal and
                    // opposite energy exchange is returned to its groups.
                    double const requested_return = dE_compton;
                    double actual_return = requested_return;
                    if (requested_return < 0.0) {
                        double const available = std::max(Erad_tot, 0.0);
                        actual_return = -std::min(-requested_return, available);
                    }
                    std::vector<double> weights(ENERGY_GROUPS_NUM, 0.0);
                    double weight_sum = 0.0;
                    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                        weights[g] = std::max(extensives[i].Eg[g], 0.0);
                        weight_sum += weights[g];
                    }
                    if (weight_sum <= 0.0) {
                        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                            weights[g] = std::max(planck_integal_group[i][g], 0.0);
                            weight_sum += weights[g];
                        }
                    }
                    if (weight_sum <= 0.0)
                        weight_sum = static_cast<double>(ENERGY_GROUPS_NUM);
                    if (requested_return >= 0.0) {
                        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                            extensives[i].Eg[g] += actual_return * weights[g] / weight_sum;
                    } else {
                        double const remove = -actual_return;
                        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                            double const fraction = (Erad_tot > 0.0)
                                ? std::max(extensives[i].Eg[g], 0.0) / Erad_tot : 0.0;
                            extensives[i].Eg[g] -= remove * fraction;
                        }
                    }
                    extensives[i].internal_energy -= actual_return;
                    extensives[i].energy -= actual_return;
                    Erad_tot = 0.0;
                    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                        cells[i].Eg[g] = extensives[i].Eg[g] / extensives[i].mass;
                        Erad_tot += extensives[i].Eg[g];
                    }
                    extensives[i].Erad = Erad_tot;
                    cells[i].Erad = Erad_tot / extensives[i].mass;
                    split_compton_cells_[i] = true;
                    if (extensives[i].internal_energy < 0.0) {
                        double const injected = -extensives[i].internal_energy;
                        extensives[i].internal_energy = 0.0;
                        extensives[i].energy += injected;
                        split_injected_energy_ += injected;
                    }
                } else if (compton_deferred_[i]) {
                    if (extensives[i].internal_energy < 0.0) {
                        double const injected = -extensives[i].internal_energy;
                        extensives[i].internal_energy = 0.0;
                        extensives[i].energy += injected;
                        split_injected_energy_ += injected;
                        split_compton_cells_[i] = true;
                    }
                } else {
                    postcg_unrecoverable_ = true;
                }
                if (postcg_unrecoverable_) {
                    good_end = 0;
                    ++local_rejected_count;
                    if (getLastStepFailureReason().empty()) {
                        log_postcg_crash_precursor(rank,
                            "negative thermal energy after absorption/emission",
                            cells[i], tess.GetMeshPoint(i), old_e_therm,
                            dE_absorption_emission, dE_compton, cells[i].internal_energy);
                        setCellLocalStepFailure(
                            "negative thermal energy after absorption/emission",
                            cells[i].ID);
                    }
                    break;
                }
            }
            cells[i].internal_energy = extensives[i].internal_energy / extensives[i].mass;
            cells[i].Erad = extensives[i].Erad / extensives[i].mass;
            if (minimum_temperature_ > 0) {
                double const min_e_therm = eos_.dT2e(cells[i].density, minimum_temperature_, cells[i].tracers, ComputationalCell3D::tracerNames);
                if (cells[i].internal_energy < min_e_therm) {
                    double const Trad = std::pow(cells[i].Erad * cells[i].density * mass_scale_ / (units::arad * length_scale_ * pow<2>(time_scale_)), 0.25);
                    if (cells[i].temperature > Trad &&
                        ((extensives[i].Erad * cells[i].temperature > 1e2 * old_e_therm * Trad) ||
                            (fleck_factor[i] < 0.75 &&
                                extensives[i].Erad > min_e_therm *extensives[i].mass)) &&
                        Trad > minimum_temperature_) {
                        double const min_e_therm2 = eos_.dT2e(cells[i].density, Trad, cells[i].tracers, ComputationalCell3D::tracerNames);
                        double const delta_e = min_e_therm2 - cells[i].internal_energy;
                        cells[i].internal_energy += delta_e;
                        double const dE_change = delta_e * extensives[i].mass;
                        double const ratio = (extensives[i].Erad - dE_change) / extensives[i].Erad;
                        if (ratio > 0) {
                            min_T_E_added += dE_change;
                            extensives[i].energy += dE_change;
                            extensives[i].internal_energy += dE_change;
                            extensives[i].Erad *= ratio;
                            cells[i].Erad *= ratio;
                            for (size_t k = 0; k < ENERGY_GROUPS_NUM; ++k) {
                                extensives[i].Eg[k] *= ratio;
                                cells[i].Eg[k] *= ratio;
                            }
                        }
                    }
                }
                if (min_e_therm > cells[i].internal_energy) {
                    if (cells[i].temperature < 2e4) {
                        double const delta_e = min_e_therm - cells[i].internal_energy;
                        cells[i].internal_energy += delta_e;
                        min_T_E_added += delta_e * extensives[i].mass;
                        extensives[i].energy += delta_e * extensives[i].mass;
                        extensives[i].internal_energy += delta_e * extensives[i].mass;
                    }
                    if (cells[i].internal_energy < 0) {
                        if (getLastStepFailureReason().empty()) {
                            log_postcg_crash_precursor(rank,
                                "negative thermal energy after radiation coupling",
                                cells[i], tess.GetMeshPoint(i), old_e_therm,
                                dE_absorption_emission, dE_compton, cells[i].internal_energy);
                            setCellLocalStepFailure(
                                "negative thermal energy after radiation coupling",
                                cells[i].ID);
                        }
                        postcg_unrecoverable_ = true;
                        good_end = 0;
                        ++local_rejected_count;
                        break;
                    }
                }
            }
            cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].pressure = eos_.de2p(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].velocity = extensives[i].momentum / extensives[i].mass;
            if (with_entropy) {
                double new_entropy = eos_.dp2s(cells[i].density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
                cells[i].tracers[entropy_index] = new_entropy;
                extensives[i].tracers[entropy_index] = new_entropy * extensives[i].mass;
            }
        }
        catch (UniversalError const&) {
            postcg_unrecoverable_ = true;
            good_end = 0;
            ++local_rejected_count;
            if (getLastStepFailureReason().empty())
                setCellLocalStepFailure(
                    "EOS error in PostCG", cells[i].ID);
            break;
        }
    }

    if(postcg_face_geometry_option_requested) {
        unsigned long long counters[12] = {
            local_postcg_face_geometry_cells,
            local_postcg_face_geometry_candidate_cells,
            local_postcg_face_geometry_selected_cells,
            local_postcg_face_geometry_shadow_cells,
            local_postcg_face_geometry_fallback_cells,
            local_postcg_face_geometry_unsupported_cells,
            local_postcg_face_geometry_faces,
            local_postcg_face_geometry_group_face_uses,
            local_postcg_face_geometry_candidate_avoided,
            local_postcg_face_geometry_selected_avoided,
            local_postcg_face_geometry_mismatch_cells,
            local_postcg_face_geometry_mismatch_values};
        double timing_sums[2] = {
            local_postcg_face_geometry_candidate_seconds,
            local_postcg_face_geometry_legacy_shadow_seconds};
        double timing_maxima[2] = {
            local_postcg_face_geometry_candidate_seconds,
            local_postcg_face_geometry_legacy_shadow_seconds};
        unsigned int route_state =
            (local_postcg_face_geometry_supported ? 0u : 1u) |
            (postcg_face_geometry_failed_closed_ ? 2u : 0u) |
            (local_postcg_face_geometry_mismatch_cells != 0 ? 4u : 0u);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, counters, 12,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, timing_sums, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, timing_maxima, 2, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &route_state, 1, MPI_UNSIGNED, MPI_BOR,
                      MPI_COMM_WORLD);
#endif
        bool const collective_supported = (route_state & 1u) == 0u;
        bool const collective_failed_closed = (route_state & 2u) != 0u;
        bool const collective_mismatch = (route_state & 4u) != 0u;
        if(collective_failed_closed)
            postcg_face_geometry_failed_closed_ = true;
        char const* outcome = "shadow_match";
        if(!collective_supported)
            outcome = "unsupported_fail_closed";
        else if(collective_mismatch)
            outcome = "mismatch_fail_closed";
        else if(collective_failed_closed)
            outcome = "fail_closed";
        else if(counters[0] == 0)
            outcome = "no_active_hydro_cells";
        else if(postcg_face_geometry_requested_)
            outcome = "selected";
        if(rank == 0)
            std::clog
                << "MG_POSTCG_FACE_GEOMETRY scope=global"
                << " requested="
                << (postcg_face_geometry_requested_ ? 1 : 0)
                << " shadow_requested="
                << (postcg_face_geometry_shadow_requested_ ? 1 : 0)
                << " supported=" << (collective_supported ? 1 : 0)
                << " enabled="
                << (postcg_face_geometry_requested_ &&
                    collective_supported && !collective_failed_closed ? 1 : 0)
                << " shadow_enabled="
                << (postcg_face_geometry_shadow_requested_ &&
                    collective_supported && !collective_failed_closed ? 1 : 0)
                << " outcome=" << outcome
                << " cells_total=" << counters[0]
                << " candidate_cells=" << counters[1]
                << " selected_cells=" << counters[2]
                << " shadow_cells=" << counters[3]
                << " fallback_cells=" << counters[4]
                << " unsupported_cells=" << counters[5]
                << " faces_precomputed=" << counters[6]
                << " group_face_uses=" << counters[7]
                << " candidate_geometry_evaluations=" << counters[6]
                << " candidate_evaluations_avoided=" << counters[8]
                << " selected_evaluations_avoided=" << counters[9]
                << " mismatch_cells=" << counters[10]
                << " mismatch_values=" << counters[11]
                << " failed_closed="
                << (collective_failed_closed ? 1 : 0)
                << " candidate_seconds_sum=" << timing_sums[0]
                << " candidate_seconds_max=" << timing_maxima[0]
                << " legacy_shadow_seconds_sum=" << timing_sums[1]
                << " legacy_shadow_seconds_max=" << timing_maxima[1]
                << std::endl;
    }

    if(local_postcg_compton_shadow_mismatch != 0) {
        postcg_unrecoverable_ = true;
        good_end = 0;
        ++local_rejected_count;
        if(getLastStepFailureReason().empty())
            setCellLocalStepFailure(
                "Compton bulk coefficient shadow mismatch in PostCG",
                first_postcg_compton_shadow_mismatch_cell_id);
    }

    int const local_good_end = good_end;
    int const local_unrecoverable = postcg_unrecoverable_ ? 1 : 0;
    int global_rejected_count = local_rejected_count;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &min_T_E_added, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &d_Ek, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_rejected_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (rank == 0 && global_rejected_count > 0)
        std::clog << "PostCG rejected " << global_rejected_count << " cells" << std::endl;

    int global_unrecoverable = local_unrecoverable;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_unrecoverable, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    postcg_unrecoverable_ = global_unrecoverable != 0;

    if((postcg_unrecoverable_ || good_end == 0) &&
       local_unrecoverable == 0 && local_good_end != 0 &&
       getLastStepFailureReason().empty()) {
        setStepFailure("PostCG rejection on another MPI rank");
        markStepFailureRemote();
    }

    if (postcg_unrecoverable_ || good_end == 0) {
        if (postcg_unrecoverable_)
            return;
        // A candidate that failed the collective validation is not committed.
        // Let the transactional caller restore the accepted state and report a
        // controlled solver failure instead of aborting from this rank.
        postcg_unrecoverable_ = true;
        return;
    }

    double Efinal = 0;
    for (std::size_t i=0; i<N; ++i) {
        Efinal += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Efinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

// #ifdef DEBUG
    if (rank == 0) {
        std::clog << std::setprecision(14) << "Einit = " << Einit << ", Efinal = " << Efinal <<" min_T_E_added = "<<min_T_E_added<<" d_Ek "<<d_Ek<<std::endl;
        std::clog << std::setprecision(16) << "|Einit-Efinal|/Einit = " << std::abs(Einit - Efinal) / Einit << std::endl;
    }
// #endif
}

void MultigroupDiffusion::calculate_fleck_factor(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double dt_cgs) const
{
    size_t const N = tess.GetPointNo();
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    unsigned long long local_absorption_only_cells = 0;
    unsigned long long representative_cell_id = 0;
    double representative_values[5] = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < N; ++i) {
        double const dt_cell_cgs = individualScheduledTimeStep(
            i, dt_cgs / time_scale_) * time_scale_;
        double const sigma_planck = sigma_absorption_planck[i];
        double const sigma_planck_safe = (std::isfinite(sigma_planck) && sigma_planck > 0.0)
            ? sigma_planck : 0.0;
        double const T = old_Tm[i];
        // A non-positive/non-finite thermodynamic baseline cannot be passed to
        // the EOS or Compton table.  Disable coupled Compton for this cell and
        // keep a valid, absorption-only Fleck factor; the split solver will
        // subsequently handle only cells with a usable state.
        if (!std::isfinite(T) || T <= 0.0) {
            compton_deferred_[i] = true;
            use_n_zero[i] = false;
            compton_occupation_mode_[i] = ComptonOccupationMode::Off;
            fleck_factor[i] = 1.0;
            Gammas[i] = std::max(sigma_planck_safe, std::numeric_limits<double>::min());
            upsilon_[i] = 0.0;
            continue;
        }
        double cv = 0.0;
        try {
            cv = eos_.dT2cv(cells[i].density, T, cells[i].tracers, ComputationalCell3D::tracerNames);
        }
        catch (UniversalError const&) {
            compton_deferred_[i] = true;
            use_n_zero[i] = false;
            compton_occupation_mode_[i] = ComptonOccupationMode::Off;
            fleck_factor[i] = 1.0;
            Gammas[i] = std::max(sigma_planck_safe, std::numeric_limits<double>::min());
            upsilon_[i] = 0.0;
            continue;
        }

        double const material_energy_density =
            cells[i].internal_energy * cells[i].density;
        double const energy_ratio =
            std::isfinite(material_energy_density) &&
            material_energy_density > 0.0 && std::isfinite(cv) && cv > 0.0
                ? cv * T / material_energy_density
                : 1.0;
        double const beta_scale = std::max(1.0, 0.5 * energy_ratio);
        cv *= mass_scale_ / (pow<2>(time_scale_)*length_scale_);
        double const radiation_cv = get_radiation_cv(T);
        double cv_bar = (std::isfinite(radiation_cv) && radiation_cv > 0.0)
            ? cv / radiation_cv : std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(cv_bar) || cv_bar <= 0.0) {
            compton_deferred_[i] = true;
            cv_bar = 1.0;
        }

        double Gamma = sigma_planck_safe;
        double upsilon = 0;
        bool did_compton = false;
        ComptonOccupationMode occupation_mode = ComptonOccupationMode::RadiationField;
        compton_jacobian_frozen_[i] = false;
        upsilon_erad_[i] = std::numeric_limits<double>::quiet_NaN();
        upsilon_lte_[i] = std::numeric_limits<double>::quiet_NaN();
        upsilon_n0_[i] = std::numeric_limits<double>::quiet_NaN();
        if (!compton_deferred_[i] && compton_on_ &&
            (sigma_planck * dt_cell_cgs * CG::speed_of_light < compton_optical_depth_turn_off)) {
            did_compton = true;
            generate_S_and_dSdUm_matrices(cells[i], i, dt_cell_cgs, ComptonOccupationMode::RadiationField);
            double const upsilon_erad = calculate_Upsilon(cells[i]);
            upsilon_erad_[i] = upsilon_erad;
            upsilon = upsilon_erad;

            double const beta = beta_scale / cv_bar;
            double const coupling_opacity = std::max(sigma_planck_safe, std::abs(upsilon));
                double const coupling = coupling_opacity * beta * CG::speed_of_light * dt_cell_cgs;

            if (std::abs(upsilon) > 0.1 * sigma_planck_safe && coupling > 0.1) {
                    generate_S_and_dSdUm_matrices(cells[i], i, dt_cell_cgs, ComptonOccupationMode::PlanckFunction);
                double const upsilon_lte = calculate_Upsilon(cells[i]);
                upsilon_lte_[i] = upsilon_lte;

                double best_upsilon = upsilon_erad;
                occupation_mode = ComptonOccupationMode::RadiationField;
                if (upsilon_lte > best_upsilon) {
                    best_upsilon = upsilon_lte;
                    occupation_mode = ComptonOccupationMode::PlanckFunction;
                }

                if (best_upsilon < 0.0) {
                    generate_S_and_dSdUm_matrices(cells[i], i, dt_cell_cgs, ComptonOccupationMode::Zero);
                    double const upsilon_n0 = calculate_Upsilon(cells[i]);
                    upsilon_n0_[i] = upsilon_n0;
                    if (upsilon_n0 > best_upsilon) {
                        best_upsilon = upsilon_n0;
                        occupation_mode = ComptonOccupationMode::Zero;
                    }
                }

                upsilon = best_upsilon;
                use_n_zero[i] = occupation_mode == ComptonOccupationMode::Zero;
                    generate_S_and_dSdUm_matrices(cells[i], i, dt_cell_cgs, occupation_mode);
            } else if (upsilon < 0.0) {
                        generate_S_and_dSdUm_matrices(cells[i], i, dt_cell_cgs, ComptonOccupationMode::Zero);
                upsilon = calculate_Upsilon(cells[i]);
                upsilon_n0_[i] = upsilon;
                occupation_mode = ComptonOccupationMode::Zero;
                use_n_zero[i] = true;
            } else {
                use_n_zero[i] = false;
            }

            Gamma += upsilon;
            if (upsilon < 0.0) {
                compton_jacobian_frozen_[i] = true;
                fill_zero(dSdUm);
                upsilon = 0.0;
                Gamma = sigma_planck_safe;
            }
            compton_occupation_mode_[i] = occupation_mode;
        }

        double f = CG::FleckFactor(
            dt_cell_cgs, beta_scale / cv_bar, Gamma);

        // If the selected occupation model still gives an inadmissible
        // Jacobian, defer Compton for this cell and fall back to absorption-only.
        if (!compton_deferred_[i] &&
            (!std::isfinite(Gamma) || Gamma <= 0.0 || !std::isfinite(f) || f <= 0.0 || f > 1.0)) {
            compton_deferred_[i] = true;
            use_n_zero[i] = false;
            compton_jacobian_frozen_[i] = false;
            compton_occupation_mode_[i] = ComptonOccupationMode::Off;
            Gamma = sigma_planck_safe;
                f = CG::FleckFactor(
                    dt_cell_cgs,
                    beta_scale /
                        std::max(cv_bar, std::numeric_limits<double>::min()),
                    Gamma);
        } else if (did_compton) {
            use_n_zero[i] = occupation_mode == ComptonOccupationMode::Zero;
        }

        if (!std::isfinite(f) || f <= 0.0 || f > 1.0) {
            // A malformed EOS/opacity state cannot be allowed to inject a
            // non-positive Fleck factor into the matrix. Freeze absorption
            // for this cell; the caller's post-step positivity check will
            // force the ordinary timestep failure path if its EOS is still
            // invalid.
            f = 1.0;
            Gamma = std::max(sigma_planck_safe, std::numeric_limits<double>::min());
        }

        if (did_compton && sigma_scattering_group[i][0] < 1e-50)
            fillComptonScatteringRates(i, tau, n);

        fleck_factor[i] = f;
        Gammas[i] = Gamma;
        upsilon_[i] = Gamma - sigma_planck_safe;
        if (!did_compton)
            compton_occupation_mode_[i] = ComptonOccupationMode::Off;
        if(compton_deferred_[i] && i < split_compton_cells_.size() &&
           split_compton_cells_[i]) {
            if(local_absorption_only_cells == 0) {
                representative_cell_id = static_cast<unsigned long long>(
                    cells[i].ID);
                representative_values[0] = dt_cell_cgs / time_scale_;
                representative_values[1] = 1.0 / cv_bar;
                representative_values[2] = sigma_planck_safe;
                representative_values[3] = Gamma;
                representative_values[4] = f;
            }
            ++local_absorption_only_cells;
        }
    }

    unsigned long long global_absorption_only_cells =
        local_absorption_only_cells;
    int representative_rank = local_absorption_only_cells > 0
        ? rank : std::numeric_limits<int>::max();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_absorption_only_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &representative_rank, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    if(global_absorption_only_cells > 0) {
#ifdef RICH_MPI
        MPI_Bcast(&representative_cell_id, 1, MPI_UNSIGNED_LONG_LONG,
                  representative_rank, MPI_COMM_WORLD);
        MPI_Bcast(representative_values, 5, MPI_DOUBLE,
                  representative_rank, MPI_COMM_WORLD);
#endif
        if(rank == 0)
            std::clog << std::setprecision(17)
                      << "MG_COMPTON_ABSORPTION_ONLY_RETRY"
                      << " scope=aggregate"
                      << " cells=" << global_absorption_only_cells
                      << " rank=" << representative_rank
                      << " cell_id=" << representative_cell_id
                      << " dt=" << representative_values[0]
                      << " beta=" << representative_values[1]
                      << " kappa_planck=" << representative_values[2]
                      << " gamma=" << representative_values[3]
                      << " fleck=" << representative_values[4]
                      << " compton_in_fleck=0"
                      << " occupation=off"
                      << std::endl;
    }

    // Second pass: fill Compton scattering rates for cells skipped by the
    // main loop (Compton turned off by optical depth, or ghost cells i >= N).
    // We check group 0 as a proxy -- if it was filled above, all groups were.
    if (compton_on_) {
        auto const Ntotal = cells.size();
        double constexpr fac_n = pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);
        std::vector<std::vector<double>> local_tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0));
        std::vector<std::vector<double>> local_dtau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)); // required by get_tau_matrix, unused
        std::vector<double> local_n(ENERGY_GROUPS_NUM, 0.0);
        for (std::size_t i = 0; i < Ntotal; ++i) {
            if (i < N && compton_deferred_[i])
                continue;
            if (sigma_scattering_group[i][0] > 1e-50)
                continue;
            if (i >= N && tess.IsPointOutsideBox(i))
                continue;
            auto const& cell = cells[i];
            if (cell.temperature <= 0.0 || cell.density <= 0.0)
                continue;
            double const raw_T = (i < N) ? old_Tm[i] : cell.temperature;
            if (!std::isfinite(raw_T) || raw_T <= 0.0)
                continue;
            double const T = std::clamp(raw_T,
                compton_matrix_gen.get_minimum_temperature_grid() * 1.0001,
                compton_matrix_gen.get_maximum_temperature_grid() * 0.9999);
            double const rho_cgs = cell.density * mass_scale_ / pow<3>(length_scale_);
            compton_matrix_gen.get_tau_matrix(T, rho_cgs, 1.0, 1.0, local_tau, local_dtau);

            for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                double const dnu = energy_groups_width[g] / units::planck_constant;
                double const nu = energy_groups_center[g] / units::planck_constant;
                double const Eg = cell.Eg[g] * cell.density * mass_scale_ /
                                  (length_scale_ * pow<2>(time_scale_));
                local_n[g] = std::min(100.0, fac_n * Eg / (pow<3>(nu) * dnu));
            }

            fillComptonScatteringRates(i, local_tau, local_n);
        }
    }
}

void MultigroupDiffusion::calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                                 std::vector<ComputationalCell3D> const& cells,
                                                                                 double const dt) const {
    auto const N = tess.GetPointNo();
    auto const Ntotal = cells.size();
    sigma_absorption_group.resize(Ntotal);
    sigma_scattering_group.resize(Ntotal);
    std::vector<std::size_t> cooling_neighbors;
    face_vec cooling_faces;
    for (std::size_t i=0; i < N; ++i) {
        double const dt_cell = std::max(
            individualScheduledTimeStep(i, dt / time_scale_) * time_scale_,
            std::numeric_limits<double>::min() * 1e100);
        double const Trad = std::pow(cells[i].Erad * cells[i].density / CG::radiation_constant, 0.25);
        double cv = eos_.dT2cv(cells[i].density * pow<3>(length_scale_) / mass_scale_, cells[i].temperature) * mass_scale_ / (pow<2>(time_scale_)*length_scale_);
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);

        sigma_absorption_group[i].resize(ENERGY_GROUPS_NUM);
        sigma_scattering_group[i].resize(ENERGY_GROUPS_NUM);

        auto const& cell = cells[i];
        double const kT_1 = 1.0 / (CG::boltzmann_constant * std::max(cell.temperature, 1e-200));
        double const Um = get_radiation_energy_density(cell.temperature);
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {

            sigma_absorption_group[i][g] = std::min(coefficient_calculator.CalcAbsorptionOpacity(cell, energy_groups_center[g]),
                CG::max_coupling_strength / (CG::speed_of_light * dt_cell));
            if (protections_on_) {
                if (Trad > 1.1 * cells[i].temperature && cv < 0.1 * get_radiation_cv(Trad)) {
                    sigma_absorption_group[i][g] = std::min(sigma_absorption_group[i][g],
                        cv * Trad / (CG::speed_of_light * dt_cell * cells[i].Erad * cells[i].density));
                }
            }

            double const a = energy_groups_boundary[g] * kT_1;
            double const b = energy_groups_boundary[g+1] * kT_1;
            double const bg = planck_integral::planck_integral(a, b);
            if (!std::isfinite(sigma_absorption_group[i][g]) || sigma_absorption_group[i][g] < 0.) {
                sigma_absorption_group[i][g] = 0.0;
                compton_deferred_[i] = true;
            }

            sigma_scattering_group[i][g] = coefficient_calculator.CalcScatteringOpacity(cell, energy_groups_center[g]);

            if (!std::isfinite(sigma_scattering_group[i][g]) || sigma_scattering_group[i][g] < 0.) {
                sigma_scattering_group[i][g] = 0.0;
                compton_deferred_[i] = true;
            }
        }

        // Limit the cell-total material cooling, not every group separately.
        // The former per-group cap allowed each group to remove twice the
        // thermal energy and scaled opacity as 1/dt, so a rejected candidate
        // could remain nonphysical under arbitrarily many timestep halvings.
        // A common scale preserves the opacity spectrum and bounds the total
        // explicit cooling estimate independently of the runtime group count.
        long double net_absorption_cooling = 0.0;
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            double const a = energy_groups_boundary[g] * kT_1;
            double const b = energy_groups_boundary[g + 1] * kT_1;
            double const bg = planck_integral::planck_integral(a, b);
            double const radiation_energy_density =
                cell.Eg[g] * cell.density;
            net_absorption_cooling +=
                static_cast<long double>(CG::speed_of_light * dt_cell) *
                static_cast<long double>(sigma_absorption_group[i][g]) *
                static_cast<long double>(bg * Um - radiation_energy_density);
        }
        double const material_energy_density = std::max(
            cell.internal_energy * cell.density,
            0.0);
        double const maximum_absorption_cooling =
            0.5 * material_energy_density;
        if(std::isfinite(static_cast<double>(net_absorption_cooling)) &&
           net_absorption_cooling > maximum_absorption_cooling &&
           net_absorption_cooling > 0.0) {
            double const opacity_scale = std::clamp(
                maximum_absorption_cooling /
                    static_cast<double>(net_absorption_cooling),
                0.0, 1.0);
            for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                sigma_absorption_group[i][g] *= opacity_scale;
        }

        if(cooling_time_limiter_on_)
        {
            tess.GetNeighbors(i, cooling_neighbors);
            cooling_faces = tess.GetCellFaces(i);
            auto &neighbors = cooling_neighbors;
            auto &faces = cooling_faces;

            double div_v = 0;
            for(std::size_t j = 0; j < neighbors.size(); ++j)
            {
                if(j >= faces.size())
                    continue;
                std::size_t const neigh = neighbors[j];
                Vector3D const r_ij = normalize(tess.GetMeshPoint(i) - tess.GetMeshPoint(neigh));
                Vector3D vel_j = cells[i].velocity;
                if(neigh < N || !tess.IsPointOutsideBox(neigh))
                    vel_j = cells[neigh].velocity;
                div_v -= 0.5 * ScalarProd(cells[i].velocity + vel_j, r_ij) *
                         tess.GetArea(faces[j]) * length_scale_ * length_scale_;
            }
            div_v /= std::max(volume, 1e-200);

            double const speed = fastabs(cells[i].velocity);
            double const compression_speed = std::max(-div_v, 0.0) * cell_width;
            if(speed > 1.0 && compression_speed > 0.25 * speed && compression_speed * speed > cells[i].internal_energy * 0.25)
            {
                double const hydro_time = 1.0 / std::max(-div_v, 1e-200);
                double const T_local = std::max(cells[i].temperature, 1.0);
                double const inv_kT = 1.0 / (CG::boltzmann_constant * T_local);
                double const Um_local = get_radiation_energy_density(T_local);
                double planck_exchange = 0.0;
                double compton_exchange = 0.0;
                for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                {
                    double const a = energy_groups_boundary[g] * inv_kT;
                    double const b = energy_groups_boundary[g + 1] * inv_kT;
                    double const bg = planck_integral::planck_integral(a, b);
                    double const Eg_density = cells[i].Eg[g] * cells[i].density;

                    planck_exchange += CG::speed_of_light * sigma_absorption_group[i][g] * (bg * Um_local - Eg_density);
                }

                // Use the full multigroup Compton operator for the gas-radiation
                // exchange estimate instead of a gray (Tgas-Trad) approximation.
                if(compton_on_)
                {
                    ComputationalCell3D cell_for_compton = cells[i];
                    cell_for_compton.density *= pow<3>(length_scale_) / mass_scale_;
                    cell_for_compton.internal_energy *= pow<2>(time_scale_) / pow<2>(length_scale_);
                    cell_for_compton.Erad *= pow<2>(time_scale_) / pow<2>(length_scale_);
                    cell_for_compton.velocity *= time_scale_ / length_scale_;
                    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                    {
                        cell_for_compton.Eg[g] *= pow<2>(time_scale_) / pow<2>(length_scale_);
                    }

                    ComptonOccupationMode const occupation_mode =
                        compton_occupation_mode_[i] == ComptonOccupationMode::Off
                            ? ComptonOccupationMode::RadiationField
                            : compton_occupation_mode_[i];
                    generate_S_and_dSdUm_matrices(cell_for_compton, i, dt_cell, occupation_mode);
                    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                    {
                        double const Eg_density = cells[i].Eg[g] * cells[i].density;
                        for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt)
                        {
                            compton_exchange -= CG::speed_of_light * S[g][gt] * Eg_density;
                        }
                    }
                }

                double const net_cooling_power = planck_exchange + compton_exchange;
                if(net_cooling_power > 0)
                {
                    double const thermal_energy = std::max(cells[i].internal_energy * cells[i].density, 1e-200);
                    double const cool_time = thermal_energy / net_cooling_power;
                    double const target_cool_time = 2.0 * hydro_time;
                    if(cool_time < target_cool_time)
                    {
                        double const target_cooling_power = thermal_energy / target_cool_time;
                        double const opacity_scale = std::max(target_cooling_power / std::max(net_cooling_power, 1e-200), 1e-6);
                        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                        {
                            sigma_absorption_group[i][g] *= opacity_scale;
                        }
                        if(compton_on_)
                            compton_limiter_scale_[i] = opacity_scale;
                    }
                }
            }
        }
    }

    for (std::size_t i = N; i < Ntotal; ++i) {
        sigma_absorption_group[i].resize(ENERGY_GROUPS_NUM, 0.0);
        sigma_scattering_group[i].resize(ENERGY_GROUPS_NUM, 0.0);
        if (tess.IsPointOutsideBox(i))
            continue;
        auto const& cell = cells[i];
        if (cell.temperature <= 0.0)
            continue;
        double dt_reference = individualScheduledTimeStep(
            i, dt / time_scale_) * time_scale_;
        dt_reference = std::max(dt_reference,
                                std::numeric_limits<double>::min() * 1e100);
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            sigma_absorption_group[i][g] = std::min(
                coefficient_calculator.CalcAbsorptionOpacity(cell, energy_groups_center[g]),
                CG::max_coupling_strength / (CG::speed_of_light * dt_reference));
            sigma_scattering_group[i][g] = coefficient_calculator.CalcScatteringOpacity(cell, energy_groups_center[g]);
        }
    }
}

void MultigroupDiffusion::calculate_planck_integrals(Tessellation3D const& tess,
                                                     std::vector<ComputationalCell3D> const& cells) const {

    auto const N = tess.GetPointNo();

    planck_integal_group.resize(N);
    for (std::size_t i=0; i<N; ++i) {
        planck_integal_group[i].resize(ENERGY_GROUPS_NUM);
        double const planck_T = (std::isfinite(old_Tm[i]) && old_Tm[i] > 0.0)
            ? old_Tm[i] : 1e-200;
        double const kT = CG::boltzmann_constant * planck_T;
        double planck_sum = 0.0;
        for (std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g) {

            double const a = energy_groups_boundary[g] / kT;
            double const b = energy_groups_boundary[g+1] / kT;

            double const bg = planck_integral::planck_integral(a, b);

            planck_integal_group[i][g] = bg;
            planck_sum += bg;
        }

        if (planck_sum < (1. - 1e-2) && not displayed_warning_) {
            displayed_warning_ = true;
            std::clog << "bad groups! planckian not covered well! cell " << i << " T " << old_Tm[i] <<" ID "<<cells[i].ID<<std::endl;
            std::clog << "bad planck_sum " << planck_sum << std::endl;
            // throw UniversalError("bad groups! planckian not covered well!");
        }
    }
}

void MultigroupDiffusion::calculate_planck_absorption_coefficient(Tessellation3D const& tess,
                                                                                std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();
    std::fill(sigma_absorption_planck.begin(), sigma_absorption_planck.end(), 0.0);

    for (std::size_t i=0; i<N; ++i) {
        auto const& cell = cells[i];
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
            double const sigma = sigma_absorption_group[i][g];
            double const bg = planck_integal_group[i][g];

            sigma_absorption_planck[i] += sigma * bg;
        }
    }
}

void MultigroupDiffusion::generate_S_and_dSdUm_matrices(ComputationalCell3D const& cell,
                                                          std::size_t const cell_index,
                                                          double const dt_cgs,
                                                          ComptonOccupationMode const occupation_mode) const {
    cell_id_of_compton_matrices = cell.ID;
    compton_occupation_mode_[cell_index] = occupation_mode;

    double constexpr fac = pow<3>(units::clight) / (8.0*M_PI*units::planck_constant);

    double const raw_T = old_Tm[cell_index];
    double const safe_T = (std::isfinite(raw_T) && raw_T > 0.0) ? raw_T : 1e-200;
    // Below the tabulated range the redistribution kernel approaches the
    // cold-electron edge.  Clamp only the kernel lookup; keep the material
    // temperature unchanged for EOS and emission/coupling terms.
    double const T = std::clamp(safe_T,
        compton_matrix_gen.get_minimum_temperature_grid() * 1.0001,
        compton_matrix_gen.get_maximum_temperature_grid() * 0.9999);
    bool const use_planck_lte = occupation_mode == ComptonOccupationMode::PlanckFunction;
    double const T_lte = use_planck_lte ? compute_lte_temperature(cell) : safe_T;
    double const Um_lte = CG::radiation_constant * pow<4>(T_lte);
    std::vector<double> lte_planck_fraction(ENERGY_GROUPS_NUM, 0.0);
    if (use_planck_lte) {
        double planck_integral_total = 0.0;
        double const kT = CG::boltzmann_constant * T_lte;
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            double const a = energy_groups_boundary[g] / kT;
            double const b = energy_groups_boundary[g + 1] / kT;
            lte_planck_fraction[g] = planck_integral::planck_integral(a, b);
            planck_integral_total += lte_planck_fraction[g];
        }
        if (planck_integral_total > 0.0) {
            for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                lte_planck_fraction[g] /= planck_integral_total;
        }
    }

    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        if (occupation_mode == ComptonOccupationMode::RadiationField) {
            double const dnu = energy_groups_width[g]/units::planck_constant;
            double const nu = energy_groups_center[g]/units::planck_constant;

            double const Eg = cell.Eg[g] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));

            n[g] = std::min(100.0, fac * Eg / (pow<3>(nu)*dnu));
        } else if (use_planck_lte) {
            double const dnu = energy_groups_width[g]/units::planck_constant;
            double const nu = energy_groups_center[g]/units::planck_constant;
            double const Eg = Um_lte * lte_planck_fraction[g];
            double const occupation = fac * Eg / (pow<3>(nu) * dnu);
            n[g] = std::clamp(occupation, 0.0, 100.0);
        } else {
            n[g] = 0.0;
        }
    }

    double const A = 1.0;
    double const Z = 1.0;
    compton_matrix_gen.get_tau_matrix(T, cell.density*mass_scale_/pow<3>(length_scale_), A, Z, tau, dtau_dUm);

    // transform dtau_dT to dtau_dUm
    // double const beta = 1.0 / (4.0*CG::radiation_constant*pow<3>(T));
    // for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
    //     for (auto& val : dtau_dUm[g]) {
    //         val *= beta;
    //     }
    // }

    auto const [up_scattering_last, down_scattering_last] = compton_matrix_gen.get_last_group_upscattering_and_downscattering(T, cell.density*mass_scale_/pow<3>(length_scale_), A, Z);

    fill_zero(S);
    fill_zero(dSdUm);

    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
            if (g+1 == ENERGY_GROUPS_NUM and gt+1 == ENERGY_GROUPS_NUM) {
                S[g][g] += (up_scattering_last - down_scattering_last)*(1.0 + n[g]);
                dSdUm[g][g] += dtau_dUm[g][g] * (1.0 + n[g]);
                continue;
            }

            // in scattering
            double const in_scattering_factor = energy_groups_center[g] / energy_groups_center[gt] * (1.0 + n[g]);
            double const in_scattering_factor_dsdum = energy_groups_center[g] / energy_groups_center[gt] * (1.0 + n[g]);
            S[gt][g] += tau[gt][g] * in_scattering_factor;
            dSdUm[gt][g] += dtau_dUm[gt][g] * in_scattering_factor_dsdum;

            // out scattering
            double const out_scattering_factor = 1.0 + n[gt];
            S[g][g] -= tau[g][gt] * out_scattering_factor;
            dSdUm[g][g] -= dtau_dUm[g][gt] * (1 + n[gt]);
        }
    }
    double const T_for_um = (std::isfinite(cell.temperature) && cell.temperature > 0.0)
        ? cell.temperature : 1e-200;
    double const Um = CG::radiation_constant * pow<4>(T_for_um);
    double const Um_factor = 1.0 / (4 * CG::radiation_constant * pow<3>(T_for_um));
    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
            dSdUm[g][gt] *= Um_factor;
        }
    }

    if (protections_on_) {
        double dE = 0;
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
            for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                dE -= S[g][gt] * cell.Eg[g] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
            }
        }
        dE *= dt_cgs * units::clight;

        double dE_dT = 0;
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
            for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                dE_dT -= dSdUm[g][gt] * cell.Eg[g] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
            }
        }
        dE_dT *= dt_cgs * units::clight*Um;
        double const E_cell = cell.internal_energy * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
        double const Trad = std::pow(cell.Erad * cell.density * mass_scale_ / (units::arad * length_scale_ * pow<2>(time_scale_)), 0.25);
        if (dE > E_cell) {
            if ((cell.internal_energy * Trad < 0.1 * cell.Erad * cell.temperature) && Trad > cell.temperature) {
                double max_dE = cell.density * cell.internal_energy * (Trad - cell.temperature) / cell.temperature;
                max_dE *= mass_scale_ / (length_scale_ * pow<2>(time_scale_));
                double const reduce_factor = max_dE / dE;
                if (reduce_factor < 1) {
                    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
                        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                            S[gt][g] *= reduce_factor;
                            dSdUm[gt][g] *= reduce_factor;
                        }
                    }
                }
            }
        } else {
            double const dE_factor = 0.4;
            if (dE < -dE_factor * E_cell) {
                double const reduce_factor = std::abs(dE_factor * E_cell / dE);
                if (reduce_factor < 1)
                {
                    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
                        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                            S[gt][g] *= reduce_factor;
                            dSdUm[gt][g] *= reduce_factor;
                        }
                    }
                }
            }
        }
    }

    if(cooling_time_limiter_on_ && compton_limiter_scale_[cell_index] < 1.0)
    {
        double const scale = compton_limiter_scale_[cell_index];
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt)
            {
                S[g][gt] *= scale;
                dSdUm[g][gt] *= scale;
            }
        }
    }

    if (cell_index < compton_jacobian_frozen_.size() && compton_jacobian_frozen_[cell_index])
        fill_zero(dSdUm);
}

double MultigroupDiffusion::calculate_Upsilon(ComputationalCell3D const& cell) const {
    assert(cell_id_of_compton_matrices == cell.ID);

    double Upsilon = 0.0;

    for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
        for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
            Upsilon += dSdUm[gt][gtt] * cell.Eg[gt] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
        }
    }

    return Upsilon;
}

double MultigroupDiffusion::compute_lte_temperature(ComputationalCell3D const& cell) const {
    double const e_tot = cell.internal_energy + cell.Erad;
    if (e_tot <= 0.0 || !std::isfinite(e_tot))
        return std::max(cell.temperature, 1e-200);

    double const T_max = compton_matrix_gen.get_maximum_temperature_grid() * 0.9999;
    double const Trad = std::pow(
        std::max(cell.Erad, 0.0) * cell.density * mass_scale_
            / (CG::radiation_constant * length_scale_ * pow<2>(time_scale_)),
        0.25);
    double T = std::clamp(std::max(cell.temperature, Trad), 1e-30, T_max);

    for (int iter = 0; iter < 50; ++iter) {
        double const e_matter = eos_.dT2e(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames);
        double const e_radiation = CG::radiation_constant * pow<4>(T) * pow<2>(time_scale_) * length_scale_
            / (cell.density * mass_scale_);
        double const residual = e_matter + e_radiation - e_tot;
        if (std::abs(residual) <= 1e-10 * std::max(e_tot, 1e-30))
            break;

        double const derivative = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames)
            + 4.0 * CG::radiation_constant * pow<3>(T) * pow<2>(time_scale_) * length_scale_
                / (cell.density * mass_scale_);
        if (std::abs(derivative) <= 1e-30)
            break;

        T = std::clamp(T - residual / derivative, 1e-30, T_max);
    }

    return T;
}

void MultigroupDiffusion::fillImplicitComptonCellCoefficients(
    Tessellation3D const& tess,
    ComputationalCell3D const& cell,
    std::size_t const cell_index,
    double const dt_cgs,
    ImplicitComptonCellCoefficients& coefficients) const
{
    assert(cell_id_of_compton_matrices == cell.ID);

    double const volume =
        tess.GetVolume(cell_index) * pow<3>(length_scale_);
    double const cdt = CG::speed_of_light * dt_cgs;
    double const T = old_Tm[cell_index];
    double const cv = eos_.dT2cv(
        cell.density, T, cell.tracers,
        ComputationalCell3D::tracerNames) * mass_scale_ /
        (pow<2>(time_scale_) * length_scale_);
    double const cv_bar = cv / get_radiation_cv(T);
    double const cdt_cv_bar = cdt / cv_bar;
    double const f = fleck_factor[cell_index];
    double const Um_old = get_radiation_energy_density(T);
    double const kp = sigma_absorption_planck[cell_index];

    std::array<double, ENERGY_GROUPS_NUM> sum_dSdUm_Egtt;
    std::array<double, ENERGY_GROUPS_NUM> A;
    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        double sum = 0.0;
        for(std::size_t gtt = 0; gtt < ENERGY_GROUPS_NUM; ++gtt)
            sum += dSdUm[gtt][g] * cell.Eg[gtt] * cell.density *
                mass_scale_ / (length_scale_ * pow<2>(time_scale_));
        sum_dSdUm_Egtt[g] = sum;

        double const kg = sigma_absorption_group[cell_index][g];
        double const kgbg = kg * planck_integal_group[cell_index][g];
        A[g] = volume * cdt * cdt_cv_bar * f * (kgbg + sum);
        coefficients.delta_b[g] = volume * cdt * Um_old *
            (kgbg * (1.0 - (1.0 + cdt_cv_bar * kp) * f)
             - kp * cdt_cv_bar * f * sum);
    }

    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt) {
            double contribution = 0.0;
            contribution -= volume * cdt * S[gt][g];
            contribution -= volume * cdt * cdt_cv_bar * f *
                sum_dSdUm_Egtt[g] *
                sigma_absorption_group[cell_index][gt];
            // Keep the original sequential multiply-add order. Factoring this
            // into A[g] * sum(S[gt]) is algebraically equivalent but changes
            // floating-point rounding and the legacy shadow comparison.
            for(std::size_t gtt = 0; gtt < ENERGY_GROUPS_NUM; ++gtt)
                contribution += A[g] * S[gt][gtt];
            coefficients.delta_A[g * ENERGY_GROUPS_NUM + gt] = contribution;
        }
}

bool MultigroupDiffusion::
fillLegacyImplicitComptonCellCoefficientsAndCompare(
    Tessellation3D const& tess,
    ComputationalCell3D const& cell,
    std::size_t const cell_index,
    double const dt_cgs,
    ImplicitComptonCellCoefficients const& bulk,
    ImplicitComptonCellCoefficients& legacy,
    char const* const phase) const
{
    bool matches = true;
    std::size_t mismatch_count = 0;
    bool first_is_rhs = false;
    std::size_t first_g = 0;
    std::size_t first_gt = 0;
    double first_bulk = 0.0;
    double first_legacy = 0.0;

    auto const record = [&](bool const is_rhs,
                            std::size_t const g,
                            std::size_t const gt,
                            double const bulk_value,
                            double const legacy_value) {
        if(mgDoubleBits(bulk_value) == mgDoubleBits(legacy_value))
            return;
        if(matches) {
            first_is_rhs = is_rhs;
            first_g = g;
            first_gt = gt;
            first_bulk = bulk_value;
            first_legacy = legacy_value;
        }
        matches = false;
        ++mismatch_count;
    };

    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        legacy.delta_b[g] = get_implicit_compton_contribution_to_b(
            tess, cell, cell_index, g, dt_cgs);
        record(true, g, 0, bulk.delta_b[g], legacy.delta_b[g]);
        for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt) {
            std::size_t const index = g * ENERGY_GROUPS_NUM + gt;
            legacy.delta_A[index] = get_implicit_compton_contribution(
                tess, cell, cell_index, g, gt, dt_cgs);
            record(false, g, gt, bulk.delta_A[index], legacy.delta_A[index]);
        }
    }

    if(!matches) {
        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        std::clog << "MG_COMPTON_BULK_SHADOW mismatch"
                  << " phase=" << phase
                  << " rank=" << rank
                  << " cell_id=" << cell.ID
                  << " kind=" << (first_is_rhs ? "b" : "A")
                  << " g=" << first_g;
        if(!first_is_rhs)
            std::clog << " gt=" << first_gt;
        std::clog << " bulk=" << first_bulk
                  << " legacy=" << first_legacy
                  << " bulk_bits=" << mgDoubleBits(first_bulk)
                  << " legacy_bits=" << mgDoubleBits(first_legacy)
                  << " ulp_distance="
                  << mgUlpDistance(first_bulk, first_legacy)
                  << " mismatches=" << mismatch_count
                  << " comparisons="
                  << ENERGY_GROUPS_NUM * (ENERGY_GROUPS_NUM + 1)
                  << std::endl;
    }
    return matches;
}

double MultigroupDiffusion::get_implicit_compton_contribution(Tessellation3D const& tess, ComputationalCell3D const& cell, std::size_t const cell_index, std::size_t const g, std::size_t const gt, double const dt_cgs) const {
    assert(cell_id_of_compton_matrices == cell.ID);

    double const volume = tess.GetVolume(cell_index) * pow<3>(length_scale_);

    double const cdt = CG::speed_of_light*dt_cgs;

    double const T = old_Tm[cell_index];
    double const cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames)*mass_scale_ / (pow<2>(time_scale_)*length_scale_);

    double const cv_bar = cv / get_radiation_cv(T);
    double const cdt_cv_bar = cdt / cv_bar;

    double const kg = sigma_absorption_group[cell_index][g];
    double const kgbg = kg*planck_integal_group[cell_index][g];

    double const f = fleck_factor[cell_index];


    double implicit_contribution = 0.0;

    double const coeff_1 = volume*cdt*cdt_cv_bar*kgbg*f;

    double const Um_old = get_radiation_energy_density(T);

    implicit_contribution -= volume*cdt*S[gt][g];
    double sum_dSdUm_Egtt = 0.0;
    for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
        sum_dSdUm_Egtt += dSdUm[gtt][g]*cell.Eg[gtt]*cell.density*mass_scale_ / (length_scale_ * pow<2>(time_scale_));
    }

    double const A = volume*cdt*cdt_cv_bar*f*(kgbg + sum_dSdUm_Egtt);

    implicit_contribution -= volume*cdt*cdt_cv_bar*f*sum_dSdUm_Egtt*sigma_absorption_group[cell_index][gt];
    for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
        implicit_contribution += A*S[gt][gtt];
    }

    return implicit_contribution;
}

double MultigroupDiffusion::get_implicit_compton_contribution_to_b(Tessellation3D const& tess, ComputationalCell3D const& cell, std::size_t const cell_index, std::size_t const g, double const dt_cgs) const {
    assert(cell_id_of_compton_matrices == cell.ID);

    double const volume = tess.GetVolume(cell_index) * pow<3>(length_scale_);

    double const cdt = CG::speed_of_light*dt_cgs;

    double const T = old_Tm[cell_index];
    double const cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames)*mass_scale_ / (pow<2>(time_scale_)*length_scale_);

    double const cv_bar = cv / get_radiation_cv(T);
    double const cdt_cv_bar = cdt / cv_bar;

    double const kg = sigma_absorption_group[cell_index][g];
    double const kgbg = kg*planck_integal_group[cell_index][g];
    double const kp = sigma_absorption_planck[cell_index];

    double const f = fleck_factor[cell_index];
    double const Um_old = get_radiation_energy_density(T);

    double sum_dSdUm_Egtt = 0.0;
    for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
        sum_dSdUm_Egtt += dSdUm[gtt][g]*cell.Eg[gtt]*cell.density*mass_scale_ / (length_scale_ * pow<2>(time_scale_));
    }

    // Keep this algebraic form to avoid dividing by kp when Planck opacity
    // vanishes; it is the exact cancellation of the original expression.
    double const contribution_to_b = volume*cdt*Um_old *
        (kgbg*(1.0 - (1.0 + cdt_cv_bar*kp)*f)
         - kp * cdt_cv_bar * f * sum_dSdUm_Egtt);

    return contribution_to_b;
}

double MultigroupDiffusion::calcEffectiveDiffusionCoefficient(
    ComputationalCell3D const& cell,
    std::size_t cell_index,
    std::size_t group) const
{
    double D = coefficient_calculator.CalcDiffusionCoefficient(cell, energy_groups_center[group]);
    if (compton_on_ && !coefficient_calculator.ComptonIncludedInTransport()) {
        if (cell_index < sigma_scattering_group.size() &&
            group < sigma_scattering_group[cell_index].size() &&
            sigma_scattering_group[cell_index][group] > 1e-50)
        {
            double sigma_transport = CG::speed_of_light / (3.0 * D);
            sigma_transport += sigma_scattering_group[cell_index][group];
            D = CG::speed_of_light / (3.0 * sigma_transport);
        }
    }
    return D;
}

double MultigroupDiffusion::applyComptonTransportCorrection(
    double diffusion_coefficient,
    std::size_t cell_index,
    std::size_t group) const
{
    double D = diffusion_coefficient;
    if (compton_on_ && !coefficient_calculator.ComptonIncludedInTransport()) {
        if (cell_index < sigma_scattering_group.size() &&
            group < sigma_scattering_group[cell_index].size() &&
            sigma_scattering_group[cell_index][group] > 1e-50)
        {
            double sigma_transport = CG::speed_of_light / (3.0 * D);
            sigma_transport += sigma_scattering_group[cell_index][group];
            D = CG::speed_of_light / (3.0 * sigma_transport);
        }
    }
    return D;
}

void MultigroupDiffusion::fillComptonScatteringRates(
    std::size_t cell_index,
    std::vector<std::vector<double>> const& tau_mat,
    std::vector<double> const& occ) const
{
    sigma_scattering_group[cell_index] =
        comptonTransportScatteringExtinction(tau_mat, occ);
}
