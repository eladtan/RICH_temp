#include "Diffusion.hpp"
#include "newtonian/three_dimensional/simulation/RuntimeLog.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
#include "misc/utils.hpp"
#include <boost/math/special_functions.hpp>
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cfenv>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <stdexcept>

#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace
{
double reduce_grey_reference_scale(double local_maximum)
{
	local_maximum = std::max(local_maximum, 0.0);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &local_maximum, 1, MPI_DOUBLE, MPI_MAX,
		MPI_COMM_WORLD);
#endif
	return local_maximum;
}

ComputationalCell3D radiationCellInCgs(
    ComputationalCell3D const& cell,
    double const length_scale,
    double const time_scale,
    double const mass_scale)
{
    ComputationalCell3D result(cell);
    result.density *= mass_scale /
        (length_scale * length_scale * length_scale);
    result.Erad *= length_scale * length_scale /
        (time_scale * time_scale);
    result.Erad_dt *= length_scale * length_scale /
        (time_scale * time_scale * time_scale);
    result.Erad_dt_dt *= length_scale * length_scale /
        (time_scale * time_scale * time_scale * time_scale);
    result.velocity *= length_scale / time_scale;
    return result;
}

}

namespace CG
{
     // LP flux limiter taken from "EQUATIONS AND ALGORITHMS FOR MIXED-FRAME FLUX-LIMITED DIFFUSION RADIATION HYDRODYNAMICS"
    double CalcSingleFluxLimiter(Vector3D const& grad, double const D, double const cell_value)
    {
        double const R = std::max(3 * abs(grad) * D / (cell_value * CG::speed_of_light + 1e-200), 1e-200);
        
        // series expansion
        if(R < 1e-2) return 1 - R * R / 15 + 2 * R * R * R * R /315;
        
        return 3 * (1.0 / std::tanh(R) - 1.0 / R) / R;
    }

    double FleckFactor(double const dt, double const beta, double const sigma_a)
    {
        return 1.0 / (1 + beta * dt * sigma_a * CG::speed_of_light);
    }

    double FleckFactorCompton(double const dt, double const beta, double const sigma_a, double const sigma_s, double const Erad, double const Cv)
    {
        return 1.0 / (1 + beta * dt * sigma_a * CG::speed_of_light + dt * 16 * sigma_s * CG::boltzmann_constant * Erad / (CG::electron_mass * CG::speed_of_light * Cv));
    }
}

Diffusion::Diffusion(OpacityCalculator const& D_coefficient_calc, 
                     DiffusionBoundaryCalculator const& boundary_calc,
                     EquationOfState const& eos, 
                     std::vector<std::string> const zero_cells, 
                     bool const flux_limiter, 
                     bool const hydro_on, 
                     bool const compton_on,
                     bool const cooling_time_limiter_on,
                     double const max_planck_opacity_factor) : 
                                             RadiationDriver(eos, 
                                                            zero_cells, 
                                                            flux_limiter, 
                                                            hydro_on, 
                                                            compton_on),
                                             D_coefficient_calcualtor(D_coefficient_calc),
                                             boundary_calc_(boundary_calc), 
                                             sigma_planck(),
                                             sigma_s(), 
                                             fleck_factor(),
                                             D(),
                                             R2(),
                                             cell_flux_limiter(),
                                             new_Er(),
                                             new_Er_full(),
                                             old_Er(),
                                             cells_temp(),
                                             extensives_temp(),
                                             cooling_time_limiter_on_(cooling_time_limiter_on),
                                             max_planck_opacity_factor_(max_planck_opacity_factor) {}

double Diffusion::GetSingleFleckFactor(ComputationalCell3D const& cell, double const dt)const
{
    ComputationalCell3D cell_cgs(cell);
    cell_cgs.density *= mass_scale_ / (length_scale_ * length_scale_ * length_scale_);
    double const Er = cell.Erad * cell.density *  mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
    double sigma_planck = D_coefficient_calcualtor.CalcPlanckOpacity(cell_cgs);
    sigma_planck = std::min(sigma_planck, max_planck_opacity_factor_ / (CG::speed_of_light * dt * time_scale_)); // avoid too large opacities
    double const sigma_s = D_coefficient_calcualtor.CalcScatteringOpacity(cell_cgs);
    double const T = cell.temperature;
    double Cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames);
    double const energy_ratio = Cv * cell.temperature / (cell.internal_energy * cell.density);
    Cv *= mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
    double const beta = std::max(1.0, 0.5 * energy_ratio) * 4 * CG::radiation_constant * T * T * T / Cv;
    return compton_on_ ? FleckFactorCompton(dt * time_scale_, beta, sigma_planck, sigma_s, Er, Cv) : FleckFactor(dt * time_scale_, beta, sigma_planck);
}

bool Diffusion::prestep(Tessellation3D const& tess,
                        std::vector<ComputationalCell3D> const& cells) const {
    MEMORY_PROFILE_SCOPE("diffusion prestep");
    auto const N = tess.GetPointNo();
    
    sigma_planck.resize(N, 0.0);
    conditional_shrink(sigma_planck);
    sigma_s.resize(N, 0.0);
    conditional_shrink(sigma_s);
    fleck_factor.resize(N, 0.0);
    conditional_shrink(fleck_factor);
    D.resize(N, 0.0);
    conditional_shrink(D);
    R2.resize(N, 0.0);
    conditional_shrink(R2);
    cell_flux_limiter.resize(N, 0.0);
    conditional_shrink(cell_flux_limiter);

    new_Er.resize(N, 0.0);
    conditional_shrink(new_Er);
    new_Er_full.resize(N, 0.0);
    conditional_shrink(new_Er_full);
	    old_Er.resize(N, 0.0);
	    conditional_shrink(old_Er);
	    old_T.resize(N, 0.0);
	    conditional_shrink(old_T);

    cells_temp.resize(N);
    conditional_shrink(cells_temp);
    extensives_temp.resize(N);
    conditional_shrink(extensives_temp);

	    for(std::size_t i=0; i < N; ++i){
	        old_Er[i] = cells[i].Erad * cells[i].density;
	        old_T[i] = cells[i].temperature;

        if(old_Er[i] < 0.0){
            UniversalError eo("negative Erad");
 			eo.addEntry("i", i);
			eo.addEntry("old_Er", old_Er[i]);
			eo.addEntry("ID", cells[i].ID);
			eo.addEntry("density", cells[i].density);
            throw eo;
        }
    }

	    return true;
	}

bool Diffusion::prestepIndividual(
	Tessellation3D const& tess,
	std::vector<ComputationalCell3D> const& cells,
	IndividualStepContext const&) const
{
	bool const result = prestep(tess, cells);
	individual_event_old_Er = old_Er;
	individual_event_old_T = old_T;
	individual_event_old_Eint.resize(cells.size());
	for(std::size_t i = 0; i < cells.size(); ++i)
		individual_event_old_Eint[i] = cells[i].internal_energy * cells[i].density;
	return result;
}

void Diffusion::prepareIndividualCandidate(
	Tessellation3D const& tess,
	std::vector<ComputationalCell3D> const& cells) const
{
	// A rejected candidate is rolled back to the latest accepted state before
	// this hook runs.  Refresh candidate-local radiation/material baselines here;
	// keep individual_event_old_* fixed for event-level timestep feedback.
	if(!prestep(tess, cells))
		throw std::runtime_error("grey individual candidate preparation failed");
}

void Diffusion::ReleaseDormantGlobalSolverStorage() const
{
    bool const had_storage =
        cg_workspace_.HasAllocatedStorage() ||
        new_Er.capacity() != 0 ||
        new_Er_full.capacity() != 0;
    if(cg_workspace_.HasAllocatedStorage())
        cg_workspace_.Release();
    release_container_memory(new_Er);
    release_container_memory(new_Er_full);
    if(had_storage)
        rich_trim_after_rare_spike();
}

bool Diffusion::poststep() const {    
    const size_t N = cells_temp.size();
    const size_t Nnz = cg_workspace_.A_values.size();
    cells_temp.clear();
    extensives_temp.clear();
    release_if_very_stale(cells_temp, N);
    release_if_very_stale(extensives_temp, N);
    release_if_very_stale(cg_workspace_.b, N);
    release_if_very_stale(cg_workspace_.sub_x, N);
    release_if_very_stale(cg_workspace_.M, N);
    release_if_very_stale(cg_workspace_.r_old, N);
    release_if_very_stale(cg_workspace_.sub_a_times_p, N);
    release_if_very_stale(cg_workspace_.sub_r, N);
    release_if_very_stale(cg_workspace_.sub_p, N);
    release_if_very_stale(cg_workspace_.sub_r0, N);
    release_if_very_stale(cg_workspace_.A_diag, N);
    release_if_very_stale(cg_workspace_.y, N);
    release_if_very_stale(cg_workspace_.z, N);
    release_if_very_stale(cg_workspace_.v, N);
    release_if_very_stale(cg_workspace_.h, N);
    release_if_very_stale(cg_workspace_.s, N);
    release_if_very_stale(cg_workspace_.t, N);
    release_if_very_stale(cg_workspace_.scratch_rescale1, N);
    release_if_very_stale(cg_workspace_.scratch_rescale2, N);
    release_if_very_stale(cg_workspace_.old_x, N);
    release_if_very_stale(cg_workspace_.A_row_ptr, N + 1);
    release_if_very_stale(cg_workspace_.A_col_idx, Nnz);
    release_if_very_stale(cg_workspace_.A_values, Nnz);

    return false;
}

double Diffusion::calculate_dt(double const dt,
                               Tessellation3D& tess, 
                               std::vector<ComputationalCell3D>& cells) const {

    int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

		double local_max_Er = 0;
		for(ComputationalCell3D const& cell : cells)
			local_max_Er = std::max(
				local_max_Er, cell.Erad * cell.density);
		double const max_Er = reduce_grey_reference_scale(local_max_Er);

    auto const N = tess.GetPointNo();            
    size_t const Nzero = zero_cells_.size();
	std::vector<size_t> zero_indeces;
	for(size_t i = 0; i < Nzero; ++i)
		zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
	double max_diff = -1;
	size_t max_loc = max_size_t;
	last_cell_dt_limits_.assign(N, std::numeric_limits<double>::infinity());
	for(size_t i = 0; i < N; ++i)
	{
		bool to_calc = true;
		for(size_t j = 0; j < Nzero; ++j)
			if(cells[i].stickers[zero_indeces[j]])
				to_calc = false;
		if(not to_calc)
			continue;
		double const equlibrium_factor = std::abs(cells[i].temperature - std::pow(new_Er[i] / CG::radiation_constant, 0.25)) < 0.02 * cells[i].temperature ? 0.05 : 1;
		double const normalization =
			std::max(old_Er[i], cells[i].Erad * cells[i].density) +
			0.02 * max_Er;
		double diff = normalization > 0 ?
			equlibrium_factor *
			std::abs(cells[i].Erad * cells[i].density - old_Er[i]) /
			normalization : 0;
		if(fleck_factor[i] < 0.4)
			diff *= 0.2;
		if(!std::isfinite(diff))
			continue;
		// The grid-wide limit below is the smallest of these (before its
		// 1.25 growth cap); individual steps apply the same rule per cell.
		// Stored as infinity when not representable, tested before dividing
		// (diff < 1 keeps diff * max finite).
		double const numerator = dt * 0.15;
		if(diff > 0 && (diff >= 1 ||
			numerator < diff * std::numeric_limits<double>::max()))
			last_cell_dt_limits_[i] = numerator / diff;
		if(max_loc == max_size_t || diff > max_diff)
		{
			max_diff = diff;
			max_loc = i;
		}
	}

	struct
    {
        double val;
        int mpi_id;
    }max_data;
    
	max_data.mpi_id = max_loc == max_size_t ?
		std::numeric_limits<int>::max() : rank;
	max_data.val = max_loc == max_size_t ? -1 : max_diff;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
	MPI_exchange_data(tess, cells, true);	
#endif
	bool const has_limiting_cell =
		max_data.mpi_id != std::numeric_limits<int>::max();
	max_diff = has_limiting_cell ? max_data.val : 0;
	double const difference_scale =
		std::max(max_diff, std::numeric_limits<double>::min());
	double const suggested_dt =
		dt * std::min(1.25, 0.15 / difference_scale);
	bool const detailed_runtime_log = RuntimeLogDetailed();
	unsigned long long limiting_cell_id =
		std::numeric_limits<unsigned long long>::max();
	if(detailed_runtime_log && has_limiting_cell && rank == max_data.mpi_id)
		limiting_cell_id =
			static_cast<unsigned long long>(cells[max_loc].ID);
#ifdef RICH_MPI
	if(detailed_runtime_log && has_limiting_cell)
		MPI_Bcast(&limiting_cell_id, 1, MPI_UNSIGNED_LONG_LONG,
			max_data.mpi_id, MPI_COMM_WORLD);
#endif
	std::string limiting_cell_details;
	if(detailed_runtime_log && has_limiting_cell)
	{
		if(rank == max_data.mpi_id)
		{
			std::ostringstream details;
			details<<"Radiation time step ID "<<cells[max_loc].ID
				<<" old Er "<<old_Er[max_loc]
				<<" new Er "<<cells[max_loc].Erad * cells[max_loc].density
				<<" diff "<<max_diff<<" Tgas "<<cells[max_loc].temperature
				<<" Trad "<<std::pow(new_Er[max_loc] /
					CG::radiation_constant, 0.25)
				<<" max_Er "<<max_Er<<" rank "<<rank
				<<" density "<<cells[max_loc].density
				<<" width "<<tess.GetWidth(max_loc)
				<<" Tgas_old "<<old_T[max_loc]
				<<" location "<<tess.GetMeshPoint(max_loc)<<std::endl;
			std::streambuf* const original_buffer = std::cout.rdbuf(
				details.rdbuf());
			try
			{
				PrintDebugData(max_loc);
			}
			catch(...)
			{
				std::cout.rdbuf(original_buffer);
				throw;
			}
			std::cout.rdbuf(original_buffer);
			details<<"Next time step is "<<suggested_dt<<std::endl;
			limiting_cell_details = details.str();
		}
#ifdef RICH_MPI
		unsigned long long detail_size =
			static_cast<unsigned long long>(limiting_cell_details.size());
		MPI_Bcast(&detail_size, 1, MPI_UNSIGNED_LONG_LONG,
			max_data.mpi_id, MPI_COMM_WORLD);
		if(detail_size >
		   static_cast<unsigned long long>(std::numeric_limits<int>::max()))
			throw std::overflow_error(
				"Grey timestep diagnostic is too large for MPI");
		if(rank != max_data.mpi_id)
			limiting_cell_details.resize(static_cast<std::size_t>(detail_size));
		if(detail_size > 0)
			MPI_Bcast(&limiting_cell_details[0], static_cast<int>(detail_size),
				MPI_CHAR, max_data.mpi_id, MPI_COMM_WORLD);
#endif
	}
	if(detailed_runtime_log && rank == 0)
	{
		std::cout<<limiting_cell_details;
		std::cout<<"GREY_TIMESTEP_LIMIT mode=global current_dt="<<dt
			<<" suggested_dt="<<suggested_dt
			<<" difference="<<(has_limiting_cell ? max_diff : 0)
			<<" max_Er="<<max_Er<<" growth_cap=1.25"
			<<" reference_scope=mesh_global cell_id=";
		if(has_limiting_cell)
			std::cout<<limiting_cell_id<<" rank="<<max_data.mpi_id;
		else
			std::cout<<"none rank=-1";
		std::cout<<std::endl;
	}

    return suggested_dt;
}


namespace
{
	// Relative-increment limit (decision 2026-09-22 D5, R1'): the fraction of a
	// cell's internal or radiation energy that one radiation update may change;
	// the next step is bounded by applied_dt * fraction / r with
	// r = max(|de_int| / e_int0, |dE_r| / E_r0) over the event.  0 (unset) = off;
	// 0.15 is the value D5 specified.  Must agree on every rank.
	double IndividualRadiationIncrementLimit()
	{
		static double const value = []()
		{
			char const* const text = std::getenv("RICH_INDIVIDUAL_RADIATION_INCREMENT_LIMIT");
			if(text == nullptr || text[0] == '\0')
				return 0.0;
			char* end = nullptr;
			double const parsed = std::strtod(text, &end);
			if(end == text || *end != '\0' || !std::isfinite(parsed) || parsed < 0)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_RADIATION_INCREMENT_LIMIT must be a finite number >= 0");
			return parsed;
		}();
		return value;
	}

	// RICH_RADIATION_MOMENTUM_POSITIVITY caps: the largest fraction of a cell's
	// available radiation energy the radiation force's kinetic-energy gain, and
	// of its gas internal energy the relativistic exchange, may take in one
	// update (the analogue of RICH_INDIVIDUAL_THERMAL_LOSS_FRACTION).  Default
	// 0.5; RICH_RADIATION_MOMENTUM_KINETIC_LOSS_FRACTION in (0, 1).  First call
	// is collective (PostCG start on every rank) and requires one value on all ranks.
	double RadiationMomentumKineticLossFraction()
	{
		static double const value = []()
		{
			char const* const text = std::getenv("RICH_RADIATION_MOMENTUM_KINETIC_LOSS_FRACTION");
			double parsed = 0.5;
			int invalid = 0;
			if(text != nullptr && text[0] != '\0')
			{
				int const traps = fegetexcept();
				fedisableexcept(FE_ALL_EXCEPT);
				char* end = nullptr;
				errno = 0;
				parsed = std::strtod(text, &end);
				bool const range_error = errno == ERANGE;
				feclearexcept(FE_ALL_EXCEPT);
				feenableexcept(traps);
				if(end == text || *end != '\0' || range_error || !std::isfinite(parsed) || !(parsed > 0) ||
				   !(parsed < 1))
				{
					invalid = 1;
					parsed = 0.5;
				}
			}
			bool agreed = true;
#ifdef RICH_MPI
			double extrema[2] = {parsed, -parsed};
			MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			agreed = extrema[0] == -extrema[1];
#endif
			if(invalid != 0 || !agreed)
				throw std::invalid_argument(
					"RICH_RADIATION_MOMENTUM_KINETIC_LOSS_FRACTION must be one number in (0, 1) on every rank");
			return parsed;
		}();
		return value;
	}
}

void Diffusion::calculateIndividualTimeSteps(
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
    int canonical_mapping_valid =
        (canonical_owned_cells == nullptr) == (local_to_global == nullptr) ?
        1 : 0;
    bool const canonical_reference = canonical_owned_cells != nullptr &&
        local_to_global != nullptr;
    std::vector<unsigned char> canonical_active;
    if(canonical_reference) {
        canonical_active.assign(canonical_owned_cells->size(), 0);
        if(local_to_global->size() != tess.GetPointNo())
            canonical_mapping_valid = 0;
        for(std::size_t local : context.active_indices) {
            if(local >= cells.size() || local >= local_to_global->size()) {
                canonical_mapping_valid = 0;
                continue;
            }
            std::size_t const global = local_to_global->at(local);
            if(global >= canonical_owned_cells->size() ||
               canonical_active[global] != 0 ||
               canonical_owned_cells->at(global).ID != cells[local].ID) {
                canonical_mapping_valid = 0;
                continue;
            }
            canonical_active[global] = 1;
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &canonical_mapping_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
#endif
    if(canonical_mapping_valid == 0)
        throw std::runtime_error(
            "grey individual canonical-owned mapping is invalid");

    double local_max_Er = 0.0;
    if(canonical_reference) {
        for(std::size_t global = 0;
            global < canonical_owned_cells->size(); ++global)
            if(canonical_active[global] == 0) {
                ComputationalCell3D const& cell =
                    canonical_owned_cells->at(global);
                local_max_Er = std::max(
                    local_max_Er, cell.Erad * cell.density);
            }
        for(std::size_t local : context.active_indices)
            local_max_Er = std::max(
                local_max_Er, cells[local].Erad * cells[local].density);
    }
    else
        for(ComputationalCell3D const& cell : cells)
            local_max_Er = std::max(
                local_max_Er, cell.Erad * cell.density);
    double const max_Er = reduce_grey_reference_scale(local_max_Er);

    std::vector<double> const& event_old_Er =
        individual_event_old_Er.empty() ? old_Er : individual_event_old_Er;
    std::vector<std::size_t> zero_indices;
    zero_indices.reserve(zero_cells_.size());
    for(std::string const& name : zero_cells_)
        zero_indices.push_back(binary_index_find(ComputationalCell3D::stickerNames, name));

	double local_max_difference = -1;
	std::size_t representative = cells.size();
	double representative_suggested_dt = 0;
	unsigned long long active_cells = 0;
	double const increment_fraction = IndividualRadiationIncrementLimit();
	unsigned long long increment_limited = 0;
	double increment_tightest = std::numeric_limits<double>::infinity();
	unsigned long long increment_example_id = 0;
	double increment_example_r = 0;
    for(std::size_t i : context.active_indices) {
        if(i >= cells.size() || i >= time_step_limits.size() ||
           i >= event_old_Er.size())
            throw std::out_of_range(
                "grey individual timestep cell is out of range");
        bool calculate = true;
        for(std::size_t sticker : zero_indices)
            if(cells[i].stickers[sticker])
                calculate = false;
        if(!calculate)
            continue;
		++active_cells;

        double const new_Er_cell = cells[i].Erad * cells[i].density;
        // In cgs, as the global rule's new_Er: the radiation constant is cgs
        // and cells hold code units (without the conversion the radiation
        // temperature came out ~1e4 x too low on the TDE, so equilibrated
        // cells lost the 0.05 equilibrium factor: a 20 x shorter limit).
        double const radiation_temperature = std::pow(std::max(new_Er_cell, 0.0) *
            mass_scale_ / (time_scale_ * time_scale_ * length_scale_) / CG::radiation_constant, 0.25);
        double const equilibrium_factor =
            std::abs(cells[i].temperature - radiation_temperature) <
                    0.02 * cells[i].temperature
                ? 0.05
                : 1.0;
        double difference = equilibrium_factor *
            std::abs(new_Er_cell - event_old_Er.at(i)) /
            (std::max(event_old_Er.at(i), new_Er_cell) + 0.02 * max_Er +
             std::numeric_limits<double>::min());
        if(fleck_factor.at(i) < 0.4)
            difference *= 0.2;
        double const applied_dt = context.cellTimeStep(i);
        double const nominal_dt = context.nominalCellTimeStep(i);
        double const suggested_dt = std::min(
            applied_dt * 0.15 /
                std::max(difference, std::numeric_limits<double>::min()),
            nominal_dt * 2.0);
        time_step_limits.at(i) = std::min(
			time_step_limits.at(i), suggested_dt);
		// Relative-increment limit on the net change of this event: both
		// budgets positive and finite, else the cell is skipped.  Shortens
		// only; relaxes as the coupling weakens.
		if(increment_fraction > 0 && i < individual_event_old_Eint.size())
		{
			double const e_old = individual_event_old_Eint[i];
			double const e_new = cells[i].internal_energy * cells[i].density;
			double const er_old = event_old_Er.at(i);
			if(e_old > 0 && er_old > 0 && std::isfinite(e_old) && std::isfinite(er_old) &&
				std::isfinite(e_new) && std::isfinite(new_Er_cell) && applied_dt > 0)
			{
				double const r = std::max(std::abs(e_new - e_old) / e_old,
					std::abs(new_Er_cell - er_old) / er_old);
				if(r > 0)
				{
					double const limit = applied_dt * increment_fraction / r;
					if(limit < time_step_limits.at(i))
					{
						time_step_limits.at(i) = limit;
						++increment_limited;
						if(limit / applied_dt < increment_tightest)
						{
							increment_tightest = limit / applied_dt;
							increment_example_id = static_cast<unsigned long long>(cells[i].ID);
							increment_example_r = r;
						}
					}
				}
			}
		}
		if(difference > local_max_difference)
		{
			local_max_difference = difference;
			representative = i;
			representative_suggested_dt = suggested_dt;
		}
    }

	struct
	{
		double val;
		int mpi_id;
	} max_data = {local_max_difference, rank};
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
		MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &active_cells, 1, MPI_UNSIGNED_LONG_LONG,
		MPI_SUM, MPI_COMM_WORLD);
#endif
	if(increment_fraction > 0)
	{
		// Rank-0 aggregate: cells limited this event and the tightest one.
		unsigned long long limited_total = increment_limited;
		struct { double val; int mpi_id; } tightest = {increment_tightest, rank};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &limited_total, 1, MPI_UNSIGNED_LONG_LONG,
			MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &tightest, 1, MPI_DOUBLE_INT, MPI_MINLOC,
			MPI_COMM_WORLD);
		MPI_Bcast(&increment_example_id, 1, MPI_UNSIGNED_LONG_LONG, tightest.mpi_id,
			MPI_COMM_WORLD);
		MPI_Bcast(&increment_example_r, 1, MPI_DOUBLE, tightest.mpi_id, MPI_COMM_WORLD);
#endif
		if(rank == 0 && limited_total > 0)
			std::cout << std::setprecision(6)
				<< "INDIVIDUAL_RADIATION_INCREMENT_LIMITED event_time=" << context.event_time
				<< " cells=" << limited_total << " fraction=" << increment_fraction
				<< " tightest_limit_over_interval=" << tightest.val
				<< " example_cell_id=" << increment_example_id
				<< " example_r=" << increment_example_r
				<< " example_rank=" << tightest.mpi_id << std::endl;
	}
	bool const detailed_runtime_log = RuntimeLogDetailed();
	unsigned long long representative_cell_id =
		std::numeric_limits<unsigned long long>::max();
	double representative_timesteps[2] = {0, 0};
	if(detailed_runtime_log && active_cells > 0 && rank == max_data.mpi_id)
	{
		if(representative >= cells.size())
			throw std::logic_error(
				"Grey timestep representative is missing on the winning rank");
		representative_cell_id =
			static_cast<unsigned long long>(cells[representative].ID);
		representative_timesteps[0] = context.cellTimeStep(representative);
		representative_timesteps[1] = representative_suggested_dt;
	}
#ifdef RICH_MPI
	if(detailed_runtime_log && active_cells > 0)
	{
		MPI_Bcast(&representative_cell_id, 1, MPI_UNSIGNED_LONG_LONG,
			max_data.mpi_id, MPI_COMM_WORLD);
		MPI_Bcast(representative_timesteps, 2, MPI_DOUBLE,
			max_data.mpi_id, MPI_COMM_WORLD);
	}
#endif
	if(detailed_runtime_log && active_cells > 0 && rank == 0)
	{
		std::cout<<"GREY_TIMESTEP_LIMIT mode=individual event_time="
			<<context.event_time<<" active_cells="<<active_cells
			<<" cell_id="<<representative_cell_id
			<<" rank="<<max_data.mpi_id
			<<" current_dt="<<representative_timesteps[0]
			<<" suggested_dt="<<representative_timesteps[1]
			<<" difference="<<max_data.val<<" max_Er="<<max_Er
			<<" growth_cap=2 reference_scope="
			<<(canonical_reference ? "canonical_owned_global" : "mesh_global")
			<<std::endl;
	}
}

bool Diffusion::MatrixBuildRejected() const
{
    if(!MomentumPositivityEnabled() || !hydro_on_)
        return false;
    int rejected = momentum_positivity_reject_ ? 1 : 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &rejected, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(rejected == 0)
        return false;
    if(momentum_positivity_reject_)
        setStepFailure(momentum_positivity_reason_, momentum_positivity_reject_cell_);
    else
        setStepFailure("momentum positivity certificate failed on another rank");
    return true;
}

bool Diffusion::step(double const tolerance, 
                     int& total_iters, 
                     Tessellation3D const& tess, 
                     std::vector<ComputationalCell3D>& cells,
                      std::vector<Conserved3D>& extensives,
                      double const dt,
                      double const time) const {
    MEMORY_PROFILE_SCOPE("diffusion step");
    clearStepFailure();
    
    int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    extensives_temp = extensives;
    cells_temp = cells;

    std::size_t const N = tess.GetPointNo();
    bool good_end = false;
    // RICH_RADIATION_MOMENTUM_POSITIVITY: an uncertified matrix makes BiCGSTAB
    // return good_end = false before preconditioning (MatrixBuildRejected);
    // the caller halves and retries as for any rejected radiation step.
    new_Er = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Er_full, good_end, cg_workspace_);
    MEMORY_DEBUG_PRINT("diffusion: after BiCGSTAB");
    if(not good_end) {
        CG::HistoricalMGResidualCorrectionDiagnostics const& diagnostic =
            cg_workspace_.historical_correction;
        if(!diagnostic.failure_reason.empty()) {
            std::ostringstream reason;
            reason << diagnostic.failure_reason
                   << " failure_class="
                   << RadiationPositivity::SpectralRepairFailureLabel(
                          diagnostic.failure_class)
                   << " failure_cell_id=" << diagnostic.failure_cell_id
                   << " group=" << diagnostic.failure_group
                   << " signed_group_extent="
                   << diagnostic.failure_signed_group_extent
                   << " negative_extent="
                   << diagnostic.failure_negative_extent
                   << " positive_extent="
                   << diagnostic.failure_positive_extent
                   << " relative_deficit="
                   << diagnostic.failure_relative_deficit
                   << " global_E_max="
                   << diagnostic.failure_global_maximum_cell_extent
                   << " solver_iterations=" << total_iters
                   << " minimum_correction_scale="
                   << diagnostic.minimum_scale
                   << " limiting_cell_id=" << diagnostic.limiting_cell_id
                   << " limiting_group=" << diagnostic.limiting_group
                   << " limiting_before_extent="
                   << diagnostic.limiting_before_extent
                   << " limiting_unscaled_after_extent="
                   << diagnostic.limiting_unscaled_after_extent
                   << " limiting_applied_after_extent="
                   << diagnostic.limiting_applied_after_extent;
            setCellLocalStepFailure(reason.str(), diagnostic.failure_cell_id);
        }
        return false;
    }
    
	    double max_Er = new_Er.empty() ? 0.0 :
	        *std::max_element(new_Er.begin(), new_Er.end());

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

   struct MinErData {
        double value;
        int rank;
    };
    MinErData minErData = {
        std::numeric_limits<double>::infinity(), rank};
    size_t min_index = max_size_t;
    for(std::size_t i=0; i < N; ++i){
        if(new_Er[i] < 0.0 && std::abs(new_Er[i]) < 1e-9 * max_Er){
            new_Er[i] = std::min(1e-8 * max_Er, CG::radiation_constant * cells[i].temperature * cells[i].temperature * cells[i].temperature * cells[i].temperature);
        }

        if(new_Er[i] < minErData.value) {
            minErData.value = new_Er[i];
            min_index = i;
        }
    }

#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &minErData, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
#endif

    if(minErData.value < 0) {
        if(rank == minErData.rank && min_index < N) {
            setCellLocalStepFailure(
                "negative radiation energy after diffusion solve",
                cells[min_index].ID);
            std::clog << "Negative Er! Rank: " << minErData.rank << ", Index: " << min_index <<" location "<<tess.GetMeshPoint(min_index)<<" Er value "<<minErData.value<<" cell "<<cells[min_index]<<std::endl;
        }

        return false;
    }

    try {
        PostCG(tess, extensives, dt, cells, new_Er, new_Er_full);
        MEMORY_DEBUG_PRINT("diffusion: after PostCG");
    } catch(UniversalError const& eo) {
        if(rank == 0){
            std::clog<< "PostCG Exception:" << std::endl;
            std::clog<< eo.getErrorMessage() << std::endl;
        }
        
        extensives = std::move(extensives_temp);
        cells = cells_temp;
        return false;
    }

    commitResidualCorrectionAccounting(cg_workspace_.historical_correction);
    return true;
}

bool Diffusion::MomentumPositivityEnabled()
{
    static bool const enabled = []()
    {
        char const* const text = std::getenv("RICH_RADIATION_MOMENTUM_POSITIVITY");
        int value = text != nullptr && std::string(text) == "1" ? 1 : 0;
        int invalid = text != nullptr && text[0] != '\0' && std::string(text) != "0" && std::string(text) != "1" ? 1 : 0;
        int extrema[3] = {value, -value, invalid};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, extrema, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(extrema[2] != 0 || extrema[0] != -extrema[1])
            throw std::invalid_argument("RICH_RADIATION_MOMENTUM_POSITIVITY must be 0 or 1 on every rank");
        return extrema[0] != 0;
    }();
    return enabled;
}

void Diffusion::BuildMatrix(Tessellation3D const& tess, mat& A, size_t_mat& A_indeces, std::vector<ComputationalCell3D> const& cells,
    double const dt, std::vector<double>& b, std::vector<double>& x0, double const current_time) const
{
    double const max_v = 0.1 * CG::speed_of_light * length_scale_ / time_scale_;
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells_cgs(cells);
    for(size_t i = 0; i < cells_cgs.size(); ++i)
    {
        cells_cgs[i].density *= mass_scale_ / (length_scale_ * length_scale_ * length_scale_);
        cells_cgs[i].Erad *= length_scale_ * length_scale_ / (time_scale_ * time_scale_);
        cells_cgs[i].Erad_dt *= length_scale_ * length_scale_ / (time_scale_ * time_scale_ * time_scale_);
        cells_cgs[i].Erad_dt_dt *= length_scale_ * length_scale_ / (time_scale_ * time_scale_ * time_scale_ * time_scale_);
        cells_cgs[i].velocity *= length_scale_ / time_scale_;
     }
#ifdef RICH_MPI
	// RadiationStep refreshes compact individual-mesh primitives with
	// SyncPartialBuildData.  MadVoro's sparse send indices address the canonical
	// all-point array, so the legacy compact-vector exchange is valid only for
	// the full/global path.
	if(individual_context_ == nullptr)
		MPI_exchange_data(tess, cells_cgs, true);
#endif
    b.resize(Nlocal, 0);
    x0.resize(individual_context_ == nullptr ? Nlocal : cells_cgs.size(), 0);
    D.resize(Nlocal);
    fleck_factor.resize(Nlocal);
    sigma_planck.resize(Nlocal);
    sigma_s.resize(Nlocal);
    std::vector<size_t> neighbors;
    face_vec faces;
    std::vector<size_t> zero_indeces;
    size_t const Nzero = zero_cells_.size();
    for(size_t i = 0; i < Nzero; ++i)
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
    double const zero_value = 1e-10;
    std::vector<double> new_Er(Nlocal, 0), Er_for_limit(Nlocal, 0);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const dt_cell = individualCellTimeStep(i, dt);
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
        bool set_to_zero = false;
        for(size_t j = 0; j < Nzero; ++j)
            if(cells_cgs[i].stickers[zero_indeces[j]])
                set_to_zero = true;
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density * (set_to_zero ? zero_value : 1);
        new_Er[i] = Er;

        D[i] = D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
        if(D[i] < 0)
            throw UniversalError("Negative D");
        double const T = cells_cgs[i].temperature;
        sigma_planck[i] = D_coefficient_calcualtor.CalcPlanckOpacity(cells_cgs[i]);
        if(sigma_planck[i] < 0)
            throw UniversalError("Negative sigma_planck");
        if(dt_cell > 0)
            sigma_planck[i] = std::min(sigma_planck[i], max_planck_opacity_factor_ / (CG::speed_of_light * dt_cell * time_scale_)); // avoid too large opacities
        sigma_s[i] = D_coefficient_calcualtor.CalcScatteringOpacity(cells_cgs[i]);

        // Optional cooling limiter for under-resolved post-shock cooling layers.
        if(cooling_time_limiter_on_)
        {
            faces = tess.GetCellFaces(i);
            tess.GetNeighbors(i, neighbors);
            double div_v = 0;
            for(size_t j = 0; j < neighbors.size(); ++j)
            {
                size_t const neigh = neighbors[j];
                Vector3D const r_ij = normalize(tess.GetMeshPoint(i) - tess.GetMeshPoint(neigh));
                Vector3D vel_j = cells_cgs[i].velocity;
                if(neigh < Nlocal || !tess.IsPointOutsideBox(neigh))
                    vel_j = cells_cgs[neigh].velocity;
                div_v -= 0.5 * ScalarProd(cells_cgs[i].velocity + vel_j, r_ij) * tess.GetArea(faces[j]) * length_scale_ * length_scale_;
            }
            div_v /= std::max(volume, 1e-200);

            double const speed = fastabs(cells_cgs[i].velocity);
            double const compression_speed = std::max(-div_v, 0.0) * cell_width;
            if(speed > 1.0 && compression_speed > 0.25 * speed && compression_speed * speed > cells[i].internal_energy * 0.25)
            {
                double const hydro_time = 1.0 / std::max(-div_v, 1e-200);
                double const T_local = std::max(cells_cgs[i].temperature, 1.0);
                double const radiation_eq = CG::radiation_constant * std::pow(T_local, 4);
                double const planck_exchange = CG::speed_of_light * sigma_planck[i] * (radiation_eq - Er);
                double compton_exchange = 0.0;
                if(compton_on_)
                {
                    double const Tr = std::pow(std::max(Er / CG::radiation_constant, 1e-200), 0.25);
                    compton_exchange = 16.0 * sigma_s[i] * CG::boltzmann_constant * Er *
                                       (T_local - Tr) / (CG::electron_mass * CG::speed_of_light);
                }

                // Include both cooling and heating terms: only limit when net effect is cooling.
                double const net_cooling_power = planck_exchange + compton_exchange;
                if(net_cooling_power > 0)
                {
                    double const thermal_energy = std::max(cells_cgs[i].internal_energy * cells_cgs[i].density , 1e-200);
                    double const cool_time = thermal_energy / net_cooling_power;
                    double const target_cool_time = 2.0 * hydro_time;
                    if(cool_time < target_cool_time)
                    {
                        // Preserve relative Planck/Compton term weighting while reducing net cooling.
                        double const target_cooling_power = thermal_energy / target_cool_time;
                        double const opacity_scale = std::max(target_cooling_power / std::max(net_cooling_power, 1e-200), 1e-8);
                        sigma_planck[i] *= opacity_scale;
                        if(compton_on_)
                            sigma_s[i] *= opacity_scale;
                    }
                }
            }
        }

        double Cv = eos_.dT2cv(cells[i].density, T, cells[i].tracers, ComputationalCell3D::tracerNames);
        double const energy_ratio = Cv * cells[i].temperature / (cells[i].internal_energy * cells[i].density);
        Cv *= mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
        double const beta = std::max(1.0, 0.5 * energy_ratio) * 4 * CG::radiation_constant * T * T * T / Cv;
        fleck_factor[i] = compton_on_ ? FleckFactorCompton(dt_cell * time_scale_, beta, sigma_planck[i], sigma_s[i], Er, Cv) : FleckFactor(dt_cell * time_scale_, beta, sigma_planck[i]);
        if(fleck_factor[i] < 0)
            throw UniversalError("Negative fleck_factor");
        b[i] = volume * Er;
        double const Um = CG::radiation_constant * T * T * T * T;
        if(fleck_factor[i] < 0.8 && Um > Er)
        {
            double const prefactor = fleck_factor[i] * dt_cell * CG::speed_of_light * sigma_planck[i];
            x0[i] = (Er + prefactor * Um) / (1 + prefactor);
        }
        else
            x0[i] = std::min(2 * Er, std::max(0.5 * Er, Er + cells_cgs[i].Erad_dt * cells_cgs[i].density * dt_cell * time_scale_  + 0.5 * cells_cgs[i].Erad_dt_dt * cells_cgs[i].density * dt_cell * dt_cell * time_scale_ * time_scale_));//std::max(Er + 0.5 * std::min(fleck_factor[i] * dt_cell * sigma_planck[i] * CG::speed_of_light * time_scale_, 1.0) * (CG::radiation_constant * T * T * T * T - Er), 0.25 * Er);
        b[i] += volume * fleck_factor[i] * dt_cell * CG::speed_of_light * sigma_planck[i] * T * T * T * T * CG::radiation_constant * time_scale_;
        Er_for_limit[i] = std::min(Er, std::max(1e-5 * Er, Er + dt_cell * time_scale_ * fleck_factor[i] * sigma_planck[i] * CG::speed_of_light * (CG::radiation_constant * T * T * T * T - Er)));
    }
    if(individual_context_ != nullptr)
        for(size_t i = Nlocal; i < cells_cgs.size(); ++i)
        {
            bool set_to_zero = false;
            for(size_t j = 0; j < Nzero; ++j)
                if(cells_cgs[i].stickers[zero_indeces[j]])
                    set_to_zero = true;
            x0[i] = cells_cgs[i].Erad * cells_cgs[i].density *
                    (set_to_zero ? zero_value : 1);
        }
#ifdef RICH_MPI
	if(individual_context_ == nullptr)
		MPI_exchange_data(tess, D, true);
	else
	{
		// A diffusion coefficient is cell-local.  Computing compact ghost values
		// from the already synchronized primitive state is exact and avoids using
		// canonical all-point indices on a compact array.
		D.resize(cells_cgs.size());
		for(size_t i = Nlocal; i < cells_cgs.size(); ++i)
		{
			// Physical boundary generators have no primitive state and their
			// diffusion coefficient is never used as an interior-cell value.
			if(tess.IsPointOutsideBox(i))
				continue;
			D[i] = D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
			if(D[i] < 0)
				throw UniversalError("Negative D");
		}
	}
#endif
    size_t max_neigh = 0;
    // Find maximum number of neighbors and allocate data
    for(size_t i = 0; i < Nlocal; ++i)
        max_neigh = std::max(max_neigh, tess.GetCellFaces(i).size());
    ++max_neigh;
    A.resize(Nlocal);
    A_indeces.resize(Nlocal);
    for(size_t row = 0; row < Nlocal; ++row) {
        A[row].clear();
        A_indeces[row].clear();
    }
    for(size_t i = 0; i < Nlocal; ++i) {
        A[i].reserve(max_neigh);
        A_indeces[i].reserve(max_neigh);
    }
    R2.clear();
    R2.resize(Nlocal, 0);
    conditional_shrink(R2);
    cell_flux_limiter.clear();
    cell_flux_limiter.resize(Nlocal, 0);
    conditional_shrink(cell_flux_limiter);

    // Build the matrix
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const dt_cell = individualCellTimeStep(i, dt);
        A_indeces[i].push_back(i);
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        double const T = cells_cgs[i].temperature;
        A[i].push_back(volume * (1 + fleck_factor[i] * dt_cell * CG::speed_of_light * sigma_planck[i] * time_scale_));
        if(compton_on_ && cells[i].tracers[1] > 0.5)
        {
            double const Tr = std::pow(new_Er[i] / CG::radiation_constant, 0.25);
	        double const pre_factor = fleck_factor[i] * dt_cell * time_scale_ * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light);
            double const compton_term = pre_factor * (Tr - T);
            double const theta = (fleck_factor[i] < 0.5 || std::abs(compton_term) > 1e-3) ? 1 : 0.1;
            A[i][0] += pre_factor * (Tr - (1 - theta) * T) * volume;
            b[i] += pre_factor * volume * theta * T * new_Er[i];
        }
        if(A[i][0] < 0)
	        std::clog<<"Negative A in matrix build, density "<<cells_cgs[i].density<<" T "<<cells_cgs[i].temperature<<" fleck "<<fleck_factor[i]<<
	        " sig_P "<<sigma_planck[i]<<" sig_s "<<sigma_s[i]<<" dt "<<dt_cell * time_scale_<<" Erad "<<cells_cgs[i].Erad * cells_cgs[i].density<<" compton term "<<fleck_factor[i] * dt_cell * time_scale_ * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light)<<std::endl;
    }

    std::vector<double> max_R;
    max_R.reserve(Nlocal);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double max_R_local = 0;
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density;
        Vector3D const CM = tess.GetCellCM(i);
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            if(neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j))
            {
                bool set_to_zero = false;
                for(size_t k = 0; k < Nzero; ++k)
                    if(cells_cgs[neighbor_j].stickers[zero_indeces[k]])
                        set_to_zero = true;
                double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density * (set_to_zero ? zero_value : 1);
                Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                Vector3D const grad_E = cm_ij * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));                
                max_R_local = std::max(max_R_local, std::abs(fastabs(grad_E) * (Er - Er_j)));
            }
        }
        max_R.push_back(max_R_local);
    }
#ifdef RICH_MPI
	if(individual_context_ == nullptr)
		MPI_exchange_data(tess, max_R, true);
	else
	{
		// Only constructed owner cells have a complete stencil for max_R.
		// Sync them through canonical index space; non-target geometric supports
		// retain zero, matching the partial-build volume convention.
		std::vector<double> canonical_max_R;
		tess.SyncPartialBuildData(max_R, canonical_max_R);
	}
#endif
    Vector3D dummy_v;
    std::vector<Vector3D> gradE(Nlocal);
    std::vector<double> max_neighbor_R(Nlocal, 0);
    // RICH_RADIATION_MOMENTUM_POSITIVITY state of this build (see the header).
    bool const momentum_positivity = MomentumPositivityEnabled() && hydro_on_;
    momentum_positivity_reject_ = false;
    momentum_positivity_reason_.clear();
    momentum_positivity_reject_cell_ = std::numeric_limits<std::size_t>::max();
    momentum_face_weight_.assign(momentum_positivity ? Nlocal : 0, std::vector<double>());
    momentum_v_ratio_.assign(momentum_positivity ? Nlocal : 0, 0.0);
    momentum_row_boundary_.assign(momentum_positivity ? Nlocal : 0, 0);
    momentum_face_term_.assign(momentum_positivity ? Nlocal : 0, std::vector<double>());
    momentum_face_excess_.assign(momentum_positivity ? Nlocal : 0, std::vector<double>());
    momentum_minimum_verification_rhs_ = std::numeric_limits<double>::infinity();
    // rows changed, faces lumped, rejected rows (unexplained positive
    // couplings), row-sum or diagonal violations, lumped faces with alpha > 1,
    // rows with boundary faces, rows with a positive diffusion coupling,
    // row-sum or diagonal violations in (uncertified) boundary rows
    double momentum_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double momentum_maxima[2] = {0, 0}; // max w, max lumped excess / diagonal
    double momentum_minima[1] = {std::numeric_limits<double>::infinity()}; // min row sum / diagonal
    // With RICH_RADIATION_MOMENTUM_POSITIVITY a rank-local exception in either
    // assembly loop (neither contains a collective) becomes the build's
    // rejection state, so every rank reaches the same collectives: the aggregate
    // below, then MatrixBuildRejected (global) or collectiveAllTrue (individual).
    try
    {
    for(size_t i = 0; i < Nlocal; ++i)
    {
        if(!individualCellActive(i))
            continue;
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const CM = tess.GetCellCM(i);
        Vector3D const point = tess.GetMeshPoint(i);
        gradE[i] = Vector3D(0, 0, 0) ;
        double const Dcell = D[i];
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density;
        bool self_zero = false;
        for(size_t k = 0; k < Nzero; ++k)
            if(cells_cgs[i].stickers[zero_indeces[k]])
                self_zero = true;
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            double const dt_face = individualFaceTimeStep(i, neighbor_j, dt);
            Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
            double const r_ij_size = abs(r_ij);
            r_ij *= 1.0 / r_ij_size;
            double Er_j = 0;
            if(!tess.IsPointOutsideBox(neighbor_j))
            {
                bool set_to_zero = false;
                for(size_t k = 0; k < Nzero; ++k)
                    if(cells_cgs[neighbor_j].stickers[zero_indeces[k]])
                        set_to_zero = true;
                Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density * (set_to_zero ? zero_value : 1);
                bool const neighbor_active = individualCellActive(neighbor_j);
                bool const assemble_face = individual_context_ == nullptr
                    ? i < neighbor_j
                    : (!neighbor_active || i < neighbor_j);
                if(assemble_face)
                {
                    if(individual_context_ != nullptr && neighbor_j >= max_R.size())
                        throw std::runtime_error(
                            "partial radiation mesh is missing an active face neighbor");
                    Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                    Vector3D const grad_E = cm_ij * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));                
                    
                    double const T1 = cells_cgs[i].temperature;
                    double const T2 = cells_cgs[neighbor_j].temperature;
                    double const maxT = std::pow(0.5 * (pow<4>(T1) + pow<4>(T2)), 0.25);
                    cells_cgs[i].temperature = maxT;
                    double const D1 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
                    cells_cgs[i].temperature = T1;
                    cells_cgs[neighbor_j].temperature = maxT;
                    double const D2 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[neighbor_j]);
                    cells_cgs[neighbor_j].temperature = T2;
                    double mid_D = 2 * D1 * D2 / (D1 + D2);

                    // double mid_D = 0.5 * (D[neighbor_j] + Dcell);
                    double const grad_magnitude = std::max(std::numeric_limits<double>::min() * 1e40, std::abs(fastabs(grad_E) * (Er - Er_j)));
                    double grad_factor = 1;
                    max_neighbor_R[i] = std::max(max_neighbor_R[i], max_R[neighbor_j]);
                    if(grad_magnitude < 0.15 * (max_R[i] + max_R[neighbor_j]))
                        grad_factor = 0.15 * (max_R[i] + max_R[neighbor_j]) / grad_magnitude;
                    double const flux_limiter = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * ((Er - Er_j) * grad_factor), mid_D, 0.5 * (Er + Er_j)) : 1;
                    mid_D *= flux_limiter;
                    double const flux = ((self_zero || set_to_zero) ? tess.GetArea(faces[j]) * dt_face * CG::speed_of_light * 0.5 : ScalarProd(grad_E, r_ij) * tess.GetArea(faces[j]) * dt_face * mid_D) * length_scale_ * length_scale_ * time_scale_;
                    recordIndividualFaceCoefficient(i, neighbor_j, 0, flux);
                    A[i][0] += flux;
                    A[i].push_back(-flux);
                    A_indeces[i].push_back(neighbor_j);
                    if(neighbor_j < Nlocal &&
                       (individual_context_ == nullptr || neighbor_active))
                    {
                        A[neighbor_j].push_back(-flux);
                        A_indeces[neighbor_j].push_back(i);
                        A[neighbor_j][0] += flux;
                    }
                }
            }
            else
            {
                if(individual_context_ != nullptr || i < neighbor_j)
                    boundary_calc_.SetBoundaryValues(tess, i, neighbor_j, dt_face * time_scale_, cells_cgs, tess.GetArea(faces[j]) * length_scale_ * length_scale_, A[i][0], b[i], faces[j]);
                boundary_calc_.GetOutSideValues(tess, cells_cgs, i, neighbor_j, new_Er, Er_j, dummy_v);
            }
            gradE[i] += r_ij * (tess.GetArea(faces[j]) * 0.5 * (Er + Er_j) * length_scale_ * length_scale_);
        }
    }
    for(size_t i = 0; i < Nlocal; ++i)
    {
        if(!individualCellActive(i))
            continue;
        double const dt_cell = individualCellTimeStep(i, dt);
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_* length_scale_;
        gradE[i] *= -1.0 / volume;
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const point = tess.GetMeshPoint(i);
        double const Dcell = D[i];
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density; 

        double const grad_magnitude = std::max(std::numeric_limits<double>::min() * 1e40, std::abs(fastabs(gradE[i])));
        if(grad_magnitude < 0.5 * max_neighbor_R[i])
            gradE[i] *= 0.5 * max_neighbor_R[i] / grad_magnitude;
        Vector3D grad_for_limiter = gradE[i];
        if(flux_limiter_)
        {
            double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
            double const min_grad = std::abs(Er_for_limit[i]) / (1000.0 * cell_width);
            double const grad_abs = std::abs(fastabs(grad_for_limiter));
            if(grad_abs < min_grad)
            {
                if(grad_abs > 0)
                    grad_for_limiter *= min_grad / grad_abs;
                else
                    grad_for_limiter = Vector3D(min_grad, 0, 0);
            }
        }
        double const flux_limiter = flux_limiter_ ? CalcSingleFluxLimiter(grad_for_limiter, Dcell, Er_for_limit[i]) : 1;
        cell_flux_limiter[i] = flux_limiter;
        Vector3D const CM = tess.GetCellCM(i);
        double const v_ratio = std::min(1.0, 0.05 * CG::speed_of_light / (fastabs(cells_cgs[i].velocity) + 1e-2));
        // Velocity coefficient T_ij of each interior face (0 on boundary faces).
        std::vector<double> face_momentum_term(momentum_positivity ? Nneigh : 0, 0.0);
        bool row_has_boundary = false;
        // Before this row's velocity terms are added, its off-diagonal slots hold
        // only diffusion couplings, each of which must be non-positive.
        bool diffusion_positive = false;
        if(momentum_positivity)
            for(size_t slot = 1; slot < A[i].size(); ++slot)
                if(A[i][slot] > 0)
                    diffusion_positive = true;
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            double const dt_face = individualFaceTimeStep(i, neighbor_j, dt);
            if(tess.IsPointOutsideBox(neighbor_j))
                row_has_boundary = true;
            if(!tess.IsPointOutsideBox(neighbor_j))
            {
                Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
                double const r_ij_size = abs(r_ij);
                r_ij *= 1.0 / r_ij_size;
                Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                Vector3D const grad_E = r_ij * ScalarProd(r_ij, cm_ij) * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));   
                // double mid_D = 0.5 * (D[neighbor_j] + Dcell);
                // double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density;
                // double const flux_limiter_face = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j)) : 1;

                // double const T1 = cells_cgs[i].temperature;
                // double const T2 = cells_cgs[neighbor_j].temperature;
                // double const maxT = std::max(T1, T2);
                // cells_cgs[i].temperature = maxT;
                // double const D1 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
                // cells_cgs[i].temperature = T1;
                // cells_cgs[neighbor_j].temperature = maxT;
                // double const D2 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[neighbor_j]);
                // cells_cgs[neighbor_j].temperature = T2;
                // double mid_D = 2 * D1 * D2 / (D1 + D2);
                double const mid_D = Dcell;
                double const momentum_relativity_term = -0.5 * dt_face * flux_limiter * tess.GetArea(faces[j]) * (v_ratio * fleck_factor[i] * 2 * 3 * sigma_planck[i] * mid_D / CG::speed_of_light - 1) * length_scale_ * length_scale_ * time_scale_
                    * ScalarProd(cells_cgs[i].velocity, r_ij) / 3;
                A[i][0] += momentum_relativity_term;
                if(momentum_positivity)
                    face_momentum_term[j] = momentum_relativity_term;
                auto it = std::find(A_indeces[i].begin(), A_indeces[i].end(), neighbor_j);
                if(it == A_indeces[i].end())
                    throw UniversalError("Key not equal in diffusion");
                size_t const neigh_counter = static_cast<size_t>(it - A_indeces[i].begin());
                if(A_indeces[i][neigh_counter] != neighbor_j)
                    throw UniversalError("Key not equal value in diffusion");
                A[i][neigh_counter] += momentum_relativity_term;
            }
            else
                boundary_calc_.SetMomentumTermBoundary(tess, i, neighbor_j, dt_face * time_scale_, cells_cgs[i],
                    tess.GetArea(faces[j]) * length_scale_ * length_scale_, A[i][0], b[i], faces[j], fleck_factor[i],
                    flux_limiter, Dcell, sigma_planck[i]);
        }
        R2[i] = flux_limiter_ ? flux_limiter / 3 + boost::math::pow<2>(flux_limiter * abs(gradE[i]) * Dcell / (CG::speed_of_light * Er)) : 1.0 / 3.0;
        A[i][0] -= volume * fleck_factor[i] * dt_cell * 0.5 * (3 - R2[i]) * sigma_planck[i] * std::min(0.01 * CG::speed_of_light * CG::speed_of_light, ScalarProd(cells_cgs[i].velocity, cells_cgs[i].velocity)) * time_scale_ / CG::speed_of_light;
        if(momentum_positivity)
        {
            // Minimal lumping (design section 2): per interior neighbour column,
            // the positive part e of the aggregate assembled coupling moves to
            // the diagonal, at most the column's positive velocity coefficients
            // P; the faces carrying them get w = e / P.  A positive coupling the
            // velocity term does not explain, a non-positive diagonal or a
            // non-positive row sum rejects the candidate.
            std::vector<double>& weights = momentum_face_weight_[i];
            weights.assign(Nneigh, 0.0);
            momentum_v_ratio_[i] = v_ratio;
            momentum_row_boundary_[i] = row_has_boundary ? 1 : 0;
            momentum_face_term_[i] = face_momentum_term;
            momentum_face_excess_[i].assign(Nneigh, 0.0);
            double const alpha = 6 * fleck_factor[i] * v_ratio * sigma_planck[i] * Dcell / CG::speed_of_light;
            bool row_changed = false;
            bool row_rejected = diffusion_positive;
            for(size_t j = 0; j < Nneigh; ++j)
            {
                size_t const column = neighbors[j];
                if(tess.IsPointOutsideBox(column))
                    continue;
                bool first_face_of_column = true;
                for(size_t k = 0; k < j; ++k)
                    if(neighbors[k] == column && !tess.IsPointOutsideBox(neighbors[k]))
                        first_face_of_column = false;
                if(!first_face_of_column)
                    continue;
                double assembled = 0;
                double assembled_abs = 0;
                size_t first_slot = 0;
                for(size_t slot = 1; slot < A_indeces[i].size() && slot < A[i].size(); ++slot)
                    if(A_indeces[i][slot] == column)
                    {
                        assembled += A[i][slot];
                        assembled_abs += std::abs(A[i][slot]);
                        if(first_slot == 0)
                            first_slot = slot;
                    }
                double positive_velocity = 0;
                double velocity_abs = 0;
                for(size_t k = 0; k < Nneigh; ++k)
                    if(neighbors[k] == column)
                    {
                        velocity_abs += std::abs(face_momentum_term[k]);
                        if(face_momentum_term[k] > 0)
                            positive_velocity += face_momentum_term[k];
                    }
                if(!(assembled > 0) || first_slot == 0)
                    continue;
                // With non-positive diffusion (checked above) the positive
                // aggregate cannot exceed the positive velocity terms except by
                // the summation's roundoff; beyond 1e-12 of the magnitudes, reject.
                // Within it, w = assembled / P may exceed 1 by that roundoff, so
                // the exchange represents the lumped amount exactly.
                double const tolerance = 1e-12 * (assembled_abs + velocity_abs);
                if(assembled > positive_velocity + tolerance || !(positive_velocity > 0))
                {
                    row_rejected = true;
                    continue;
                }
                // Lump the whole positive aggregate: the column becomes exactly 0.
                for(size_t slot = 1; slot < A_indeces[i].size() && slot < A[i].size(); ++slot)
                    if(A_indeces[i][slot] == column)
                        A[i][slot] = 0;
                A[i][0] += assembled;
                momentum_face_excess_[i][j] = assembled;
                row_changed = true;
                double const w = assembled / positive_velocity;
                for(size_t k = 0; k < Nneigh; ++k)
                    if(neighbors[k] == column && face_momentum_term[k] > 0)
                    {
                        weights[k] = w;
                        momentum_counts[1] += 1;
                        if(alpha > 1)
                            momentum_counts[4] += 1;
                    }
                momentum_maxima[0] = std::max(momentum_maxima[0], w);
                if(A[i][0] > 0)
                    momentum_maxima[1] = std::max(momentum_maxima[1], assembled / A[i][0]);
            }
            // Final sign check of every aggregated interior column.
            for(size_t j = 0; j < Nneigh && !row_rejected; ++j)
            {
                size_t const column = neighbors[j];
                if(tess.IsPointOutsideBox(column))
                    continue;
                double final_value = 0;
                for(size_t slot = 1; slot < A_indeces[i].size() && slot < A[i].size(); ++slot)
                    if(A_indeces[i][slot] == column)
                        final_value += A[i][slot];
                if(final_value > 0)
                    row_rejected = true;
            }
            if(row_rejected)
                momentum_counts[2] += 1;
            if(diffusion_positive)
                momentum_counts[6] += 1;
            double row_sum = A[i][0];
            for(size_t slot = 1; slot < A[i].size(); ++slot)
                row_sum += A[i][slot];
            // Rows with boundary faces are outside the certificate (design
            // section 1): counted, never rejected, as for the RHS gate.
            bool const row_violation = !(A[i][0] > 0) || !(row_sum > 0) || !std::isfinite(row_sum);
            bool const certificate_violated = row_violation && !row_has_boundary;
            if(certificate_violated)
                momentum_counts[3] += 1;
            else if(row_violation)
                momentum_counts[7] += 1;
            else
                momentum_minima[0] = std::min(momentum_minima[0], row_sum / A[i][0]);
            if(row_changed)
                momentum_counts[0] += 1;
            if(row_has_boundary)
                momentum_counts[5] += 1;
            if((row_rejected || certificate_violated) && !momentum_positivity_reject_)
            {
                momentum_positivity_reject_ = true;
                momentum_positivity_reason_ = diffusion_positive ?
                    "momentum positivity: positive diffusion coupling" : row_rejected ?
                    "momentum positivity: positive coupling not explained by the velocity term" :
                    "momentum positivity: non-positive diagonal or row sum";
                momentum_positivity_reject_cell_ = cells[i].ID;
            }
        }
    }
    }
    catch(UniversalError const& error)
    {
        if(!momentum_positivity)
            throw;
        momentum_positivity_reject_ = true;
        momentum_positivity_reason_ = std::string("momentum positivity: matrix assembly failed: ") + error.getErrorMessage();
        momentum_positivity_reject_cell_ = std::numeric_limits<std::size_t>::max();
    }
    catch(std::exception const& error)
    {
        if(!momentum_positivity)
            throw;
        momentum_positivity_reject_ = true;
        momentum_positivity_reason_ = std::string("momentum positivity: matrix assembly failed: ") + error.what();
        momentum_positivity_reject_cell_ = std::numeric_limits<std::size_t>::max();
    }
    catch(...)
    {
        if(!momentum_positivity)
            throw;
        momentum_positivity_reject_ = true;
        momentum_positivity_reason_ = "momentum positivity: matrix assembly failed";
        momentum_positivity_reject_cell_ = std::numeric_limits<std::size_t>::max();
    }
    if(momentum_positivity)
    {
        // One rank-0 aggregate per build (design section 5, matrix part).
        double counts[8];
        double maxima[2];
        double minima[1];
#ifdef RICH_MPI
        MPI_Allreduce(momentum_counts, counts, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(momentum_maxima, maxima, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(momentum_minima, minima, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#else
        std::copy(momentum_counts, momentum_counts + 8, counts);
        std::copy(momentum_maxima, momentum_maxima + 2, maxima);
        std::copy(momentum_minima, momentum_minima + 1, minima);
#endif
        if(rank == 0)
            std::cout << std::setprecision(4) << "RICH_RADIATION_MOMENTUM_POSITIVITY stage=matrix time=" << current_time
                      << " rows_changed=" << counts[0] << " faces_lumped=" << counts[1]
                      << " unexplained_positive_rows=" << counts[2] << " positive_diffusion_rows=" << counts[6]
                      << " certificate_violations=" << counts[3] << " boundary_row_violations=" << counts[7]
                      << " lumped_faces_alpha_gt_1=" << counts[4] << " rows_with_boundary_faces=" << counts[5]
                      << " max_w=" << maxima[0] << " max_excess_over_diagonal=" << maxima[1]
                      << " min_row_margin=" << minima[0] << std::endl;
    }
    for(size_t i = 0; i < Nlocal; ++i)
    {
        A[i].resize(max_neigh, 0);
        A_indeces[i].resize(max_neigh, max_size_t);
	if(A[i][0] < 0)
	  std::clog<<"Negative A in matrix build, density "<<cells_cgs[i].density<<" T "<<cells_cgs[i].temperature<<" fleck "<<fleck_factor[i]<<
	  " sig_P "<<sigma_planck[i]<<" dt "<<individualCellTimeStep(i, dt) * time_scale_<<" Erad "<<cells_cgs[i].Erad * cells_cgs[i].density<<std::endl;
    }
}

void Diffusion::PostCG(Tessellation3D const& tess, std::vector<Conserved3D>& extensives, double const dt, std::vector<ComputationalCell3D>& cells,
        std::vector<double>const& full_CG_result, std::vector<double> const& CG_result) const
{
    double const max_v = 0.1 * CG::speed_of_light * time_scale_ / length_scale_;
    Vector3D dummy_v;
    std::vector<size_t> neighbors;
    face_vec faces;
    size_t const N = tess.GetPointNo();
    bool const entropy = !(std::find(ComputationalCell3D::tracerNames.begin(), ComputationalCell3D::tracerNames.end(), std::string("Entropy")) ==
		ComputationalCell3D::tracerNames.end());
    size_t const entropy_index = static_cast<size_t>(std::find(ComputationalCell3D::tracerNames.begin(),
        ComputationalCell3D::tracerNames.end(), std::string("Entropy")) - ComputationalCell3D::tracerNames.begin());
    std::vector<size_t> zero_indeces;
    size_t const Nzero = zero_cells_.size();
    for(size_t i = 0; i < Nzero; ++i)
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));

    int good_end = 1;
    std::string local_failure_details;
    old_T.resize(N, 0);
    // Mean active cell volume (code units) for the failing-cell record
    // (decision 2026-09-22 D5); computed only when the detailed log or the D5
    // trace switch below is on; no effect on the update.
    // RICH_INDIVIDUAL_D5_TRACE=1 emits the D5 aggregate at production speed;
    // RICH_RUNTIME_LOG=detailed (which slows every phase ~11x on the TDE)
    // additionally prints the per-cell detail blocks.
    static bool const d5_trace = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_D5_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    double mean_active_volume = 0;
    if(RuntimeLogDetailed() || d5_trace)
    {
        std::size_t active_count = 0;
        for(size_t i = 0; i < N; ++i)
            if(individualCellActive(i))
            {
                mean_active_volume += tess.GetVolume(i);
                ++active_count;
            }
        double sum_count[2] = {mean_active_volume, static_cast<double>(active_count)};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, sum_count, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        mean_active_volume = sum_count[1] > 0 ? sum_count[0] / sum_count[1] : 0;
    }
    // Per-rank first-failure record, gathered to rank 0 below so the failing
    // population (one cell per failing rank) can be summarized, not just one
    // representative.  Plain doubles so it is trivially MPI-gatherable.
    struct D5Record
    {
        double has = 0, stage = 0, cell = 0, dt_cell = 0, nominal_dt = 0, volume = 0,
            volume_ratio = 0, e_int0 = 0, erad0 = 0, de_int = 0, derad = 0, erad_de = 0;
    };
    static_assert(sizeof(D5Record) == 12 * sizeof(double), "D5Record must be 12 doubles");
    D5Record d5{};
    // RICH_RADIATION_MOMENTUM_POSITIVITY exchange sums for the rank-0 record:
    // signed and absolute lumping energy sum e(E_i - E_j), interior pressure
    // work, relativity exchange to the gas, kinetic-energy change.
    bool const momentum_positivity_post = MomentumPositivityEnabled() && hydro_on_;
    int rank_postcg = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_postcg);
#endif
    // Collective first use, before the cell loop (zero-active ranks included).
    double const loss_fraction = momentum_positivity_post ? RadiationMomentumKineticLossFraction() : 0.5;
    // signed and absolute lumping energy delta Q, interior pressure work,
    // relativity to gas, kinetic change, capped cells, impulse dropped, impulse
    // total, kinetic energy kept in radiation by the cap, |(v_primitive -
    // v_conserved).dP_interior|, cells with no radiation energy available,
    // thermal-capped cells, energy the thermal cap moved from radiation to gas,
    // sum |coefficient closure|, boundary returned work, boundary relativity to
    // gas, sum |reservoir closure| (interior-only rows)
    double momentum_exchange_sums[17] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double momentum_minimum_scale = 1;
    double momentum_maximum_closure = 0; // max per-row |coefficient closure| / its scale
    double momentum_maximum_reservoir_closure = 0; // max per-row |reservoir closure| / cell energy
    for(size_t i = 0; i < N; ++i)
    {
        if(!individualCellActive(i))
            continue;
        double const dt_cell = individualCellTimeStep(i, dt);
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_* length_scale_;
        double const Erad0 = extensives[i].Erad;
        extensives[i].Erad = CG_result[i] * volume * time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        double const T = cells[i].temperature;
        old_T[i] = T;
        double dE = fleck_factor[i] * CG::speed_of_light * dt_cell * sigma_planck[i] * (full_CG_result[i] - T * T * T * T * CG::radiation_constant
            -0.5 * (3 - R2[i]) * std::min(max_v * max_v, ScalarProd(cells[i].velocity, cells[i].velocity)) * full_CG_result[i] * length_scale_ * length_scale_ / (CG::speed_of_light * CG::speed_of_light * time_scale_ * time_scale_)) * volume * time_scale_;
	    double old_Tr = 0;
        double compton_term = 0;
        double e_absorb = fleck_factor[i] * CG::speed_of_light * dt_cell * sigma_planck[i] * full_CG_result[i] * volume * time_scale_;
        double e_emitt = -fleck_factor[i] * CG::speed_of_light * dt_cell * sigma_planck[i] * T * T * T * T * CG::radiation_constant * volume * time_scale_;
        double e_v2 =  fleck_factor[i] * CG::speed_of_light * dt_cell * sigma_planck[i] * (-0.5 * (3 - R2[i]) * std::min(max_v * max_v, ScalarProd(cells[i].velocity, cells[i].velocity)) * full_CG_result[i] * length_scale_ * length_scale_ / (CG::speed_of_light * CG::speed_of_light * time_scale_ * time_scale_)) * volume * time_scale_;
        if(compton_on_)
        {
            double const old_Er = cells[i].Erad * cells[i].density * mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
            old_Tr = std::pow(old_Er / CG::radiation_constant, 0.25);
            double const pre_factor = fleck_factor[i] * dt_cell * time_scale_ * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light);
            compton_term = pre_factor * (old_Tr - T);
            double const theta = (fleck_factor[i] < 0.5 || std::abs(compton_term) > 1e-3) ? 1 : 0.1;
            compton_term = pre_factor * volume * (full_CG_result[i] * (old_Tr - T * (1 - theta)) - T * theta * old_Er);
            dE += pre_factor * volume * (full_CG_result[i] * (old_Tr - T * (1 - theta)) - T * theta * old_Er);
        }
        if(compton_on_)
            compton_term *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        dE *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        e_absorb *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        e_emitt *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        e_v2 *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        extensives[i].energy += dE;
        extensives[i].internal_energy += dE;
        if(extensives[i].internal_energy < 0 ||
            !std::isfinite(extensives[i].internal_energy) ||
            !(extensives[i].Erad > 0) ||
            !std::isfinite(extensives[i].Erad))
        {
            if(!RuntimeLogDetailed())
            {
                double const nominal_dt = individual_context_ != nullptr ?
                    individual_context_->nominalCellTimeStep(i) : dt_cell;
                d5 = D5Record{1, 1, static_cast<double>(cells[i].ID), dt_cell, nominal_dt,
                    tess.GetVolume(i), mean_active_volume > 0 ? tess.GetVolume(i) / mean_active_volume : 0,
                    old_e_therm, Erad0, extensives[i].internal_energy - old_e_therm,
                    extensives[i].Erad - Erad0, 0.0};
            }
            setCellLocalStepFailure(
                "negative or invalid energy during radiation update",
                cells[i].ID);
            good_end = 0;
            if(RuntimeLogDetailed())
            {
                std::ostringstream details;
                details<<"RICH_RADIATION_DETAIL stage=postcg1 cell_id="<<cells[i].ID<<std::endl;
                details<<"  state | internal_energy="<<extensives[i].internal_energy
                    <<" | T="<<T<<" | velocity="<<fastabs(cells[i].velocity)
                    <<" | mass="<<extensives[i].mass<<std::endl;
                details<<"  solve | CG_result="<<CG_result[i]
                    <<" | full_CG_result="<<full_CG_result[i]
                    <<" | old_Er="<<cells[i].Erad * cells[i].density * mass_scale_ /
                        (time_scale_ * time_scale_ * length_scale_)<<std::endl;
                details<<"  energy | dE="<<dE<<" | R2="<<R2[i]
                    <<" | old_e_therm="<<old_e_therm
                    <<" | compton_term="<<compton_term
                    <<" | old_Tr="<<old_Tr<<std::endl;
                {
                    double const nominal_dt = individual_context_ != nullptr ?
                        individual_context_->nominalCellTimeStep(i) : dt_cell;
                    d5 = D5Record{1, 1, static_cast<double>(cells[i].ID), dt_cell, nominal_dt,
                        tess.GetVolume(i), mean_active_volume > 0 ? tess.GetVolume(i) / mean_active_volume : 0,
                        old_e_therm, Erad0, extensives[i].internal_energy - old_e_therm,
                        extensives[i].Erad - Erad0, 0.0};
                    details<<"  d5 | dt_cell="<<dt_cell<<" | nominal_dt="<<nominal_dt
                        <<" | applied_over_nominal="<<(nominal_dt > 0 ? dt_cell / nominal_dt : 0)
                        <<" | volume="<<tess.GetVolume(i)
                        <<" | mean_active_volume="<<mean_active_volume
                        <<" | volume_ratio="<<d5.volume_ratio
                        <<" | E_int0="<<old_e_therm<<" | Erad0="<<Erad0
                        <<" | dE_int="<<d5.de_int<<" | dErad="<<d5.derad
                        <<" | Erad_dE="<<0.0<<" | e_absorb="<<e_absorb<<" | e_emitt="<<e_emitt
                        <<" | fleck="<<fleck_factor[i]<<std::endl;
                }
                details<<"  cell | "<<cells[i]<<std::endl;
                details<<"  extensive | "<<extensives[i]<<std::endl;
                local_failure_details = details.str();
            }
            break;
        }

        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        faces = tess.GetCellFaces(i);
        Vector3D const point = tess.GetMeshPoint(i);
        Vector3D gradE(0, 0, 0);
        double const Dcell = D[i];
        Vector3D r_ij;
        Vector3D const CM = tess.GetCellCM(i);
        double const Erad_factor = mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
        double const cell_old_Er = Erad_factor * cells[i].Erad * cells[i].density;

        double total_relativity = 0;
        double etherm_mid = extensives[i].internal_energy;
        double const v_ratio = std::min(1.0, 0.05 * CG::speed_of_light / (fastabs(cells[i].velocity) * length_scale_ / time_scale_ + 1e-2));
        // RICH_RADIATION_MOMENTUM_POSITIVITY (design section 3): interior faces
        // use the lumped face value E'_f and the assembled face timestep for
        // both the relativity exchange and the radiation-force impulse; their
        // pressure work W'_i is returned to the radiation explicitly below.
        // Boundary faces keep today's form (gradE_boundary, dt_cell).
        bool const momentum_exchange = momentum_positivity_post && i < momentum_face_weight_.size();
        std::vector<double> const* const face_weights = momentum_exchange ? &momentum_face_weight_[i] : nullptr;
        // The matrix's v_ratio, so the relativity exchange carries the matrix's alpha exactly.
        double const matrix_v_ratio = momentum_exchange && i < momentum_v_ratio_.size() ? momentum_v_ratio_[i] : v_ratio;
        double const matrix_alpha = 6 * fleck_factor[i] * matrix_v_ratio * sigma_planck[i] * Dcell / CG::speed_of_light;
        std::vector<double> const* const face_terms = momentum_exchange && i < momentum_face_term_.size() ?
            &momentum_face_term_[i] : nullptr;
        std::vector<double> const* const face_excess = momentum_exchange && i < momentum_face_excess_.size() ?
            &momentum_face_excess_[i] : nullptr;
        // Reservoir totals at the start of the velocity exchange (radiation as solved,
        // gas after absorption/emission), for the reservoir closure below.
        double const reservoir_start = momentum_exchange ? extensives[i].Erad + extensives[i].internal_energy +
            0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass : 0.0;
        double matrix_action_total = 0;
        Vector3D gradE_boundary(0, 0, 0);
        Vector3D dP_interior(0, 0, 0);
        double work_interior = 0;
        // Coefficient closure: the assembled matrix's velocity action on the
        // solution, sum T_f (E_i + E_j) + e_f (E_i - E_j) with the stored
        // coefficients and lumped excesses (erg, converted to code energy), minus
        // the exchange's (1 - alpha) W'; it checks the weights and alpha.
        double closure = 0, closure_scale = 0, boundary_relativity = 0;
        double const code_energy = time_scale_ * time_scale_ / (mass_scale_ * length_scale_ * length_scale_);
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            r_ij = point - tess.GetMeshPoint(neighbor_j);
            double const r_ij_size = abs(r_ij);
            r_ij *= 1.0 / r_ij_size;
            double Er_j = 0;
            bool const boundary_face = tess.IsPointOutsideBox(neighbor_j);
            if(boundary_face)
                boundary_calc_.GetOutSideValues(tess, cells, i, neighbor_j, full_CG_result, Er_j, dummy_v);
            else
                Er_j = full_CG_result[neighbor_j];
  

            gradE += (0.5 * tess.GetArea(faces[j]) * (Er_j + full_CG_result[i])) * r_ij * length_scale_ * length_scale_;
            if(momentum_exchange && !boundary_face)
            {
                double const w = face_weights != nullptr && j < face_weights->size() ? (*face_weights)[j] : 0.0;
                double const E_face = 0.5 * ((1 + w) * full_CG_result[i] + (1 - w) * Er_j);
                double const dt_face = individualFaceTimeStep(i, neighbor_j, dt);
                double const momentum_term = (dt_face * cell_flux_limiter[i] * tess.GetArea(faces[j]) * ScalarProd(cells[i].velocity, r_ij) * E_face / 3) * (time_scale_ * time_scale_ * length_scale_ / mass_scale_);
                double const central_term = (dt_face * cell_flux_limiter[i] * tess.GetArea(faces[j]) * ScalarProd(cells[i].velocity, r_ij) * 0.5 * (Er_j + full_CG_result[i]) / 3) * (time_scale_ * time_scale_ * length_scale_ / mass_scale_);
                double const relativity_term = -matrix_v_ratio * momentum_term * 2 * 3 * sigma_planck[i] * Dcell / CG::speed_of_light;
                extensives[i].energy += fleck_factor[i] * relativity_term;
                extensives[i].internal_energy += fleck_factor[i] * relativity_term;
                total_relativity += fleck_factor[i] * relativity_term;
                work_interior += momentum_term;
                dP_interior += ((cell_flux_limiter[i] * dt_face * time_scale_ / 3) * tess.GetArea(faces[j]) * E_face * length_scale_ * length_scale_) * r_ij * (time_scale_ / (length_scale_ * mass_scale_));
                // delta Q = (1 - alpha) (W' - W): the radiation energy the lumping moved.
                double const lumping_energy = (1 - matrix_alpha) * (momentum_term - central_term);
                double const T_face = face_terms != nullptr && j < face_terms->size() ? (*face_terms)[j] : 0.0;
                double const e_face = face_excess != nullptr && j < face_excess->size() ? (*face_excess)[j] : 0.0;
                double const matrix_action = (T_face * (full_CG_result[i] + Er_j) + e_face * (full_CG_result[i] - Er_j)) *
                    code_energy;
                matrix_action_total += matrix_action;
                closure += matrix_action - (1 - matrix_alpha) * momentum_term;
                closure_scale += std::abs(matrix_action) + std::abs(momentum_term);
                momentum_exchange_sums[0] += lumping_energy;
                momentum_exchange_sums[1] += std::abs(lumping_energy);
                momentum_exchange_sums[3] += fleck_factor[i] * relativity_term;
                continue;
            }
            if(momentum_exchange)
                gradE_boundary += (0.5 * tess.GetArea(faces[j]) * (Er_j + full_CG_result[i])) * r_ij * length_scale_ * length_scale_;
            double const momentum_term = (0.5 * dt_cell * cell_flux_limiter[i] * tess.GetArea(faces[j]) * ScalarProd(cells[i].velocity, r_ij) * (Er_j + full_CG_result[i]) / 3) * (time_scale_ * time_scale_ * length_scale_ / mass_scale_);
            double const relativity_term = -v_ratio * momentum_term * 2 * 3 * sigma_planck[i] * Dcell / CG::speed_of_light;
            extensives[i].energy += /*momentum_term + */fleck_factor[i] * relativity_term;
            extensives[i].internal_energy += fleck_factor[i] * relativity_term;
            total_relativity += fleck_factor[i] * relativity_term;
            if(momentum_exchange)
                boundary_relativity += fleck_factor[i] * relativity_term;
        }
        if(momentum_exchange)
        {
            momentum_exchange_sums[13] += std::abs(closure);
            momentum_exchange_sums[15] += boundary_relativity;
            if(closure_scale > 0)
                momentum_maximum_closure = std::max(momentum_maximum_closure, std::abs(closure) / closure_scale);
            // Thermal cap (energy-conserving): the relativistic exchange may take
            // at most the loss fraction of the gas internal energy the cell has
            // after absorption and emission (etherm_mid).  In near-vacuum cells
            // the stiff exchange (fleck*dt saturated) can otherwise exceed it at
            // any timestep; the excess is paid by the cell's radiation instead.
            double const floor_energy = (1 - loss_fraction) * etherm_mid;
            if(etherm_mid > 0 && extensives[i].internal_energy < floor_energy)
            {
                double const shift = floor_energy - extensives[i].internal_energy;
                extensives[i].internal_energy = floor_energy;
                extensives[i].energy += shift;
                extensives[i].Erad -= shift;
                momentum_exchange_sums[11] += 1;
                momentum_exchange_sums[12] += shift;
            }
        }
        Vector3D dP;
        double Erad_dE = 0;
        double cap_available = 0, cap_linear = 0, cap_quadratic = 0, cap_scale = 1; // failure trace only
        if(hydro_on_ && momentum_exchange)
        {
            // Exact exchange (design section 3): radiation gets back the interior
            // pressure work W'_i and the boundary work v.dP_boundary, and pays the
            // actual kinetic-energy change; no uncompensated undo.
            double const mass = extensives[i].mass;
            double const old_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass;
            Vector3D const dP_boundary = (cell_flux_limiter[i] * dt_cell * time_scale_ / 3) * gradE_boundary * (time_scale_ / (length_scale_ * mass_scale_));
            dP = dP_interior + dP_boundary;
            double const boundary_work = ScalarProd(dP_boundary, extensives[i].momentum) / mass;
            Erad_dE = work_interior + boundary_work;
            // Kinetic cap (energy-conserving): the kinetic change of an impulse
            // s*dP is s*linear + s^2*quadratic.  When the full impulse would take
            // more than the loss fraction of the radiation energy the cell has
            // (after the returned work), the impulse is scaled to the largest s
            // that takes exactly that fraction.  Energy stays exact for any s;
            // the dropped impulse is momentum the gas does not receive.
            double const linear = ScalarProd(dP, extensives[i].momentum) / mass;
            double const quadratic = 0.5 * ScalarProd(dP, dP) / mass;
            double const available = extensives[i].Erad + Erad_dE;
            double scale = 1;
            bool cap_valid = std::isfinite(linear) && std::isfinite(quadratic) && std::isfinite(available) &&
                quadratic >= 0;
            double cap_magnitude = 0; // scale of the cap's coefficients (0: not evaluated)
            double scaled_withheld = 0; // kinetic energy kept in radiation / cap_magnitude
            if(cap_valid && available > 0)
            {
                // Everything is evaluated with the coefficients divided by their
                // largest magnitude m (one of a, |b|, c is then 1), so no sum,
                // product or quotient below can overflow or divide 0/0 (the entry
                // point traps FP overflow and invalid operations).
                double const budget = loss_fraction * available;
                cap_magnitude = std::max(std::max(std::abs(linear), quadratic), budget);
                // m = 0 only with no impulse work and a budget that underflowed: nothing to cap.
                double const a = cap_magnitude > 0 ? quadratic / cap_magnitude : 0.0;
                double const b = cap_magnitude > 0 ? linear / cap_magnitude : 0.0;
                double const c = cap_magnitude > 0 ? budget / cap_magnitude : 0.0;
                if(a + b > c)
                {
                    // Largest s in [0, 1] with a s^2 + b s = c, cancellation-free for
                    // each sign of b.  b < 0: a > c - b > 0, so a = 1 after scaling
                    // and (D - b) / (2a) is safe for any c >= 0.  b >= 0: if c
                    // underflowed below the smallest normal number, the root is below
                    // max(c / b, sqrt(c / a)) < 1.5e-154 (one of a, b is 1) and s = 0
                    // is used, keeping energy exact; otherwise b + D > 0.
                    double root = 0;
                    if(b < 0 || c >= std::numeric_limits<double>::min())
                    {
                        double const discriminant = std::hypot(b, 2 * std::sqrt(a * c));
                        root = b >= 0 ? 2 * c / (b + discriminant) : (discriminant - b) / (2 * a);
                    }
                    if(std::isfinite(root) && root >= 0)
                    {
                        scale = std::min(1.0, root);
                        scaled_withheld = (1 - scale) * b + (1 - scale * scale) * a;
                    }
                    else
                        cap_valid = false;
                }
            }
            else if(cap_valid)
                momentum_exchange_sums[10] += 1;
            if(!cap_valid)
            {
                setCellLocalStepFailure("momentum positivity: invalid kinetic cap", cells[i].ID);
                good_end = 0;
                break;
            }
            if(scale < 1)
            {
                momentum_exchange_sums[5] += 1;
                momentum_exchange_sums[6] += (1 - scale) * fastabs(dP);
                // Ledger only; skipped for coefficients too large to multiply back safely.
                if(cap_magnitude < 1e300)
                    momentum_exchange_sums[8] += scaled_withheld * cap_magnitude;
                momentum_minimum_scale = std::min(momentum_minimum_scale, scale);
            }
            momentum_exchange_sums[7] += fastabs(dP);
            cap_available = available;
            cap_linear = linear;
            cap_quadratic = quadratic;
            cap_scale = scale;
            momentum_exchange_sums[9] += std::abs(work_interior - ScalarProd(dP_interior, extensives[i].momentum) / mass);
            dP *= scale;
            extensives[i].momentum += dP;
            // Radiation pays the rounded kinetic change of the updated momentum
            // (exact bookkeeping); if rounding exhausts the budget, the final
            // Erad > 0 check below rejects the candidate.
            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass;
            double const dE = -new_Ek + old_Ek + Erad_dE;
            extensives[i].Erad += dE;
            extensives[i].energy = extensives[i].internal_energy +  ScalarProd(extensives[i].momentum, extensives[i].momentum) / (2 * mass);
            momentum_exchange_sums[2] += work_interior;
            momentum_exchange_sums[4] += new_Ek - old_Ek;
            momentum_exchange_sums[14] += boundary_work;
            // Reservoir closure (interior-only rows): the actual change of
            // radiation + internal + kinetic energy through the velocity exchange,
            // thermal and kinetic caps included, must equal the matrix action.
            if(i < momentum_row_boundary_.size() && momentum_row_boundary_[i] == 0)
            {
                double const reservoir_end = extensives[i].Erad + extensives[i].internal_energy + new_Ek;
                double const reservoir_closure = reservoir_end - reservoir_start - matrix_action_total;
                double const reservoir_scale = std::abs(reservoir_start) + std::abs(matrix_action_total);
                momentum_exchange_sums[16] += std::abs(reservoir_closure);
                if(reservoir_scale > 0)
                    momentum_maximum_reservoir_closure = std::max(momentum_maximum_reservoir_closure,
                        std::abs(reservoir_closure) / reservoir_scale);
            }
        }
        else if(hydro_on_)
        {
            double const old_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
            dP = (cell_flux_limiter[i] * dt_cell * time_scale_ / 3) * gradE * (time_scale_ / (length_scale_ * mass_scale_));
            Erad_dE = ScalarProd(dP, extensives[i].momentum) / extensives[i].mass;
            extensives[i].momentum += dP;
            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
            double const dE = -new_Ek + old_Ek + Erad_dE;
            extensives[i].Erad += dE;
            if(extensives[i].Erad < 0 && dE < 0 && std::abs(dE) < extensives[i].energy * 0.01)
                 extensives[i].Erad -= dE;
            extensives[i].energy = extensives[i].internal_energy +  ScalarProd(extensives[i].momentum, extensives[i].momentum) / (2 * extensives[i].mass);
        }
        if(!(extensives[i].Erad > 0) ||
            !std::isfinite(extensives[i].Erad) ||
            extensives[i].internal_energy < 0 ||
            !std::isfinite(extensives[i].internal_energy) ||
            !(cells[i].Erad > 0) ||
            !std::isfinite(cells[i].Erad))
        {
            if(RuntimeLogDetailed())
            {
                std::ostringstream details;
                details<<"RICH_RADIATION_DETAIL stage=postcg2 cell_id="<<cells[i].ID<<std::endl;
                details<<"  state | internal_energy="<<extensives[i].internal_energy
                    <<" | Erad="<<extensives[i].Erad<<" | T="<<T
                    <<" | velocity="<<fastabs(cells[i].velocity)
                    <<" | mass="<<extensives[i].mass<<std::endl;
                details<<"  solve | CG_result="<<CG_result[i]
                    <<" | full_CG_result="<<full_CG_result[i]
                    <<" | sigma_planck="<<sigma_planck[i]
                    <<" | sigma_r="<<CG::speed_of_light / (3 * Dcell)<<std::endl;
                {
                    double const nominal_dt = individual_context_ != nullptr ?
                        individual_context_->nominalCellTimeStep(i) : dt_cell;
                    d5 = D5Record{1, 2, static_cast<double>(cells[i].ID), dt_cell, nominal_dt,
                        tess.GetVolume(i), mean_active_volume > 0 ? tess.GetVolume(i) / mean_active_volume : 0,
                        old_e_therm, Erad0, extensives[i].internal_energy - old_e_therm,
                        extensives[i].Erad - Erad0, Erad_dE};
                    details<<"  d5 | dt_cell="<<dt_cell<<" | nominal_dt="<<nominal_dt
                        <<" | applied_over_nominal="<<(nominal_dt > 0 ? dt_cell / nominal_dt : 0)
                        <<" | volume="<<tess.GetVolume(i)
                        <<" | mean_active_volume="<<mean_active_volume
                        <<" | volume_ratio="<<d5.volume_ratio
                        <<" | E_int0="<<old_e_therm<<" | Erad0="<<Erad0
                        <<" | dE_int="<<d5.de_int<<" | dErad="<<d5.derad
                        <<" | Erad_dE="<<Erad_dE<<" | e_absorb="<<e_absorb<<" | e_emitt="<<e_emitt
                        <<" | fleck="<<fleck_factor[i]<<std::endl;
                }
                details<<"  energy | Erad_dE="<<Erad_dE
                    <<" | e_absorb="<<e_absorb<<" | e_emitt="<<e_emitt
                    <<" | e_v2="<<e_v2<<" | total_relativity="<<total_relativity
                    <<" | etherm_mid="<<etherm_mid<<std::endl;
                details<<"  motion | dP="<<dP.x<<","<<dP.y<<","<<dP.z
                    <<" | momentum="<<extensives[i].momentum.x<<","<<extensives[i].momentum.y<<","<<extensives[i].momentum.z
                    <<" | flux_limiter="<<cell_flux_limiter[i]
                    <<" | gradE="<<gradE * (1.0 / tess.GetVolume(i))<<std::endl;
                for(size_t j = 0; j < Nneigh; ++j)
                {
                    size_t const neighbor_j = neighbors[j];
                    r_ij = point - tess.GetMeshPoint(neighbor_j);
                    double const r_ij_size = abs(r_ij);
                    r_ij *= 1.0 / r_ij_size;
                    double Er_j = 0;
                    if(tess.IsPointOutsideBox(neighbor_j))
                        boundary_calc_.GetOutSideValues(tess, cells, i, neighbor_j, CG_result, Er_j, dummy_v);
                    else
                        Er_j = full_CG_result[neighbor_j];

                    Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                    Vector3D const grad_E = r_ij * ScalarProd(r_ij, cm_ij) * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));
                    double mid_D = 0.5 * (D[neighbor_j] + Dcell);
                    double const flux_limiter_face = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * (CG_result[i] - Er_j), mid_D, 0.5 * (CG_result[i] + Er_j)) : 1;
                    double const max_local_v = std::min(max_v, std::max(-max_v, ScalarProd(cells[i].velocity, r_ij)));
                    double const v_ratio = max_local_v / ScalarProd(cells[i].velocity, r_ij);
                    double const momentum_term = (0.5 * dt_cell * cell_flux_limiter[i] * tess.GetArea(faces[j]) * ScalarProd(cells[i].velocity, r_ij) * (Er_j + full_CG_result[i]) / 3) * (time_scale_ * time_scale_ * length_scale_ / mass_scale_);
                    double const relativity_term = -v_ratio * fleck_factor[i] * momentum_term * 2 * 3 * sigma_planck[i] * Dcell / CG::speed_of_light;
                    details<<"  neighbor | cell_id="<<cells[neighbor_j].ID
                        <<" | relativity_term="<<relativity_term
                        <<" | flux_limiter="<<flux_limiter_face
                        <<" | Er="<<Er_j
                        <<" | initial_Er="<<cells[neighbor_j].density * cells[neighbor_j].Erad * mass_scale_ /
                            (time_scale_ * time_scale_ * length_scale_)
                        <<" | v_ratio="<<v_ratio
                        <<" | area="<<tess.GetArea(faces[j])<<std::endl;
                }
                local_failure_details = details.str();
            }

            if(!RuntimeLogDetailed())
            {
                double const nominal_dt = individual_context_ != nullptr ?
                    individual_context_->nominalCellTimeStep(i) : dt_cell;
                d5 = D5Record{1, 2, static_cast<double>(cells[i].ID), dt_cell, nominal_dt,
                    tess.GetVolume(i), mean_active_volume > 0 ? tess.GetVolume(i) / mean_active_volume : 0,
                    old_e_therm, Erad0, extensives[i].internal_energy - old_e_therm,
                    extensives[i].Erad - Erad0, Erad_dE};
            }
            // RICH_INDIVIDUAL_D5_TRACE with RICH_RADIATION_MOMENTUM_POSITIVITY: one
            // rank-local line per failing rank with the terms of the final energies.
            if(d5_trace && momentum_exchange)
                std::cout << std::setprecision(6) << "RICH_RADIATION_MOMENTUM_FAILURE rank=" << rank_postcg
                          << " cell_id=" << cells[i].ID << " Erad=" << extensives[i].Erad
                          << " internal_energy=" << extensives[i].internal_energy << " Erad0=" << Erad0
                          << " e_int0=" << old_e_therm << " cell_Erad=" << cells[i].Erad << " mass=" << extensives[i].mass
                          << " Erad_dE=" << Erad_dE << " total_relativity=" << total_relativity
                          << " e_absorb=" << e_absorb << " e_emitt=" << e_emitt << " e_v2=" << e_v2
                          << " cap_available=" << cap_available << " cap_linear=" << cap_linear
                          << " cap_quadratic=" << cap_quadratic << " cap_scale=" << cap_scale
                          << " fleck=" << fleck_factor[i] << " dt_cell=" << dt_cell << std::endl;
            setCellLocalStepFailure(
                "negative or invalid energy after radiation update",
                cells[i].ID);
            good_end = 0;
            break;
        }

        double const old_Edot = cells[i].Erad_dt;
        cells[i].Erad_dt = (extensives[i].Erad / extensives[i].mass - cells[i].Erad) / dt_cell;
        cells[i].Erad_dt_dt = (cells[i].Erad_dt - old_Edot) / dt_cell;
        if(!std::isfinite(cells[i].Erad_dt) || !std::isfinite(cells[i].Erad_dt_dt))
        {
            std::clog<<"Bad Edot "<<cells[i].Erad_dt<<" edot_dt_dt "<<cells[i].Erad_dt_dt<<" i "<<i<<" ID "<<cells[i].ID<<" dt "<<dt_cell<<" Erad "<<extensives[i].Erad<<" m "<<extensives[i].mass<<" old_Edot "<<old_Edot<<std::endl;
            setCellLocalStepFailure(
                "non-finite radiation derivative after radiation update",
                cells[i].ID);
            good_end = 0;
            break;
        }
        cells[i].Erad = extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy = extensives[i].internal_energy / extensives[i].mass;
        try
        {
            cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].pressure = eos_.de2p(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].velocity = extensives[i].momentum / extensives[i].mass;    
            if(entropy)
            {
                cells[i].tracers[entropy_index] = eos_.dp2s(cells[i].density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
                extensives[i].tracers[entropy_index] = cells[i].tracers[entropy_index] * extensives[i].mass;
            }
        }
        catch(UniversalError &eo)
        {
            if(RuntimeLogDetailed())
            {
                std::ostringstream details;
                details<<"RICH_RADIATION_DETAIL stage=eos cell_id="<<cells[i].ID<<std::endl;
                details<<"  error | message="<<eo.getErrorMessage()<<std::endl;
                details<<"  state | density="<<cells[i].density
                    <<" | internal_energy="<<cells[i].internal_energy
                    <<" | Erad="<<cells[i].Erad<<std::endl;
                local_failure_details = details.str();
            }
            setCellLocalStepFailure(eo.getErrorMessage(), cells[i].ID);
            good_end = 0;
            break;
        }
    }
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    if(momentum_positivity_post)
    {
        // One rank-0 aggregate per candidate (design section 5, exchange part).
        double sums[17];
        double minima[2] = {momentum_minimum_scale, momentum_minimum_verification_rhs_};
        double maxima[2] = {momentum_maximum_closure, momentum_maximum_reservoir_closure};
#ifdef RICH_MPI
        MPI_Allreduce(momentum_exchange_sums, sums, 17, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, minima, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
        std::copy(momentum_exchange_sums, momentum_exchange_sums + 17, sums);
#endif
        double const minimum_scale = minima[0];
        // stage_success: this PostCG stage only; later candidate gates may still reject.
        if(rank == 0)
        {
            std::cout << std::setprecision(6) << "RICH_RADIATION_MOMENTUM_POSITIVITY stage=exchange stage_success="
                      << good_end << " lumping_energy=" << sums[0] << " lumping_energy_abs=" << sums[1]
                      << " interior_pressure_work=" << sums[2] << " relativity_to_gas=" << sums[3]
                      << " kinetic_change=" << sums[4] << " kinetic_capped_cells=" << sums[5]
                      << " impulse_dropped=" << sums[6] << " impulse_total=" << sums[7]
                      << " kinetic_kept_in_radiation=" << sums[8] << " min_impulse_scale=" << minimum_scale
                      << " velocity_mismatch_work=" << sums[9] << " no_radiation_available_cells=" << sums[10]
                      << " thermal_capped_cells=" << sums[11] << " thermal_cap_energy=" << sums[12]
                      << " coefficient_closure_abs=" << sums[13] << " max_row_coefficient_closure=" << maxima[0]
                      << " reservoir_closure_abs=" << sums[16] << " max_row_reservoir_closure=" << maxima[1]
                      << " boundary_work=" << sums[14] << " boundary_relativity_to_gas=" << sums[15]
                      << " min_verification_rhs=";
            if(std::isfinite(minima[1]))
                std::cout << minima[1];
            else
                std::cout << "none";
            std::cout << std::endl;
        }
    }
    if((RuntimeLogDetailed() || d5_trace) && good_end == 0)
    {
        int world_size = 1;
#ifdef RICH_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
        std::vector<D5Record> all_d5(rank == 0 ? static_cast<std::size_t>(world_size) : 1);
#ifdef RICH_MPI
        MPI_Gather(&d5, 12, MPI_DOUBLE, all_d5.data(), 12, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
        all_d5[0] = d5;
#endif
        if(rank == 0)
        {
            std::vector<D5Record> failing;
            for(D5Record const& record : all_d5)
                if(record.has > 0)
                    failing.push_back(record);
            auto median_of = [&](double D5Record::* field)
            {
                std::vector<double> values;
                for(D5Record const& record : failing)
                    values.push_back(record.*field);
                if(values.empty())
                    return 0.0;
                std::sort(values.begin(), values.end());
                return values[values.size() / 2];
            };
            std::size_t below_nominal = 0, de_int_negative = 0, derad_negative = 0, stage1 = 0;
            double ratio_min = std::numeric_limits<double>::infinity(), ratio_max = 0;
            for(D5Record const& record : failing)
            {
                if(record.dt_cell < record.nominal_dt * (1 - 1e-9))
                    ++below_nominal;
                if(record.de_int < 0)
                    ++de_int_negative;
                if(record.derad < 0)
                    ++derad_negative;
                if(record.stage == 1)
                    ++stage1;
                ratio_min = std::min(ratio_min, record.volume_ratio);
                ratio_max = std::max(ratio_max, record.volume_ratio);
            }
            std::cout << std::setprecision(6)
                      << "RICH_RADIATION_D5_AGGREGATE failing_ranks=" << failing.size()
                      << " of=" << world_size
                      << " stage_during=" << stage1
                      << " stage_after=" << (failing.size() - stage1)
                      << " volume_ratio_median=" << median_of(&D5Record::volume_ratio)
                      << " volume_ratio_min=" << (failing.empty() ? 0 : ratio_min)
                      << " volume_ratio_max=" << ratio_max
                      << " dt_cell_median=" << median_of(&D5Record::dt_cell)
                      << " nominal_dt_median=" << median_of(&D5Record::nominal_dt)
                      << " below_nominal=" << below_nominal
                      << " de_int_negative=" << de_int_negative
                      << " derad_negative=" << derad_negative
                      << " e_int0_median=" << median_of(&D5Record::e_int0)
                      << " erad0_median=" << median_of(&D5Record::erad0)
                      << std::endl;
        }
        if(RuntimeLogDetailed())
        {
#ifdef RICH_MPI
        int detail_rank = local_failure_details.empty() ?
            std::numeric_limits<int>::max() : rank;
        MPI_Allreduce(MPI_IN_PLACE, &detail_rank, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD);
        if(detail_rank != std::numeric_limits<int>::max())
        {
            unsigned long long detail_size = rank == detail_rank ?
                static_cast<unsigned long long>(local_failure_details.size()) : 0;
            MPI_Bcast(&detail_size, 1, MPI_UNSIGNED_LONG_LONG, detail_rank,
                MPI_COMM_WORLD);
            if(detail_size > static_cast<unsigned long long>(
                std::numeric_limits<int>::max()))
                throw std::overflow_error(
                    "Radiation failure diagnostic is too large for MPI");
            if(rank != detail_rank)
                local_failure_details.resize(
                    static_cast<std::size_t>(detail_size));
            if(detail_size > 0)
                MPI_Bcast(&local_failure_details[0],
                    static_cast<int>(detail_size), MPI_CHAR, detail_rank,
                    MPI_COMM_WORLD);
        }
#endif
        if(rank == 0 && !local_failure_details.empty())
            std::cout<<local_failure_details<<std::endl;
        }
    }
    if(good_end == 0)
    {
        throw UniversalError("Negative energy in POSTCG");
    }

}

void DiffusionSideBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
        std::vector<ComputationalCell3D> const& /*cells*/, double const Area, double& A, double &b, size_t const /*face_index*/)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += 0.5 * CG::speed_of_light * dt * Area;
        b += 2 * Area * dt * CG::stefan_boltzman * T_ * T_ * T_ * T_;
    }
}

void DiffusionSideBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, double const fleck_factor,
        double const flux_limiter, double const D, double const sigma_planck)const
{
    double const R = tess.GetWidth(index);
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += momentum_relativity_term;
        b -= momentum_relativity_term * CG::radiation_constant * T_ * T_ * T_ * T_;
    }
    else
        A += 2 * momentum_relativity_term;
}

void DiffusionSideBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
        E_outside = CG::radiation_constant * T_ * T_ * T_ * T_;
    else
        E_outside = new_E[index];
    v_outside = cells[index].velocity;
}

bool DiffusionDirichletBoundary::IsLeftXBoundary(
    Tessellation3D const& tess, size_t const index, size_t const outside_point) const
{
    double const R = tess.GetWidth(index);
    return tess.GetMeshPoint(index).x > tess.GetMeshPoint(outside_point).x + R * 1e-4;
}

double DiffusionDirichletBoundary::BoundaryEnergyDensity() const
{
    return CG::radiation_constant * T_ * T_ * T_ * T_;
}

double DiffusionDirichletBoundary::CalcFaceDiffusionCoefficient(
    std::vector<ComputationalCell3D> const& cells, size_t const index) const
{
    ComputationalCell3D boundary_cell = cells[index];
    boundary_cell.density = cells[index].density;
    boundary_cell.temperature = T_;
    boundary_cell.Erad = BoundaryEnergyDensity() / std::max(boundary_cell.density, 1e-300);

    double const D_cell = D_calc_.CalcDiffusionCoefficient(cells[index]);
    double const D_boundary = D_calc_.CalcDiffusionCoefficient(boundary_cell);

    return 2.0 * D_cell * D_boundary / std::max(D_cell + D_boundary, 1e-300);
}

void DiffusionDirichletBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const /*face_index*/)const
{
    if (!IsLeftXBoundary(tess, index, outside_point))
        return;

    double const E_boundary = BoundaryEnergyDensity();
    double const dx = std::max(
        tess.GetCellCM(index).x - tess.GetBoxCoordinates().first.x, 1e-200);

    double const D_face = CalcFaceDiffusionCoefficient(cells, index);
    double const coef = dt * D_face * Area / dx;

    A += coef;
    b += coef * E_boundary;
}

void DiffusionDirichletBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const /*face_index*/, double const fleck_factor,
        double const flux_limiter, double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = std::max(abs(r_ij), 1e-200);
    r_ij *= 1.0 / r_ij_size;

    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2.0 * 3.0 * sigma_planck * D / CG::speed_of_light - 1.0) * ScalarProd(cell.velocity, r_ij) / 3.0;

    if (IsLeftXBoundary(tess, index, outside_point))
    {
        double const E_boundary = BoundaryEnergyDensity();
        A += momentum_relativity_term;
        b -= momentum_relativity_term * E_boundary;
    }
    else
    {
        A += 2.0 * momentum_relativity_term;
    }
}

void DiffusionDirichletBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    if (IsLeftXBoundary(tess, index, outside_point))
    {
        E_outside = BoundaryEnergyDensity();
        v_outside = cells[index].velocity;
        return;
    }

    E_outside = new_E[index];
    Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
    v_outside = cells[index].velocity;
    v_outside -= 2.0 * normal * ScalarProd(normal, v_outside);
}

void DiffusionClosedBox::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, 
        double& /*b*/, size_t const /*face_index*/, double const fleck_factor, double const flux_limiter, 
        double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    A += 2 * momentum_relativity_term;
}

void DiffusionClosedBox::SetBoundaryValues(Tessellation3D const& /*tess*/, size_t const /*index*/, size_t const /*outside_point*/, double const /*dt*/, 
        std::vector<ComputationalCell3D> const& /*cells*/, double const /*Area*/, double& /*A*/, double& /*b*/, size_t const /*face_index*/)const
{}

void DiffusionClosedBox::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    E_outside = new_E[index];
    Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
    v_outside = cells[index].velocity;
    v_outside -= 2 * normal * ScalarProd(normal, v_outside);
}

double PowerLawOpacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell) const
{
    return D0_ * std::pow(cell.density, alpha_) * std::pow(cell.temperature, beta_);
}

double PowerLawOpacity::CalcPlanckOpacity(ComputationalCell3D const& cell) const
{
    return planck0_ * std::pow(cell.density, alpha_planck_) * std::pow(cell.temperature, beta_planck_);
}

void DiffusionXInflowBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        double const Er_j = left_state_.Erad * left_state_.density;
        double const Er = cells[index].Erad * cells[index].density;
        double const dx = tess.GetCellCM(index).x - tess.GetBoxCoordinates().first.x;
        Vector3D const grad_E = Vector3D(1, 0, 0) * (1.0 / (2 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
        double mid_D = 0.5 * (D_calc_.CalcDiffusionCoefficient(cells[index]) + D_calc_.CalcDiffusionCoefficient(left_state_));
        double const flux_limiter = CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j));
        mid_D *= flux_limiter;
        double const flux = ScalarProd(grad_E, r_ij * (tess.GetArea(face_index) / abs(r_ij))) * dt * mid_D; 
        A += flux;
        b += flux * Er_j;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            double const Er_j = right_state_.Erad * right_state_.density;
            double const Er = cells[index].Erad * cells[index].density;
            double const dx = tess.GetBoxCoordinates().second.x - tess.GetCellCM(index).x;
            Vector3D const grad_E = Vector3D(-1, 0, 0) * (1.0 / (2 * dx));
            Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
            double mid_D = 0.5 * (D_calc_.CalcDiffusionCoefficient(cells[index]) + D_calc_.CalcDiffusionCoefficient(right_state_));
            double const flux_limiter = CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j));
            mid_D *= flux_limiter;
            double const flux = ScalarProd(grad_E, r_ij * (tess.GetArea(face_index) / abs(r_ij))) * dt * mid_D; 
            A += flux;
            b += flux * Er_j;
        }
    }
}

void DiffusionXInflowBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, double const fleck_factor,
        double const flux_limiter, double const D, double const sigma_planck)const
{
    double const R = tess.GetWidth(index);
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += momentum_relativity_term;
        b -= momentum_relativity_term * left_state_.Erad * left_state_.density;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            A += momentum_relativity_term;
            b -= momentum_relativity_term * right_state_.Erad * right_state_.density;
        }
        else
            A += 2 * momentum_relativity_term;
    }
}

void DiffusionXInflowBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        E_outside = left_state_.Erad * left_state_.density;
        v_outside = left_state_.velocity;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            E_outside = right_state_.Erad * right_state_.density;
            v_outside = right_state_.velocity;
        }
        else
        {
            E_outside = new_E[index];
            Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
            v_outside = cells[index].velocity;
            v_outside -= 2 * normal * ScalarProd(normal, v_outside);
        }
    }
}

void DiffusionOpenBoundary::SetBoundaryValues(Tessellation3D const& /*tess*/, size_t const /*index*/, size_t const /*outside_point*/, double const dt,
    std::vector<ComputationalCell3D> const& /*cells*/, double const Area, double& A, double& /*b*/, size_t const /*face_index*/)const
{
    A += Area * dt * 0.5 * CG::speed_of_light;
}

void DiffusionOpenBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    E_outside = new_E[index] * 1e-20;
    v_outside = cells[index].velocity;
}

void DiffusionOpenBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
    double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * (2 * 3 * 
        sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    A += momentum_relativity_term;
}

// --- DiffusionMovingMarshakBoundary ---

double DiffusionMovingMarshakBoundary::BoundaryEnergyDensity() const
{
    return CG::radiation_constant * T_bath_ * T_bath_ * T_bath_ * T_bath_;
}

bool DiffusionMovingMarshakBoundary::IsLeftXBoundary(
    Tessellation3D const& tess, size_t const index, size_t const outside_point) const
{
    double const R = tess.GetWidth(index);
    return tess.GetMeshPoint(index).x > tess.GetMeshPoint(outside_point).x + R * 1e-4;
}

bool DiffusionMovingMarshakBoundary::IsRightXBoundary(
    Tessellation3D const& tess, size_t const index, size_t const outside_point) const
{
    double const R = tess.GetWidth(index);
    return tess.GetMeshPoint(index).x < tess.GetMeshPoint(outside_point).x - R * 1e-4;
}

bool DiffusionMovingMarshakBoundary::IsActiveBathBoundary(
    Tessellation3D const& tess, size_t const index, size_t const outside_point) const
{
    if (IsLeftXBoundary(tess, index, outside_point))
        return true;
    if (enable_right_boundary_ && IsRightXBoundary(tess, index, outside_point))
        return true;
    return false;
}

Vector3D DiffusionMovingMarshakBoundary::FaceVelocity(
    Tessellation3D const& tess, size_t const index, size_t const outside_point) const
{
    if (IsLeftXBoundary(tess, index, outside_point))
        return left_face_velocity_;
    if (IsRightXBoundary(tess, index, outside_point))
        return right_face_velocity_;
    return Vector3D(0.0, 0.0, 0.0);
}

void DiffusionMovingMarshakBoundary::SetBoundaryValues(
    Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const /*face_index*/)const
{
    if (!IsActiveBathBoundary(tess, index, outside_point))
        return;

    Vector3D const n_out = normalize(
        tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));

    Vector3D const v_face = FaceVelocity(tess, index, outside_point);

    double const E_b = BoundaryEnergyDensity();
    double const alpha = 0.5 * CG::speed_of_light;

    double const v_n = ScalarProd(cells[index].velocity, n_out);
    double const w_n = ScalarProd(v_face, n_out);

    double const coef = alpha - w_n + (4.0 / 3.0) * v_n;
    double const safe_coef = std::max(coef, 0.0);

    A += dt * Area * safe_coef;
    b += dt * Area * alpha * E_b;

    if (debug_) {
        std::cerr << std::scientific
                  << "MOVING_MARSHAK_ENERGY"
                  << " index=" << index
                  << " E_b=" << E_b
                  << " alpha=" << alpha
                  << " v_n=" << v_n
                  << " w_n=" << w_n
                  << " coef=" << coef
                  << " rel_coef=" << (coef - alpha) / alpha
                  << " dt=" << dt
                  << " Area=" << Area
                  << "\n";
    }
}

void DiffusionMovingMarshakBoundary::GetOutSideValues(
    Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    if (IsActiveBathBoundary(tess, index, outside_point)) {
        E_outside = BoundaryEnergyDensity();
        v_outside = FaceVelocity(tess, index, outside_point);
        return;
    }

    E_outside = new_E[index];
    Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
    v_outside = cells[index].velocity;
    v_outside -= 2.0 * normal * ScalarProd(normal, v_outside);
}

void DiffusionMovingMarshakBoundary::SetMomentumTermBoundary(
    Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const /*face_index*/,
    double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = std::max(abs(r_ij), 1e-200);
    r_ij *= 1.0 / r_ij_size;

    if (IsActiveBathBoundary(tess, index, outside_point)) {
        Vector3D const v_face = FaceVelocity(tess, index, outside_point);
        Vector3D const v_rel = cell.velocity - v_face;

        double const momentum_relativity_term =
            -0.5 * fleck_factor * dt * flux_limiter * Area *
            (2.0 * 3.0 * sigma_planck * D / CG::speed_of_light - 1.0) *
            ScalarProd(v_rel, r_ij) / 3.0;

        double const E_b = BoundaryEnergyDensity();

        A += momentum_relativity_term;
        b -= momentum_relativity_term * E_b;

        if (debug_) {
            std::cerr << std::scientific
                      << "MOVING_MARSHAK_MOMENTUM"
                      << " index=" << index
                      << " sigma_planck=" << sigma_planck
                      << " D=" << D
                      << " fleck=" << fleck_factor
                      << " flux_limiter=" << flux_limiter
                      << " v_cell_dot=" << ScalarProd(cell.velocity, r_ij)
                      << " v_face_dot=" << ScalarProd(v_face, r_ij)
                      << " v_rel_dot=" << ScalarProd(v_rel, r_ij)
                      << " mom_term=" << momentum_relativity_term
                      << "\n";
        }
    }
    else {
        double const momentum_relativity_term =
            -0.5 * fleck_factor * dt * flux_limiter * Area *
            (2.0 * 3.0 * sigma_planck * D / CG::speed_of_light - 1.0) *
            ScalarProd(cell.velocity, r_ij) / 3.0;

        A += 2.0 * momentum_relativity_term;
    }
}
