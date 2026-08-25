#include "hdsim_3d.hpp"
#include "CourantFriedrichsLewy.hpp"
#include "CFL1D.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
#include "3D/tessellation/voronoi/exception/MadVoroException.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "spherical_symmetry/SphericalShellProjector3D.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


namespace
{
	struct StrictBooleanEnvironment
	{
		bool value;
		bool valid;
	};

	StrictBooleanEnvironment ReadStrictBooleanEnvironment(char const* name)
	{
		char const* const value = std::getenv(name);
		if(value == nullptr || value[0] == '\0' ||
		   std::strcmp(value, "0") == 0 ||
		   std::strcmp(value, "false") == 0 ||
		   std::strcmp(value, "off") == 0 ||
		   std::strcmp(value, "no") == 0)
			return {false, true};
		if(std::strcmp(value, "1") == 0 ||
		   std::strcmp(value, "true") == 0 ||
		   std::strcmp(value, "on") == 0 ||
		   std::strcmp(value, "yes") == 0)
			return {true, true};
		return {false, false};
	}

	enum class IndividualEventMeshReuseMode
	{
		Off = 0,
		Selected = 1,
		Shadow = 2,
		Conflict = 3,
		Invalid = 4
	};

	char const* IndividualEventMeshReuseModeName(
		IndividualEventMeshReuseMode mode)
	{
		switch(mode)
		{
		case IndividualEventMeshReuseMode::Off:
			return "off";
		case IndividualEventMeshReuseMode::Selected:
			return "selected";
		case IndividualEventMeshReuseMode::Shadow:
			return "shadow";
		case IndividualEventMeshReuseMode::Conflict:
			return "conflict";
		case IndividualEventMeshReuseMode::Invalid:
			return "invalid";
		}
		return "invalid";
	}

	struct IndividualEventMeshReuseValidationState
	{
		size_t epoch = 0;
		size_t cached_epoch = std::numeric_limits<size_t>::max();
		vector<size_t> canonical_ids;
		size_t attempts = 0;
		size_t selected_hits = 0;
		size_t shadow_checks = 0;
		size_t failed_closed = 0;
	};

	std::unordered_map<HDSim3D const*, IndividualEventMeshReuseValidationState>&
	IndividualEventMeshReuseValidationStates()
	{
		static std::unordered_map<HDSim3D const*,
			IndividualEventMeshReuseValidationState> states;
		return states;
	}

	#ifdef RICH_MPI
	double get_time()
	{
		return MPI_Wtime();
	}
	#else
	std::chrono::time_point<std::chrono::high_resolution_clock> get_time()
	{
		return std::chrono::high_resolution_clock::now();
	}
	#endif

	template <class T>
	void DisplayTime(T const& t1, T const& t2, std::string const& msg)
	{
		#ifdef RICH_MPI
			int rank = -1;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if(rank == 0)
				std::cout<<msg<<" "<<t2 - t1<<" seconds"<<std::endl;
		#else
			std::cout<<msg<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<" mseconds"<<std::endl;
		#endif
	}

	double IndividualHydroWallTime()
	{
		#ifdef RICH_MPI
			return MPI_Wtime();
		#else
			return std::chrono::duration<double>(
				std::chrono::steady_clock::now().time_since_epoch()).count();
		#endif
	}

	constexpr size_t individual_hydro_phase_count = 12;
	constexpr size_t individual_hydro_total_index = individual_hydro_phase_count;
	constexpr size_t individual_hydro_active_index = individual_hydro_phase_count + 1;
	constexpr size_t individual_hydro_canonical_index = individual_hydro_phase_count + 2;
	constexpr size_t individual_hydro_record_count = individual_hydro_phase_count + 3;
	using IndividualHydroPhaseRecord =
		std::array<double, individual_hydro_record_count>;

	void ReportIndividualHydroPhaseTiming(
		IndividualHydroPhaseRecord const& local_record)
	{
		static std::array<char const*, individual_hydro_phase_count> const names = {{
			"setup", "first_mesh", "first_gather_sync",
			"first_update_source_scatter", "generator_prediction", "event_mesh",
			"event_gather_sync", "cell_face", "flux", "extensive", "source",
			"final_update_scatter_reuse"
		}};
		int rank = 0;
		int rank_count = 1;
		#ifdef RICH_MPI
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
		#endif
		std::vector<double> gathered;
		if(rank == 0)
			gathered.resize(static_cast<size_t>(rank_count) *
				individual_hydro_record_count);
		#ifdef RICH_MPI
			MPI_Gather(local_record.data(),
				static_cast<int>(individual_hydro_record_count), MPI_DOUBLE,
				rank == 0 ? gathered.data() : nullptr,
				static_cast<int>(individual_hydro_record_count), MPI_DOUBLE,
				0, MPI_COMM_WORLD);
		#else
			gathered.assign(local_record.begin(), local_record.end());
		#endif
		if(rank != 0)
			return;

		auto statistic = [&](size_t field, double quantile)
		{
			std::vector<double> values(static_cast<size_t>(rank_count));
			for(int source_rank = 0; source_rank < rank_count; ++source_rank)
				values[static_cast<size_t>(source_rank)] =
					gathered[static_cast<size_t>(source_rank) *
						individual_hydro_record_count + field];
			std::sort(values.begin(), values.end());
			if(quantile == 0.5 && values.size() % 2 == 0)
				return 0.5 * (values[values.size() / 2 - 1] +
					values[values.size() / 2]);
			size_t const index = std::min(values.size() - 1,
				static_cast<size_t>(std::ceil(quantile * values.size())) - 1);
			return values[index];
		};
		double active_cells_global = 0;
		double canonical_cells_global = 0;
		for(int source_rank = 0; source_rank < rank_count; ++source_rank)
		{
			size_t const offset = static_cast<size_t>(source_rank) *
				individual_hydro_record_count;
			active_cells_global += gathered[offset + individual_hydro_active_index];
			canonical_cells_global +=
				gathered[offset + individual_hydro_canonical_index];
		}
		std::cout << "INDIVIDUAL_HYDRO_PHASE_TIMING"
			<< " ranks=" << rank_count
			<< " active_cells_global=" << active_cells_global
			<< " canonical_cells_global=" << canonical_cells_global;
		for(size_t phase = 0; phase < individual_hydro_phase_count; ++phase)
			std::cout << " " << names[phase] << "_median=" << statistic(phase, 0.5)
				<< " " << names[phase] << "_p95=" << statistic(phase, 0.95)
				<< " " << names[phase] << "_max=" << statistic(phase, 1.0);
		std::cout << " total_median=" << statistic(individual_hydro_total_index, 0.5)
			<< " total_p95=" << statistic(individual_hydro_total_index, 0.95)
			<< " total_max=" << statistic(individual_hydro_total_index, 1.0)
			<< std::endl;
	}

	bool HasPositiveThermalMargin(Conserved3D const& state,
		double mass_threshold)
	{
		if (!(state.mass > 0.0) || !(state.mass > mass_threshold) ||
			!std::isfinite(state.mass) ||
			!std::isfinite(state.energy) ||
			!std::isfinite(abs(state.momentum)))
			return false;
		double const kinetic_energy =
			ScalarProd(state.momentum, state.momentum) /
			(2.0 * state.mass);
		double const internal_energy = state.energy - kinetic_energy;
		double const energy_scale =
			std::max(std::abs(state.energy), std::abs(kinetic_energy));
		return internal_energy > 1e-12 * energy_scale &&
			std::isfinite(internal_energy);
	}

	struct IndividualFaceGeometry
	{
		size_t neighbor;
		double area;
		Vector3D centroid;
		Vector3D normal;
	};

	struct IndividualCellGeometry
	{
		size_t global;
		double volume;
		Vector3D centroid;
		vector<IndividualFaceGeometry> faces;
	};

	size_t GlobalNeighbor(Tessellation3D const& tess,
		size_t local_neighbor,
		size_t global_cell_count)
	{
		auto const& map = tess.GetIndicesInAllPoints();
		auto const found = map.find(local_neighbor);
		if(found == map.end() || found->second >= global_cell_count)
			return std::numeric_limits<size_t>::max();
		return found->second;
	}

	IndividualCellGeometry CaptureCellGeometry(Tessellation3D const& tess,
		size_t local,
		size_t global,
		size_t global_cell_count)
	{
		IndividualCellGeometry result;
		result.global = global;
		result.volume = tess.GetVolume(local);
		result.centroid = tess.GetCellCM(local);
		vector<size_t> neighbors;
		tess.GetNeighbors(local, neighbors);
		face_vec const& faces = tess.GetCellFaces(local);
		if(neighbors.size() != faces.size())
			throw std::logic_error("Voronoi face/neighbor count mismatch during partial parity check");
		result.faces.reserve(faces.size());
		for(size_t i = 0; i < faces.size(); ++i)
		{
			Vector3D normal = tess.Normal(faces[i]);
			double const normal_size = abs(normal);
			if(!(normal_size > 0))
				throw std::logic_error("Zero Voronoi face normal during partial parity check");
			normal *= 1.0 / normal_size;
			if(neighbors[i] < tess.getMeshPoints().size())
			{
				Vector3D const outward = tess.GetMeshPoint(neighbors[i]) -
					tess.GetMeshPoint(local);
				if(ScalarProd(normal, outward) < 0)
					normal *= -1;
			}
			result.faces.push_back({GlobalNeighbor(tess, neighbors[i], global_cell_count),
				tess.GetArea(faces[i]), tess.FaceCM(faces[i]), normal});
		}
		return result;
	}

	vector<IndividualCellGeometry> CaptureActiveGeometry(
		Tessellation3D const& tess,
		ActiveMeshView const& view,
		vector<size_t> const& active,
		size_t global_cell_count)
	{
		vector<IndividualCellGeometry> result;
		result.reserve(active.size());
		for(size_t global : active)
		{
			if(!view.containsGlobal(global))
				throw std::logic_error("Active cell missing during partial parity check");
			result.push_back(CaptureCellGeometry(tess, view.globalToLocal(global),
				global, global_cell_count));
		}
		return result;
	}

	bool GeometryClose(double left, double right, double relative = 2e-8)
	{
		return std::abs(left - right) <= relative *
			std::max({1.0, std::abs(left), std::abs(right)});
	}

	bool GeometryClose(Vector3D const& left, Vector3D const& right,
		double relative = 2e-8)
	{
		return GeometryClose(left.x, right.x, relative) &&
			GeometryClose(left.y, right.y, relative) &&
			GeometryClose(left.z, right.z, relative);
	}

	bool GeometryCloseAtScale(Vector3D const& left, Vector3D const& right,
		double scale, double relative = 2e-8)
	{
		double const safe_scale = std::max(1.0, std::abs(scale));
		return GeometryClose(left.x / safe_scale, right.x / safe_scale, relative) &&
			GeometryClose(left.y / safe_scale, right.y / safe_scale, relative) &&
			GeometryClose(left.z / safe_scale, right.z / safe_scale, relative);
	}

	bool ActiveGeometryMatches(vector<IndividualCellGeometry> const& partial,
		Tessellation3D const& full,
		ActiveMeshView const& full_view,
		size_t global_cell_count,
		bool report_mismatch = true)
	{
		for(IndividualCellGeometry const& expected : partial)
		{
			size_t const full_local = full_view.globalToLocal(expected.global);
			IndividualCellGeometry const candidate = CaptureCellGeometry(full,
				full_local, expected.global,
				global_cell_count);
			double const cell_width = full.GetWidth(full_local);
			if(!GeometryClose(expected.volume, candidate.volume))
			{
				if(report_mismatch)
					std::cout << "Individual partial parity mismatch: cell "
						<< expected.global << " volume partial=" << expected.volume
						<< " full=" << candidate.volume << std::endl;
				return false;
			}
			if(!GeometryCloseAtScale(expected.centroid, candidate.centroid,
				cell_width))
			{
				if(report_mismatch)
					std::cout << "Individual partial parity mismatch: cell "
						<< expected.global << " centroid partial=" << expected.centroid
						<< " full=" << candidate.centroid << std::endl;
				return false;
			}
			if(expected.faces.size() != candidate.faces.size())
			{
				if(report_mismatch)
					std::cout << "Individual partial parity mismatch: cell "
						<< expected.global << " face count partial="
						<< expected.faces.size() << " full=" << candidate.faces.size()
						<< std::endl;
				return false;
			}
			vector<unsigned char> used(candidate.faces.size(), 0);
			for(IndividualFaceGeometry const& expected_face : expected.faces)
			{
				size_t best = candidate.faces.size();
				double best_distance = std::numeric_limits<double>::max();
				for(size_t i = 0; i < candidate.faces.size(); ++i)
				{
					if(used[i] || candidate.faces[i].neighbor != expected_face.neighbor)
						continue;
					double const distance = abs(candidate.faces[i].centroid -
						expected_face.centroid);
					if(distance < best_distance)
					{
						best = i;
						best_distance = distance;
					}
				}
				if(best == candidate.faces.size())
				{
					if(report_mismatch)
						std::cout << "Individual partial parity mismatch: cell "
							<< expected.global << " has no full face for neighbor "
							<< expected_face.neighbor << std::endl;
					return false;
				}
				if(!GeometryClose(expected_face.area, candidate.faces[best].area) ||
				   !GeometryCloseAtScale(expected_face.centroid,
					candidate.faces[best].centroid, cell_width) ||
				   !GeometryClose(expected_face.normal, candidate.faces[best].normal))
				{
					if(report_mismatch)
						std::cout << "Individual partial parity mismatch: cell "
							<< expected.global << " neighbor " << expected_face.neighbor
							<< " face area partial=" << expected_face.area
							<< " full=" << candidate.faces[best].area
							<< " centroid partial=" << expected_face.centroid
							<< " full=" << candidate.faces[best].centroid
							<< " normal partial=" << expected_face.normal
							<< " full=" << candidate.faces[best].normal << std::endl;
					return false;
				}
				used[best] = 1;
			}
		}
		return true;
	}
}

void InvalidateIndividualEventMeshReuseValidation(HDSim3D const& simulation)
{
	IndividualEventMeshReuseValidationState& state =
		IndividualEventMeshReuseValidationStates()[&simulation];
	++state.epoch;
	state.cached_epoch = std::numeric_limits<size_t>::max();
	state.canonical_ids.clear();
}

Tessellation3D& HDSim3D::getTessellation(void)
{
	return tess_;
}

vector<ComputationalCell3D>& HDSim3D::getCells(void)
{
	return cells_;
}

vector<Conserved3D>& HDSim3D::getExtensives(void)
{
	return extensive_;
}

const vector<Conserved3D>& HDSim3D::getExtensives(void) const
{
	return extensive_;
}

void HDSim3D::SetSphericalShellProjector(
	std::shared_ptr<SphericalShellProjector3D> projector,
	FluxCalculator3D const& perturbation_flux_calculator)
{
	spherical_shell_projector_ = std::move(projector);
	spherical_shell_geometry_ = spherical_shell_projector_ ?
		spherical_shell_projector_->GetShellGeometry() : nullptr;
	spherical_perturbation_flux_calculator_ =
		spherical_shell_projector_ ? &perturbation_flux_calculator : nullptr;
	RefreshSphericalShellGeometry("initial mesh", true);
}

void HDSim3D::RefreshSphericalShellGeometry(
	char const* update_name,
	bool always_report)
{
	if (!spherical_shell_geometry_)
		return;
	bool const mesh_changed = spherical_shell_geometry_->Update(tess_);
	if (!always_report && !mesh_changed)
		return;
#ifdef RICH_MPI
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank != 0)
		return;
#endif
	std::cout << "Spherical shell update (" << update_name << "): "
		<< spherical_shell_geometry_->GetShellRadii().size()
		<< " shells" << std::endl;
}

FluxCalculator3D const&
HDSim3D::GetSphericalPerturbationFluxCalculator(void) const
{
	if (!spherical_perturbation_flux_calculator_)
		throw UniversalError(
			"Spherical symmetry requires a perturbation flux calculator");
	return *spherical_perturbation_flux_calculator_;
}

FluxCalculator3D const& HDSim3D::GetFullStateFluxCalculator(void) const
{
	return spherical_shell_projector_ ?
		GetSphericalPerturbationFluxCalculator() : fc_;
}

void HDSim3D::ApplySphericalBackgroundCorrection(
	vector<ComputationalCell3D> const& stage_input_cells,
	vector<Conserved3D> const& stage_input_extensives,
	vector<Vector3D> const& face_velocities,
	vector<Vector3D> const& point_velocities,
	double time,
	double dt,
	bool source_before_extensive_update,
	vector<Conserved3D>& full_candidate)
{
	if (!spherical_shell_projector_)
		return;

	size_t const local_count = tess_.GetPointNo();
	vector<Conserved3D> background_extensives;
	spherical_shell_projector_->ProjectExtensives(tess_,
		stage_input_extensives, background_extensives);
	double maximum_background_deviation = 0;
	for (size_t i = 0; i < local_count; ++i) {
		double scale =
			std::abs(background_extensives[i].mass) +
			abs(background_extensives[i].momentum) +
			std::abs(background_extensives[i].energy) +
			std::abs(background_extensives[i].internal_energy);
		double deviation =
			std::abs(stage_input_extensives[i].mass -
				background_extensives[i].mass) +
			abs(stage_input_extensives[i].momentum -
				background_extensives[i].momentum) +
			std::abs(stage_input_extensives[i].energy -
				background_extensives[i].energy) +
			std::abs(stage_input_extensives[i].internal_energy -
				background_extensives[i].internal_energy);
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer) {
			scale += std::abs(background_extensives[i].tracers[tracer]);
			deviation += std::abs(stage_input_extensives[i].tracers[tracer] -
				background_extensives[i].tracers[tracer]);
		}
		maximum_background_deviation =
			std::max(maximum_background_deviation,
				deviation / std::max(scale, 1e-300));
	}
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &maximum_background_deviation, 1,
		MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
	bool const symmetric_to_roundoff =
		maximum_background_deviation <= 1e-12;
	vector<ComputationalCell3D> background_cells = stage_input_cells;
	background_extensives.resize(local_count);
	background_cells.resize(local_count);
	cu_(background_cells, eos_, tess_, background_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, background_extensives, true);
	MPI_exchange_data(tess_, background_cells, true);
#endif

	vector<Conserved3D> background_fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
		background_face_values;
	fc_.Calculate(background_fluxes, tess_, face_velocities,
		background_cells, background_extensives, eos_, time, dt,
		background_face_values);

	vector<Conserved3D> background_candidate = background_extensives;
	if (source_before_extensive_update) {
		source_(tess_, background_cells, background_fluxes, point_velocities,
			time, dt, background_candidate);
		eu_(background_fluxes, tess_, dt, background_cells,
			background_candidate, time, face_velocities,
			point_velocities, background_face_values);
	}
	else {
		eu_(background_fluxes, tess_, dt, background_cells,
			background_candidate, time, face_velocities,
			point_velocities, background_face_values);
		source_(tess_, background_cells, background_fluxes, point_velocities,
			time, dt, background_candidate);
	}

	vector<Conserved3D> background_rate = background_candidate;
	for (size_t i = 0; i < local_count; ++i)
		background_rate[i] -= background_extensives[i];
	vector<Conserved3D> projected_background_rate;
	spherical_shell_projector_->ProjectRates(tess_,
		background_rate, projected_background_rate);

	// Evolve a nonspherical state with the ordinary 3-D operator and only
	// replace the angular error of the spherical-background contribution:
	// P L_S(PU) + L(U) - L_S(PU).
	if (!symmetric_to_roundoff)
		++spherical_perturbation_evaluation_count_;

	for (size_t i = 0; i < local_count; ++i) {
		if (symmetric_to_roundoff) {
			full_candidate[i] = background_extensives[i] +
				projected_background_rate[i];
		}
		else {
			Conserved3D const full_rate =
				full_candidate[i] - stage_input_extensives[i];
			full_candidate[i] = stage_input_extensives[i] +
				projected_background_rate[i] +
				(full_rate - background_rate[i]);
		}
		if (full_candidate[i].mass > 0)
			full_candidate[i].internal_energy =
				full_candidate[i].energy -
				ScalarProd(full_candidate[i].momentum,
					full_candidate[i].momentum) /
				(2.0 * full_candidate[i].mass);
	}
}

HDSim3D::HDSim3D(Tessellation3D& tess,
	vector<ComputationalCell3D>& cells,
	vector<Conserved3D>& extensives,
	const EquationOfState& eos,
	ProgressTracker &pt,
	const PointMotion3D& pm,
	TimeStepFunction3D& tsc,
	const FluxCalculator3D& fc,
	const CellUpdater3D& cu,
	const ExtensiveUpdater3D& eu,
	const SourceTerm3D& source,
	const pair<vector<string>, vector<string> >& tsn,
	bool SR
	#ifdef RICH_MPI
	, std::shared_ptr<CostCalculator3D> cost_calc
	#endif // RICH_MPI
	) :
	tess_(tess),
	eos_(eos), cells_(cells), extensive_(extensives), pm_(pm), tsc_(tsc), fc_(fc), cu_(cu), eu_(eu), source_(source), pt_(pt),
	special_relativity_(SR)
	#ifdef RICH_MPI
	, cost_calc_(cost_calc), exchange_chain_(MPI_COMM_WORLD)
	#endif // RICH_MPI
{
	const bool validity_check = tess.GetPointNo() <= cells.size();
	assert(validity_check);
	assert(tsn.second.size() <= MAX_STICKERS);
	assert(tsn.first.size() <= MAX_TRACERS);
	// sort tracers and stickers
	size_t N = tess.GetPointNo();
	vector<size_t> tindex = sort_index(tsn.first);
	vector<size_t> sindex = sort_index(tsn.second);
	ComputationalCell3D::tracerNames = VectorValues(tsn.first, tindex);
	ComputationalCell3D::stickerNames = VectorValues(tsn.second, sindex);
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < tindex.size(); ++j)
			cells_[i].tracers[j] = cells[i].tracers[tindex[j]];
		for (size_t j = 0; j < sindex.size(); ++j)
			cells_[i].stickers[j] = cells[i].stickers[sindex[j]];
	}

#ifdef RICH_MPI
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess_, cells_, true);
#endif
	extensive_.resize(N);
	if (SR)
	{
		for (size_t i = 0; i < N; ++i)
			PrimitiveToConservedSR(cells_[i], tess.GetVolume(i), extensive_[i], eos_);
	}
	else
	{
		for (size_t i = 0; i < N; ++i)
			PrimitiveToConserved(cells_[i], tess.GetVolume(i), extensive_[i]);
	}
}

namespace
{
	void CalcFaceVelocities(Tessellation3D const& tess, vector<Vector3D> const& point_vel, vector<Vector3D>& res)
	{
		size_t N = tess.GetTotalFacesNumber();
		res.resize(N);
		for (size_t i = 0; i < N; ++i)
		{
			if (tess.BoundaryFace(i))
				res[i] = Vector3D();
			else
			{
				try
				{
					res[i] = tess.CalcFaceVelocity(i, point_vel[tess.GetFaceNeighbors(i).first], point_vel[tess.GetFaceNeighbors(i).second]);
				}
				catch (UniversalError & /*eo*/)
				{
					throw;
				}
			}
		}
	}

	void MovePoints(Tessellation3D& tess, std::vector<Vector3D> const& point_vel, double const dt)
	{
		size_t const N = tess.GetPointNo();
		std::vector<Vector3D>& points = tess.accessMeshPoints();
		for(size_t i = 0; i < N; ++i)
			points[i] += point_vel[i] * dt;
	}

	#ifdef RICH_MPI
		void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, ExchangeChain &chain, vector<Vector3D> &points, std::vector<Vector3D> const* orgpoints = nullptr)
	#else // RICH_MPI
		void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, vector<Vector3D> &points, std::vector<Vector3D> const* orgpoints = nullptr)
	#endif // RICH_MPI
	{
		MEMORY_PROFILE_SCOPE("tessellation rebuild");
		if (orgpoints == nullptr)
		{
			const vector<Vector3D> &mesh = tess.getMeshPoints();
			points.assign(mesh.begin(), mesh.end());
		}
		else
			points = *orgpoints;
		points.resize(tess.GetPointNo());
		if(orgpoints != nullptr)
		{
			size_t const N = points.size();
			for (size_t i = 0; i < N; ++i)
				points[i] += point_vel[i] * dt;
		}
		
		#ifdef RICH_MPI
		tess.BuildParallel(points);
		chain.Exchange(tess.GetSentProcs(), tess.GetSentPoints(), tess.GetSelfIndex());
		#else // RICH_MPI
		tess.Build(points);
		#endif // RICH_MPI
	}

	void ExtensiveAvg(vector<Conserved3D>& res, vector<Conserved3D> const& other)
	{
		assert(res.size() == other.size());
		size_t N = res.size();
		for (size_t i = 0; i < N; ++i)
		{
			res[i] += other[i];
			res[i] *= 0.5;
			double kinetic = 0.5 * ScalarProd(res[i].momentum, res[i].momentum) / res[i].mass;
    		res[i].internal_energy = res[i].energy - kinetic;
		}
	}

	std::pair<std::vector<size_t>, std::vector<size_t>>
	FindXBoundaryFaces(const Tessellation3D& tess)
	{
		auto box = tess.GetBoxCoordinates();
		Vector3D ll = box.first;
		Vector3D ur = box.second;
		size_t Norg = tess.GetPointNo();
		std::vector<size_t> left_faces, right_faces;
		size_t Nfaces = tess.GetTotalFacesNumber();
		for (size_t i = 0; i < Nfaces; ++i)
		{
			if (!tess.BoundaryFace(i))
				continue;
			Vector3D normal = normalize(tess.Normal(i));
			if (std::abs(normal.x) < 0.5)
				continue;
			size_t n0 = tess.GetFaceNeighbors(i).first;
			size_t n1 = tess.GetFaceNeighbors(i).second;
			size_t ghost = (n0 < Norg) ? n1 : n0;
			Vector3D ghost_point = tess.GetMeshPoint(ghost);
			if (ghost_point.x < ll.x)
				left_faces.push_back(i);
			else if (ghost_point.x > ur.x)
				right_faces.push_back(i);
		}
		return std::make_pair(left_faces, right_faces);
	}

	double SetBoundaryFaceVelocities(
		const std::vector<size_t>& face_indices,
		const Tessellation3D& tess,
		const std::vector<ComputationalCell3D>& cells,
		const EquationOfState& eos,
		const Hllc3D& hllc,
		const ComputationalCell3D* external_state,
		std::vector<Vector3D>& face_vel)
	{
		double weighted_vx = 0;
		double total_area = 0;
		size_t Norg = tess.GetPointNo();
		for (size_t fi : face_indices)
		{
			const std::pair<size_t, size_t>& neighbors = tess.GetFaceNeighbors(fi);
			size_t n0 = neighbors.first;
			size_t n1 = neighbors.second;
			size_t interior = (n0 < Norg) ? n0 : n1;
			Vector3D raw_normal = normalize(tess.Normal(fi));
			double sign = (n0 < Norg) ? 1.0 : -1.0;
			double nx_sign = (raw_normal.x * sign > 0) ? 1.0 : -1.0;
			Vector3D outward_normal(nx_sign, 0, 0);

			double contact_speed;
			if (external_state == nullptr)
			{
				const ComputationalCell3D& cell = cells[interior];
				ComputationalCell3D vacuum_state = cell;
				double vac_factor = 1e-10;
				vacuum_state.velocity = Vector3D(0, 0, 0);
				vacuum_state.density = cell.density * vac_factor;
				vacuum_state.pressure = cell.pressure * vac_factor * 0.01;
				vacuum_state.internal_energy = eos.dp2e(vacuum_state.density,
					vacuum_state.pressure, cell.tracers, ComputationalCell3D::tracerNames);
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cell, vacuum_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
			}
			else
			{
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cells[interior], *external_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
			}
			face_vel[fi] = contact_speed * outward_normal;
			double area = tess.GetArea(fi);
			weighted_vx += contact_speed * outward_normal.x * area;
			total_area += area;
		}
		return (total_area > 0) ? (weighted_vx / total_area) : 0.0;
	}

	double OverrideBoundaryFluxes(
		const std::vector<size_t>& face_indices,
		const Tessellation3D& tess,
		const std::vector<ComputationalCell3D>& cells,
		const EquationOfState& eos,
		const Hllc3D& hllc,
		const ComputationalCell3D* external_state,
		std::vector<Conserved3D>& fluxes,
		std::vector<Vector3D>& face_vel)
	{
		double weighted_vx = 0;
		double total_area = 0;
		size_t Norg = tess.GetPointNo();
		for (size_t fi : face_indices)
		{
			const std::pair<size_t, size_t>& neighbors = tess.GetFaceNeighbors(fi);
			size_t n0 = neighbors.first;
			size_t n1 = neighbors.second;
			size_t interior = (n0 < Norg) ? n0 : n1;
			Vector3D raw_normal = normalize(tess.Normal(fi));
			double sign = (n0 < Norg) ? 1.0 : -1.0;
			double nx_sign = (raw_normal.x * sign > 0) ? 1.0 : -1.0;
			Vector3D outward_normal(nx_sign, 0, 0);

			double contact_speed;
			if (external_state == nullptr)
			{
				const ComputationalCell3D& cell = cells[interior];
				ComputationalCell3D vacuum_state = cell;
				double vac_factor = 1e-10;
				vacuum_state.density = cell.density * vac_factor;
				vacuum_state.pressure = cell.pressure * vac_factor;
				vacuum_state.internal_energy = eos.dp2e(vacuum_state.density,
					vacuum_state.pressure, cell.tracers, ComputationalCell3D::tracerNames);
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cell, vacuum_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
				fluxes[fi] = Conserved3D();
			}
			else
			{
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cells[interior], *external_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
				Vector3D fv = contact_speed * outward_normal;
				RotateSolveBack3D(outward_normal, cells[interior], *external_state,
					fv, hllc, fluxes[fi], eos);
			}
			face_vel[fi] = contact_speed * outward_normal;
			double area = tess.GetArea(fi);
			weighted_vx += contact_speed * outward_normal.x * area;
			total_area += area;
		}
		return (total_area > 0) ? (weighted_vx / total_area) : 0.0;
	}

	void SetBoxAndRebuild(Tessellation3D& tess, const Vector3D& new_ll, const Vector3D& new_ur,
		std::vector<Vector3D>& points_scratch
	#ifdef RICH_MPI
		, ExchangeChain& chain
	#endif
	)
	{
		tess.SetBox(new_ll, new_ur);
		const vector<Vector3D>& mesh = tess.getMeshPoints();
		points_scratch.assign(mesh.begin(), mesh.end());
		points_scratch.resize(tess.GetPointNo());
	#ifdef RICH_MPI
		tess.BuildParallel(points_scratch);
		chain.Exchange(tess.GetSentProcs(), tess.GetSentPoints(),
			tess.GetSelfIndex());
	#else
		tess.Build(points_scratch);
	#endif
	}
}


void HDSim3D::timeAdvance2(void)
{
	RefreshSphericalShellGeometry("timeAdvance2 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance2");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	MEMORY_DEBUG_PRINT("hydro: after MPI reset");
	const double time = pt_.getTime();
	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	if (auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&tsc_))
		cfl->SetPointVelocities(&point_vel);
	else if (auto* cfl1d = dynamic_cast<CFL1D*>(&tsc_))
		cfl1d->SetPointVelocities(&point_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	MEMORY_DEBUG_PRINT("hydro: after CFL + face velocities");
	auto t1 = get_time();
	auto t2 = t1;
	GetFullStateFluxCalculator().Calculate(
		fluxes, tess_, face_vel, cells_, extensive_,
		eos_, time, dt, face_values);
	MEMORY_DEBUG_PRINT("hydro: after flux calc");
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time,
		face_vel, point_vel, face_values);
	MEMORY_DEBUG_PRINT("hydro: after extensive update");
	t1 = get_time();
	source_(tess_, cells_, fluxes, point_vel, time, dt,
		mid_extensives);
	t2 = get_time();
	DisplayTime(t1, t2, "Source time ");
	ApplySphericalBackgroundCorrection(cells_, extensive_, face_vel,
		point_vel, time, dt, false, mid_extensives);
	MEMORY_DEBUG_PRINT("hydro: after source terms");
	// if (pt_.getCycle() % 10 == 0 && pm_.MovedPoints())
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
// #ifdef RICH_MPI
		// tess_.PreparePoints(mesh, order);
// #endif
	// }
	Conserved3D edummy;
	ComputationalCell3D cdummy;
	if(pm_.MovedPoints())
	{
		MovePoints(tess_, point_vel, dt);
		t1 = get_time();
		#ifdef RICH_MPI
			UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
		#else // RICH_MPI
			UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
		#endif // RICH_MPI
		RefreshSphericalShellGeometry("timeAdvance2 mesh build", true);
		t2 = get_time();
		DisplayTime(t1, t2, "Voronoi build time ");
#ifdef RICH_MPI
		MPI_exchange_data(tess_, mid_extensives, false);
		MPI_exchange_data(tess_, extensive_, false);
		MPI_exchange_data(tess_, cells_, false);
		MPI_exchange_data(tess_, point_vel, false);
		MPI_exchange_data(tess_, point_vel, true);
#endif
	}
	MEMORY_DEBUG_PRINT("hydro: after Voronoi rebuild");
cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro: after cell update (1st half)");

	CalcFaceVelocities(tess_, point_vel, face_vel);
	vector<Conserved3D> const stage_two_input_extensives = mid_extensives;
	GetFullStateFluxCalculator().Calculate(
		fluxes, tess_, face_vel, cells_,
		mid_extensives, eos_, time + dt, dt, face_values);
	t1 = get_time();
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt,
		mid_extensives);
	t2 = get_time();
	DisplayTime(t1, t2, "Second source time ");
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt,
		face_vel, point_vel, face_values);
	ApplySphericalBackgroundCorrection(cells_,
		stage_two_input_extensives, face_vel, point_vel,
		time + dt, dt, true, mid_extensives);
	ExtensiveAvg(extensive_, mid_extensives);
cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif
MEMORY_DEBUG_PRINT("hydro: after cell update (2nd half)");
}

void HDSim3D::timeAdvanceIndividual(const IndividualStepContext& context)
{
	if(special_relativity_ || spherical_shell_projector_ != nullptr)
		throw std::logic_error("Individual timesteps support only Newtonian Cartesian 3D hydro");
	if(!eu_.SupportsIndividualTimeSteps())
		throw std::logic_error("Configured extensive updater does not support individual timesteps");
	if(!source_.SupportsIndividualTimeSteps())
		throw std::logic_error("Configured source term does not support individual timesteps");
	if(!GetFullStateFluxCalculator().SupportsIndividualTimeSteps())
		throw std::logic_error("Configured flux reconstruction does not support second-order individual timesteps");
	MEMORY_PROFILE_SCOPE("hydro timeAdvanceIndividual");
	static StrictBooleanEnvironment const phase_trace_environment =
		ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_PERF_TRACE");
	if(!phase_trace_environment.valid)
		throw std::invalid_argument(
			"RICH_INDIVIDUAL_PERF_TRACE must be a strict boolean");
	bool const phase_trace = phase_trace_environment.value;
	IndividualHydroPhaseRecord phase_record = {{}};
	double phase_start = phase_trace ? IndividualHydroWallTime() : 0;
	auto finish_phase = [&](size_t phase)
	{
		if(!phase_trace)
			return;
		double const finish = IndividualHydroWallTime();
		phase_record.at(phase) += finish - phase_start;
		phase_start = finish;
	};
	const size_t canonical_count = extensive_.size();
	if(context.active_mask.size() != canonical_count)
		throw std::logic_error(
			"Individual hydro context does not match the canonical owned-cell count");
	auto any_rank = [](bool value)
	{
#ifdef RICH_MPI
		int flag = value ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		return flag != 0;
#else
		return value;
#endif
	};
	auto all_ranks = [](bool value)
	{
#ifdef RICH_MPI
		int flag = value ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		return flag != 0;
#else
		return value;
#endif
	};
	vector<Vector3D>& point_vel = point_vel_scratch_;
	vector<Vector3D>& face_vel = face_vel_scratch_;
	vector<Conserved3D>& fluxes = fluxes_scratch_;
	auto& face_values = face_values_scratch_;

	if(context.point_velocities.size() != canonical_count)
		context.point_velocities.assign(canonical_count, Vector3D());
	if(context.cached_accelerations.size() != canonical_count)
		context.cached_accelerations.assign(canonical_count, Vector3D());
	if(context.gravity_half_kick_pending.size() != canonical_count)
		context.gravity_half_kick_pending.assign(canonical_count, 0);

	if(individual_points_.size() != canonical_count)
	{
		if(tess_.GetPointNo() != canonical_count)
			throw std::logic_error(
				"Cannot initialize canonical individual generator positions from a partial mesh");
		individual_points_.resize(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			individual_points_[global] = tess_.GetMeshPoint(global);
	}
	if(individual_centroids_.size() != canonical_count)
	{
		if(tess_.GetPointNo() != canonical_count)
			throw std::logic_error(
				"Cannot initialize canonical individual cell centroids from a partial mesh");
		individual_centroids_.resize(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			individual_centroids_[global] = tess_.GetCellCM(global);
	}
	vector<Vector3D> all_points = individual_points_;
	context.gravity_source_points = individual_centroids_;
	context.gravity_source_masses.resize(extensive_.size());
	context.gravity_source_ids.resize(extensive_.size());
	for(size_t global = 0; global < extensive_.size(); ++global)
	{
		context.gravity_source_masses[global] = extensive_[global].mass;
		context.gravity_source_ids[global] =
			static_cast<std::uint64_t>(cells_[global].ID);
	}

	bool const globally_all_active = all_ranks(
		context.active_indices.size() == canonical_count &&
		std::all_of(context.active_mask.begin(), context.active_mask.end(),
			[](unsigned char value) { return value != 0; }));
	bool const partial_requested =
		context.mesh_build_policy == IndividualMeshBuildPolicy::AutoPartial &&
		source_.SupportsPartialMesh() && spherical_shell_projector_ == nullptr &&
		!globally_all_active;
	if(individual_mesh_restore_pending_)
	{
		bool const globally_has_target = any_rank(!individual_mesh_target_ids_.empty());
		bool restore_failed = false;
		vector<size_t> restore_target;
		if(partial_requested && globally_has_target)
		{
			std::unordered_map<size_t, size_t> id_to_global;
			id_to_global.reserve(cells_.size());
			for(size_t global = 0; global < cells_.size(); ++global)
				id_to_global[cells_[global].ID] = global;
			restore_target.reserve(individual_mesh_target_ids_.size());
			for(size_t id : individual_mesh_target_ids_)
			{
				auto const found = id_to_global.find(id);
				if(found == id_to_global.end())
				{
					restore_failed = true;
					break;
				}
				restore_target.push_back(found->second);
			}

			if(!any_rank(restore_failed))
			{
				try
				{
#ifdef RICH_MPI
					tess_.BuildPartiallyParallel(all_points, restore_target, true, true);
#else
					tess_.BuildPartially(all_points, restore_target);
#endif
					ActiveMeshView restored_view(tess_, canonical_count);
					(void)restored_view;
				}
				catch(...)
				{
					restore_failed = true;
				}
			}
		}

		if(partial_requested && globally_has_target && any_rank(restore_failed))
		{
#ifdef RICH_MPI
			tess_.BuildParallel(all_points, true, true);
#else
			tess_.Build(all_points);
#endif
			individual_mesh_target_ids_.clear();
			std::cerr << "Individual partial Voronoi restart fallback: "
				"saved target closure could not be restored" << std::endl;
		}
		else if(!partial_requested)
			individual_mesh_target_ids_.clear();
		individual_mesh_restore_pending_ = false;
	}

	vector<size_t> seed;
	vector<size_t> warm_target;
	if(partial_requested)
	{
	seed = context.active_indices;
	vector<unsigned char> seeded(canonical_count, 0);
	for(size_t global : seed)
		seeded.at(global) = 1;
	// Warm-start distributed closure from the last accepted target set.  Keep
	// this separate from the current reconstruction seed: cached geometric
	// support must not be promoted to a depth-one stencil cell.  Stable IDs make
	// the cache safe across local reordering and AMR; removed IDs are ignored.
	if(partial_requested && !individual_mesh_target_ids_.empty())
	{
		std::unordered_map<size_t, size_t> current_index_by_id;
		current_index_by_id.reserve(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			current_index_by_id.emplace(cells_[global].ID, global);
		for(size_t id : individual_mesh_target_ids_)
		{
			auto const found = current_index_by_id.find(id);
			if(found != current_index_by_id.end() && !seeded[found->second])
				warm_target.push_back(found->second);
		}
	}
	// Reuse the previous face halo.  Synchronizing the previous activity mask is
	// essential in MPI: a locally passive cell adjacent to a remote active cell
	// must be a target so its reconstruction slope can be sent to the face owner.
	std::unique_ptr<ActiveMeshView> previous_view;
	bool previous_mapping_failed = false;
	try
	{
		previous_view.reset(new ActiveMeshView(tess_, canonical_count));
	}
	catch(...)
	{
		previous_mapping_failed = true;
	}
	if(!any_rank(previous_mapping_failed))
	{
		std::unique_ptr<IndividualStepContext> previous_context;
		bool previous_sync_failed = false;
		try
		{
			previous_context.reset(new IndividualStepContext(
				previous_view->remapContext(context, false)));
		}
		catch(...)
		{
			previous_sync_failed = true;
		}
		if(!any_rank(previous_sync_failed))
		{
			for(size_t face = 0; face < tess_.GetTotalFacesNumber(); ++face)
			{
				auto const neighbors = tess_.GetFaceNeighbors(face);
				if(!previous_context->isActive(neighbors.first) &&
				   !previous_context->isActive(neighbors.second))
					continue;
				for(size_t local : {neighbors.first, neighbors.second})
				{
					if(local >= tess_.GetPointNo())
						continue;
					size_t const global = previous_view->localToGlobal(local);
					if(global < canonical_count && !seeded[global])
					{
						seeded[global] = 1;
						seed.push_back(global);
					}
				}
			}
		}
	}
	}

	auto remember_partial_target = [&](vector<size_t> const& target)
	{
		individual_mesh_target_ids_.clear();
		individual_mesh_target_ids_.reserve(target.size());
		for(size_t global : target)
			individual_mesh_target_ids_.push_back(cells_.at(global).ID);
	};
	auto build_event_mesh = [&](vector<Vector3D> const& points,
		vector<size_t> const& initial_seed, vector<size_t>& cached_target,
		char const* stage, bool force_full) -> ActiveMeshView
	{
		auto full_build = [&](char const* reason) -> ActiveMeshView
		{
			individual_mesh_target_ids_.clear();
			cached_target.clear();
			auto const start = get_time();
#ifdef RICH_MPI
			tess_.BuildParallel(points, true, true);
#else
			tess_.Build(points);
#endif
			auto const finish = get_time();
			DisplayTime(start, finish, string("Individual full Voronoi build (") + reason + ") ");
			RefreshSphericalShellGeometry(stage, true);
			return ActiveMeshView(tess_, canonical_count);
		};

		if(force_full)
			return full_build("event mesh reuse shadow reference");
		if(!partial_requested)
			return full_build("policy or source requires full geometry");

		vector<unsigned char> included(canonical_count, 0);
		vector<unsigned char> depth(canonical_count, 255);
		vector<size_t> target;
		target.reserve(initial_seed.size() * 2 + 8);
		for(size_t global : context.active_indices)
		{
			included.at(global) = 1;
			depth[global] = 0;
			target.push_back(global);
		}
		for(size_t global : initial_seed)
			if(!included.at(global))
			{
				included[global] = 1;
				depth[global] = 1;
				target.push_back(global);
			}
		for(size_t global : cached_target)
			if(!included.at(global))
			{
				included[global] = 1;
				target.push_back(global);
			}

		size_t const full_threshold = static_cast<size_t>(
			context.partial_build_fraction * static_cast<double>(canonical_count));
		if(any_rank(target.size() > full_threshold))
			return full_build("closure threshold");

		for(;;)
		{
			bool partial_build_failed = false;
			string partial_build_error;
			try
			{
				auto const start = get_time();
#ifdef RICH_MPI
				tess_.BuildPartiallyParallel(points, target, true, true);
#else
				tess_.BuildPartially(points, target);
#endif
				auto const finish = get_time();
				DisplayTime(start, finish, "Individual partial Voronoi build ");
			}
			catch(MadVoro::Exception::MadVoroException const& error)
			{
				partial_build_failed = true;
				partial_build_error = error.getErrorMessage();
			}
			catch(UniversalError const& error)
			{
				partial_build_failed = true;
				partial_build_error = error.getErrorMessage();
			}
			catch(std::exception const& error)
			{
				partial_build_failed = true;
				partial_build_error = error.what();
			}
			catch(...)
			{
				partial_build_failed = true;
				partial_build_error = "non-standard exception";
			}
			if(any_rank(partial_build_failed))
			{
#ifdef RICH_MPI
				int rank = 0;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				int reporting_rank = partial_build_failed ?
					rank : std::numeric_limits<int>::max();
				MPI_Allreduce(MPI_IN_PLACE, &reporting_rank, 1, MPI_INT,
					MPI_MIN, MPI_COMM_WORLD);
				if(rank == reporting_rank)
#endif
					std::cerr << "Individual partial Voronoi build failed"
#ifdef RICH_MPI
						<< " on rank " << rank
#endif
						<< ": " << partial_build_error << std::endl;
				return full_build("partial build exception");
			}

			std::unique_ptr<ActiveMeshView> view_ptr;
			bool mapping_failed = false;
			try
			{
				view_ptr.reset(new ActiveMeshView(tess_, canonical_count));
			}
			catch(...)
			{
				mapping_failed = true;
			}
			if(any_rank(mapping_failed))
				return full_build("partial mapping invariant");
			ActiveMeshView& view = *view_ptr;
			vector<size_t> additions;
			vector<size_t> neighbors;
			bool closure_changed = false;
#ifdef RICH_MPI
			// Map each remote ghost to the canonical index on its owning rank.
			// A newly appearing cross-rank face must make that endpoint a target
			// on its owner; otherwise its cached centroid can be stale even when
			// the active cell geometry itself looks complete.
			int rank = 0;
			int rank_count = 1;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
			vector<int> ghost_owner(tess_.getMeshPoints().size(), -1);
			auto const& duplicate_ranks = tess_.GetDuplicatedProcs();
			auto const& ghost_indices = tess_.GetGhostIndeces();
			bool ghost_mapping_valid =
				duplicate_ranks.size() == ghost_indices.size();
			for(size_t peer = 0; peer < ghost_indices.size(); ++peer)
			{
				if(peer >= duplicate_ranks.size())
				{
					ghost_mapping_valid = false;
					continue;
				}
				for(size_t ghost : ghost_indices[peer])
				{
					if(duplicate_ranks[peer] < 0 ||
					   duplicate_ranks[peer] >= rank_count ||
					   ghost >= ghost_owner.size() ||
					   (ghost_owner[ghost] >= 0 &&
					    ghost_owner[ghost] != duplicate_ranks[peer]))
					{
						ghost_mapping_valid = false;
						continue;
					}
					ghost_owner[ghost] = duplicate_ranks[peer];
				}
			}
			if(!all_ranks(ghost_mapping_valid))
				return full_build("remote ghost ownership invariant");

			vector<size_t> canonical_indices(canonical_count);
			std::iota(canonical_indices.begin(), canonical_indices.end(), 0);
			vector<size_t> mesh_owner_indices(
				tess_.GetPointNo(), ActiveMeshView::invalidIndex());
			for(size_t local = 0; local < view.localSize(); ++local)
				mesh_owner_indices[local] = view.localToGlobal(local);
			tess_.SyncPartialBuildData(mesh_owner_indices, canonical_indices);
			vector<unsigned char> canonical_target = included;
			vector<unsigned char> mesh_target(tess_.GetPointNo(), 0);
			for(size_t local = 0; local < view.localSize(); ++local)
				mesh_target[local] = included[view.localToGlobal(local)];
			tess_.SyncPartialBuildData(mesh_target, canonical_target);
			vector<vector<size_t> > remote_requests(
				static_cast<size_t>(rank_count));
#endif
			vector<size_t> frontier;
			frontier.reserve(target.size());
			for(size_t global : target)
				if(depth[global] < 2 && view.containsGlobal(global))
					frontier.push_back(global);
			vector<unsigned char> expanded_depth(canonical_count, 255);
			while(!frontier.empty())
			{
				size_t const global = frontier.back();
				frontier.pop_back();
				if(depth[global] >= 2 ||
				   expanded_depth[global] <= depth[global] ||
				   !view.containsGlobal(global))
					continue;
				expanded_depth[global] = depth[global];
				size_t const local = view.globalToLocal(global);
				if(local >= tess_.GetPointNo())
					continue;
				tess_.GetNeighbors(local, neighbors);
				for(size_t neighbor : neighbors)
				{
					size_t const neighbor_global = view.meshLocalToGlobal(neighbor);
					unsigned char const neighbor_depth =
						static_cast<unsigned char>(depth[global] + 1);
					if(neighbor_global >= canonical_count)
					{
#ifdef RICH_MPI
						if(neighbor < ghost_owner.size() &&
						   ghost_owner[neighbor] >= 0 &&
						   ghost_owner[neighbor] != rank &&
						   neighbor < mesh_owner_indices.size() &&
						   mesh_owner_indices[neighbor] !=
						       ActiveMeshView::invalidIndex())
						{
							size_t const owner = static_cast<size_t>(
								ghost_owner[neighbor]);
							remote_requests[owner].push_back(
								mesh_owner_indices[neighbor]);
							remote_requests[owner].push_back(
								static_cast<size_t>(neighbor_depth));
						}
#endif
						continue;
					}
					if(!included[neighbor_global])
					{
						included[neighbor_global] = 1;
						depth[neighbor_global] = neighbor_depth;
						additions.push_back(neighbor_global);
					}
					else if(neighbor_depth < depth[neighbor_global])
					{
						depth[neighbor_global] = neighbor_depth;
						if(neighbor_depth < 2 && view.containsGlobal(neighbor_global))
							frontier.push_back(neighbor_global);
					}
				}
			}

#ifdef RICH_MPI
			vector<vector<size_t> > const incoming_requests =
				MPI_Exchange_all_to_all(remote_requests, MPI_COMM_WORLD);
			bool remote_request_valid =
				incoming_requests.size() == static_cast<size_t>(rank_count);
			for(auto const& requests : incoming_requests)
			{
				if(requests.size() % 2 != 0)
				{
					remote_request_valid = false;
					continue;
				}
				for(size_t request = 0; request < requests.size(); request += 2)
				{
					size_t const global = requests[request];
					size_t const requested_depth = requests[request + 1];
					if(global >= canonical_count || requested_depth > 2)
					{
						remote_request_valid = false;
						continue;
					}
					unsigned char const new_depth =
						static_cast<unsigned char>(requested_depth);
					if(!included[global])
					{
						included[global] = 1;
						depth[global] = new_depth;
						additions.push_back(global);
						closure_changed = true;
					}
					else if(new_depth < depth[global])
					{
						depth[global] = new_depth;
						closure_changed = true;
					}
				}
			}
			if(!all_ranks(remote_request_valid))
				return full_build("remote closure request invariant");
#endif
			closure_changed = closure_changed || !additions.empty();

			if(!any_rank(closure_changed))
			{
				bool closed = true;
				size_t closure_active = ActiveMeshView::invalidIndex();
				size_t closure_neighbor = ActiveMeshView::invalidIndex();
				size_t closure_neighbor_global = ActiveMeshView::invalidIndex();
				for(size_t global : context.active_indices)
				{
					if(!view.containsGlobal(global))
					{
						closed = false;
						closure_active = global;
						break;
					}
					size_t const local = view.globalToLocal(global);
					tess_.GetNeighbors(local, neighbors);
					for(size_t neighbor : neighbors)
					{
						size_t const neighbor_global = view.meshLocalToGlobal(neighbor);
						if(neighbor_global < canonical_count &&
						   neighbor >= tess_.GetPointNo())
						{
							closed = false;
							closure_active = global;
							closure_neighbor = neighbor;
							closure_neighbor_global = neighbor_global;
							break;
						}
#ifdef RICH_MPI
						if(neighbor_global >= canonical_count &&
						   neighbor < ghost_owner.size() &&
						   ghost_owner[neighbor] >= 0 &&
						   (neighbor >= mesh_target.size() ||
						    mesh_target[neighbor] == 0))
						{
							closed = false;
							closure_active = global;
							closure_neighbor = neighbor;
							closure_neighbor_global =
								neighbor < mesh_owner_indices.size() ?
								mesh_owner_indices[neighbor] :
								ActiveMeshView::invalidIndex();
							break;
						}
#endif
					}
					if(!closed)
						break;
				}
				bool const globally_closed = all_ranks(closed);
				if(globally_closed)
				{
						if(!context.verify_partial_build)
						{
							cached_target = target;
							remember_partial_target(target);
							return std::move(*view_ptr);
					}

					vector<IndividualCellGeometry> const partial_geometry =
						CaptureActiveGeometry(tess_, view, context.active_indices,
							canonical_count);
					auto const reference_start = get_time();
#ifdef RICH_MPI
					tess_.BuildParallel(points, true, true);
#else
					tess_.Build(points);
#endif
					auto const reference_finish = get_time();
					DisplayTime(reference_start, reference_finish,
						"Individual full-reference Voronoi parity build ");
					std::unique_ptr<ActiveMeshView> full_view;
					bool full_mapping_failed = false;
					try
					{
						full_view.reset(new ActiveMeshView(tess_, canonical_count));
					}
					catch(...)
					{
						full_mapping_failed = true;
					}
					if(any_rank(full_mapping_failed))
						return full_build("full-reference mapping invariant");
					bool const geometry_matches = ActiveGeometryMatches(partial_geometry,
						tess_, *full_view, canonical_count);
						if(!all_ranks(geometry_matches))
						{
							std::cout << "Individual partial Voronoi fallback: debug parity mismatch"
								<< std::endl;
							cached_target.clear();
							individual_mesh_target_ids_.clear();
							return std::move(*full_view);
					}
					bool rebuild_failed = false;
					try
					{
#ifdef RICH_MPI
						tess_.BuildPartiallyParallel(points, target, true, true);
#else
						tess_.BuildPartially(points, target);
#endif
					}
					catch(...)
					{
						rebuild_failed = true;
					}
					if(any_rank(rebuild_failed))
						return full_build("partial rebuild after parity check failed");
					std::unique_ptr<ActiveMeshView> rebuilt_view;
					bool rebuilt_mapping_failed = false;
					try
					{
						rebuilt_view.reset(new ActiveMeshView(tess_, canonical_count));
					}
					catch(...)
					{
						rebuilt_mapping_failed = true;
					}
						if(any_rank(rebuilt_mapping_failed))
							return full_build("partial rebuild mapping invariant");
						cached_target = target;
						remember_partial_target(target);
					return std::move(*rebuilt_view);
				}
				if(!closed)
				{
#ifdef RICH_MPI
					int rank = 0;
					MPI_Comm_rank(MPI_COMM_WORLD, &rank);
					std::cerr << "Individual partial closure incomplete on rank " << rank
						<< ": active=" << closure_active
						<< " neighbor_local=" << closure_neighbor
						<< " neighbor_global=" << closure_neighbor_global
						<< " target_size=" << target.size();
					if(closure_neighbor_global < included.size())
						std::cerr << " included="
							<< static_cast<int>(included[closure_neighbor_global]);
					if(closure_neighbor < tess_.getMeshPoints().size())
						std::cerr << " geometric_owner="
							<< tess_.GetOwner(tess_.GetMeshPoint(closure_neighbor));
					std::cerr << std::endl;
#endif
				}
				return full_build("incomplete active reconstruction closure");
			}

			target.insert(target.end(), additions.begin(), additions.end());
			if(any_rank(target.size() > full_threshold))
				return full_build("expanded closure threshold");
		}
	};

	StrictBooleanEnvironment const selected_flag =
		ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_REUSE_EVENT_MESH");
	StrictBooleanEnvironment const shadow_flag =
		ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_REUSE_EVENT_MESH_SHADOW");
	IndividualEventMeshReuseMode local_reuse_mode =
		IndividualEventMeshReuseMode::Off;
	if(!selected_flag.valid || !shadow_flag.valid)
		local_reuse_mode = IndividualEventMeshReuseMode::Invalid;
	else if(selected_flag.value && shadow_flag.value)
		local_reuse_mode = IndividualEventMeshReuseMode::Conflict;
	else if(selected_flag.value)
		local_reuse_mode = IndividualEventMeshReuseMode::Selected;
	else if(shadow_flag.value)
		local_reuse_mode = IndividualEventMeshReuseMode::Shadow;

	int reuse_mode_extrema[2] = {
		static_cast<int>(local_reuse_mode),
		-static_cast<int>(local_reuse_mode)};
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, reuse_mode_extrema, 2, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
#endif
	int const minimum_reuse_mode = -reuse_mode_extrema[1];
	int const maximum_reuse_mode = reuse_mode_extrema[0];
	bool const reuse_mode_consistent =
		minimum_reuse_mode == maximum_reuse_mode;
	IndividualEventMeshReuseMode const reuse_mode = reuse_mode_consistent ?
		static_cast<IndividualEventMeshReuseMode>(minimum_reuse_mode) :
		IndividualEventMeshReuseMode::Invalid;
	bool const reuse_gate_requested = reuse_mode_consistent &&
		(reuse_mode == IndividualEventMeshReuseMode::Selected ||
		 reuse_mode == IndividualEventMeshReuseMode::Shadow);

	IndividualEventMeshReuseValidationState* reuse_state = nullptr;
	if(reuse_gate_requested)
	{
		reuse_state = &IndividualEventMeshReuseValidationStates()[this];
		++reuse_state->attempts;
	}
	bool local_cache_state_failure = false;
	bool local_mapping_failure = false;
	bool local_epoch_failure = false;
	unsigned long long local_id_mismatches = 0;
	unsigned long long local_coordinate_mismatches = 0;
	std::unique_ptr<ActiveMeshView> reusable_view;
	if(reuse_gate_requested)
	{
		local_cache_state_failure = !individual_event_mesh_reusable_ ||
			!individual_mesh_target_ids_.empty() ||
			tess_.GetPointNo() != canonical_count;
		local_epoch_failure = reuse_state->cached_epoch != reuse_state->epoch;
		if(reuse_state->canonical_ids.size() != canonical_count)
			local_id_mismatches = 1;
		else
			for(size_t global = 0; global < canonical_count; ++global)
				if(reuse_state->canonical_ids[global] != cells_[global].ID)
					++local_id_mismatches;
		if(!local_cache_state_failure)
		{
			try
			{
				reusable_view.reset(new ActiveMeshView(tess_, canonical_count));
				if(reusable_view->localSize() != canonical_count ||
				   reusable_view->globalSize() != canonical_count)
					local_mapping_failure = true;
				for(size_t local = 0;
					!local_mapping_failure && local < reusable_view->localSize(); ++local)
				{
					size_t const global = reusable_view->localToGlobal(local);
					if(global >= canonical_count ||
					   !reusable_view->containsGlobal(global) ||
					   reusable_view->globalToLocal(global) != local)
					{
						local_mapping_failure = true;
						break;
					}
					Vector3D const& cached = tess_.GetMeshPoint(local);
					Vector3D const& expected = all_points[global];
					if(cached.x != expected.x || cached.y != expected.y ||
					   cached.z != expected.z)
						++local_coordinate_mismatches;
				}
			}
			catch(...)
			{
				local_mapping_failure = true;
				reusable_view.reset();
			}
		}
	}
	bool const local_reusable_event_mesh = reuse_gate_requested &&
		!local_cache_state_failure && !local_mapping_failure &&
		!local_epoch_failure && local_id_mismatches == 0 &&
		local_coordinate_mismatches == 0 && reusable_view.get() != nullptr;
	// Eligibility and mode are collective: every rank either reuses the cached
	// full mesh or every rank follows the ordinary rebuilding path.
	bool const reuse_event_mesh = reuse_gate_requested &&
		!any_rank(!local_reusable_event_mesh);
	bool shadow_attempted = false;
	bool shadow_geometry_matches = false;
	finish_phase(0);
	vector<ComputationalCell3D> local_cells;
	vector<Conserved3D> local_extensives;
	const double event_dt = context.event_time - context.previous_event_time;
	{
	ActiveMeshView first_view = [&]() -> ActiveMeshView
	{
		if(!reuse_event_mesh)
			return build_event_mesh(all_points, seed, warm_target,
				"timeAdvanceIndividual first-half mesh", false);
		if(reuse_mode == IndividualEventMeshReuseMode::Shadow)
		{
			shadow_attempted = true;
			vector<IndividualCellGeometry> cached_geometry;
			bool local_capture_failure = false;
			try
			{
				vector<size_t> canonical_indices(canonical_count);
				std::iota(canonical_indices.begin(), canonical_indices.end(), 0);
				cached_geometry = CaptureActiveGeometry(tess_, *reusable_view,
					canonical_indices, canonical_count);
			}
			catch(...)
			{
				local_capture_failure = true;
			}
			bool const capture_failure = any_rank(local_capture_failure);
			ActiveMeshView rebuilt = build_event_mesh(all_points, seed, warm_target,
				"timeAdvanceIndividual first-half mesh shadow", true);
			if(!capture_failure)
			{
				bool local_geometry_matches = false;
				try
				{
					local_geometry_matches = ActiveGeometryMatches(cached_geometry,
						tess_, rebuilt, canonical_count, false);
				}
				catch(...)
				{
					local_geometry_matches = false;
				}
				shadow_geometry_matches = all_ranks(local_geometry_matches);
			}
			return rebuilt;
		}
		auto const start = get_time();
		ActiveMeshView result = std::move(*reusable_view);
		auto const finish = get_time();
		DisplayTime(start, finish,
			"Individual full Voronoi reuse (previous event mesh) ");
		return result;
	}();
	finish_phase(1);
	bool const gate_failed_closed = !reuse_mode_consistent ||
		reuse_mode == IndividualEventMeshReuseMode::Conflict ||
		reuse_mode == IndividualEventMeshReuseMode::Invalid ||
		(reuse_gate_requested && !reuse_event_mesh) ||
		(shadow_attempted && !shadow_geometry_matches);
	if(reuse_state != nullptr)
	{
		if(reuse_mode == IndividualEventMeshReuseMode::Selected && reuse_event_mesh)
			++reuse_state->selected_hits;
		if(shadow_attempted)
			++reuse_state->shadow_checks;
		if(gate_failed_closed)
			++reuse_state->failed_closed;
	}
	if(minimum_reuse_mode != static_cast<int>(IndividualEventMeshReuseMode::Off) ||
	   maximum_reuse_mode != static_cast<int>(IndividualEventMeshReuseMode::Off))
	{
		unsigned long long aggregate[6] = {
			local_reusable_event_mesh ? 1ULL : 0ULL,
			local_cache_state_failure ? 1ULL : 0ULL,
			local_mapping_failure ? 1ULL : 0ULL,
			local_epoch_failure ? 1ULL : 0ULL,
			local_id_mismatches,
			local_coordinate_mismatches};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, aggregate, 6, MPI_UNSIGNED_LONG_LONG,
			MPI_SUM, MPI_COMM_WORLD);
		int rank = 0;
		int rank_count = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
#else
		int const rank = 0;
		int const rank_count = 1;
#endif
		if(rank == 0)
		{
			char const* const mode_name = reuse_mode_consistent ?
				IndividualEventMeshReuseModeName(reuse_mode) : "inconsistent";
			std::cout << "INDIVIDUAL_EVENT_MESH_REUSE mode=" << mode_name
				<< " ranks=" << rank_count
				<< " eligible_ranks=" << aggregate[0]
				<< " cache_state_failures=" << aggregate[1]
				<< " mapping_failures=" << aggregate[2]
				<< " epoch_failures=" << aggregate[3]
				<< " id_mismatches=" << aggregate[4]
				<< " coordinate_mismatches=" << aggregate[5]
				<< " selected="
				<< (reuse_mode == IndividualEventMeshReuseMode::Selected &&
					reuse_event_mesh ? 1 : 0)
				<< " shadow_checked=" << (shadow_attempted ? 1 : 0)
				<< " shadow_match="
				<< (shadow_attempted ? (shadow_geometry_matches ? 1 : 0) : -1)
				<< " failed_closed=" << (gate_failed_closed ? 1 : 0);
			if(reuse_state != nullptr)
				std::cout << " attempts_total=" << reuse_state->attempts
					<< " selected_total=" << reuse_state->selected_hits
					<< " shadow_total=" << reuse_state->shadow_checks
					<< " failed_closed_total=" << reuse_state->failed_closed;
			std::cout << std::endl;
		}
	}
		for(size_t local = 0; local < first_view.localSize(); ++local)
		{
			size_t const global = first_view.localToGlobal(local);
			if(context.isActive(global))
				individual_centroids_[global] = tess_.GetCellCM(local);
		}
	context.gravity_source_points = individual_centroids_;
	IndividualStepContext first_context = first_view.remapContext(context);
	first_view.gatherOwnedInto(cells_, local_cells);
	first_view.gatherOwnedInto(extensive_, local_extensives);
	tess_.SyncPartialBuildData(local_cells, cells_);
	tess_.SyncPartialBuildData(local_extensives, extensive_);
	finish_phase(2);
	fluxes.assign(tess_.GetTotalFacesNumber(), Conserved3D());
	point_vel.clear();
	face_vel.clear();
	face_values.clear();
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, first_context);
	pm_(tess_, local_cells, context.previous_event_time, point_vel);
	bool const first_mesh_is_partial = any_rank(
		first_view.localSize() != canonical_count);
	if(first_mesh_is_partial)
	{
		vector<Vector3D> all_point_velocities = context.point_velocities;
		pm_.ApplyFixIndividual(tess_, local_cells, cells_,
			context.previous_event_time, event_dt, point_vel,
			all_point_velocities);
	}
	else
		pm_.ApplyFix(tess_, local_cells, context.previous_event_time,
			event_dt, point_vel);
	// Point velocities are predictor state.  Updating passive closure cells here
	// would make their future generator and centroid predictions depend on whether
	// this event used a full or partial mesh.  Only cells whose interval begins at
	// this event may replace their cached velocity.
	for(size_t local : first_context.active_indices)
	{
		if(local >= point_vel.size())
			throw std::logic_error(
				"Individual point-velocity update is outside the event mesh");
		context.point_velocities.at(first_view.localToGlobal(local)) =
			point_vel[local];
	}
	source_.ApplyIndividual(tess_, local_cells, fluxes, point_vel,
		context.previous_event_time, first_context, IndividualSourcePhase::FirstHalf,
		local_extensives);
	for(size_t local : first_context.active_indices)
	{
		size_t const global = first_view.localToGlobal(local);
		context.cached_accelerations[global] = first_context.cached_accelerations[local];
		context.gravity_half_kick_pending[global] =
			first_context.gravity_half_kick_pending[local];
	}
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, first_context);
	first_view.scatterOwned(local_cells, cells_);
	first_view.scatterOwned(local_extensives, extensive_);
	finish_phase(3);
	}

	auto const prediction_start = get_time();
	if(pm_.MovedPoints())
		for(size_t global = 0; global < all_points.size(); ++global)
		{
			all_points[global] += context.point_velocities[global] * event_dt;
			individual_centroids_[global] +=
				context.point_velocities[global] * event_dt;
		}
	context.gravity_source_points = individual_centroids_;
	auto const prediction_finish = get_time();
	DisplayTime(prediction_start, prediction_finish, "Individual generator prediction ");
	finish_phase(4);

	ActiveMeshView event_view = build_event_mesh(all_points, seed, warm_target,
		"timeAdvanceIndividual event mesh", false);
		for(size_t local = 0; local < event_view.localSize(); ++local)
		{
			size_t const global = event_view.localToGlobal(local);
			if(context.isActive(global))
				individual_centroids_[global] = tess_.GetCellCM(local);
		}
	context.gravity_source_points = individual_centroids_;
	finish_phase(5);
	IndividualStepContext local_context = event_view.remapContext(context);
	event_view.gatherOwnedInto(cells_, local_cells);
	event_view.gatherOwnedInto(extensive_, local_extensives);
	tess_.SyncPartialBuildData(local_cells, cells_);
	tess_.SyncPartialBuildData(local_extensives, extensive_);
	event_view.gatherOwnedInto(context.point_velocities, point_vel);
	vector<Vector3D> all_point_velocities = context.point_velocities;
	tess_.SyncPartialBuildData(point_vel, all_point_velocities);
	context.point_velocities = all_point_velocities;
	finish_phase(6);

	fluxes.assign(tess_.GetTotalFacesNumber(), Conserved3D());
	face_values.clear();
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, local_context);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	if(auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&tsc_))
		cfl->SetPointVelocities(&point_vel);
	else if(auto* cfl1d = dynamic_cast<CFL1D*>(&tsc_))
		cfl1d->SetPointVelocities(&point_vel);
	finish_phase(7);
	GetFullStateFluxCalculator().CalculateIndividual(fluxes, tess_, face_vel,
		local_cells, local_extensives, eos_, local_context, face_values);
	finish_phase(8);
	eu_.UpdateIndividual(fluxes, tess_, local_context, local_cells, local_extensives,
		context.event_time, face_vel, point_vel, face_values, &cells_, &extensive_);
	if(local_context.gravity_source_masses.size() != extensive_.size())
		throw std::logic_error(
			"Individual gravity source mass cache has the wrong canonical size");
	for(size_t global = 0; global < extensive_.size(); ++global)
		local_context.gravity_source_masses[global] = extensive_[global].mass;
	// Second-half source terms may depend on primitive fields; recover them
	// from the flux-updated extensive state before source evaluation.
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, local_context);
	finish_phase(9);
	source_.ApplyIndividual(tess_, local_cells, fluxes, point_vel,
		context.event_time, local_context, IndividualSourcePhase::SecondHalf,
		local_extensives);
	for(size_t local = 0; local < event_view.localSize(); ++local)
	{
		size_t const global = event_view.localToGlobal(local);
		context.cached_accelerations[global] = local_context.cached_accelerations[local];
		context.gravity_half_kick_pending[global] =
			local_context.gravity_half_kick_pending[local];
	}
	finish_phase(10);
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, local_context);
	event_view.scatterOwned(local_cells, cells_);
	event_view.scatterOwned(local_extensives, extensive_);
	individual_points_ = all_points;
	individual_event_mesh_reusable_ =
		event_view.localSize() == canonical_count &&
		individual_mesh_target_ids_.empty();
	if(reuse_state != nullptr)
	{
		if(individual_event_mesh_reusable_)
		{
			reuse_state->canonical_ids.resize(canonical_count);
			for(size_t global = 0; global < canonical_count; ++global)
				reuse_state->canonical_ids[global] = cells_[global].ID;
			reuse_state->cached_epoch = reuse_state->epoch;
		}
		else
		{
			reuse_state->canonical_ids.clear();
			reuse_state->cached_epoch = std::numeric_limits<size_t>::max();
		}
	}
	finish_phase(11);
	if(phase_trace)
	{
		phase_record[individual_hydro_total_index] = std::accumulate(
			phase_record.begin(),
			phase_record.begin() + individual_hydro_phase_count, 0.0);
		phase_record[individual_hydro_active_index] =
			static_cast<double>(context.active_indices.size());
		phase_record[individual_hydro_canonical_index] =
			static_cast<double>(canonical_count);
		ReportIndividualHydroPhaseTiming(phase_record);
	}
}

void HDSim3D::suggestIndividualTimeSteps(const IndividualStepContext& context,
	vector<double>& time_step_limits) const
{
	ActiveMeshView const view(tess_, context.active_mask.size());
	IndividualStepContext const local_context = view.remapContext(context, false);
	vector<ComputationalCell3D> local_cells = view.gatherOwned(cells_);
	vector<ComputationalCell3D> all_cells = cells_;
	tess_.SyncPartialBuildData(local_cells, all_cells);
	vector<double> local_limits(tess_.GetPointNo(),
		std::numeric_limits<double>::max());
	tsc_.SuggestIndividualTimeSteps(tess_, local_cells, eos_, face_vel_scratch_,
		context.event_time, local_context, local_limits);
	for(size_t local = 0; local < view.localSize(); ++local)
	{
		size_t const global = view.localToGlobal(local);
		time_step_limits.at(global) = std::min(
			time_step_limits.at(global), local_limits.at(local));
	}
}

void HDSim3D::timeAdvanceLagrangian1D(
	const ComputationalCell3D* left_external,
	const ComputationalCell3D* right_external)
{
	RefreshSphericalShellGeometry("timeAdvanceLagrangian1D entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvanceLagrangian1D");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();
	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();

	std::pair<Vector3D, Vector3D> orig_box = tess_.GetBoxCoordinates();
	Vector3D orig_ll = orig_box.first;
	Vector3D orig_ur = orig_box.second;
	Hllc3D hllc_local;

	std::pair<std::vector<size_t>, std::vector<size_t> > xfaces = FindXBoundaryFaces(tess_);
	std::vector<size_t>& left_faces = xfaces.first;
	std::vector<size_t>& right_faces = xfaces.second;

	// ---- Phase A: predictor at time t ----
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	SetBoundaryFaceVelocities(left_faces, tess_, cells_, eos_, hllc_local, left_external, face_vel);
	SetBoundaryFaceVelocities(right_faces, tess_, cells_, eos_, hllc_local, right_external, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	SetBoundaryFaceVelocities(left_faces, tess_, cells_, eos_, hllc_local, left_external, face_vel);
	SetBoundaryFaceVelocities(right_faces, tess_, cells_, eos_, hllc_local, right_external, face_vel);
	if (auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&tsc_))
		cfl->SetPointVelocities(&point_vel);
	else if (auto* cfl1d = dynamic_cast<CFL1D*>(&tsc_))
		cfl1d->SetPointVelocities(&point_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);

	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);

	double vx_left_A = OverrideBoundaryFluxes(
		left_faces, tess_, cells_, eos_, hllc_local, left_external, fluxes, face_vel);
	double vx_right_A = OverrideBoundaryFluxes(
		right_faces, tess_, cells_, eos_, hllc_local, right_external, fluxes, face_vel);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &vx_left_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vx_right_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	auto t1 = get_time();
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);
	auto t2 = get_time();
	DisplayTime(t1, t2, "Source time ");

	// if (pt_.getCycle() % 10 == 0 && pm_.MovedPoints())
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
// #ifdef RICH_MPI
		// tess_.PreparePoints(mesh, order);
// #endif
	// }

	if (pm_.MovedPoints())
		MovePoints(tess_, point_vel, dt);
	Vector3D new_ll = orig_ll;
	Vector3D new_ur = orig_ur;
	new_ll.x += vx_left_A * dt;
	new_ur.x += vx_right_A * dt;
#ifdef RICH_MPI
	SetBoxAndRebuild(tess_, new_ll, new_ur, this->tessellation_points_scratch_, this->exchange_chain_);
#else
	SetBoxAndRebuild(tess_, new_ll, new_ur, this->tessellation_points_scratch_);
#endif
	t1 = get_time();
	t2 = get_time();

#ifdef RICH_MPI
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif

	cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	// ---- Phase B: corrector at time t + dt ----
	std::pair<std::vector<size_t>, std::vector<size_t> > xfaces_B = FindXBoundaryFaces(tess_);
	std::vector<size_t>& left_faces_B = xfaces_B.first;
	std::vector<size_t>& right_faces_B = xfaces_B.second;

	CalcFaceVelocities(tess_, point_vel, face_vel);
	SetBoundaryFaceVelocities(left_faces_B, tess_, cells_, eos_, hllc_local, left_external, face_vel);
	SetBoundaryFaceVelocities(right_faces_B, tess_, cells_, eos_, hllc_local, right_external, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);

	double vx_left_B = OverrideBoundaryFluxes(
		left_faces_B, tess_, cells_, eos_, hllc_local, left_external, fluxes, face_vel);
	double vx_right_B = OverrideBoundaryFluxes(
		right_faces_B, tess_, cells_, eos_, hllc_local, right_external, fluxes, face_vel);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &vx_left_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vx_right_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	t1 = get_time();
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	t2 = get_time();
	DisplayTime(t1, t2, "Second source time ");
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	ExtensiveAvg(extensive_, mid_extensives);

	Vector3D final_ll = orig_ll;
	Vector3D final_ur = orig_ur;
	final_ll.x += 0.5 * (vx_left_A + vx_left_B) * dt;
	final_ur.x += 0.5 * (vx_right_A + vx_right_B) * dt;
#ifdef RICH_MPI
	SetBoxAndRebuild(tess_, final_ll, final_ur, this->tessellation_points_scratch_, this->exchange_chain_);
#else
	SetBoxAndRebuild(tess_, final_ll, final_ur, this->tessellation_points_scratch_);
#endif

#ifdef RICH_MPI
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif

	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
}

void HDSim3D::timeAdvance(void)
{
	RefreshSphericalShellGeometry("timeAdvance entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance");
	MEMORY_DEBUG_PRINT("hydro1: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	const double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, dt, extensive_);
	eu_(fluxes, tess_, dt, cells_, extensive_, time, face_vel, point_vel, face_values);
	if(pm_.MovedPoints())
	{
	MovePoints(tess_, point_vel, dt);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance mesh build", true);

	#ifdef RICH_MPI
	// Keep relevant points
	ComputationalCell3D cdummy;
	Conserved3D edummy;
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif
	}
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro1: end");
}


void HDSim3D::timeAdvance3(void)
{
	RefreshSphericalShellGeometry("timeAdvance3 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance3");
	MEMORY_DEBUG_PRINT("hydro3: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Vector3D> &oldpoints = this->oldpoints_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	vector<Conserved3D> &u1 = this->u1_scratch_;
	vector<Conserved3D> &u2 = this->u2_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, 0.5 * dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, 0.5 * dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt * 0.5);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance3 first mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	//MPI_exchange_data(tess_, du1, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	u1 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, 2 * dt, face_values);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, 2 * dt,  mid_extensives);
	eu_(fluxes, tess_, 2 * dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);
	mid_extensives = mid_extensives - 3 * (u1 - extensive_);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance3 second mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, u1, false);
	//MPI_exchange_data(tess_, du2, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	u2 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt / 6, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt / 6,  mid_extensives);
	eu_(fluxes, tess_, dt / 6, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	extensive_ = mid_extensives - (1.0 / 3.0) * (2 * u2 + extensive_) + u1;
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro3: end");
}

void HDSim3D::timeAdvance33(void)
{
	RefreshSphericalShellGeometry("timeAdvance33 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance33");
	MEMORY_DEBUG_PRINT("hydro33: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Vector3D> &oldpoints = this->oldpoints_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance33 first mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	mid_extensives = 0.25 * mid_extensives + 0.75 * extensive_;

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt / 2, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt / 2, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance33 second mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);
	extensive_ = 0.33333333333333333333333 * (2 * mid_extensives + extensive_);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance33 third mesh build", true);

#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif

	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro33: end");
}

void HDSim3D::timeAdvance32(void)
{
	RefreshSphericalShellGeometry("timeAdvance32 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance32");
	MEMORY_DEBUG_PRINT("hydro32: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	vector<Conserved3D> &u1 = this->u1_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, 0.5 * dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	MovePoints(tess_, point_vel, dt);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance32 mesh build", true);

	#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	u1 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	mid_extensives = 0.5 * (mid_extensives + extensive_);
	cu_(cells_, eos_, tess_, mid_extensives);

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	//extensive_ = 0.333333333333333333*(extensive_ + u1 + mid_extensives);
	extensive_ = 0.333333333333333333 * (extensive_ + u1 + mid_extensives);
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro32: end");
}

void HDSim3D::timeAdvance4(void)
{
	RefreshSphericalShellGeometry("timeAdvance4 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance4");
	MEMORY_DEBUG_PRINT("hydro4: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Vector3D> &oldpoints = this->oldpoints_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	vector<Conserved3D> &du1 = this->u1_scratch_;
	vector<Conserved3D> &du2 = this->u2_scratch_;
	vector<Conserved3D> &du3 = this->u3_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, 0.5 * dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, 0.5 * dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt * 0.5);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance4 first mesh build", true);

#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	//MPI_exchange_data(tess_, du1, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);
	du1 = mid_extensives - extensive_;

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, 0.5 * dt, face_values);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, 0.5 * dt, mid_extensives);
	mid_extensives = mid_extensives - du1;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);
	cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	du2 = mid_extensives - extensive_;

	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, dt, mid_extensives);
	mid_extensives = mid_extensives - du2;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance4 second mesh build", true);

#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, du1, false);
	MPI_exchange_data(tess_, du2, false);
	//MPI_exchange_data(tess_, du3, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);
	du3 = mid_extensives - extensive_;

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt / 6, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt / 6,  mid_extensives);
	mid_extensives = mid_extensives - du3;
	eu_(fluxes, tess_, dt / 6, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	extensive_ = mid_extensives + (1.0 / 6.0) * (2 * du1 + 4 * du2 + 2 * du3);
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro4: end");
}

const Tessellation3D& HDSim3D::getTessellation(void) const
{
	return tess_;
}

const vector<ComputationalCell3D>& HDSim3D::getCells(void) const
{
	return cells_;
}

double HDSim3D::getTime(void) const
{
	return pt_.getTime();
}

size_t HDSim3D::getCycle(void) const
{
	return static_cast<size_t>(pt_.getCycle());
}
