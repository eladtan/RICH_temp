#include "hdsim_3d.hpp"
#include "CourantFriedrichsLewy.hpp"
#include "CFL1D.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
#include "Rusanov3D.hpp"
#include "spherical_symmetry/SphericalShellProjector3D.hpp"


namespace
{
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
	std::shared_ptr<SphericalShellProjector3D> projector)
{
	spherical_shell_projector_ = std::move(projector);
}

void HDSim3D::SetSphericalPositivityPreservingStage(bool enabled)
{
	spherical_positivity_preserving_ = enabled;
}

void HDSim3D::CalculateSphericalLowOrderFluxes(
	vector<ComputationalCell3D> const& cells,
	vector<Vector3D> const& face_velocities,
	vector<Conserved3D>& fluxes) const
{
	size_t const face_count = tess_.GetTotalFacesNumber();
	fluxes.resize(face_count);
	Rusanov3D solver;
	for (size_t face = 0; face < face_count; ++face) {
		auto const neighbors = tess_.GetFaceNeighbors(face);
		bool first_valid = neighbors.first < cells.size();
		bool second_valid = neighbors.second < cells.size();
		if (tess_.BoundaryFace(face)) {
			// MPI halo entries extend cells beyond GetPointNo().  A physical
			// boundary sentinel can therefore be numerically smaller than
			// cells.size(); use the tessellation's boundary orientation instead.
			first_valid = neighbors.first <= tess_.GetPointNo();
			second_valid = !first_valid;
		}
		if (!(first_valid || second_valid))
			throw UniversalError("Low-order face has no valid neighbor");

		Vector3D const normal = normalize(tess_.Normal(face));
		ComputationalCell3D left =
			first_valid ? cells[neighbors.first] : cells[neighbors.second];
		ComputationalCell3D right =
			second_valid ? cells[neighbors.second] : cells[neighbors.first];
		if (!first_valid) {
			left = right;
			left.velocity -= 2.0 *
				ScalarProd(left.velocity, normal) * normal;
		}
		if (!second_valid) {
			right = left;
			right.velocity -= 2.0 *
				ScalarProd(right.velocity, normal) * normal;
		}
		fluxes[face] = solver(left, right,
			ScalarProd(normal, face_velocities[face]), eos_, normal);
	}
}

void HDSim3D::ApplyFluxesWithoutValidation(
	vector<Conserved3D> const& fluxes,
	vector<ComputationalCell3D> const&,
	double dt,
	vector<Conserved3D>& candidate) const
{
	size_t const local_count = tess_.GetPointNo();
	for (size_t face = 0; face < fluxes.size(); ++face) {
		Conserved3D delta =
			fluxes[face] * (dt * tess_.GetArea(face));
		delta.internal_energy = 0;
		auto const neighbors = tess_.GetFaceNeighbors(face);
		if (neighbors.first < local_count)
			candidate[neighbors.first] -= delta;
		if (neighbors.second < local_count)
			candidate[neighbors.second] += delta;
	}
	for (size_t i = 0; i < local_count; ++i) {
		if (candidate[i].mass > 0)
			candidate[i].internal_energy = candidate[i].energy -
				ScalarProd(candidate[i].momentum,
					candidate[i].momentum) /
				(2.0 * candidate[i].mass);
	}
	candidate.resize(local_count);
}

void HDSim3D::ApplySphericalBackgroundCorrection(
	vector<ComputationalCell3D> const& stage_input_cells,
	vector<Conserved3D> const& stage_input_extensives,
	vector<Vector3D> const& face_velocities,
	vector<Vector3D> const& point_velocities,
	double time,
	double dt,
	bool source_before_extensive_update,
	vector<Conserved3D>& full_candidate,
	bool low_order)
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
	if (low_order)
		CalculateSphericalLowOrderFluxes(background_cells,
			face_velocities, background_fluxes);
	else
		fc_.Calculate(background_fluxes, tess_, face_velocities,
			background_cells, background_extensives, eos_, time, dt,
			background_face_values);

	vector<Conserved3D> background_candidate = background_extensives;
	if (source_before_extensive_update) {
		source_(tess_, background_cells, background_fluxes, point_velocities,
			time, dt, background_candidate);
		if (spherical_positivity_preserving_)
			ApplyFluxesWithoutValidation(background_fluxes,
				background_cells, dt, background_candidate);
		else
			eu_(background_fluxes, tess_, dt, background_cells,
				background_candidate, time, face_velocities,
				point_velocities, background_face_values);
	}
	else {
		if (spherical_positivity_preserving_)
			ApplyFluxesWithoutValidation(background_fluxes,
				background_cells, dt, background_candidate);
		else
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

void HDSim3D::BuildSphericalStageCandidate(
	vector<ComputationalCell3D> const& stage_input_cells,
	vector<Conserved3D> const& stage_input_extensives,
	vector<Vector3D> const& face_velocities,
	vector<Vector3D> const& point_velocities,
	double time,
	double dt,
	bool source_before_extensive_update,
	bool low_order,
	vector<Conserved3D>& candidate)
{
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>>
		face_values;
	if (low_order)
		CalculateSphericalLowOrderFluxes(stage_input_cells,
			face_velocities, fluxes);
	else
		fc_.Calculate(fluxes, tess_, face_velocities, stage_input_cells,
			stage_input_extensives, eos_, time, dt, face_values);

	candidate = stage_input_extensives;
	if (source_before_extensive_update)
		source_(tess_, stage_input_cells, fluxes, point_velocities,
			time, dt, candidate);
	ApplyFluxesWithoutValidation(fluxes, stage_input_cells, dt, candidate);
	if (!source_before_extensive_update)
		source_(tess_, stage_input_cells, fluxes, point_velocities,
			time, dt, candidate);
	ApplySphericalBackgroundCorrection(stage_input_cells,
		stage_input_extensives, face_velocities, point_velocities,
		time, dt, source_before_extensive_update, candidate, low_order);
}

void HDSim3D::BlendSphericalStageCandidates(
	vector<Conserved3D> const& low_order,
	vector<Conserved3D> const& high_order,
	vector<Conserved3D>& result)
{
	size_t const local_count = tess_.GetPointNo();
	vector<Conserved3D> projected_low_order;
	vector<Conserved3D> projected_high_order;
	spherical_shell_projector_->ProjectExtensives(tess_, low_order,
		projected_low_order);
	spherical_shell_projector_->ProjectExtensives(tess_, high_order,
		projected_high_order);
	bool low_order_is_admissible = true;
	double local_theta = 1.0;
	for (size_t i = 0; i < local_count; ++i) {
		double const mass_threshold = 1e-12 * low_order[i].mass;
		double const projected_mass_threshold =
			1e-12 * projected_low_order[i].mass;
		bool const low_good =
			HasPositiveThermalMargin(low_order[i], mass_threshold) &&
			HasPositiveThermalMargin(projected_low_order[i],
				projected_mass_threshold);
		low_order_is_admissible =
			low_order_is_admissible && low_good;
		if (!low_good)
			continue;

		auto admissible = [&](double theta) {
			Conserved3D const state = low_order[i] +
				theta * (high_order[i] - low_order[i]);
			Conserved3D const projected_state = projected_low_order[i] +
				theta *
				(projected_high_order[i] - projected_low_order[i]);
			return HasPositiveThermalMargin(state, mass_threshold) &&
				HasPositiveThermalMargin(projected_state,
					projected_mass_threshold);
		};
		if (admissible(1.0))
			continue;
		double lower = 0;
		double upper = 1;
		for (size_t iteration = 0; iteration < 60; ++iteration) {
			double const middle = 0.5 * (lower + upper);
			if (admissible(middle))
				lower = middle;
			else
				upper = middle;
		}
		local_theta = std::min(local_theta, lower);
	}

#ifdef RICH_MPI
	int low_good = low_order_is_admissible ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &low_good, 1, MPI_INT, MPI_MIN,
		MPI_COMM_WORLD);
	low_order_is_admissible = low_good == 1;
	MPI_Allreduce(MPI_IN_PLACE, &local_theta, 1, MPI_DOUBLE, MPI_MIN,
		MPI_COMM_WORLD);
#endif
	if (!low_order_is_admissible)
		throw UniversalError(
			"Spherical low-order stage candidate is not admissible");

	last_spherical_positivity_theta_ = local_theta;
	minimum_spherical_positivity_theta_ =
		std::min(minimum_spherical_positivity_theta_, local_theta);
	if (local_theta < 1.0 - 1e-14) {
		++spherical_positivity_activation_count_;
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		if (rank == 0)
			std::cout << "Spherical positivity blend theta "
				<< local_theta << std::endl;
	}

	result.resize(local_count);
	for (size_t i = 0; i < local_count; ++i) {
		result[i] = low_order[i] +
			local_theta * (high_order[i] - low_order[i]);
		result[i].internal_energy = result[i].energy -
			ScalarProd(result[i].momentum, result[i].momentum) /
			(2.0 * result[i].mass);
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
	eos_(eos), cells_(cells), extensive_(extensives), pm_(pm), tsc_(tsc), fc_(fc), cu_(cu), eu_(eu), source_(source), pt_(pt)
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
	if (spherical_positivity_preserving_) {
		vector<Conserved3D> high_order_candidate;
		vector<Conserved3D> low_order_candidate;
		BuildSphericalStageCandidate(cells_, extensive_, face_vel,
			point_vel, time, dt, false, false,
			high_order_candidate);
		BuildSphericalStageCandidate(cells_, extensive_, face_vel,
			point_vel, time, dt, false, true,
			low_order_candidate);
		BlendSphericalStageCandidates(low_order_candidate,
			high_order_candidate, mid_extensives);
	}
	else {
		fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_,
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
			point_vel, time, dt, false, mid_extensives, false);
	}
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
	if (spherical_positivity_preserving_) {
		vector<Conserved3D> high_order_candidate;
		vector<Conserved3D> low_order_candidate;
		BuildSphericalStageCandidate(cells_,
			stage_two_input_extensives, face_vel, point_vel,
			time + dt, dt, true, false,
			high_order_candidate);
		BuildSphericalStageCandidate(cells_,
			stage_two_input_extensives, face_vel, point_vel,
			time + dt, dt, true, true,
			low_order_candidate);
		BlendSphericalStageCandidates(low_order_candidate,
			high_order_candidate, mid_extensives);
	}
	else {
		fc_.Calculate(fluxes, tess_, face_vel, cells_,
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
			time + dt, dt, true, mid_extensives, false);
	}
	ExtensiveAvg(extensive_, mid_extensives);
cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif
MEMORY_DEBUG_PRINT("hydro: after cell update (2nd half)");
}

void HDSim3D::timeAdvanceLagrangian1D(
	const ComputationalCell3D* left_external,
	const ComputationalCell3D* right_external)
{
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
