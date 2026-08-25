#include "default_extensive_updater.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
struct PendingIndividualHydroDelta
{
	std::size_t cell_index = 0;
	std::size_t counterpart_cell_id = 0;
	int counterpart_owner = -1;
	Conserved3D delta;
};

#ifdef RICH_MPI
struct IndividualHydroDelta : public Serializable
{
	size_t cell_id = 0;
	size_t counterpart_cell_id = 0;
	int counterpart_owner = -1;
	Conserved3D delta;

	force_inline size_t dump(Serializer* serializer) const override
	{
		size_t bytes = 0;
		bytes += serializer->insert(cell_id);
		bytes += serializer->insert(counterpart_cell_id);
		bytes += serializer->insert(counterpart_owner);
		bytes += serializer->insert(delta);
		return bytes;
	}

	force_inline size_t load(const Serializer* serializer, size_t byte_offset) override
	{
		size_t bytes = 0;
		bytes += serializer->extract(cell_id, byte_offset);
		bytes += serializer->extract(counterpart_cell_id, byte_offset + bytes);
		bytes += serializer->extract(counterpart_owner, byte_offset + bytes);
		bytes += serializer->extract(delta, byte_offset + bytes);
		return bytes;
	}
};
#endif
}

DefaultExtensiveUpdater::DefaultExtensiveUpdater(void){}

void DefaultExtensiveUpdater::UpdateIndividual(
	const vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const IndividualStepContext& context,
	const vector<ComputationalCell3D>& cells,
	vector<Conserved3D>& extensives,
	double /*time*/,
	const vector<Vector3D>& /*edge_velocities*/,
	const vector<Vector3D>& /*point_velocities*/,
	const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& /*interp_values*/,
	const vector<ComputationalCell3D>* canonical_cells,
	vector<Conserved3D>* canonical_extensives) const
{
	const std::size_t norg = tess.GetPointNo();
	if(extensives.size() < norg || cells.size() < norg ||
		context.active_mask.size() < norg ||
		context.cell_time_steps.size() < norg)
		throw std::invalid_argument("Individual extensive update has inconsistent cell counts");

	std::vector<PendingIndividualHydroDelta> local_deltas;
	std::unordered_map<std::size_t, std::size_t> owned_index_by_id;
	unsigned local_mapping_issues = 0;
	if((canonical_cells == nullptr) != (canonical_extensives == nullptr))
		throw std::invalid_argument(
			"Individual hydro canonical cell and extent arrays must be supplied together");
	const std::size_t canonical_count = canonical_cells == nullptr ? 0 :
		canonical_cells->size();
	if(canonical_extensives != nullptr &&
		canonical_extensives->size() != canonical_count)
		throw std::invalid_argument(
			"Individual hydro canonical cell and extent counts differ");
	owned_index_by_id.reserve(std::max(norg, canonical_count));
	for(std::size_t i = 0; i < norg; ++i)
		if(!owned_index_by_id.emplace(cells[i].ID, i).second)
			local_mapping_issues |= 128u;

	// A partial target set need not contain an inactive owned recipient of a
	// remote face flux.  Append such canonical extents to this transaction so
	// conservation and the radiation positivity limiter remain cell-local,
	// while the passive primitive stays untouched.
	std::vector<std::pair<std::size_t, std::size_t> > canonical_work_indices;
	std::unordered_map<std::size_t, std::size_t> canonical_index_by_id;
	if(canonical_cells != nullptr)
	{
		canonical_index_by_id.reserve(canonical_count);
		for(std::size_t canonical = 0; canonical < canonical_count; ++canonical)
			if(!canonical_index_by_id.emplace(
				(*canonical_cells)[canonical].ID, canonical).second)
				local_mapping_issues |= 256u;
		for(auto const& entry : owned_index_by_id)
		{
			auto const canonical = canonical_index_by_id.find(entry.first);
			if(canonical == canonical_index_by_id.end())
			{
				local_mapping_issues |= 512u;
				continue;
			}
			canonical_work_indices.emplace_back(entry.second,
				canonical->second);
		}
		for(std::size_t canonical = 0; canonical < canonical_count; ++canonical)
		{
			const std::size_t cell_id = (*canonical_cells)[canonical].ID;
			if(owned_index_by_id.find(cell_id) != owned_index_by_id.end())
				continue;
			const std::size_t work_index = extensives.size();
			extensives.push_back((*canonical_extensives)[canonical]);
			owned_index_by_id.emplace(cell_id, work_index);
			canonical_work_indices.emplace_back(work_index, canonical);
		}
	}
	const std::size_t work_count = extensives.size();
	oldEk_.assign(work_count, 0);
	oldEtherm_.assign(work_count, 0);
	oldE_.assign(work_count, 0);
	std::vector<unsigned char> touched(work_count, 0);
	std::vector<std::size_t> work_cell_ids(work_count, 0);
	for(std::size_t i = 0; i < cells.size() && i < work_count; ++i)
		work_cell_ids[i] = cells[i].ID;
	if(canonical_cells != nullptr)
		for(auto const& mapping : canonical_work_indices)
			work_cell_ids[mapping.first] =
				(*canonical_cells)[mapping.second].ID;
#ifndef RICH_MPI
	if(local_mapping_issues != 0)
		throw std::runtime_error(
			"Individual hydro stable cell IDs are not unique");
#endif

	int rank = 0;
	int rank_count = 1;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

	const std::size_t point_count = tess.getMeshPoints().size();
	std::vector<int> point_owner(point_count, -1);
	Tessellation3D::AllPointsMap const& local_to_global =
		tess.GetIndicesInAllPoints();
	if(local_to_global.size() < norg)
		local_mapping_issues |= 1u;
	for(auto const& mapping : local_to_global)
	{
		if(mapping.first >= point_owner.size())
		{
			local_mapping_issues |= 2u;
			continue;
		}
		point_owner[mapping.first] = rank;
	}
	const std::vector<int> duplicated_procs = tess.GetDuplicatedProcs();
	const std::vector<std::vector<std::size_t> >& ghost_indices = tess.GetGhostIndeces();
	if(duplicated_procs.size() != ghost_indices.size())
		local_mapping_issues |= 4u;
	for(std::size_t i = 0; i < duplicated_procs.size() && i < ghost_indices.size(); ++i)
	{
		if(duplicated_procs[i] < 0 || duplicated_procs[i] >= rank_count)
			local_mapping_issues |= 8u;
		for(std::size_t ghost : ghost_indices[i])
		{
			if(ghost >= point_owner.size())
			{
				local_mapping_issues |= 16u;
				continue;
			}
			// SyncPartialBuildData uses the same last-peer-wins order for
			// duplicate mesh slots after many-rank rebuilds.
			point_owner[ghost] = duplicated_procs[i];
		}
	}
	for(std::size_t face = 0; face < fluxes.size(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		if(!context.isActive(neighbors.first) &&
			!context.isActive(neighbors.second))
			continue;
		if(neighbors.first >= norg && neighbors.second >= norg)
			continue;
		const std::size_t endpoints[2] = {neighbors.first, neighbors.second};
		for(std::size_t endpoint : endpoints)
		{
			if(endpoint >= point_owner.size() || endpoint >= cells.size() ||
				endpoint >= context.active_mask.size() ||
				endpoint >= context.cell_time_steps.size())
			{
				local_mapping_issues |= 32u;
				continue;
			}
			if(!tess.IsPointOutsideBox(endpoint) && point_owner[endpoint] < 0)
				local_mapping_issues |= 64u;
		}
	}
	unsigned mapping_issues = local_mapping_issues;
	MPI_Allreduce(MPI_IN_PLACE, &mapping_issues, 1, MPI_UNSIGNED, MPI_BOR,
		MPI_COMM_WORLD);
	if(mapping_issues != 0)
		throw std::runtime_error(
			"Individual hydro MPI ownership mapping is inconsistent (issue mask " +
			std::to_string(mapping_issues) + ")");
	std::vector<std::vector<IndividualHydroDelta> > outgoing(rank_count);
#endif

	auto make_applied_delta = [&cells](Conserved3D const& face_delta,
		std::size_t index, double sign)
	{
		Conserved3D applied = face_delta * sign;
		applied.internal_energy = sign * (face_delta.energy -
			ScalarProd(cells[index].velocity, face_delta.momentum) +
			0.5 * ScalarProd(cells[index].velocity, cells[index].velocity) * face_delta.mass);
		return applied;
	};

	for(std::size_t face = 0; face < fluxes.size(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		const bool first_active = context.isActive(neighbors.first);
		const bool second_active = context.isActive(neighbors.second);
		if(!first_active && !second_active)
			continue;

		double face_dt = 0;
		const bool first_physical =
			neighbors.first < context.cell_time_steps.size() &&
			!tess.IsPointOutsideBox(neighbors.first);
		const bool second_physical =
			neighbors.second < context.cell_time_steps.size() &&
			!tess.IsPointOutsideBox(neighbors.second);
		if(first_physical && second_physical)
			face_dt = context.faceTimeStep(neighbors.first, neighbors.second);
		else if(first_physical)
			face_dt = context.cellTimeStep(neighbors.first);
		else if(second_physical)
			face_dt = context.cellTimeStep(neighbors.second);
		else
			continue;

		const bool first_owned = neighbors.first < norg;
		const bool second_owned = neighbors.second < norg;
		if(!first_owned && !second_owned)
			continue;

#ifdef RICH_MPI
		if(first_physical && second_physical && first_owned != second_owned)
		{
			const int first_rank = point_owner[neighbors.first];
			const int second_rank = point_owner[neighbors.second];
			// A lone active endpoint owns its face.  This guarantees that a rank
			// with no locally active cells need not build a passive target cell.
			// Active-active faces use the stable endpoint key as the tie-breaker.
			const bool first_is_owner = first_active != second_active ? first_active :
				(cells[neighbors.first].ID < cells[neighbors.second].ID ||
				 (cells[neighbors.first].ID == cells[neighbors.second].ID && first_rank < second_rank));
			const int face_owner = first_is_owner ? first_rank : second_rank;
			if(face_owner != rank)
				continue;
		}
#endif

		Conserved3D delta = fluxes[face] * (face_dt * tess.GetArea(face));
		delta.internal_energy = 0;
		Conserved3D first_delta;
		Conserved3D second_delta;
		if(first_physical)
			first_delta = make_applied_delta(delta, neighbors.first, -1);
		if(second_physical)
			second_delta = make_applied_delta(delta, neighbors.second, 1);
		if(first_owned)
		{
			PendingIndividualHydroDelta pending;
			pending.cell_index = neighbors.first;
			pending.delta = first_delta;
			if(second_physical)
			{
				pending.counterpart_cell_id = cells[neighbors.second].ID;
				pending.counterpart_owner = rank;
#ifdef RICH_MPI
				pending.counterpart_owner = point_owner[neighbors.second];
#endif
			}
			local_deltas.push_back(pending);
		}
#ifdef RICH_MPI
		else if(first_physical)
		{
			outgoing[point_owner[neighbors.first]].emplace_back();
			IndividualHydroDelta& packet = outgoing[point_owner[neighbors.first]].back();
			packet.cell_id = cells[neighbors.first].ID;
			if(second_physical)
			{
				packet.counterpart_cell_id = cells[neighbors.second].ID;
				packet.counterpart_owner = point_owner[neighbors.second];
			}
			packet.delta = first_delta;
		}
#endif
		if(second_owned)
		{
			PendingIndividualHydroDelta pending;
			pending.cell_index = neighbors.second;
			pending.delta = second_delta;
			if(first_physical)
			{
				pending.counterpart_cell_id = cells[neighbors.first].ID;
				pending.counterpart_owner = rank;
#ifdef RICH_MPI
				pending.counterpart_owner = point_owner[neighbors.first];
#endif
			}
			local_deltas.push_back(pending);
		}
#ifdef RICH_MPI
		else if(second_physical)
		{
			outgoing[point_owner[neighbors.second]].emplace_back();
			IndividualHydroDelta& packet = outgoing[point_owner[neighbors.second]].back();
			packet.cell_id = cells[neighbors.second].ID;
			if(first_physical)
			{
				packet.counterpart_cell_id = cells[neighbors.first].ID;
				packet.counterpart_owner = point_owner[neighbors.first];
			}
			packet.delta = second_delta;
		}
#endif
	}

#ifdef RICH_MPI
	const std::vector<std::vector<IndividualHydroDelta> > incoming =
		MPI_Exchange_all_to_all(outgoing, MPI_COMM_WORLD);
	bool local_packet_error = false;
	for(const auto& rank_packets : incoming)
		for(const IndividualHydroDelta& packet : rank_packets)
			if(owned_index_by_id.find(packet.cell_id) == owned_index_by_id.end())
				local_packet_error = true;
	int packet_error = local_packet_error ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &packet_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if(packet_error != 0)
		throw std::runtime_error("Individual hydro received a delta for an unknown stable cell ID");
	for(const auto& rank_packets : incoming)
		for(const IndividualHydroDelta& packet : rank_packets)
		{
			PendingIndividualHydroDelta pending;
			pending.cell_index = owned_index_by_id.at(packet.cell_id);
			pending.counterpart_cell_id = packet.counterpart_cell_id;
			pending.counterpart_owner = packet.counterpart_owner;
			pending.delta = packet.delta;
			local_deltas.push_back(pending);
		}
#endif

	constexpr std::size_t radiation_group_count = ENERGY_GROUPS_NUM;
	static_assert(radiation_group_count > 0,
		"Individual hydro radiation limiting requires at least one group");
	if(work_count > std::numeric_limits<std::size_t>::max() /
		radiation_group_count)
		throw std::overflow_error("Individual hydro radiation limiter size overflow");
	const std::size_t radiation_value_count =
		work_count * radiation_group_count;
	std::vector<double> radiation_loss(radiation_value_count, 0);
	std::vector<double> radiation_scale(radiation_value_count, 1);
	for(const PendingIndividualHydroDelta& pending : local_deltas)
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			const double proposed = pending.delta.Eg[group];
			if(!std::isfinite(proposed))
				throw std::runtime_error(
					"Individual hydro proposed a non-finite radiation-group delta");
			if(proposed < 0)
				radiation_loss[pending.cell_index * radiation_group_count + group]
					-= proposed;
		}
	for(auto const& owned : owned_index_by_id)
	{
		const std::size_t cell = owned.second;
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			const std::size_t key = cell * radiation_group_count + group;
			const double available = extensives[cell].Eg[group];
			if(!std::isfinite(available) || available < 0)
				throw std::runtime_error(
					"Individual hydro radiation limiter found invalid pre-event extent for cell ID " +
					std::to_string(work_cell_ids[cell]) + " group " +
					std::to_string(group) + " extent " +
					std::to_string(available));
			if(radiation_loss[key] > available)
			{
				const double ratio = available > 0 ?
					available / radiation_loss[key] : 0;
				radiation_scale[key] = ratio > 0 ?
					std::nextafter(std::min(1.0, ratio), 0.0) : 0;
			}
		}
	}

	std::vector<PendingIndividualHydroDelta> limiter_local_corrections;
#ifdef RICH_MPI
	std::vector<std::vector<IndividualHydroDelta> >
		limiter_correction_outgoing(rank_count);
	bool local_limiter_mapping_error = false;
#endif
	for(PendingIndividualHydroDelta& pending : local_deltas)
	{
		Conserved3D counterpart_correction;
		bool corrected = false;
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			const double proposed = pending.delta.Eg[group];
			if(proposed >= 0)
				continue;
			const std::size_t key =
				pending.cell_index * radiation_group_count + group;
			const double applied = proposed * radiation_scale[key];
			const double local_correction = applied - proposed;
			if(local_correction == 0)
				continue;
			pending.delta.Eg[group] = applied;
			pending.delta.Erad += local_correction;
			counterpart_correction.Eg[group] -= local_correction;
			counterpart_correction.Erad -= local_correction;
			corrected = true;
		}
		if(!corrected || pending.counterpart_owner < 0)
			continue;
		if(pending.counterpart_owner == rank)
		{
			const auto found =
				owned_index_by_id.find(pending.counterpart_cell_id);
			if(found == owned_index_by_id.end())
			{
#ifdef RICH_MPI
				local_limiter_mapping_error = true;
#else
				throw std::runtime_error(
					"Individual hydro limiter could not map a local counterpart ID");
#endif
				continue;
			}
			PendingIndividualHydroDelta correction;
			correction.cell_index = found->second;
			correction.delta = counterpart_correction;
			limiter_local_corrections.push_back(correction);
		}
#ifdef RICH_MPI
		else if(pending.counterpart_owner < rank_count)
		{
			IndividualHydroDelta correction;
			correction.cell_id = pending.counterpart_cell_id;
			correction.delta = counterpart_correction;
			limiter_correction_outgoing[pending.counterpart_owner].push_back(
				correction);
		}
		else
			local_limiter_mapping_error = true;
#endif
	}
	local_deltas.insert(local_deltas.end(),
		limiter_local_corrections.begin(), limiter_local_corrections.end());
#ifdef RICH_MPI
	int limiter_mapping_error = local_limiter_mapping_error ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &limiter_mapping_error, 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	if(limiter_mapping_error != 0)
		throw std::runtime_error(
			"Individual hydro limiter could not map a conservative counterpart");
	const std::vector<std::vector<IndividualHydroDelta> >
		limiter_correction_incoming = MPI_Exchange_all_to_all(
			limiter_correction_outgoing, MPI_COMM_WORLD);
	bool local_limiter_packet_error = false;
	for(const auto& rank_packets : limiter_correction_incoming)
		for(const IndividualHydroDelta& packet : rank_packets)
			if(owned_index_by_id.find(packet.cell_id) == owned_index_by_id.end())
				local_limiter_packet_error = true;
	int limiter_packet_error = local_limiter_packet_error ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &limiter_packet_error, 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	if(limiter_packet_error != 0)
		throw std::runtime_error(
			"Individual hydro limiter received an unknown counterpart ID");
	for(const auto& rank_packets : limiter_correction_incoming)
		for(const IndividualHydroDelta& packet : rank_packets)
		{
			PendingIndividualHydroDelta correction;
			correction.cell_index = owned_index_by_id.at(packet.cell_id);
			correction.delta = packet.delta;
			local_deltas.push_back(correction);
		}
#endif

	for(const PendingIndividualHydroDelta& indexed_delta : local_deltas)
	{
		const std::size_t index = indexed_delta.cell_index;
		if(!touched[index])
		{
			touched[index] = 1;
			oldEk_[index] = 0.5 * ScalarProd(extensives[index].momentum,
				extensives[index].momentum) / extensives[index].mass;
			oldEtherm_[index] = extensives[index].internal_energy;
			oldE_[index] = extensives[index].energy;
		}
		extensives[index] += indexed_delta.delta;
	}

	for(std::size_t i = 0; i < touched.size(); ++i)
	{
		if(!touched[i])
			continue;
		const double dEtherm = extensives[i].internal_energy - oldEtherm_[i];
		const double Eknew = 0.5 * ScalarProd(extensives[i].momentum,
			extensives[i].momentum) / extensives[i].mass;
		const double dEk = Eknew - oldEk_[i];
		const double dE = extensives[i].energy - oldE_[i];
		if(dEtherm * (dE - dEk) > 0 &&
			std::abs(dEtherm) > 0.95 * std::abs(dE - dEk) &&
			std::abs(dEtherm) < 1.05 * std::abs(dE - dEk))
			extensives[i].internal_energy = extensives[i].energy - Eknew;
	}
	if(canonical_extensives != nullptr)
		for(auto const& mapping : canonical_work_indices)
			if(touched[mapping.first])
				(*canonical_extensives)[mapping.second] =
					extensives[mapping.first];
	extensives.resize(norg);
}

void DefaultExtensiveUpdater::operator()(const vector<Conserved3D>& fluxes, const Tessellation3D& tess,
	const double dt, const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, double /*time*/,
				 std::vector<Vector3D> const& /*face_vel*/,
				 const vector<Vector3D>& /*point_velocities*/,
				 std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& /*interp_values*/) const
{
	size_t N = tess.GetPointNo();
	size_t Nfluxes = fluxes.size();
	Conserved3D delta;
	oldEk_.assign(N, 0);
	oldEtherm_.assign(N, 0);
	oldE_.assign(N, 0);
	for (size_t i = 0; i < N; ++i)
	{
		oldEk_[i] = 0.5*ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		oldEtherm_[i] = extensives[i].internal_energy;
		oldE_[i] = extensives[i].energy;
	}
	for (size_t i = 0; i < Nfluxes; ++i)
	{
		delta = fluxes[i] * dt*tess.GetArea(i);
		size_t n0 = tess.GetFaceNeighbors(i).first;
		size_t n1 = tess.GetFaceNeighbors(i).second;
#ifdef RICH_DEBUG
		bool good = true;
		if (!std::isfinite(fluxes[i].energy))
			good = false;
		if (!std::isfinite(fluxes[i].internal_energy))
			good = false;
		if (!std::isfinite(fluxes[i].momentum.x))
			good = false;
		if (!std::isfinite(fluxes[i].momentum.y))
			good = false;
		if (!std::isfinite(fluxes[i].momentum.z))
			good = false;
		if (!std::isfinite(fluxes[i].mass))
			good = false;
		for (size_t j = 0; j < delta.tracers.size(); ++j)
		{
			if (!std::isfinite(fluxes[i].tracers[j]))
				good = false;
		}
		if (!good)
		{
			UniversalError eo("Bad flux");
			eo.addEntry("Face index", i);
			eo.addEntry("Area", tess.GetArea(i));
			eo.addEntry("First neigh", n0);
			eo.addEntry("Second neigh", n1);
			eo.addEntry("Norg", N);
			eo.addEntry("Energy flux", fluxes[i].energy);
			eo.addEntry("Internal Energy flux", fluxes[i].internal_energy);
			eo.addEntry("Mass flux", fluxes[i].mass);
			eo.addEntry("Momentum x flux", fluxes[i].momentum.x);
			eo.addEntry("Momentum y flux", fluxes[i].momentum.y);
			eo.addEntry("Momentum z flux", fluxes[i].momentum.z);
			for (size_t j = 0; j < delta.tracers.size(); ++j)
				eo.addEntry("Tracer flux", fluxes[i].tracers[j]);
			eo.addEntry("Left cell density", cells[n0].density);
			eo.addEntry("Left cell pressure", cells[n0].pressure);
			eo.addEntry("Left cell Vx", cells[n0].velocity.x);
			eo.addEntry("Left cell Vy", cells[n0].velocity.y);
			eo.addEntry("Left cell Vz", cells[n0].velocity.z);
			eo.addEntry("Left cell internal energy", cells[n0].internal_energy);
			eo.addEntry("Left cell ID", cells[n0].ID);
			eo.addEntry("Right cell density", cells[n1].density);
			eo.addEntry("Right cell pressure", cells[n1].pressure);
			eo.addEntry("Right cell Vx", cells[n1].velocity.x);
			eo.addEntry("Right cell Vy", cells[n1].velocity.y);
			eo.addEntry("Right cell Vz", cells[n1].velocity.z);
			eo.addEntry("Right cell internal energy", cells[n1].internal_energy);
			eo.addEntry("Right cell ID", cells[n1].ID);
			throw eo;
		}
#endif
		delta.internal_energy = 0;
		if (n0 < N)
		{
			extensives[n0] -= delta;
			extensives[n0].internal_energy -= delta.energy - ScalarProd(cells[n0].velocity, delta.momentum) +
				0.5*ScalarProd(cells[n0].velocity, cells[n0].velocity)*delta.mass;
		}
		if (n1 < N)
		{
			extensives[n1] += delta;
			extensives[n1].internal_energy += delta.energy - ScalarProd(cells[n1].velocity, delta.momentum) +
				0.5*ScalarProd(cells[n1].velocity, cells[n1].velocity)*delta.mass;
		}
	}
	for (size_t i = 0; i < N; ++i)
	{
		double dEtherm = extensives[i].internal_energy - oldEtherm_[i];
		double Eknew = 0.5*ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		double dEk = Eknew - oldEk_[i];
		double dE = extensives[i].energy - oldE_[i];
		if (dEtherm*(dE - dEk) > 0)
		{
			if (std::abs(dEtherm) > 0.95 *std::abs(dE - dEk) && std::abs(dEtherm) < 1.05*std::abs(dE - dEk))
				extensives[i].internal_energy = extensives[i].energy - Eknew;
			else
				extensives[i].energy = extensives[i].internal_energy + Eknew;
		}
		else
			extensives[i].energy = extensives[i].internal_energy + Eknew;
	}
	extensives.resize(tess.GetPointNo());
}
