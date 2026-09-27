#include "ConditionExtensiveUpdater3D.hpp"
#include "default_extensive_updater.hpp"
#include "../../misc/utils.hpp"
#include <iostream>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
std::size_t diagnostic_cell_id(const vector<ComputationalCell3D>& cells,
	std::size_t index)
{
	return index < cells.size() ? cells[index].ID :
		std::numeric_limits<std::size_t>::max();
}

void report_individual_invalid_mass(
	const vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const IndividualStepContext& context,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& extensives,
	const vector<Vector3D>& face_velocities,
	const vector<Vector3D>& point_velocities,
	const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >&
		face_values,
	std::size_t index,
	double time,
	int rank)
{
	const Conserved3D& updated = extensives[index];
	const double volume = tess.GetVolume(index);
	const double primitive_mass = cells[index].density * volume;
	const std::uint64_t no_tick = std::numeric_limits<std::uint64_t>::max();
	const unsigned int time_bin = index < context.cell_time_bins.size() ?
		static_cast<unsigned int>(context.cell_time_bins[index]) :
		std::numeric_limits<unsigned int>::max();
	const std::uint64_t primitive_tick =
		index < context.primitive_ticks.size() ?
		context.primitive_ticks[index] : no_tick;
	const std::string record_id = std::to_string(rank) + ':' +
		std::to_string(context.event_tick) + ':' +
		std::to_string(cells[index].ID);
	std::ostringstream diagnostic_record;

	std::ostringstream header;
	header << std::setprecision(17)
		<< "INDIVIDUAL_HYDRO_INVALID_MASS"
		<< " record_id=" << record_id
		<< " rank=" << rank
		<< " local_cell=" << index
		<< " cell_id=" << cells[index].ID
		<< " active=" << (context.isActive(index) ? 1 : 0)
		<< " previous_event_tick=" << context.previous_event_tick
		<< " event_tick=" << context.event_tick
		<< " primitive_tick=" << primitive_tick
		<< " time_bin=" << time_bin
		<< " previous_event_time=" << context.previous_event_time
		<< " event_time=" << context.event_time
		<< " update_time=" << time
		<< " cell_dt=" << context.cellTimeStep(index)
		<< " volume=" << volume
		<< " primitive_density=" << cells[index].density
		<< " primitive_pressure=" << cells[index].pressure
		<< " primitive_internal_energy=" << cells[index].internal_energy
		<< " primitive_mass=" << primitive_mass
		<< " updated=" << updated;
	diagnostic_record << header.str() << '\n';

	Conserved3D incident_flux_change;
	const face_vec incident_faces = tess.GetCellFaces(index);
	for(std::size_t face : incident_faces)
	{
		if(face >= fluxes.size())
		{
			std::ostringstream missing_face;
			missing_face << "INDIVIDUAL_HYDRO_INVALID_MASS_FACE"
				<< " record_id=" << record_id
				<< " rank=" << rank
				<< " cell_id=" << cells[index].ID
				<< " face=" << face
				<< " missing_flux=1";
			diagnostic_record << missing_face.str() << '\n';
			continue;
		}

		const auto neighbors = tess.GetFaceNeighbors(face);
		const bool first_physical =
			neighbors.first < context.cell_time_steps.size() &&
			!tess.IsPointOutsideBox(neighbors.first);
		const bool second_physical =
			neighbors.second < context.cell_time_steps.size() &&
			!tess.IsPointOutsideBox(neighbors.second);
		double face_dt = 0;
		if(first_physical && second_physical)
			face_dt = context.hydroFaceTimeStep(neighbors.first, neighbors.second);
		else if(first_physical)
			face_dt = context.cellTimeStep(neighbors.first);
		else if(second_physical)
			face_dt = context.cellTimeStep(neighbors.second);

		const double area = tess.GetArea(face);
		const double orientation = neighbors.first == index ? -1.0 : 1.0;
		const Conserved3D face_integral = fluxes[face] * (face_dt * area);
		Conserved3D applied_change = face_integral * orientation;
		applied_change.internal_energy = orientation *
			(face_integral.energy -
			 ScalarProd(cells[index].velocity, face_integral.momentum) +
			 0.5 * ScalarProd(cells[index].velocity, cells[index].velocity) *
			 face_integral.mass);
		incident_flux_change += applied_change;

		std::ostringstream face_record;
		face_record << std::setprecision(17)
			<< "INDIVIDUAL_HYDRO_INVALID_MASS_FACE"
			<< " record_id=" << record_id
			<< " rank=" << rank
			<< " cell_id=" << cells[index].ID
			<< " face=" << face
			<< " first_index=" << neighbors.first
			<< " first_id=" << diagnostic_cell_id(cells, neighbors.first)
			<< " first_active=" <<
				(neighbors.first < context.active_mask.size() &&
				 context.isActive(neighbors.first) ? 1 : 0)
			<< " second_index=" << neighbors.second
			<< " second_id=" << diagnostic_cell_id(cells, neighbors.second)
			<< " second_active=" <<
				(neighbors.second < context.active_mask.size() &&
				 context.isActive(neighbors.second) ? 1 : 0)
			<< " orientation=" << orientation
			<< " face_dt=" << face_dt
			<< " area=" << area
			<< " dt_area=" << face_dt * area
			<< " flux=" << fluxes[face]
			<< " applied_change=" << applied_change;
		if(face < face_velocities.size())
			face_record << " face_velocity=" << face_velocities[face];
		if(neighbors.first < point_velocities.size())
			face_record << " first_point_velocity=" <<
				point_velocities[neighbors.first];
		if(neighbors.second < point_velocities.size())
			face_record << " second_point_velocity=" <<
				point_velocities[neighbors.second];
		if(face < face_values.size())
			face_record << " reconstructed_first=" << face_values[face].first
				<< " reconstructed_second=" << face_values[face].second;
		diagnostic_record << face_record.str() << '\n';
	}

	std::ostringstream change_record;
	change_record << std::setprecision(17)
		<< "INDIVIDUAL_HYDRO_INVALID_MASS_CHANGE"
		<< " record_id=" << record_id
		<< " rank=" << rank
		<< " cell_id=" << cells[index].ID
		<< " primitive_mass=" << primitive_mass
		<< " updated_mass=" << updated.mass
		<< " mass_change_from_primitive=" << updated.mass - primitive_mass
		<< " incident_flux_change=" << incident_flux_change;
	diagnostic_record << change_record.str() << '\n';
	PersistIndividualHydroDiagnosticRecord(record_id, diagnostic_record.str());
}
}

ConditionExtensiveUpdater3D::Condition3D::~Condition3D() {}

ConditionExtensiveUpdater3D::Action3D::~Action3D() {}

ConditionExtensiveUpdater3D::~ConditionExtensiveUpdater3D() {}

ConditionExtensiveUpdater3D::ConditionExtensiveUpdater3D(const vector<pair<const Condition3D*, const Action3D*> >& sequence) :
	sequence_(sequence) {}

void ConditionExtensiveUpdater3D::UpdateIndividual(
	const vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const IndividualStepContext& context,
	const vector<ComputationalCell3D>& cells,
	vector<Conserved3D>& extensives,
	double time,
	const vector<Vector3D>& edge_velocities,
	const vector<Vector3D>& point_velocities,
	const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& interp_values,
	const vector<ComputationalCell3D>* canonical_cells,
	vector<Conserved3D>* canonical_extensives) const
{
	DefaultExtensiveUpdater regular_update;
	regular_update.UpdateIndividual(fluxes, tess, context, cells, extensives,
		time, edge_velocities, point_velocities, interp_values,
		canonical_cells, canonical_extensives);

	std::exception_ptr action_exception;
	std::size_t action_index = std::numeric_limits<std::size_t>::max();
	try
	{
		for(std::size_t index : context.active_indices)
		{
			action_index = index;
			if(index >= tess.GetPointNo() || index >= cells.size() || index >= extensives.size())
				throw std::out_of_range("Condition extensive individual cell is out of range");
			for(const auto& item : sequence_)
				if((*item.first)(index, tess, cells, time))
				{
					(*item.second)(fluxes, tess, context.cellTimeStep(index), cells,
						extensives, index, time);
					break;
				}
		}
	}
	catch(...)
	{
		action_exception = std::current_exception();
	}

	const std::size_t owned_count = tess.GetPointNo();
	std::size_t invalid_index = owned_count;
	std::uint64_t invalid_component_mask = 0;
	bool invalid_canonical = false;
	for(std::size_t index = 0; index < owned_count; ++index)
	{
		invalid_component_mask = IndividualHydroInvalidComponentMask(extensives[index]);
		if(invalid_component_mask != 0)
		{
			invalid_index = index;
			break;
		}
	}
	if(invalid_component_mask == 0 && canonical_extensives != nullptr)
		for(std::size_t index = 0; index < canonical_extensives->size(); ++index)
		{
			invalid_component_mask =
				IndividualHydroInvalidComponentMask(canonical_extensives->at(index));
			if(invalid_component_mask != 0)
			{
				invalid_index = index;
				invalid_canonical = true;
				break;
			}
		}

	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	int first_failing_rank = action_exception || invalid_component_mask != 0 ?
		rank : std::numeric_limits<int>::max();
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &first_failing_rank, 1, MPI_INT, MPI_MIN,
		MPI_COMM_WORLD);
#endif
	if(first_failing_rank == std::numeric_limits<int>::max())
		return;

	std::string failure_message = action_exception ?
		"Individual condition action failed" :
		"Individual condition extensive update failed on another rank";
	if((invalid_component_mask & 16) != 0 && !invalid_canonical)
	{
		report_individual_invalid_mass(fluxes, tess, context, cells,
			extensives, edge_velocities, point_velocities, interp_values,
			invalid_index, time, rank);
		failure_message =
			"Individual extensive update produced non-positive or non-finite mass";
	}
	else if((invalid_component_mask & 16) != 0)
	{
		const double mass = canonical_extensives->at(invalid_index).mass;
		const std::string record_id = std::to_string(rank) + ':' +
			std::to_string(context.event_tick) + ':' +
			std::to_string(canonical_cells->at(invalid_index).ID);
		std::ostringstream canonical_record;
		canonical_record << std::setprecision(17)
			<< "INDIVIDUAL_HYDRO_INVALID_MASS_CANONICAL"
			<< " record_id=" << record_id
			<< " rank=" << rank
			<< " canonical_cell=" << invalid_index
			<< " cell_id=" << canonical_cells->at(invalid_index).ID
			<< " previous_event_tick=" << context.previous_event_tick
			<< " event_tick=" << context.event_tick
			<< " previous_event_time=" << context.previous_event_time
			<< " event_time=" << context.event_time
			<< " update_time=" << time
			<< " updated_mass=" << mass
			<< " pre_action_mass_available=0"
			<< " updated=" << canonical_extensives->at(invalid_index)
			<< '\n';
		PersistIndividualHydroDiagnosticRecord(record_id,
			canonical_record.str());
		failure_message =
			"Individual canonical extensive update produced non-positive or "
			"non-finite mass";
	}
	else if(invalid_component_mask != 0)
	{
		const std::size_t cell_id = invalid_canonical ?
			canonical_cells->at(invalid_index).ID : cells[invalid_index].ID;
		const std::string record_id = std::to_string(rank) + ':' +
			std::to_string(context.event_tick) + ':' + std::to_string(cell_id);
		std::ostringstream record;
		record << std::setprecision(17)
			<< "INDIVIDUAL_HYDRO_INVALID_STATE record_id=" << record_id
			<< " rank=" << rank << " cell_id=" << cell_id
			<< " canonical=" << (invalid_canonical ? 1 : 0)
			<< " component_mask=" << invalid_component_mask
			<< " event_tick=" << context.event_tick
			<< " updated=" << (invalid_canonical ?
				canonical_extensives->at(invalid_index) : extensives[invalid_index])
			<< '\n';
		PersistIndividualHydroDiagnosticRecord(record_id, record.str());
		failure_message = "Individual condition extensive update produced an invalid state";
	}

	UniversalError error(failure_message);
	error.addEntry("rank", rank);
	error.addEntry("first failing rank", first_failing_rank);
	error.addEntry("previous event tick", context.previous_event_tick);
	error.addEntry("event tick", context.event_tick);
	error.addEntry("previous event time", context.previous_event_time);
	error.addEntry("event time", context.event_time);
	if(action_exception)
	{
		error.addEntry("action cell", action_index);
		try
		{
			std::rethrow_exception(action_exception);
		}
		catch(UniversalError const& action_error)
		{
			error.Append2ErrorMessage(": " + action_error.getErrorMessage());
			error.join(action_error);
		}
		catch(std::exception const& action_error)
		{
			error.Append2ErrorMessage(std::string(": ") + action_error.what());
		}
		catch(...)
		{
			error.Append2ErrorMessage(": unknown exception");
		}
	}
	if(invalid_component_mask != 0)
	{
		error.addEntry("component mask", invalid_component_mask);
		error.addEntry(invalid_canonical ? "canonical cell" : "local cell", invalid_index);
		error.addEntry("cell ID", invalid_canonical ?
			canonical_cells->at(invalid_index).ID : cells[invalid_index].ID);
		error.addEntry("mass", invalid_canonical ?
			canonical_extensives->at(invalid_index).mass : extensives[invalid_index].mass);
	}
#ifdef RICH_MPI
	// Finish every rank's failure-only record before any caller can abort MPI.
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	throw error;
}

void ConditionExtensiveUpdater3D::operator()(const vector<Conserved3D>& fluxes, const Tessellation3D& tess,
	const double dt, const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, double time,
	 const vector<Vector3D>& edge_velocities,
	const vector<Vector3D>& point_velocities,
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& interp_values) const
{
	size_t N = tess.GetPointNo();
	std::vector<double> oldEk(N, 0), oldEtherm(N, 0), oldE(N, 0);
	for (size_t i = 0; i < N; ++i)
	{
		oldEk[i] = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		oldEtherm[i] = extensives[i].internal_energy;
		oldE[i] = extensives[i].energy;
	}
	bool entropy = !(std::find(ComputationalCell3D::tracerNames.begin(), ComputationalCell3D::tracerNames.end(), std::string("Entropy")) ==
			 ComputationalCell3D::tracerNames.end());
	size_t entropy_index = static_cast<size_t>(std::find(ComputationalCell3D::tracerNames.begin(),
							     ComputationalCell3D::tracerNames.end(), std::string("Entropy")) - ComputationalCell3D::tracerNames.begin());
	size_t Nfluxes = fluxes.size();
	Conserved3D delta;
	for (size_t i = 0; i < Nfluxes; ++i)
	{
		delta = fluxes[i] * dt * tess.GetArea(i);
		delta.internal_energy = 0;
		size_t n0 = tess.GetFaceNeighbors(i).first;
		size_t n1 = tess.GetFaceNeighbors(i).second;
		if (n0 < N)
		{
			extensives[n0] -= delta;
			extensives[n0].internal_energy -= delta.energy - ScalarProd(cells[n0].velocity, delta.momentum) +
				0.5 * ScalarProd(cells[n0].velocity, cells[n0].velocity) * delta.mass;
		}
		if (n1 < N)
		{
			extensives[n1] += delta;
			extensives[n1].internal_energy += delta.energy - ScalarProd(cells[n1].velocity, delta.momentum) +
				0.5 * ScalarProd(cells[n1].velocity, cells[n1].velocity) * delta.mass;
		}
	}

	for (size_t i = 0; i < N; ++i)
	{
		double dEtherm = extensives[i].internal_energy - oldEtherm[i];
		double Eknew = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		double dEk = Eknew - oldEk[i];
		double dE = extensives[i].energy - oldE[i];
		// if (dEtherm * (dE - dEk) > 0)
		// {
		// 	if (std::abs(dEtherm) > 0.95 * std::abs(dE - dEk) && std::abs(dEtherm) < 1.05 * std::abs(dE - dEk))
		// 		extensives[i].internal_energy = extensives[i].energy - Eknew;
		// 	else
		// 		extensives[i].energy = extensives[i].internal_energy + Eknew;
		// }
		// else
		// 	extensives[i].energy = extensives[i].internal_energy + Eknew;

		extensives[i].internal_energy = extensives[i].energy - Eknew;

		for (size_t j = 0; j < sequence_.size(); ++j)
		{
			if (sequence_[j].first->operator()(i, tess, cells, time))
			{
				sequence_[j].second->operator()(fluxes, tess, dt, cells, extensives, i, time);
				break;
			}
		}
		for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
		{
			if(extensives[i].Eg[j] < 0 && extensives[i].mass > 0)
			{

				UniversalError eo("Negative energy group");
				eo.addEntry("ID",cells[i].ID);
				eo.addEntry("group", j);
				eo.addEntry("group energy", extensives[i].Eg[j]);
				eo.addEntry("approximate Eg", cells[i].Eg[j] * extensives[i].mass);
				eo.addEntry("mass", extensives[i].mass);
				throw eo;
			}
		}
		// check cell
		if (!(extensives[i].mass > 0) || !(extensives[i].energy > 0) || (!(extensives[i].internal_energy > 0) && (!entropy)) ||
			(!std::isfinite(fastabs(extensives[i].momentum))) || (entropy && extensives[i].tracers[entropy_index] < 0))
		{
			UniversalError eo("Bad extesnsive update");
			int rank = 0;
#ifdef RICH_MPI
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
			std::cout << "Bad cell in ExtensiveUpdater3D, cell " << i << " rank " << rank <<" dt "<<dt<<" ID "<<cells[i].ID<< std::endl;
			std::cout << "mass " << extensives[i].mass << " energy " << extensives[i].energy << " internalE " <<
				extensives[i].internal_energy << " momentum" << abs(extensives[i].momentum) << " volume " << tess.GetVolume(i)
				<< std::endl;
			std::cout << "Old cell, density " << cells[i].density << " pressure " << cells[i].pressure << " vx " <<
				cells[i].velocity.x << " vy " << cells[i].velocity.y << " vz " << cells[i].velocity.z << std::endl;
			std::cout<<"Point "<<tess.GetMeshPoint(i)<<" CM "<<tess.GetCellCM(i)<<" d "<<abs(tess.GetMeshPoint(i) - tess.GetCellCM(i)) / tess.GetWidth(i)<<std::endl;
			for (size_t j = 0; j < ComputationalCell3D::tracerNames.size(); ++j)
			{
			  std::cout << ComputationalCell3D::tracerNames[j] << " old cell " << cells[i].tracers[j] << 
					" extensive "<<extensives[i].tracers[j]<<std::endl;
			}
			face_vec temp = tess.GetCellFaces(i);
			Conserved3D old_ext(extensives[i]);
			// recover old_extensive
			for (size_t j = 0; j < temp.size(); ++j)
			{
				double Area = tess.GetArea(temp[j]) * dt;
				//				size_t N0 = tess.GetFaceNeighbors(temp[j]).first;
				size_t N1 = tess.GetFaceNeighbors(temp[j]).second;
				if (N1 == i)
				{
				  //					double newEk = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
					old_ext.mass -= Area * fluxes[temp[j]].mass;
					old_ext.momentum -= Area * fluxes[temp[j]].momentum;
					//old_ext.energy -= Area * fluxes[temp[j]].energy;
					/*					old_ext.internal_energy -= Area * fluxes[temp[j]].energy -
						newEk + 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;*/
				}
				else
				{
				  //					double newEk = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
					old_ext.mass += Area * fluxes[temp[j]].mass;
					old_ext.momentum += Area * fluxes[temp[j]].momentum;
					//					old_ext.energy += Area * fluxes[temp[j]].energy;
					/*old_ext.internal_energy += Area * fluxes[temp[j]].energy +
					  newEk - 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;*/
				}
			}
			for (size_t j = 0; j < temp.size(); ++j)
			{
				size_t N0 = tess.GetFaceNeighbors(temp[j]).first;
				size_t N1 = tess.GetFaceNeighbors(temp[j]).second;
				double Area = tess.GetArea(temp[j]) * dt;
				double Ek = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
				delta = Area * fluxes[temp[j]];
				double dEtherm1 = 0;
				if (N1 == i)
				{
					old_ext += delta;
					double Eknew1 = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
					dEtherm1 = delta.energy - (Eknew1 - Ek);
					//					old_ext.internal_energy += delta.energy - (Eknew1 - Ek);
				}
				else
				{
					old_ext -= delta;
					double Eknew1 = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
					dEtherm1 = -delta.energy - (Eknew1 - Ek);
					//					old_ext.internal_energy += dEtherm1;
				}
				Vector3D normalf = normalize(tess.Normal(temp[j]));
				std::cout << "Face " << temp[j] << " neigh " << N0 << "," << N1 << " mass=" << fluxes[temp[j]].mass * Area <<
					" energy= " << fluxes[temp[j]].energy * Area << " Etherm= " << dEtherm1 << " momentum= " << abs(fluxes[temp[j]].momentum) * Area <<
					" Area*dt " << Area << " normal " << normalf.x << "," << normalf.y << "," << normalf.z <<
					" face velocity "<<edge_velocities[temp[j]].x<<","<< edge_velocities[temp[j]].y<<","<< edge_velocities[temp[j]].z<<
					" point0 "<<tess.GetMeshPoint(N0)<<" point1 "<<tess.GetMeshPoint(N1)<<
					" point_vel0 "<<(N0 < point_velocities.size() ? point_velocities[N0] : Vector3D())<<
					" point_vel1 "<<(N1 < point_velocities.size() ? point_velocities[N1] : Vector3D())<<" Face CM "<<tess.FaceCM(temp[j])<< std::endl;
				eo.addEntry("Face", static_cast<double>(temp[j]));
				eo.addEntry("Face neigh 0", static_cast<double>(tess.GetFaceNeighbors(temp[j]).first));
				eo.addEntry("Face neigh 1", static_cast<double>(tess.GetFaceNeighbors(temp[j]).second));
				eo.addEntry("First input Density", interp_values[temp[j]].first.density);
				eo.addEntry("First input pressure", interp_values[temp[j]].first.pressure);
				eo.addEntry("First input internal energy", interp_values[temp[j]].first.internal_energy);
				eo.addEntry("First input vx", interp_values[temp[j]].first.velocity.x);
				eo.addEntry("First input vy", interp_values[temp[j]].first.velocity.y);
				eo.addEntry("First input vz", interp_values[temp[j]].first.velocity.z);
				eo.addEntry("Second input Density", interp_values[temp[j]].second.density);
				eo.addEntry("Second input pressure", interp_values[temp[j]].second.pressure);
				eo.addEntry("Second input internal energy", interp_values[temp[j]].second.internal_energy);
				eo.addEntry("Second input vx", interp_values[temp[j]].second.velocity.x);
				eo.addEntry("Second input vy", interp_values[temp[j]].second.velocity.y);
				eo.addEntry("Second input vz", interp_values[temp[j]].second.velocity.z);
				for (size_t k = 0; k < ComputationalCell3D::tracerNames.size(); ++k)
				{
				  std::cout << ComputationalCell3D::tracerNames[k] << " flux is " << fluxes[temp[j]].tracers[k] * Area << std::endl;
				}
			}
			for (size_t j = 0; j < temp.size(); ++j)
			{
				size_t N0 = tess.GetFaceNeighbors(temp[j]).first;
				size_t N1 = tess.GetFaceNeighbors(temp[j]).second;
				size_t Nother = N1 == i ? N0 : N1;
				std::cout << "Neigh cell " << Nother << ", density " << cells[Nother].density << " pressure " <<
					cells[Nother].pressure << " vx " << cells[Nother].velocity.x << " vy " << cells[Nother].velocity.y << " vz "
					<< cells[Nother].velocity.z << std::endl;
				for (size_t k = 0; k < ComputationalCell3D::tracerNames.size(); ++k)
				{
				  std::cout << ComputationalCell3D::tracerNames[k] << " of other " << cells[Nother].tracers[k] << std::endl;
				}
			}
			throw eo;
		}
	}
	extensives.resize(tess.GetPointNo());
}


void RegularExtensiveUpdate3D::operator()(const vector<Conserved3D>& /*fluxes*/, const Tessellation3D& /*tess*/, const double /*dt*/,
	const vector<ComputationalCell3D>& /*cells*/, vector<Conserved3D>& /*extensives*/,
	size_t /*index*/, double /*time*/)const
{
	//assert(extensives[index].internal_energy > 0);
	return;
}

void NoExtensiveUpdate3D::operator()(const vector<Conserved3D>& /*fluxes*/, 
				     const Tessellation3D& tess, const double /*dt*/, const vector<ComputationalCell3D>& cells, 
				     vector<Conserved3D>& extensives, size_t index, double /*time*/) const
{
	PrimitiveToConserved(cells[index], tess.GetVolume(index), extensives[index]);
}
