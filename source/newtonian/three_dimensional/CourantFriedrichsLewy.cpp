#include "CourantFriedrichsLewy.hpp"
#include "../../misc/utils.hpp"
#include "../../misc/lazy_list.hpp"
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{
	// RICH_CFL_DECISION_TRACE=1 (diagnostic, default off): rank 0 prints one
	// CFL_DECISION record per evaluation.  Parsed on the first evaluation,
	// which every rank makes collectively.
	bool CflDecisionTraceEnabled(void)
	{
		static int enabled = -1;
		if(enabled < 0)
		{
			char const* const value = std::getenv("RICH_CFL_DECISION_TRACE");
			int local = 2;
			if(value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0)
				local = 0;
			else if(std::strcmp(value, "1") == 0)
				local = 1;
			int lowest = local;
			int highest = local;
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &lowest, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &highest, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
			if(highest == 2 || lowest != highest)
				throw UniversalError(
					"RICH_CFL_DECISION_TRACE must be 0 or 1 and agree on every rank");
			enabled = highest;
		}
		return enabled == 1;
	}
}

CourantFriedrichsLewy::CourantFriedrichsLewy(double cfl, double SourceCFL, SourceTerm3D const& source, 
	std::vector<std::string> no_calc, bool debug) :
	cfl_(cfl), sourcecfl_(SourceCFL), source_(source), no_calc_(no_calc), debug_(debug), first_try_(true), dt_first_(-1), last_time_(-10000)
{
	assert(cfl_ < 1 && "cfl number must be smaller than 1");
}

double CourantFriedrichsLewy::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
	const EquationOfState& eos, const vector<Vector3D>& face_velocities, const double time)
{
	double res = 0.001 * std::numeric_limits<double>::max();
	size_t const N = tess.GetPointNo();
	size_t loc = 0;
	bool raw_found = false;
	size_t const N_no_calc = no_calc_.size();
	std::vector<size_t> no_calc_indeces(N_no_calc);
	for(size_t i = 0; i <N_no_calc; ++i)
		no_calc_indeces[i] = binary_index_find(ComputationalCell3D::stickerNames, no_calc_[i]);
	if (N > 0)
	{
		for (size_t i = 0; i < N; ++i)
		{
			const ComputationalCell3D &cell = cells[i];
			if(std::any_of(no_calc_indeces.cbegin(), no_calc_indeces.cend(), [&cell](const size_t &idx){return cell.stickers[idx];}))
			{
				continue;
			}
			double res_temp = 0;
			double c = 0;
#ifdef RICH_DEBUG
			try
			{
#endif
				c = eos.de2c(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
#ifdef RICH_DEBUG
			}
			catch (UniversalError& eo)
			{
				eo.addEntry("Error in CFL", 0);
				eo.addEntry("Cell number", i);
				throw eo;
			}
#endif
			Vector3D const& v = cell.velocity;
			face_vec const& faces = tess.GetCellFaces(i);
			size_t const Nloop = faces.size();
			double max_face_area = 0;
			for (size_t j = 0; j < Nloop; ++j)
			{
				Vector3D n = tess.Normal(faces[j]);
				n *= 1.0 / fastabs(n);
				res_temp = fmax(res_temp, (c + std::abs(ScalarProd(n, v - face_velocities[faces[j]]))));
				max_face_area = std::max(max_face_area, tess.GetArea(faces[j]));
			}
			double cell_effective_radius = std::min(tess.GetWidth(i), tess.GetVolume(i) / max_face_area);
			res_temp = cell_effective_radius / res_temp;
			if (res_temp < res)
			{
				res = res_temp;
				loc = i;
				raw_found = true;
			}
		}
	}
	res *= cfl_;
	double old_res = res;
	double const source_inverse = source_.SuggestInverseTimeStep();
	// std::max takes its first argument unless it is smaller: the source limit
	// wins ties.
	bool const local_force_binding = !(source_inverse / sourcecfl_ < 1.0 / res);
	res = 1.0 / std::max(source_inverse / sourcecfl_, 1.0 / res);
	double const local_limit = res;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
	double hydro_res = res;
	double const cap_value = dt_first_;
	bool const cap_armed = (first_try_ && dt_first_ > 0) ||
		(last_time_ == time && dt_first_ > 0);
	if ((first_try_ && dt_first_ > 0) || (last_time_ == time && dt_first_ > 0))
	{
		res = std::min(res, dt_first_);
		first_try_ = false;
		if (close2zero(last_time_ - time))
			dt_first_ = -1;
	}
	bool const cap_cleared = cap_armed && !(dt_first_ > 0);
	int rank = 0;
	int limiting_rank = 0;
	double raw_limit = old_res;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	struct { double val; int rank; } local_min{old_res, rank}, global_min;
	MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
	limiting_rank = global_min.rank;
	raw_limit = global_min.val;
#endif
	if(CflDecisionTraceEnabled())
	{
		// Which criterion sets this evaluation: the raw CFL (its winning cell
		// and that cell's signal inputs), the source limit (its owning rank), or
		// the cap from the previous SetTimeStep.  Collective.  In the TDE drivers
		// HDSim3D::timeAdvance2 is the only caller: two evaluations per global
		// step at its start time (eval 1 before, eval 2 after the point-velocity
		// fix, which returns the accepted step); individual events use
		// SuggestIndividualTimeSteps.
		static double trace_time = std::numeric_limits<double>::quiet_NaN();
		static std::size_t trace_evaluation = 0;
		static unsigned long long trace_call = 0;
		++trace_call;
		trace_evaluation = time == trace_time ? trace_evaluation + 1 : 1;
		trace_time = time;
		int limit_rank = 0;
		int force_binding = local_force_binding ? 1 : 0;
		// Whether every rank returns the same step and holds the same cap state
		// (the record's result and cap fields are rank 0's).
		double agreement[6] = {res, cap_value, cap_armed ? 1.0 : 0.0,
			-res, -cap_value, cap_armed ? -1.0 : 0.0};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, agreement, 6, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
		int const result_agree = agreement[0] == -agreement[3] ? 1 : 0;
		int const cap_agree = agreement[1] == -agreement[4] &&
			agreement[2] == -agreement[5] ? 1 : 0;
#ifdef RICH_MPI
		struct { double val; int rank; } limit_local{local_limit, rank}, limit_global;
		MPI_Allreduce(&limit_local, &limit_global, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
		limit_rank = limit_global.rank;
		MPI_Bcast(&force_binding, 1, MPI_INT, limit_rank, MPI_COMM_WORLD);
#endif
		// The raw winner on limiting_rank: width, effective radius, sound
		// speed, |v|, density, largest normal signal speed |n.(v - w_f)|.
		int raw_valid = rank == limiting_rank && raw_found && loc < N ? 1 : 0;
		unsigned long long raw_id = 0;
		double winner[6] = {0, 0, 0, 0, 0, 0};
		if(raw_valid != 0)
		{
			ComputationalCell3D const& cell = cells[loc];
			face_vec const& faces = tess.GetCellFaces(loc);
			double max_face_area = 0;
			double max_normal_speed = 0;
			for(size_t j = 0; j < faces.size(); ++j)
			{
				Vector3D n = tess.Normal(faces[j]);
				n *= 1.0 / fastabs(n);
				max_normal_speed = std::max(max_normal_speed,
					std::abs(ScalarProd(n, cell.velocity - face_velocities[faces[j]])));
				max_face_area = std::max(max_face_area, tess.GetArea(faces[j]));
			}
			raw_id = static_cast<unsigned long long>(cell.ID);
			winner[0] = tess.GetWidth(loc);
			winner[1] = std::min(tess.GetWidth(loc), tess.GetVolume(loc) / max_face_area);
			winner[2] = eos.de2c(cell.density, cell.internal_energy, cell.tracers,
				ComputationalCell3D::tracerNames);
			winner[3] = fastabs(cell.velocity);
			winner[4] = cell.density;
			winner[5] = max_normal_speed;
		}
#ifdef RICH_MPI
		MPI_Bcast(&raw_valid, 1, MPI_INT, limiting_rank, MPI_COMM_WORLD);
		MPI_Bcast(&raw_id, 1, MPI_UNSIGNED_LONG_LONG, limiting_rank, MPI_COMM_WORLD);
		MPI_Bcast(winner, 6, MPI_DOUBLE, limiting_rank, MPI_COMM_WORLD);
#endif
		if(rank == 0)
		{
			char const* const binding = res < hydro_res ? "cap" :
				(force_binding != 0 ? "force" : "raw");
			char line[768];
			std::snprintf(line, sizeof(line),
				"CFL_DECISION call=%llu time=%.17g eval=%zu result=%.17g binding=%s "
				"hydro=%.17g limit_rank=%d raw=%.17g raw_rank=%d raw_valid=%d raw_id=%llu "
				"width=%.9g radius=%.9g c=%.9g v=%.9g density=%.9g max_normal_speed=%.9g "
				"cap=%.17g cap_armed=%d cap_reduced=%d cap_cleared=%d "
				"result_agree=%d cap_agree=%d",
				trace_call, time, trace_evaluation, res, binding, hydro_res, limit_rank,
				raw_limit, limiting_rank, raw_valid, raw_id, winner[0], winner[1],
				winner[2], winner[3], winner[4], winner[5], cap_value,
				cap_armed ? 1 : 0, res < hydro_res ? 1 : 0, cap_cleared ? 1 : 0,
				result_agree, cap_agree);
			std::cout << line << std::endl;
		}
	}
	last_time_ = time;
	if (debug_ && rank == limiting_rank && raw_found && loc < N &&
		hydro_res < 0.9999 * dt_suggest_)
	{
		Vector3D const& v = cells[loc].velocity;
		double c = eos.dp2c(cells[loc].density, cells[loc].pressure, cells[loc].tracers, ComputationalCell3D::tracerNames);
		double source_inv = source_.SuggestInverseTimeStep();
		double dt_force = source_inv > 0 ? 1.0 / source_inv : -1;
		std::cout << "Min dt="<<res<<", cell ID " << cells[loc].ID << " width " << tess.GetWidth(loc) << " c "
			<< c << " cell v " << cells[loc].velocity.x << "," << cells[loc].velocity.y << "," << cells[loc].velocity.z <<" dt_org "<<old_res<<" dt_force "<<dt_force<<" dt_rad "<<dt_first_<<" density "<<cells[loc].density<<std::endl;
		std::cout<<"Location "<<tess.GetMeshPoint(loc)<<" CM "<<tess.GetCellCM(loc)<<std::endl;
		face_vec const& faces = tess.GetCellFaces(loc);
		size_t Nloop = faces.size();
		double max_face_area = 0;
		for (size_t j = 0; j < Nloop; ++j)
		{
			max_face_area = std::max(max_face_area, tess.GetArea(faces[j]));
			std::cout << " face_vel " << face_velocities[faces[j]] <<" dv= " <<fastabs(v - face_velocities[faces[j]]) << " ";
			Vector3D p1 = tess.GetMeshPoint(tess.GetFaceNeighbors(faces[j]).first);
			Vector3D p2 = tess.GetMeshPoint(tess.GetFaceNeighbors(faces[j]).second);
			std::cout<<"p1="<<p1.x<<","<<p1.y<<","<<p1.z<<" p2="<<p2.x<<","<<p2.y<<","<<p2.z<<std::endl;
			p1 = cells[tess.GetFaceNeighbors(faces[j]).first].velocity;
			p2 = cells[tess.GetFaceNeighbors(faces[j]).second].velocity;
			std::cout<<"v1="<<p1.x<<","<<p1.y<<","<<p1.z<<" |v1|="<<fastabs(p1)
			<<" v2="<<p2.x<<","<<p2.y<<","<<p2.z<<" |v2|="<<fastabs(p2)
			<<" A "<<tess.GetArea(faces[j])<<std::endl;
			if (point_velocities_)
			{
				size_t n1 = tess.GetFaceNeighbors(faces[j]).first;
				size_t n2 = tess.GetFaceNeighbors(faces[j]).second;
				if (n1 < point_velocities_->size() && n2 < point_velocities_->size())
					std::cout<<"pv1="<<(*point_velocities_)[n1]<<" pv2="<<(*point_velocities_)[n2]<<std::endl;
			}
		}
		double cell_effective_radius = std::min(tess.GetWidth(loc), tess.GetVolume(loc) / max_face_area);
		std::cout<<"cell_effective_radius "<<cell_effective_radius<<std::endl;
	}

	this->last_time_ = time;
	this->dt_ = res;
	this->dt_suggest_ = hydro_res;
	return res;
}

double CourantFriedrichsLewy::GetTimeStep(void) const
{
	return this->first_try_ ? this->dt_first_ : this->dt_;
}

void CourantFriedrichsLewy::SetTimeStep(double dt)
{
	dt_first_ = dt;
	first_try_ = true;
}

double CourantFriedrichsLewy::SuggestTimeStep(void) const
{
	return this->dt_suggest_;
}

void CourantFriedrichsLewy::CellTimeSteps(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const EquationOfState& eos,
	const vector<Vector3D>& face_velocities,
	vector<double>& limits,
	bool const include_source_limit) const
{
	std::size_t const N = tess.GetPointNo();
	limits.assign(N, std::numeric_limits<double>::infinity());
	std::vector<std::size_t> no_calc_indices(no_calc_.size());
	for(std::size_t i = 0; i < no_calc_.size(); ++i)
		no_calc_indices[i] = binary_index_find(ComputationalCell3D::stickerNames, no_calc_[i]);
	double const source_inverse =
		include_source_limit ? source_.SuggestInverseTimeStep() : 0;
	double const source_limit = source_inverse > 0 ?
		sourcecfl_ / source_inverse : std::numeric_limits<double>::infinity();
	for(std::size_t index = 0; index < N && index < cells.size(); ++index)
	{
		const ComputationalCell3D& cell = cells[index];
		if(std::any_of(no_calc_indices.cbegin(), no_calc_indices.cend(),
			[&cell](std::size_t sticker){return cell.stickers[sticker];}))
			continue;
		const double sound_speed = eos.de2c(cell.density, cell.internal_energy,
			cell.tracers, ComputationalCell3D::tracerNames);
		double signal_speed = 0;
		double maximum_face_area = 0;
		bool faces_valid = true;
		for(std::size_t face : tess.GetCellFaces(index))
		{
			if(face >= face_velocities.size())
			{
				faces_valid = false;
				break;
			}
			Vector3D normal = tess.Normal(face);
			normal *= 1.0 / fastabs(normal);
			signal_speed = std::max(signal_speed, sound_speed +
				std::abs(ScalarProd(normal, cell.velocity - face_velocities[face])));
			maximum_face_area = std::max(maximum_face_area, tess.GetArea(face));
		}
		double limit = source_limit;
		if(faces_valid && signal_speed > 0 && maximum_face_area > 0)
		{
			const double effective_radius = std::min(tess.GetWidth(index),
				tess.GetVolume(index) / maximum_face_area);
			limit = std::min(limit, cfl_ * effective_radius / signal_speed);
		}
		limits[index] = limit;
	}
}

void CourantFriedrichsLewy::SuggestIndividualTimeSteps(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const EquationOfState& eos,
	const vector<Vector3D>& face_velocities,
	double /*time*/,
	const IndividualStepContext& context,
	vector<double>& time_step_limits) const
{
	std::vector<double> source_limits(cells.size(), std::numeric_limits<double>::infinity());
	source_.SuggestIndividualTimeSteps(tess, cells, context, source_limits);
	for(std::size_t const index : context.active_indices)
		if(index < source_limits.size() && index < time_step_limits.size())
			time_step_limits[index] = std::min(time_step_limits[index],
				source_limits[index] * sourcecfl_);
	std::vector<std::size_t> no_calc_indices(no_calc_.size());
	for(std::size_t i = 0; i < no_calc_.size(); ++i)
		no_calc_indices[i] = binary_index_find(ComputationalCell3D::stickerNames, no_calc_[i]);

	for(std::size_t const index : context.active_indices)
	{
		const ComputationalCell3D& cell = cells.at(index);
		if(std::any_of(no_calc_indices.cbegin(), no_calc_indices.cend(),
			[&cell](std::size_t sticker){return cell.stickers[sticker];}))
			continue;

		const double sound_speed = eos.de2c(cell.density, cell.internal_energy,
			cell.tracers, ComputationalCell3D::tracerNames);
		double signal_speed = 0;
		double maximum_face_area = 0;
		const face_vec& faces = tess.GetCellFaces(index);
		for(std::size_t face : faces)
		{
			Vector3D normal = tess.Normal(face);
			normal *= 1.0 / fastabs(normal);
			signal_speed = std::max(signal_speed, sound_speed +
				std::abs(ScalarProd(normal, cell.velocity - face_velocities.at(face))));
			maximum_face_area = std::max(maximum_face_area, tess.GetArea(face));
		}
		double hydro_limit = std::numeric_limits<double>::infinity();
		if(signal_speed > 0 && maximum_face_area > 0)
		{
			const double effective_radius = std::min(tess.GetWidth(index),
				tess.GetVolume(index) / maximum_face_area);
			hydro_limit = cfl_ * effective_radius / signal_speed;
		}
		time_step_limits.at(index) = std::min(time_step_limits.at(index),
			hydro_limit);
	}
}
