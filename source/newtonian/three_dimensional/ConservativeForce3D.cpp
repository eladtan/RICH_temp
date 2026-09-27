#include "ConservativeForce3D.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{
	void RequireOnEveryRank(bool valid, char const* message)
	{
#ifdef RICH_MPI
		int valid_on_every_rank = valid ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &valid_on_every_rank, 1, MPI_INT, MPI_MIN,
			MPI_COMM_WORLD);
		valid = valid_on_every_rank != 0;
#endif
		if(!valid)
			throw std::runtime_error(message);
	}

	Vector3D MassFlux(Tessellation3D const& tess, size_t point,vector<Conserved3D> const& fluxes)
	{
		Vector3D dm;
		Vector3D center = tess.GetMeshPoint(point);
		face_vec const& faces = tess.GetCellFaces(point);
		size_t Nfaces = faces.size();

		for (size_t i = 0; i < Nfaces; ++i)
		{
			if (point == tess.GetFaceNeighbors(faces[i]).first)
				dm -= tess.GetArea(faces[i])*fluxes[faces[i]].mass*(center -
					tess.GetMeshPoint(tess.GetFaceNeighbors(faces[i]).second));
			else
				dm += tess.GetArea(faces[i])*fluxes[faces[i]].mass*(center -
					tess.GetMeshPoint(tess.GetFaceNeighbors(faces[i]).first));
		}
		return dm;
	}
}

ConservativeForce3D::ConservativeForce3D(const Acceleration3D& acc,bool mass_flux) : acc_(acc),mass_flux_(mass_flux),dt_(0) {}

ConservativeForce3D::~ConservativeForce3D(void) {}

void ConservativeForce3D::operator()(const Tessellation3D& tess,const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& fluxes,const vector<Vector3D>& point_velocities,const double t,double dt,
	vector<Conserved3D> & extensives) const
{
	size_t N = tess.GetPointNo();
	acc_buf_.clear();
	acc_(tess, cells, fluxes, t, acc_buf_);
	dt_ = 0;
	size_t loc = 0;
	int rank = 0;
 #ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	for (size_t i = 0; i < N; ++i)
	{
		double const res_temp = fastsqrt(fastabs(acc_buf_[i])/ tess.GetWidth(i));
		if (res_temp > dt_)
		{
			dt_ = res_temp;
			loc = i;
		}
		double volume = tess.GetVolume(i);
		double Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		extensives[i].momentum += volume*cells[i].density*acc_buf_[i]*dt;
		if (mass_flux_ && (fastabs(acc_buf_[i])*tess.GetWidth(i)*cells[i].density)<(0.5*cells[i].pressure))
		{
			double part0 = volume*cells[i].density*ScalarProd(point_velocities[i], acc_buf_[i]);
			double part1 = 0.5*ScalarProd(MassFlux(tess, i, fluxes), acc_buf_[i]);
			extensives[i].energy += (part0+part1)*dt;
		}
		else
		{
			double Eknew = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
			extensives[i].energy += Eknew - Ek;
		}
	}
	struct
		{
			double val;
			int mpi_id;
		}max_data;
		max_data.mpi_id = rank;
		max_data.val = dt_;
#ifdef RICH_MPI   
    MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    dt_ = max_data.val;
#endif
    if(rank == max_data.mpi_id)
        std::cout<<"ConservativeForce3D dt ID "<<cells[loc].ID<<" width "<< tess.GetWidth(loc)<<" r "<<fastabs(tess.GetMeshPoint(loc))<<" acc "<<fastabs(acc_buf_[loc])<<" next dt "<<dt_<<std::endl;

}

Acceleration3D::~Acceleration3D(void) {}

void Acceleration3D::EvaluateIndividualTargets(
	std::pair<Vector3D, Vector3D> const& /*bounds*/,
	vector<Vector3D> const& /*source_points*/,
	vector<double> const& /*source_masses*/,
	vector<std::uint64_t> const& /*source_ids*/,
	vector<Vector3D> const& /*target_points*/,
	vector<ComputationalCell3D> const& /*target_cells*/,
	double /*time*/,
	vector<Vector3D>& /*acc*/) const
{
	throw std::logic_error(
		"Acceleration provider does not support individual target evaluation");
}

double ConservativeForce3D::SuggestInverseTimeStep(void)const
{
	return dt_;
}

void ConservativeForce3D::ApplyIndividual(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& fluxes,
	const vector<Vector3D>& point_velocities,
	double time,
	const IndividualStepContext& context,
	IndividualSourcePhase phase,
	vector<Conserved3D>& extensives) const
{
	auto require_on_every_rank = [](bool valid, char const* message)
	{
#ifdef RICH_MPI
		int valid_on_every_rank = valid ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &valid_on_every_rank, 1, MPI_INT, MPI_MIN,
			MPI_COMM_WORLD);
		valid = valid_on_every_rank != 0;
#endif
		if(!valid)
			throw std::runtime_error(message);
	};

	const std::size_t norg = tess.GetPointNo();
	bool active_indices_valid = true;
	for(std::size_t index : context.active_indices)
		if(index >= norg)
		{
			active_indices_valid = false;
			break;
		}
	require_on_every_rank(
		cells.size() >= norg && extensives.size() >= norg &&
		context.cell_time_steps.size() >= norg &&
		context.cached_accelerations.size() >= norg &&
		context.gravity_half_kick_pending.size() >= norg &&
		active_indices_valid &&
		(!mass_flux_ || (point_velocities.size() >= norg &&
			fluxes.size() >= tess.GetTotalFacesNumber())),
		"Individual conservative force has inconsistent event arrays or active indices");
	bool calculate_acceleration = phase != IndividualSourcePhase::FirstHalf;
	if(!calculate_acceleration)
		for(std::size_t index : context.active_indices)
			if(context.gravity_half_kick_pending[index] == 0)
			{
				calculate_acceleration = true;
				break;
			}
#ifdef RICH_MPI
	int calculate_on_any_rank = calculate_acceleration ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &calculate_on_any_rank, 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	calculate_acceleration = calculate_on_any_rank != 0;
#endif

	if(calculate_acceleration)
	{
		individual_time_step_limits_.assign(
			norg, std::numeric_limits<double>::infinity());
		dt_ = 0;
		if(acc_.SupportsIndividualTargetEvaluation())
		{
			require_on_every_rank(
				context.gravity_source_points.size() ==
					context.gravity_source_masses.size() &&
				context.gravity_source_points.size() ==
					context.gravity_source_ids.size(),
				"Individual gravity source point/mass/ID counts differ");
			vector<Vector3D> target_points;
			target_points.reserve(context.active_indices.size());
			vector<ComputationalCell3D> target_cells;
			target_cells.reserve(context.active_indices.size());
			for(std::size_t index : context.active_indices)
			{
				target_points.push_back(tess.GetCellCM(index));
				target_cells.push_back(cells[index]);
				double const volume = tess.GetVolume(index);
				if(volume > 0 && std::isfinite(volume))
					target_cells.back().density = extensives[index].mass / volume;
			}
			vector<Vector3D> target_accelerations;
			acc_.EvaluateIndividualTargets(tess.GetBoxCoordinates(),
				context.gravity_source_points, context.gravity_source_masses,
				context.gravity_source_ids,
				target_points, target_cells, time, target_accelerations);
			require_on_every_rank(
				target_accelerations.size() == context.active_indices.size(),
				"Acceleration provider returned the wrong active-target count");
			acc_buf_ = context.cached_accelerations;
			for(std::size_t target = 0;
				target < context.active_indices.size(); ++target)
				acc_buf_[context.active_indices[target]] =
					target_accelerations[target];
		}
		else
		{
			vector<ComputationalCell3D> force_cells = cells;
			for(std::size_t index = 0; index < norg; ++index)
			{
				double const volume = tess.GetVolume(index);
				if(volume > 0 && std::isfinite(volume))
					force_cells[index].density = extensives[index].mass / volume;
			}
			acc_buf_.clear();
			acc_(tess, force_cells, fluxes, time, acc_buf_);
			require_on_every_rank(acc_buf_.size() >= norg,
				"Acceleration provider returned too few individual accelerations");
		}
		for(std::size_t index : context.active_indices)
		{
			const double acceleration = fastabs(acc_buf_[index]);
			if(acceleration <= 0)
				continue;
			const double inverse_dt =
				fastsqrt(acceleration / tess.GetWidth(index));
			dt_ = std::max(dt_, inverse_dt);
			individual_time_step_limits_[index] = 1.0 / inverse_dt;
		}
	}
	else
		acc_buf_ = context.cached_accelerations;

	const double fraction = phase == IndividualSourcePhase::Full ? 1.0 : 0.5;
	for(std::size_t index : context.active_indices)
	{
		if(calculate_acceleration)
			context.cached_accelerations[index] = acc_buf_[index];
		context.gravity_half_kick_pending[index] =
			phase == IndividualSourcePhase::FirstHalf ? 0 : 1;
		const double acceleration = fastabs(acc_buf_[index]);

		const double dt = fraction * context.cellTimeStep(index);
		const double old_kinetic = 0.5 * ScalarProd(extensives[index].momentum,
			extensives[index].momentum) / extensives[index].mass;
		extensives[index].momentum += extensives[index].mass * acc_buf_[index] * dt;
		if(mass_flux_ && acceleration * tess.GetWidth(index) * cells[index].density <
			0.5 * cells[index].pressure)
		{
			const double part0 = extensives[index].mass *
				ScalarProd(point_velocities[index], acc_buf_[index]);
			const double part1 = 0.5 * ScalarProd(MassFlux(tess, index, fluxes), acc_buf_[index]);
			extensives[index].energy += (part0 + part1) * dt;
		}
		else
		{
			const double new_kinetic = 0.5 * ScalarProd(extensives[index].momentum,
				extensives[index].momentum) / extensives[index].mass;
			extensives[index].energy += new_kinetic - old_kinetic;
		}
	}
}

bool ConservativeForce3D::IndividualFirstHalfNeedsGeometry(
	const IndividualStepContext& context) const
{
	// The mass-flux energy term reads face geometry; the cached kick does not.
	if(mass_flux_)
		return true;
	// A cell without a pending half kick has no cached acceleration for the
	// interval that is closing and must evaluate one on the interval-start
	// mesh, exactly as ApplyIndividual's FirstHalf branch does.
	for(std::size_t index : context.active_indices)
		if(index >= context.gravity_half_kick_pending.size() ||
		   index >= context.cached_accelerations.size() ||
		   context.gravity_half_kick_pending[index] == 0)
			return true;
	return false;
}

void ConservativeForce3D::ApplyIndividualFirstHalfFromCache(
	const vector<ComputationalCell3D>& cells,
	const vector<Vector3D>& /*point_velocities*/,
	double /*time*/,
	const IndividualStepContext& context,
	vector<Conserved3D>& extensives) const
{
	bool valid = !mass_flux_ && cells.size() == extensives.size() &&
		context.cell_time_steps.size() >= extensives.size() &&
		context.cached_accelerations.size() >= extensives.size() &&
		context.gravity_half_kick_pending.size() >= extensives.size();
	for(std::size_t index : context.active_indices)
		if(index >= extensives.size() ||
		   context.gravity_half_kick_pending[index] == 0)
		{
			valid = false;
			break;
		}
	RequireOnEveryRank(valid,
		"Individual conservative force cannot apply its first half from cache");
	// The same kick as ApplyIndividual's cached FirstHalf branch, on canonical
	// arrays and without a mesh: the energy change is the kinetic change.
	for(std::size_t index : context.active_indices)
	{
		Vector3D const& acceleration = context.cached_accelerations[index];
		context.gravity_half_kick_pending[index] = 0;
		const double dt = 0.5 * context.cellTimeStep(index);
		Conserved3D& extensive = extensives[index];
		const double old_kinetic = 0.5 * ScalarProd(extensive.momentum,
			extensive.momentum) / extensive.mass;
		extensive.momentum += extensive.mass * acceleration * dt;
		const double new_kinetic = 0.5 * ScalarProd(extensive.momentum,
			extensive.momentum) / extensive.mass;
		extensive.energy += new_kinetic - old_kinetic;
	}
}

void ConservativeForce3D::SuggestIndividualTimeSteps(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& /*cells*/,
	const IndividualStepContext& /*context*/,
	vector<double>& time_step_limits) const
{
	for(std::size_t index = 0; index < tess.GetPointNo(); ++index)
	{
		double limit = std::numeric_limits<double>::infinity();
		if(index < individual_time_step_limits_.size())
			limit = individual_time_step_limits_[index];
		else if(index < acc_buf_.size() && fastabs(acc_buf_[index]) > 0)
			limit = fastsqrt(tess.GetWidth(index) / fastabs(acc_buf_[index]));
		time_step_limits.at(index) = std::min(time_step_limits.at(index), limit);
	}
}

void ConservativeForce3D::RefreshIndividualAccelerations(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& extensives,
	double time,
	vector<Vector3D>& accelerations) const
{
	std::size_t const N = tess.GetPointNo();
	RequireOnEveryRank(cells.size() >= N && extensives.size() >= N,
		"Conservative-force acceleration refresh needs one entry per owned cell");
	// ApplyIndividual's evaluation with every owned cell as a target: sources
	// are the cells' centroids and masses on this mesh.
	RequireOnEveryRank(acc_.SupportsIndividualTargetEvaluation(),
		"Conservative-force acceleration refresh needs an acceleration with target evaluation");
	vector<Vector3D> points(N);
	vector<double> masses(N);
	vector<std::uint64_t> ids(N);
	vector<ComputationalCell3D> target_cells(cells.begin(),
		cells.begin() + static_cast<std::ptrdiff_t>(N));
	for(std::size_t index = 0; index < N; ++index)
	{
		points[index] = tess.GetCellCM(index);
		masses[index] = extensives[index].mass;
		ids[index] = static_cast<std::uint64_t>(cells[index].ID);
		double const volume = tess.GetVolume(index);
		if(volume > 0 && std::isfinite(volume))
			target_cells[index].density = extensives[index].mass / volume;
	}
	vector<Vector3D> acceleration;
	acc_.EvaluateIndividualTargets(tess.GetBoxCoordinates(), points, masses,
		ids, points, target_cells, time, acceleration);
	RequireOnEveryRank(acceleration.size() == N,
		"Acceleration provider returned the wrong refreshed target count");
	accelerations.swap(acceleration);
}

void ConservativeForce3D::SynchronizedIndividualLimits(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& extensives,
	double time,
	vector<double>& limits,
	vector<Vector3D>& accelerations) const
{
	std::size_t const N = tess.GetPointNo();
	RequireOnEveryRank(limits.size() >= N,
		"Synchronized conservative-force limits need one entry per owned cell");
	RefreshIndividualAccelerations(tess, cells, extensives, time, accelerations);
	for(std::size_t index = 0; index < N; ++index)
	{
		double const magnitude = fastabs(accelerations[index]);
		if(magnitude > 0)
			limits[index] = std::min(limits[index],
				1.0 / fastsqrt(magnitude / tess.GetWidth(index)));
	}
}

ConstantAcceleration3D::ConstantAcceleration3D(Vector3D const g): g_(g){}

void ConstantAcceleration3D::operator()(const Tessellation3D& tess, 
	const vector<ComputationalCell3D>& /*cells*/, const vector<Conserved3D>& 
	/*fluxes*/, const double /*time*/, vector<Vector3D>& acc) const
{
	size_t const N = tess.GetPointNo();
	acc.resize(N);
	for (size_t i = 0; i < N; ++i)
		acc[i] = g_;
}

void ConstantAcceleration3D::EvaluateIndividualTargets(
	std::pair<Vector3D, Vector3D> const& /*bounds*/,
	vector<Vector3D> const& /*source_points*/,
	vector<double> const& /*source_masses*/,
	vector<std::uint64_t> const& /*source_ids*/,
	vector<Vector3D> const& target_points,
	vector<ComputationalCell3D> const& /*target_cells*/,
	double /*time*/,
	vector<Vector3D>& acc) const
{
	acc.assign(target_points.size(), g_);
}
