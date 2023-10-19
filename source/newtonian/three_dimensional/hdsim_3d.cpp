#include <cassert>
#include "hdsim_3d.hpp"
#include "3D/hilbert/HilbertOrder3D.hpp"
#include "misc/utils.hpp"
#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif
#include <chrono>

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
			std::cout<<msg<< std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()<<" seconds"<<std::endl;
		#endif
	}
}

Tessellation3D& HDSim3D::getTesselation(void)
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

HDSim3D::ProgressTracker::ProgressTracker(void) :
	time(0), cycle(0) {}

void HDSim3D::ProgressTracker::updateTime(double dt)
{
	time += dt;
}

void HDSim3D::ProgressTracker::updateCycle()
{
	++cycle;
}

double HDSim3D::ProgressTracker::getTime(void) const
{
	return time;
}

size_t HDSim3D::ProgressTracker::getCycle(void) const
{
	return cycle;
}

HDSim3D::HDSim3D(Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const EquationOfState& eos,
	const PointMotion3D& pm,
	const TimeStepFunction3D& tsc,
	const FluxCalculator3D& fc,
	const CellUpdater3D& cu,
	const ExtensiveUpdater3D& eu,
	const SourceTerm3D& source,
	const pair<vector<string>, vector<string> >& tsn,
	bool SR, 
	bool new_start) :
	tess_(tess),
	eos_(eos), cells_(cells), extensive_(), pm_(pm), tsc_(tsc), fc_(fc), cu_(cu), eu_(eu), source_(source), pt_()
	, Max_ID_(0)
	, dt_(0)
{
#ifdef RICH_MPI
	int ws = 0, rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	const bool validity_check = tess.GetPointNo() == cells.size();
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
	// Is this a new start?
	if (new_start)
	{
		size_t nstart = 0;
#ifdef RICH_MPI
		std::vector<size_t> nrecv(static_cast<size_t>(ws), 0);
		size_t nsend = N;
		MPI_Allgather(&nsend, 1, MPI_UNSIGNED_LONG_LONG, &nrecv[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		for (int i = 0; i < rank; ++i)
			nstart += nrecv[static_cast<size_t>(i)];
#endif
		for (size_t i = 0; i < N; ++i)
			cells_[i].ID = nstart + i;
		Max_ID_ = nstart + N - 1;
#ifdef RICH_MPI
		for (size_t i = static_cast<size_t>(rank + 1); i < static_cast<size_t>(ws); ++i)
			Max_ID_ += nrecv[i];
#endif
	}
	else
	{
		size_t maxid = 0;
		for (size_t i = 0; i < N; ++i)
			maxid = std::max(maxid, cells[i].ID);
#ifdef RICH_MPI
		MPI_Allreduce(&maxid, &Max_ID_, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
		Max_ID_ = maxid;
#endif
	}

#ifdef RICH_MPI
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess_, cells_, true, &cdummy);
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

	void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, std::vector<Vector3D> const* orgpoints = nullptr)
	{
		vector<Vector3D> points;
		if (orgpoints == nullptr)
			points = tess.getMeshPoints();
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
		tess.BuildHilbert(points);
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
		}
	}
}


void HDSim3D::timeAdvance2(void)
{
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true, &vdummy);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true, &vdummy);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values = 
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), dt);
	vector<Conserved3D> mid_extensives(extensive_);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);

	if (pt_.cycle % 10 == 0)
	{
		vector<Vector3D>& mesh = tess_.accessMeshPoints();
		mesh.resize(tess_.GetPointNo());
		vector<size_t> order = HilbertOrder3D(mesh);
		mesh = VectorValues(mesh, order);
		mid_extensives = VectorValues(mid_extensives, order);
		extensive_ = VectorValues(extensive_, order);
		cells_ = VectorValues(cells_, order);
		point_vel = VectorValues(point_vel, order);
	}
	MovePoints(tess_, point_vel, dt);
	auto t1 = get_time();
	UpdateTessellation(tess_, point_vel, dt);
	auto t2 = get_time();
	DisplayTime(t1, t2, "Voronoi build time");
#ifdef RICH_MPI
	// Keep relevant points
	Conserved3D edummy;
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess_, mid_extensives, false, &edummy);
	MPI_exchange_data(tess_, extensive_, false, &edummy);
	MPI_exchange_data(tess_, cells_, false, &cdummy);
	MPI_exchange_data(tess_, point_vel, false, &vdummy);
	MPI_exchange_data(tess_, point_vel, true, &vdummy);
#endif

cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true, &cdummy);
#endif

pt_.updateTime(dt);
pt_.updateCycle();
CalcFaceVelocities(tess_, point_vel, face_vel);
face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
ExtensiveAvg(extensive_, mid_extensives);
cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true, &cdummy);
#endif
}

void HDSim3D::timeAdvance(void)
{
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true, &vdummy);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	const double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true, &vdummy);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values = 
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, extensive_);
	eu_(fluxes, tess_, dt, cells_, extensive_, pt_.getTime(), face_vel, face_values);
	MovePoints(tess_, point_vel, dt);
	UpdateTessellation(tess_, point_vel, dt);
#ifdef RICH_MPI
	// Keep relevant points
	ComputationalCell3D cdummy;
	Conserved3D edummy;
	MPI_exchange_data(tess_, extensive_, false, &edummy);
	MPI_exchange_data(tess_, cells_, false, &cdummy);
#endif
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true, &cdummy);
#endif
	pt_.updateTime(dt);
	pt_.updateCycle();
}


void HDSim3D::timeAdvance3(void)
{
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values = 
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), 0.5 * dt);
	vector<Conserved3D> mid_extensives(extensive_);
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), 0.5 * dt, mid_extensives);

	if (pt_.cycle % 10 == 0)
	{
		vector<Vector3D>& mesh = tess_.accessMeshPoints();
		mesh.resize(tess_.GetPointNo());
		vector<size_t> order = HilbertOrder3D(mesh);
		mesh = VectorValues(mesh, order);
		mid_extensives = VectorValues(mid_extensives, order);
		extensive_ = VectorValues(extensive_, order);
		cells_ = VectorValues(cells_, order);
		point_vel = VectorValues(point_vel, order);
		//du1 = VectorValues(du1, order);
	}
	std::vector<Vector3D> oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt * 0.5);
	UpdateTessellation(tess_, point_vel, 0.5 * dt);
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
	std::vector<Conserved3D> u1 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	pt_.updateTime(0.5 * dt);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), 2 * dt);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), 2 * dt,  mid_extensives);
	eu_(fluxes, tess_, 2 * dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	mid_extensives = mid_extensives - 3 * (u1 - extensive_);

	UpdateTessellation(tess_, point_vel, dt, &oldpoints);
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
	std::vector<Conserved3D> u2 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	pt_.updateTime(0.5 * dt);
	pt_.updateCycle();
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt / 6);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt / 6,  mid_extensives);
	eu_(fluxes, tess_, dt / 6, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	extensive_ = mid_extensives - (1.0 / 3.0) * (2 * u2 + extensive_) + u1;
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
}

void HDSim3D::timeAdvance33(void)
{
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values = 
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), dt);
	vector<Conserved3D> mid_extensives(extensive_);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);

	if (pt_.cycle % 10 == 0)
	{
		vector<Vector3D>& mesh = tess_.accessMeshPoints();
		mesh.resize(tess_.GetPointNo());
		vector<size_t> order = HilbertOrder3D(mesh);
		mesh = VectorValues(mesh, order);
		mid_extensives = VectorValues(mid_extensives, order);
		extensive_ = VectorValues(extensive_, order);
		cells_ = VectorValues(cells_, order);
		point_vel = VectorValues(point_vel, order);
	}
	std::vector<Vector3D> oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt);
	UpdateTessellation(tess_, point_vel, dt);
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

	pt_.updateTime(dt);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	mid_extensives = 0.25 * mid_extensives + 0.75 * extensive_;

	UpdateTessellation(tess_, point_vel, dt / 2, &oldpoints);
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
	pt_.updateTime(-0.5 * dt);
	pt_.updateCycle();
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	extensive_ = 0.33333333333333333333333 * (2 * mid_extensives + extensive_);

	UpdateTessellation(tess_, point_vel, dt, &oldpoints);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif

	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
}

void HDSim3D::timeAdvance32(void)
{
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values =
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), 0.5 * dt);
	vector<Conserved3D> mid_extensives(extensive_);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);

	if (pt_.cycle % 10 == 0)
	{
		vector<Vector3D>& mesh = tess_.accessMeshPoints();
		mesh.resize(tess_.GetPointNo());
		vector<size_t> order = HilbertOrder3D(mesh);
		mesh = VectorValues(mesh, order);
		mid_extensives = VectorValues(mid_extensives, order);
		extensive_ = VectorValues(extensive_, order);
		cells_ = VectorValues(cells_, order);
		point_vel = VectorValues(point_vel, order);
	}
	MovePoints(tess_, point_vel, dt);
	UpdateTessellation(tess_, point_vel, dt);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	std::vector<Conserved3D> u1 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	pt_.updateTime(dt);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	mid_extensives = 0.5 * (mid_extensives + extensive_);
	cu_(cells_, eos_, tess_, mid_extensives);

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	pt_.updateCycle();
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	//extensive_ = 0.333333333333333333*(extensive_ + u1 + mid_extensives);
	extensive_ = 0.333333333333333333 * (extensive_ + u1 + mid_extensives);
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
}

void HDSim3D::timeAdvance4(void)
{
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values =
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), 0.5 * dt);
	vector<Conserved3D> mid_extensives(extensive_);
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), 0.5 * dt, mid_extensives);

	if (pt_.cycle % 10 == 0)
	{
		vector<Vector3D>& mesh = tess_.accessMeshPoints();
		mesh.resize(tess_.GetPointNo());
		vector<size_t> order = HilbertOrder3D(mesh);
		mesh = VectorValues(mesh, order);
		mid_extensives = VectorValues(mid_extensives, order);
		extensive_ = VectorValues(extensive_, order);
		cells_ = VectorValues(cells_, order);
		point_vel = VectorValues(point_vel, order);
		//du1 = VectorValues(du1, order);
	}
	std::vector<Vector3D> oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt * 0.5);
	UpdateTessellation(tess_, point_vel, 0.5 * dt);
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
	std::vector<Conserved3D> du1 = mid_extensives - extensive_;

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	pt_.updateTime(0.5 * dt);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), 0.5 * dt);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), 0.5 * dt, mid_extensives);
	mid_extensives = mid_extensives - du1;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	std::vector<Conserved3D> du2 = mid_extensives - extensive_;

	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	mid_extensives = mid_extensives - du2;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);

	UpdateTessellation(tess_, point_vel, dt, &oldpoints);
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
	std::vector<Conserved3D> du3 = mid_extensives - extensive_;

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	pt_.updateTime(0.5 * dt);
	pt_.updateCycle();
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt / 6);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt / 6,  mid_extensives);
	mid_extensives = mid_extensives - du3;
	eu_(fluxes, tess_, dt / 6, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	extensive_ = mid_extensives + (1.0 / 6.0) * (2 * du1 + 4 * du2 + 2 * du3);
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
}

const Tessellation3D& HDSim3D::getTesselation(void) const
{
	return tess_;
}

const vector<ComputationalCell3D>& HDSim3D::getCells(void) const
{
	return cells_;
}

double HDSim3D::getTime(void)const
{
	return pt_.getTime();
}

size_t HDSim3D::getCycle(void)const
{
	return static_cast<size_t>(pt_.getCycle());
}

void HDSim3D::SetCycle(size_t cycle)
{
	pt_.cycle = cycle;
}

void HDSim3D::SetTime(double t)
{
	pt_.time = t;
}

size_t& HDSim3D::GetMaxID(void)
{
	return Max_ID_;
}

double HDSim3D::RadiationTimeStep(double const dt, CG::MatrixBuilder const& matrix_builder, bool const no_hydro)
{
	int total_iters = 0;
	double const CG_eps = 1e-10;
	size_t const N = tess_.GetPointNo();
	if(N == 0)
		std::cout<<"Zero cells in RadiationTimeStep"<<std::endl;
	std::vector<double> old_Er(N, 0);
	for(size_t i = 0; i < N; ++i)
	{
		old_Er[i] = cells_[i].Erad * cells_[i].density;
		if(old_Er[i] < 0)
		{
			UniversalError eo("negative Erad");
			eo.addEntry("i", i);
			eo.addEntry("old_Er", old_Er[i]);
			eo.addEntry("ID", cells_[i].ID);
			eo.addEntry("density", cells_[i].density);
			throw eo;
		}
	}
	double max_Er = *std::max_element(old_Er.begin(), old_Er.end());
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif	
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	double total_elapsed_time = 0;
	double dt_try = dt;
	std::vector<double> new_Er, new_Er_full;
	size_t reduce_counter = 0;
	while(total_elapsed_time < dt * 0.9999999)
	{
		std::vector<ComputationalCell3D> cells(cells_);
		std::vector<Conserved3D> extensives(extensive_);
		bool good_try = true;
		dt_try = std::min(dt_try, dt - total_elapsed_time);

		new_Er = CG::conj_grad_solver(CG_eps, total_iters, tess_, cells_ , dt_try, matrix_builder, pt_.getTime(), new_Er_full);
		double max_Er = *std::max_element(new_Er.begin(), new_Er.end());
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		double min_Er = 1;
		for(size_t i = 0; i < N; ++i)
		{
			if(new_Er[i] < 0 && std::abs(new_Er[i]) < 1e-9 * max_Er)
				new_Er[i] = std::min(1e-8 * max_Er, CG::radiation_constant * cells_[i].temperature * cells_[i].temperature * cells_[i].temperature * cells_[i].temperature);
			min_Er = std::min(min_Er, new_Er[i]);
		}
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &min_Er, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
		if(min_Er < 0)
		{
			reduce_counter++;
			good_try = false;
			dt_try *= 0.5;
			if(rank == 0)
				std::cout<<"Negative Er, Reducing dt, new dt "<<dt_try<<std::endl;
			if(dt_try < 0.001 * dt)
				throw UniversalError("too small dt in RadiationTimeStep");
		}
		else
		{
			int good_run = 1;
			try
			{
				matrix_builder.PostCG(tess_, extensive_, dt_try, cells_, new_Er, new_Er_full);		
			}
			catch(UniversalError const& eo)
			{
				good_run = 0;
			}
			if(good_run == 0)
			{
				good_try = false;
				dt_try *= 0.5;
				if(rank == 0)
					std::cout<<"Reducing PostCG dt, new dt "<<dt_try<<std::endl;
				extensive_ = extensives;
				cells_ = cells;
				if(dt_try < 0.001 * dt)
					throw UniversalError("too small dt in RadiationTimeStep");
			}
		}
		if(good_try)
			total_elapsed_time += dt_try;
	}
	size_t const Nzero = matrix_builder.zero_cells_.size();
	std::vector<size_t> zero_indeces;
	for(size_t i = 0; i < Nzero; ++i)
		zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, matrix_builder.zero_cells_[i]));
	double max_diff = std::numeric_limits<double>::min() * 100;
	int max_loc = 0;
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	for(size_t i = 0; i < N; ++i)
	{
		bool to_calc = true;
		for(size_t j = 0; j < Nzero; ++j)
			if(cells_[i].stickers[zero_indeces[j]])
				to_calc = false;
		if(not to_calc)
			continue;
		double const equlibrium_factor = std::abs(cells_[i].temperature - std::pow(new_Er[i] / CG::radiation_constant, 0.25)) < 0.02 * cells_[i].temperature ? 0.05 : 1;
		double const diff = equlibrium_factor * std::abs(cells_[i].Erad * cells_[i].density - old_Er[i]) / (cells_[i].Erad * cells_[i].density + 0.02 * max_Er);
		if(diff > max_diff)
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
    max_data.mpi_id = rank;
    max_data.val = max_diff;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
	max_diff = max_data.val;
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess_, cells_, true, &cdummy);	
#endif
	if(rank == max_data.mpi_id)
		std::cout<<"Radiation time step ID "<<cells_[max_loc].ID<<" old Er "<<old_Er[max_loc]<<" new Er "<<cells_[max_loc].Erad * cells_[max_loc].density<<
		" diff "<<max_diff<<" Tgas "<<cells_[max_loc].temperature<<" Trad "<<std::pow(new_Er[max_loc] / CG::radiation_constant, 0.25)<<" max_Er "<<max_Er<<" rank "<<rank<<" density "<<cells_[max_loc].density<<std::endl;
	if(no_hydro)
	{
		pt_.updateTime(dt);
		pt_.updateCycle();
	}
	return dt * std::min(0.075 / max_diff, 1.2) * std::pow(0.5, std::max(static_cast<double>(reduce_counter) - 1, 0.0));
}