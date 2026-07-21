#include "hdsim_3d.hpp"

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
	#ifdef RICH_MPI
	, exchange_chain_(MPI_COMM_WORLD)
	#endif // RICH_MPI
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
		void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, ExchangeChain &chain, std::vector<Vector3D> const* orgpoints = nullptr)
	#else // RICH_MPI
		void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, std::vector<Vector3D> const* orgpoints = nullptr)
	#endif // RICH_MPI
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
		}
	}
}


void HDSim3D::timeAdvance2(void)
{
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
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
	auto t1 = get_time();
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	auto t2 = get_time();
	DisplayTime(t1, t2, "Source time ");
	if (pt_.cycle % 10 == 0 && pm_.MovedPoints())
	{
		vector<Vector3D>& mesh = tess_.accessMeshPoints();
		mesh.resize(tess_.GetPointNo());
		vector<size_t> order = HilbertOrder3D(mesh);
		mesh = VectorValues(mesh, order);
		mid_extensives = VectorValues(mid_extensives, order);
		extensive_ = VectorValues(extensive_, order);
		cells_ = VectorValues(cells_, order);
		point_vel = VectorValues(point_vel, order);
#ifdef RICH_MPI
		tess_.PreparePoints(mesh, order);
#endif
	}
	Conserved3D edummy;
	ComputationalCell3D cdummy;
	if(pm_.MovedPoints())
	{
		MovePoints(tess_, point_vel, dt);
		t1 = get_time();
		#ifdef RICH_MPI
			UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_);
		#else // RICH_MPI
			UpdateTessellation(tess_, point_vel, dt);
		#endif // RICH_MPI
		t2 = get_time();
		DisplayTime(t1, t2, "Voronoi build time ");
#ifdef RICH_MPI
		// Keep relevant points
		MPI_exchange_data(tess_, mid_extensives, false);
		MPI_exchange_data(tess_, extensive_, false);
		MPI_exchange_data(tess_, cells_, false);
		MPI_exchange_data(tess_, point_vel, false);
		MPI_exchange_data(tess_, point_vel, true);
#endif
	}

cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif

pt_.updateTime(dt);
pt_.updateCycle();
CalcFaceVelocities(tess_, point_vel, face_vel);
face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
t1 = get_time();
source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
t2 = get_time();
DisplayTime(t1, t2, "Second source time ");
eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
ExtensiveAvg(extensive_, mid_extensives);
cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif
}

void HDSim3D::timeAdvance(void)
{
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI

	vector<Vector3D> point_vel, face_vel;
	pm_(tess_, cells_, pt_.getTime(), point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	const double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	dt_ = dt;
	pm_.ApplyFix(tess_, cells_, pt_.getTime(), dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	vector<Conserved3D> fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values = 
		fc_(fluxes, tess_, face_vel, cells_, extensive_, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, extensive_);
	eu_(fluxes, tess_, dt, cells_, extensive_, pt_.getTime(), face_vel, face_values);
	if(pm_.MovedPoints())
	{
	MovePoints(tess_, point_vel, dt);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt);
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
	pt_.updateTime(dt);
	pt_.updateCycle();
}


void HDSim3D::timeAdvance3(void)
{
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI

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
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->exchange_chain_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt);
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

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, &oldpoints);
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
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI

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
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt);
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

	pt_.updateTime(dt);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	mid_extensives = 0.25 * mid_extensives + 0.75 * extensive_;

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt / 2, this->exchange_chain_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt / 2, &oldpoints);
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
	pt_.updateTime(-0.5 * dt);
	pt_.updateCycle();
	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_values = fc_(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, pt_.getTime(), dt);
	source_(tess_, cells_, fluxes, point_vel, pt_.getTime(), dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, pt_.getTime(), face_vel, face_values);
	extensive_ = 0.33333333333333333333333 * (2 * mid_extensives + extensive_);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, &oldpoints);
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
}

void HDSim3D::timeAdvance32(void)
{
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI

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

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt);
	#endif // RICH_MPI

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
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI

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
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->exchange_chain_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt);
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

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, &oldpoints);
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

double HDSim3D::RadiationTimeStep(double const dt, RadiationDriver const& matrix_builder, bool const no_hydro)
{
	int total_iters = 0;
	double const CG_eps = 1e-11;
	size_t const N = tess_.GetPointNo();

#ifdef DEBUG
	if(N == 0) std::cout<<"Zero cells in RadiationTimeStep"<<std::endl;
#endif

	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

	double total_elapsed_time = 0;
	double dt_try = dt;
	size_t reduce_counter = 0;
	int max_iter_done = 0;


	matrix_builder.prestep(tess_, cells_);
	while(total_elapsed_time < dt * 0.9999999)
	{
		dt_try = std::min(dt_try, dt - total_elapsed_time);

		bool step_success = matrix_builder.step(CG_eps, total_iters, tess_, cells_, extensive_, dt_try, pt_.getTime());

		max_iter_done = std::max(max_iter_done, total_iters);
		
		if(not step_success)
		{
			reduce_counter++;
			dt_try *= 0.5;
			if(rank == 0)
				std::cout<<"Reducing dt, new dt "<<dt_try<<std::endl;
			
			if(dt_try < 0.001 * dt)
				throw UniversalError("too small dt in RadiationTimeStep");
		}
		else {
			total_elapsed_time += dt_try;
		}
	}

	double new_dt = matrix_builder.calculate_dt(dt, tess_, cells_);

	matrix_builder.poststep();

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	
	
	if(no_hydro)
	{
		pt_.updateTime(dt);
		pt_.updateCycle();
	}
	// double grow_factor = 1.25;
	// if(max_iter_done > 200)
	// 	grow_factor = 1.02;
	// else
	// 	if(max_iter_done > 125)
	// 		grow_factor = 1.05;

	// new_dt = std::min(new_dt, dt*grow_factor) * std::pow(0.5, std::max(static_cast<double>(reduce_counter), 0.0));
	// if(max_iter_done > 300)
	// 	new_dt = dt * 0.9;
	// return this->radiation_dt_;
	return new_dt;
}