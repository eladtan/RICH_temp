#include "hdsim_3d.hpp"
#include "CourantFriedrichsLewy.hpp"
#include "CFL1D.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"


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
				vacuum_state.density = cell.density * vac_factor;
				vacuum_state.pressure = cell.pressure * vac_factor;
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
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);
	MEMORY_DEBUG_PRINT("hydro: after flux calc");
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	MEMORY_DEBUG_PRINT("hydro: after extensive update");
	auto t1 = get_time();
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);
	auto t2 = get_time();
	DisplayTime(t1, t2, "Source time ");
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
fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
t1 = get_time();
source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
t2 = get_time();
DisplayTime(t1, t2, "Second source time ");
eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
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
