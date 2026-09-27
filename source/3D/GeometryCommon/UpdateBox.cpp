#include <cassert>
#include <cmath>
#include <stdexcept>
#include "UpdateBox.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <MeshDecomposer3D/kernels/Rectangle.hpp>

namespace
{
	void RequireOnAllRanks(bool const valid, char const* message)
	{
		int ok = valid ? 1 : 0;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
		if(ok == 0)
			throw std::logic_error(message);
	}

	struct BoxGrowthPlan
	{
		bool grow = false;
		Vector3D ll;
		Vector3D ur;
		double max_width = 0;
		double old_volume = 0;
		double new_volume = 0;
		size_t points = 0;
	};

	// The resize decision of the original UpdateBox, operation for operation
	// (global results must not change): `point(i)` and `width(i)` are the
	// generator and width of owned cell i, whose velocity is cells[i].
	template <typename PointAt, typename WidthAt>
	BoxGrowthPlan PlanBoxGrowth(size_t const N, std::vector<ComputationalCell3D> const& cells,
		PointAt const& point, WidthAt const& width, Vector3D const& cur_min, Vector3D const& cur_max,
		double const min_velocity, double const volume_fraction)
	{
		Vector3D maxv(-1e200, -1e200, -1e200), minv(1e200, 1e200, 1e200);
		double maxR = 0;
		for (size_t i = 0; i < N; ++i)
		{
			if (fastabs(cells[i].velocity) > min_velocity)
			{
				Vector3D const p = point(i);
				maxv.x = std::max(maxv.x, p.x);
				maxv.y = std::max(maxv.y, p.y);
				maxv.z = std::max(maxv.z, p.z);
				minv.x = std::min(minv.x, p.x);
				minv.y = std::min(minv.y, p.y);
				minv.z = std::min(minv.z, p.z);
				maxR = std::max(maxR, width(i));
			}
		}
		std::array<double, 7> tempvec, temprecv;
		tempvec[0] = maxR;
		tempvec[1] = maxv.x;
		tempvec[2] = maxv.y;
		tempvec[3] = maxv.z;
		tempvec[4] = -minv.x;
		tempvec[5] = -minv.y;
		tempvec[6] = -minv.z;
		temprecv = tempvec;
#ifdef RICH_MPI
		MPI_Allreduce(&tempvec[0], &temprecv[0], 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		double const oldv = (cur_max.x - cur_min.x) * (cur_max.y - cur_min.y) * (cur_max.z - cur_min.z);
		temprecv[0] = std::max(temprecv[0], std::pow(oldv, 0.333333333) * 0.03);
		// Do we need to resize?
		Vector3D recvmax(temprecv[1] + 5 * temprecv[0], temprecv[2] + 5 * temprecv[0],
			temprecv[3] + 5 * temprecv[0]), recvmin(-temprecv[4] - 5 * temprecv[0],
				-temprecv[5] - 5 * temprecv[0], -temprecv[6] - 5 * temprecv[0]);
		bool recalc = false;

		if (recvmax.x > cur_max.x)
		{
			recalc = true;
			recvmax.x = cur_max.x + 5 * temprecv[0];
		}
		if (recvmax.y > cur_max.y)
		{
			recalc = true;
			recvmax.y = cur_max.y + 5 * temprecv[0];
		}
		if (recvmax.z > cur_max.z)
		{
			recalc = true;
			recvmax.z = cur_max.z + 5 * temprecv[0];
		}
		if (recvmin.x < cur_min.x)
		{
			recalc = true;
			recvmin.x = cur_min.x - 5 * temprecv[0];
		}
		if (recvmin.y < cur_min.y)
		{
			recalc = true;
			recvmin.y = cur_min.y - 5 * temprecv[0];
		}
		if (recvmin.z < cur_min.z)
		{
			recalc = true;
			recvmin.z = cur_min.z - 5 * temprecv[0];
		}
		BoxGrowthPlan plan;
		plan.grow = recalc;
		plan.max_width = temprecv[0];
		plan.old_volume = oldv;
		if (recalc)
		{
			recvmax.x = std::max(recvmax.x, cur_max.x);
			recvmax.y = std::max(recvmax.y, cur_max.y);
			recvmax.z = std::max(recvmax.z, cur_max.z);
			recvmin.x = std::min(recvmin.x, cur_min.x);
			recvmin.y = std::min(recvmin.y, cur_min.y);
			recvmin.z = std::min(recvmin.z, cur_min.z);
			// Get how many points to add
			double const newv = (recvmax.x - recvmin.x) * (recvmax.y - recvmin.y) * (recvmax.z - recvmin.z);
			plan.new_volume = newv;
			plan.points = static_cast<size_t>((newv - oldv) / (volume_fraction * newv));
		}
		plan.ll = recvmin;
		plan.ur = recvmax;
		return plan;
	}

	void ReportResize(BoxGrowthPlan const& plan, Vector3D const& cur_min, Vector3D const& cur_max, int const rank)
	{
		if (rank == 0)
		{
			std::cout << "Doing resize rank " << rank << std::endl;
			std::cout << "Old box ll = " << cur_min.x << "," << cur_min.y << "," << cur_min.z << " ur = " << cur_max.x << "," << cur_max.y << "," << cur_max.z << std::endl;
			std::cout << "New box ll = " << plan.ll.x << "," << plan.ll.y << "," << plan.ll.z << " ur = " << plan.ur.x << "," << plan.ur.y << "," << plan.ur.z << std::endl;
			std::cout << "Max cell size " << plan.max_width << std::endl;
			std::cout << "Old vol " << plan.old_volume << " New vol " << plan.new_volume << std::endl;
			std::cout << "Point number " << plan.points << std::endl;
		}
	}

	// The new region's cells, generated on rank 0 as the original UpdateBox
	// does (random points outside the old box, seeded by the cycle), appended
	// to `points` and `cells`.  MaxID advances on every rank.
	void AppendGrowthCells(BoxGrowthPlan const& plan, Vector3D const& cur_min, Vector3D const& cur_max,
		Simulation& sim, ComputationalCell3D const& reference_cell, int const rank,
		std::vector<Vector3D>& points, std::vector<ComputationalCell3D>& cells)
	{
		size_t& MaxID = sim.GetMaxID();
		size_t const Np = plan.points;
		Vector3D const& recvmin = plan.ll;
		Vector3D const& recvmax = plan.ur;
		if(rank == 0)
		{
			boost::random::mt19937_64 generator(sim.GetCycle());
			boost::random::uniform_real_distribution<> dist;
			double ran[3];
			Vector3D point;
			size_t counter = 0;
			while (counter < Np)
			{
				ran[0] = dist(generator);
				ran[1] = dist(generator);
				ran[2] = dist(generator);
				point.x = ran[0] * (recvmax.x - recvmin.x) + recvmin.x;
				point.y = ran[1] * (recvmax.y - recvmin.y) + recvmin.y;
				point.z = ran[2] * (recvmax.z - recvmin.z) + recvmin.z;
				if (point.x<cur_min.x || point.y<cur_min.y || point.z<cur_min.z || point.x>cur_max.x || point.y>cur_max.y || point.z>cur_max.z)
				{
					points.push_back(point);
					cells.push_back(reference_cell);
					cells.back().ID = MaxID + 1 + counter;
					++counter;
				}
			}
		}
		MaxID += Np;
	}

	int WorldRank(void)
	{
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		return rank;
	}
}

void UpdateBox(Voronoi3D &tess, Simulation &sim, double const min_velocity, double const volume_fraction, ComputationalCell3D const& reference_cell)
{
	if(sim.GetTimeIntegrationMode() == TimeIntegrationMode::Individual)
	{
		UpdateBoxSynchronized(tess, sim, min_velocity, volume_fraction, reference_cell);
		return;
	}
	std::vector<ComputationalCell3D>& cells = sim.getCells();
	std::vector<Conserved3D>& extensives = sim.getExtensives();
	size_t const N = tess.GetPointNo();
	Vector3D const cur_max = tess.GetBoxCoordinates().second;
	Vector3D const cur_min = tess.GetBoxCoordinates().first;
	BoxGrowthPlan const plan = PlanBoxGrowth(N, cells,
		[&tess](size_t const i) { return tess.GetMeshPoint(i); },
		[&tess](size_t const i) { return tess.GetWidth(i); },
		cur_min, cur_max, min_velocity, volume_fraction);
	int const rank = WorldRank();
	if (plan.grow)
	{
		ReportResize(plan, cur_min, cur_max, rank);
		tess.SetBox(plan.ll, plan.ur);
		// tess.SetKernel(new Rectangle(recvmin, recvmax));
		std::vector<Vector3D> mypoints = tess.getMeshPoints();
		mypoints.resize(N);
		cells.resize(N);
		AppendGrowthCells(plan, cur_min, cur_max, sim, reference_cell, rank, mypoints, cells);
		assert(N>0);

#ifdef RICH_MPI
		tess.BuildParallel(mypoints);
		MPI_exchange_data(tess, cells, false);
		MPI_exchange_data(tess, cells, true);
#else // RICH_MPI
		tess.Build(mypoints);
#endif // RICH_MPI

		extensives.resize(tess.GetPointNo());
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
			PrimitiveToConserved(cells.at(i), tess.GetVolume(i), extensives.at(i));
	}
}

bool UpdateBoxSynchronized(Voronoi3D &tess, Simulation &sim, double const min_velocity,
	double const volume_fraction, ComputationalCell3D const& reference_cell,
	Simulation::DomainGrowthReport* report)
{
	if(sim.GetTimeIntegrationMode() != TimeIntegrationMode::Individual)
	{
		std::pair<Vector3D, Vector3D> const old_box = tess.GetBoxCoordinates();
		UpdateBox(tess, sim, min_velocity, volume_fraction, reference_cell);
		return !(old_box == tess.GetBoxCoordinates());
	}
	if(!sim.IndividualStateSynchronized())
		throw std::logic_error("Individual box growth requires a synchronized individual state");
	std::vector<ComputationalCell3D> const& cells = sim.getCells();
	size_t const N = cells.size();
	// At a synchronized state the full mesh holds every owned cell, but its
	// order need not be the canonical one: read it through the view.
	ActiveMeshView const view(tess, N);
	RequireOnAllRanks(view.localSize() == N,
		"The synchronized individual mesh omits owned cells");
	std::vector<Vector3D> positions(N);
	std::vector<double> widths(N);
	for(size_t local = 0; local < N; ++local)
	{
		size_t const global = view.localToGlobal(local);
		positions[global] = tess.GetMeshPoint(local);
		widths[global] = tess.GetWidth(local);
	}
	// The event mesh is built at the positions the event commits; the
	// growth rebuild starts from the committed ones.
	std::vector<Vector3D> const committed = sim.CommittedGeneratorPoints();
	bool same = committed.size() == N;
	for(size_t i = 0; same && i < N; ++i)
		same = committed[i].x == positions[i].x && committed[i].y == positions[i].y &&
			committed[i].z == positions[i].z;
	RequireOnAllRanks(same,
		"The synchronized individual mesh is not built at the committed generator positions");
	Vector3D const cur_max = tess.GetBoxCoordinates().second;
	Vector3D const cur_min = tess.GetBoxCoordinates().first;
	BoxGrowthPlan const plan = PlanBoxGrowth(N, cells,
		[&positions](size_t const i) { return positions[i]; },
		[&widths](size_t const i) { return widths[i]; },
		cur_min, cur_max, min_velocity, volume_fraction);
	if(!plan.grow)
		return false;
	int const rank = WorldRank();
	ReportResize(plan, cur_min, cur_max, rank);
	std::vector<Vector3D> added_points;
	std::vector<ComputationalCell3D> added_cells;
	AppendGrowthCells(plan, cur_min, cur_max, sim, reference_cell, rank, added_points, added_cells);
	Simulation::DomainGrowthReport const growth =
		sim.GrowDomainAtSynchronizedIndividualState(plan.ll, plan.ur, added_points, added_cells);
	if(report != nullptr)
		*report = growth;
	return true;
}

bool BoxGrowthDue(Simulation const& sim, double const min_velocity)
{
	RequireOnAllRanks(sim.GetTimeIntegrationMode() == TimeIntegrationMode::Individual,
		"BoxGrowthDue evaluates the committed state of an individual run");
	std::vector<ComputationalCell3D> const& cells = sim.getCells();
	std::vector<Conserved3D> const& extensives = sim.getExtensives();
	RequireOnAllRanks(extensives.size() == cells.size(),
		"BoxGrowthDue found misaligned committed state");
	size_t const N = cells.size();
	std::vector<Vector3D> const points = sim.CommittedGeneratorPoints();
	// Width from the committed volume, as Voronoi3D::GetWidth from a volume;
	// a cell without a usable volume contributes no width (the floor stays).
	auto const width = [&cells, &extensives](size_t const i)
	{
		double const volume = extensives[i].mass / cells[i].density;
		if(!(std::isfinite(volume) && volume > 0))
			return 0.0;
		return std::pow(3 * volume * 0.25 / M_PI, 0.3333333333);
	};
	std::pair<Vector3D, Vector3D> const box = sim.getTessellation().GetBoxCoordinates();
	// The volume fraction only sets the number of new points, not the decision.
	BoxGrowthPlan const plan = PlanBoxGrowth(N, cells,
		[&points](size_t const i) { return points[i]; }, width,
		box.first, box.second, min_velocity, 1.0);
	return plan.grow;
}
