#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"

namespace
{

// Deterministic seed mixing (same as amr_random)
std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b, std::uint64_t c)
{
	std::uint64_t x = a + 0x9e3779b97f4a7c15ULL;
	x ^= b + 0xbf58476d1ce4e5b9ULL + (x << 6U) + (x >> 2U);
	x ^= c + 0x94d049bb133111ebULL + (x << 7U) + (x >> 3U);
	return x;
}

// Refine exactly `count` randomly chosen cells, but only on selected ranks.
class TargetedRefine3D : public CellsToRefine3D
{
public:
	TargetedRefine3D(const std::vector<int> &active_ranks, size_t count, int rank)
		: active_ranks_(active_ranks), count_(count), rank_(rank), iter_(0) {}

	std::pair<std::vector<size_t>, std::vector<Vector3D>> ToRefine(
		Tessellation3D const& tess,
		std::vector<ComputationalCell3D> const&,
		double) const override
	{
		std::vector<size_t> refine;
		bool active = std::find(active_ranks_.begin(), active_ranks_.end(), rank_) != active_ranks_.end();
		if (active)
		{
			size_t n = tess.GetPointNo();
			if (n > 0)
			{
				std::mt19937_64 gen(mix_seed(0xCAFE0001ULL, static_cast<std::uint64_t>(rank_), iter_));
				std::uniform_int_distribution<size_t> dist(0, n - 1);

				// Pick `count_` valid candidates (skip cells whose CM is far from mesh point)
				size_t attempts = 0;
				while (refine.size() < count_ && attempts < count_ * 20)
				{
					size_t idx = dist(gen);
					++attempts;
					if (fastabs(tess.GetCellCM(idx) - tess.GetMeshPoint(idx)) > 0.2 * tess.GetWidth(idx))
						continue;
					refine.push_back(idx);
				}
				std::sort(refine.begin(), refine.end());
				refine.erase(std::unique(refine.begin(), refine.end()), refine.end());
			}
		}
		++iter_;
		return std::make_pair(refine, std::vector<Vector3D>());
	}

private:
	const std::vector<int> active_ranks_;
	const size_t count_;
	const int rank_;
	mutable std::uint64_t iter_;
};

// Remove exactly `count` randomly chosen cells, but only on selected ranks.
class TargetedRemove3D : public CellsToRemove3D
{
public:
	TargetedRemove3D(const std::vector<int> &active_ranks, size_t count, int rank)
		: active_ranks_(active_ranks), count_(count), rank_(rank), iter_(0) {}

	std::pair<std::vector<size_t>, std::vector<double>> ToRemove(
		Tessellation3D const& tess,
		std::vector<ComputationalCell3D> const&,
		double) const override
	{
		std::vector<size_t> remove;
		std::vector<double> merits;
		bool active = std::find(active_ranks_.begin(), active_ranks_.end(), rank_) != active_ranks_.end();
		if (active)
		{
			size_t n = tess.GetPointNo();
			if (n > 0)
			{
				std::mt19937_64 gen(mix_seed(0xDEAD0002ULL, static_cast<std::uint64_t>(rank_), iter_));
				std::uniform_int_distribution<size_t> idx_dist(0, n - 1);
				std::uniform_real_distribution<double> merit_dist(0.0, 1.0);

				size_t attempts = 0;
				while (remove.size() < count_ && attempts < count_ * 20)
				{
					size_t idx = idx_dist(gen);
					++attempts;
					if (fastabs(tess.GetCellCM(idx) - tess.GetMeshPoint(idx)) > 0.2 * tess.GetWidth(idx))
						continue;
					remove.push_back(idx);
					merits.push_back(merit_dist(gen));
				}
				// Sort and deduplicate
				if (!remove.empty())
				{
					std::vector<std::pair<size_t, double>> zipped;
					zipped.reserve(remove.size());
					for (size_t i = 0; i < remove.size(); ++i)
						zipped.push_back({remove[i], merits[i]});
					std::sort(zipped.begin(), zipped.end(),
						[](const std::pair<size_t, double> &a, const std::pair<size_t, double> &b)
						{ return a.first < b.first; });

					remove.clear();
					merits.clear();
					size_t cur = zipped[0].first;
					double cur_m = zipped[0].second;
					for (size_t i = 1; i < zipped.size(); ++i)
					{
						if (zipped[i].first == cur)
						{
							cur_m = std::max(cur_m, zipped[i].second);
						}
						else
						{
							remove.push_back(cur);
							merits.push_back(cur_m);
							cur = zipped[i].first;
							cur_m = zipped[i].second;
						}
					}
					remove.push_back(cur);
					merits.push_back(cur_m);
				}
			}
		}
		++iter_;
		return std::make_pair(remove, merits);
	}

private:
	const std::vector<int> active_ranks_;
	const size_t count_;
	const int rank_;
	mutable std::uint64_t iter_;
};

double sum_extensive_mass(const Tessellation3D &tess,
	const std::vector<ComputationalCell3D> &cells)
{
	double total = 0.0;
	size_t n = tess.GetPointNo();
	for (size_t i = 0; i < n; ++i)
		total += cells[i].density * tess.GetVolume(i);
	return total;
}

double sum_extensive_energy(const Tessellation3D &tess,
	const std::vector<ComputationalCell3D> &cells)
{
	double total = 0.0;
	size_t n = tess.GetPointNo();
	for (size_t i = 0; i < n; ++i)
	{
		double mass = cells[i].density * tess.GetVolume(i);
		double kinetic = 0.5 * mass * ScalarProd(cells[i].velocity, cells[i].velocity);
		total += mass * cells[i].internal_energy + kinetic;
	}
	return total;
}

} // anonymous namespace

int main()
{
	int rank = 0;
	int world_size = 1;
#ifdef RICH_MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

	try
	{
		Vector3D ll(-1.0, -1.0, -1.0);
		Vector3D ur(1.0, 1.0, 1.0);
		const size_t target_points = static_cast<size_t>(2e5);
		const size_t amr_count = 500;

		// 4 ranks refine, 5 ranks remove (deterministic, non-overlapping)
		const std::vector<int> refine_ranks = {3, 17, 31, 55};
		const std::vector<int> remove_ranks = {0, 8, 22, 40, 60};

		// Generate initial mesh
		std::vector<Vector3D> points;
		if (rank == 0)
			points = RandRectangular(target_points, ll, ur);
#ifdef RICH_MPI
		points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
		points = RoundGrid3D(points, ll, ur, 10);

		Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
		tess.BuildParallel(points);
#else
		tess.Build(points);
#endif

		// Uniform initial conditions
		IdealGas eos(5.0 / 3.0);
		ComputationalCell3D baseline;
		baseline.density = 1.0;
		baseline.internal_energy = 2.5;
		baseline.pressure = eos.de2p(baseline.density, baseline.internal_energy,
			baseline.tracers, ComputationalCell3D::tracerNames);
		baseline.velocity = Vector3D(0.0, 0.0, 0.0);

		std::vector<ComputationalCell3D> cells(tess.GetPointNo(), baseline);

		// Build the simulation object
		Hllc3D rs;
		RigidWallGenerator3D ghost;
		LinearGauss3D interp(eos, ghost);
		std::vector<std::pair<const ConditionActionFlux1::Condition3D *,
			const ConditionActionFlux1::Action3D *>> sequence;
		IsBoundaryFace3D is_boundary;
		IsBulkFace3D is_bulk;
		RigidWallFlux3D rigid_flux(rs);
		RegularFlux3D regular_flux(rs);
		sequence.push_back(std::make_pair(&is_boundary, &rigid_flux));
		sequence.push_back(std::make_pair(&is_bulk, &regular_flux));
		ConditionActionFlux1 flux(sequence, interp);
		std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
			const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
		ConditionExtensiveUpdater3D eu(eu_sequence);
		DefaultCellUpdater cu;
		ZeroForce3D force;
		CourantFriedrichsLewy tsf(0.3, 1.0, force);
		Lagrangian3D bpm;
		RoundCells3D pm(bpm, eos);
		Simulation simulation(tess, cells, eos);
		HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos,
			simulation.getTracker(), pm, tsf, flux, cu, eu, force,
			std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

		// Measure total mass and energy before AMR
		double mass_before_local = sum_extensive_mass(tess, simulation.getCells());
		double energy_before_local = sum_extensive_energy(tess, simulation.getCells());
		double mass_before = mass_before_local;
		double energy_before = energy_before_local;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &mass_before, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &energy_before, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

		if (rank == 0)
		{
			std::cout << "Initial total mass:   " << mass_before << std::endl;
			std::cout << "Initial total energy: " << energy_before << std::endl;
		}

		// Set up AMR with targeted refine/remove on specific ranks
		TargetedRefine3D refine(refine_ranks, amr_count, rank);
		TargetedRemove3D remove(remove_ranks, amr_count, rank);
		AMR3D amr(eos, refine, remove, interp);

		// Run AMR
		amr(simulation);

		// Measure total mass and energy after AMR
		double mass_after_local = sum_extensive_mass(tess, simulation.getCells());
		double energy_after_local = sum_extensive_energy(tess, simulation.getCells());
		double mass_after = mass_after_local;
		double energy_after = energy_after_local;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &mass_after, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &energy_after, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

		double mass_reldiff = std::abs(mass_after - mass_before) / std::max(std::abs(mass_before), 1e-30);
		double energy_reldiff = std::abs(energy_after - energy_before) / std::max(std::abs(energy_before), 1e-30);

		if (rank == 0)
		{
			std::cout << std::endl;
			std::cout << "After AMR total mass:   " << mass_after << std::endl;
			std::cout << "After AMR total energy: " << energy_after << std::endl;
			std::cout << "Mass   relative diff: " << mass_reldiff << std::endl;
			std::cout << "Energy relative diff: " << energy_reldiff << std::endl;
		}

		const double threshold = 1e-6;
		int passed = (mass_reldiff <= threshold && energy_reldiff <= threshold) ? 1 : 0;
		int all_passed = passed;
#ifdef RICH_MPI
		MPI_Allreduce(&passed, &all_passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

		if (rank == 0)
		{
			std::ofstream out("amr_distributed_clip_metrics.txt");
			out.setf(std::ios::scientific);
			out.precision(16);
			out << "mass_before " << mass_before << "\n";
			out << "mass_after " << mass_after << "\n";
			out << "energy_before " << energy_before << "\n";
			out << "energy_after " << energy_after << "\n";
			out << "mass_reldiff " << mass_reldiff << "\n";
			out << "energy_reldiff " << energy_reldiff << "\n";
			out << "threshold " << threshold << "\n";
			out << "pass " << all_passed << "\n";
			out.close();

			std::cout << std::endl;
			std::cout << (all_passed ? "PASSED" : "FAILED") << std::endl;
		}

#ifdef RICH_MPI
		MPI_Finalize();
#endif
		return all_passed ? 0 : 1;
	}
	catch (UniversalError const &e)
	{
		reportError(e);
#ifdef RICH_MPI
		MPI_Abort(MPI_COMM_WORLD, 2);
#endif
		return 2;
	}
}
