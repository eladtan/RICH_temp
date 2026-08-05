#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"

namespace
{
	class NoRefine3D : public CellsToRefine3D
	{
	public:
		std::pair<std::vector<size_t>, std::vector<Vector3D> > ToRefine(
			Tessellation3D const&, std::vector<ComputationalCell3D> const&, double) const override
		{
			return std::make_pair(std::vector<size_t>(), std::vector<Vector3D>());
		}
	};

	class RemoveIdPair3D : public CellsToRemove3D
	{
	public:
		RemoveIdPair3D(size_t first, size_t second): first_(first), second_(second) {}

		std::pair<std::vector<size_t>, std::vector<double> > ToRemove(
			Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
			double) const override
		{
			std::vector<size_t> result;
			std::vector<double> merits;
			for (size_t i = 0; i < tess.GetPointNo(); ++i)
			{
				if (cells[i].ID == first_ || cells[i].ID == second_)
				{
					result.push_back(i);
					// Equal merits intentionally exercise the global-ID MPI tie break.
					merits.push_back(1.0);
				}
			}
			return std::make_pair(result, merits);
		}

	private:
		size_t first_;
		size_t second_;
	};

	std::pair<size_t, size_t> FindAdjacentPair(
		Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
		int rank, int world_size)
	{
		const size_t invalid = std::numeric_limits<size_t>::max();
		size_t best_lo = invalid;
		size_t best_hi = invalid;
		std::vector<size_t> neighbors;
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
		{
			const Vector3D &point = tess.GetMeshPoint(i);
			if (point.x < 0.15 || point.x > 0.85 || point.y < 0.15 || point.y > 0.85 ||
				point.z < 0.15 || point.z > 0.85)
				continue;
			tess.GetNeighbors(i, neighbors);
			for (size_t neighbor : neighbors)
			{
				if (tess.IsPointOutsideBox(neighbor))
					continue;
#ifdef RICH_MPI
				if (world_size > 1 && tess.GetOwner(tess.GetMeshPoint(neighbor)) == rank)
					continue;
#else
				(void)rank;
				(void)world_size;
#endif
				const size_t lo = std::min(cells[i].ID, cells[neighbor].ID);
				const size_t hi = std::max(cells[i].ID, cells[neighbor].ID);
				if (lo < best_lo || (lo == best_lo && hi < best_hi))
				{
					best_lo = lo;
					best_hi = hi;
				}
			}
		}

#ifdef RICH_MPI
		std::array<unsigned long long, 2> local = {
			static_cast<unsigned long long>(best_lo), static_cast<unsigned long long>(best_hi)};
		std::vector<unsigned long long> gathered(static_cast<size_t>(world_size) * 2);
		MPI_Allgather(local.data(), 2, MPI_UNSIGNED_LONG_LONG, gathered.data(), 2,
			MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		best_lo = invalid;
		best_hi = invalid;
		for (int r = 0; r < world_size; ++r)
		{
			const size_t lo = static_cast<size_t>(gathered[static_cast<size_t>(2 * r)]);
			const size_t hi = static_cast<size_t>(gathered[static_cast<size_t>(2 * r + 1)]);
			if (lo < best_lo || (lo == best_lo && hi < best_hi))
			{
				best_lo = lo;
				best_hi = hi;
			}
		}
#endif

		if (best_lo == invalid || best_hi == invalid)
			throw UniversalError("Could not find an adjacent removal pair");
		return std::make_pair(best_lo, best_hi);
	}

	struct Totals
	{
		double mass = 0;
		double energy = 0;
		Vector3D momentum;
	};

	Totals ComputeTotals(Tessellation3D const& tess,
		std::vector<ComputationalCell3D> const& cells)
	{
		Totals result;
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
		{
			const double mass = cells[i].density * tess.GetVolume(i);
			result.mass += mass;
			result.momentum += mass * cells[i].velocity;
			result.energy += mass * (cells[i].internal_energy +
				0.5 * ScalarProd(cells[i].velocity, cells[i].velocity));
		}
		return result;
	}

	double RelativeDifference(double a, double b)
	{
		return std::abs(a - b) / std::max(std::abs(b), 1e-30);
	}
}

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
		const Vector3D ll(0, 0, 0);
		const Vector3D ur(1, 1, 1);
		std::vector<Vector3D> points;
		if (rank == 0)
			points = CartesianMesh(10, 10, 10, ll, ur);
#ifdef RICH_MPI
		points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

		Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
		tess.BuildParallel(points);
#else
		tess.Build(points);
#endif

		IdealGas eos(5.0 / 3.0);
		std::vector<ComputationalCell3D> initial_cells(tess.GetPointNo());
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
		{
			ComputationalCell3D &cell = initial_cells[i];
			const Vector3D &p = tess.GetMeshPoint(i);
			cell.density = 1.0 + 0.1 * p.x + 0.03 * p.y;
			cell.internal_energy = 2.0 + 0.05 * p.z;
			cell.velocity = Vector3D(0.02 * p.y, -0.01 * p.x, 0.015 * p.z);
			cell.pressure = eos.de2p(cell.density, cell.internal_energy,
				cell.tracers, ComputationalCell3D::tracerNames);
		}

		Simulation simulation(tess, initial_cells, eos);
		std::vector<ComputationalCell3D> &cells = simulation.getCells();
		const std::pair<size_t, size_t> pair = FindAdjacentPair(tess, cells, rank, world_size);

		std::unordered_map<size_t, double> old_volumes;
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
			old_volumes[cells[i].ID] = tess.GetVolume(i);
		size_t points_before_local = tess.GetPointNo();
		Totals before = ComputeTotals(tess, cells);

		NoRefine3D refine;
		RemoveIdPair3D remove(pair.first, pair.second);
		RigidWallGenerator3D ghost;
		LinearGauss3D interp(eos, ghost);
		AMR3D amr(eos, refine, remove, interp);
		amr(simulation);

		Totals after = ComputeTotals(tess, simulation.getCells());
		size_t points_after_local = tess.GetPointNo();
		double max_volume_growth = 0.0;
		int removed_ids_remaining = 0;
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
		{
			const size_t id = simulation.getCells()[i].ID;
			if (id == pair.first || id == pair.second)
				++removed_ids_remaining;
			auto found = old_volumes.find(id);
			if (found == old_volumes.end())
				throw UniversalError("Surviving cell ID missing from old-volume map");
			max_volume_growth = std::max(max_volume_growth, tess.GetVolume(i) / found->second);
		}

		size_t points_before = points_before_local;
		size_t points_after = points_after_local;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &before.mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &before.energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &before.momentum.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &after.mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &after.energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &after.momentum.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &points_before, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &points_after, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &removed_ids_remaining, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &max_volume_growth, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

		const double mass_error = RelativeDifference(after.mass, before.mass);
		const double energy_error = RelativeDifference(after.energy, before.energy);
		const double momentum_error = fastabs(after.momentum - before.momentum) /
			std::max(fastabs(before.momentum), 1e-30);
		const size_t removed_count = points_before - points_after;
		const double conservation_tolerance = world_size > 1 ? 1e-6 : 1e-9;
		const bool passed = removed_count == 2 && removed_ids_remaining == 0 &&
			mass_error <= conservation_tolerance && energy_error <= conservation_tolerance &&
			momentum_error <= conservation_tolerance && max_volume_growth <= 3.0 * (1.0 + 1e-8);

		if (rank == 0)
		{
			std::cout << "Adjacent pair IDs: " << pair.first << " " << pair.second << "\n";
			std::cout << "Removed count: " << removed_count << "\n";
			std::cout << "Mass relative error: " << mass_error << "\n";
			std::cout << "Energy relative error: " << energy_error << "\n";
			std::cout << "Momentum relative error: " << momentum_error << "\n";
			std::cout << "Maximum volume growth: " << max_volume_growth << "\n";
			std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

			std::ofstream metrics("amr_neighbor_remove_metrics.txt");
			metrics.setf(std::ios::scientific);
			metrics.precision(16);
			metrics << "mode " << (world_size > 1 ? "mpi" : "serial") << "\n";
			metrics << "removed_count " << removed_count << "\n";
			metrics << "removed_ids_remaining " << removed_ids_remaining << "\n";
			metrics << "mass_error " << mass_error << "\n";
			metrics << "energy_error " << energy_error << "\n";
			metrics << "momentum_error " << momentum_error << "\n";
			metrics << "max_volume_growth " << max_volume_growth << "\n";
			metrics << "volume_growth_limit 3.0\n";
			metrics << "pass " << (passed ? 1 : 0) << "\n";
		}

#ifdef RICH_MPI
		MPI_Finalize();
#endif
		return passed ? 0 : 1;
	}
	catch (UniversalError const& error)
	{
		reportError(error);
#ifdef RICH_MPI
		MPI_Abort(MPI_COMM_WORLD, 2);
#endif
		return 2;
	}
}
