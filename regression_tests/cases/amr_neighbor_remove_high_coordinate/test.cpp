#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/tessellation/utils/PolyClip.hpp"
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
		RemoveIdPair3D(size_t First, size_t Second): First_(First), Second_(Second) {}

		std::pair<std::vector<size_t>, std::vector<double> > ToRemove(
			Tessellation3D const& Tess, std::vector<ComputationalCell3D> const& Cells, double) const override
		{
			std::vector<size_t> Result;
			std::vector<double> Merits;
			for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
			{
				if (Cells[Index].ID == First_ || Cells[Index].ID == Second_)
				{
					Result.push_back(Index);
					Merits.push_back(1.0);
				}
			}
			return std::make_pair(Result, Merits);
		}

	private:
		size_t First_;
		size_t Second_;
	};

	std::pair<size_t, size_t> FindSafeCrossRankPair(Tessellation3D const& Tess,
		std::vector<ComputationalCell3D> const& Cells, Vector3D const& Lower, Vector3D const& Upper,
		int Rank, int WorldSize)
	{
#ifndef RICH_MPI
		(void)Tess;
		(void)Cells;
		(void)Lower;
		(void)Upper;
		(void)Rank;
		(void)WorldSize;
		throw UniversalError("High-coordinate neighboring-removal regression requires MPI");
#else
		size_t PointCount = Tess.GetPointNo();
		MPI_Allreduce(MPI_IN_PLACE, &PointCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		std::vector<double> Volumes(PointCount, 0.0);
		std::vector<int> Owners(PointCount, -1);
		std::vector<int> Eligible(PointCount, 0);
		std::vector<unsigned long long> LocalEdges;
		std::vector<size_t> Neighbors;
		const Vector3D Width = Upper - Lower;
		for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
		{
			const size_t Id = Cells[Index].ID;
			Volumes[Id] = Tess.GetVolume(Index);
			Owners[Id] = Rank;
			const Vector3D Relative = Tess.GetMeshPoint(Index) - Lower;
			Eligible[Id] = Relative.x > 0.2 * Width.x && Relative.x < 0.8 * Width.x &&
				Relative.y > 0.2 * Width.y && Relative.y < 0.8 * Width.y &&
				Relative.z > 0.2 * Width.z && Relative.z < 0.8 * Width.z;
			Tess.GetNeighbors(Index, Neighbors);
			for (size_t Neighbor : Neighbors)
			{
				if (Tess.IsPointOutsideBox(Neighbor))
					continue;
				const size_t NeighborId = Cells[Neighbor].ID;
				const size_t First = std::min(Id, NeighborId);
				const size_t Second = std::max(Id, NeighborId);
				if (First == Id)
				{
					LocalEdges.push_back(static_cast<unsigned long long>(First));
					LocalEdges.push_back(static_cast<unsigned long long>(Second));
				}
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, Volumes.data(), static_cast<int>(Volumes.size()), MPI_DOUBLE,
			MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, Owners.data(), static_cast<int>(Owners.size()), MPI_INT,
			MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, Eligible.data(), static_cast<int>(Eligible.size()), MPI_INT,
			MPI_MAX, MPI_COMM_WORLD);

		const int LocalEdgeValueCount = static_cast<int>(LocalEdges.size());
		std::vector<int> Counts(static_cast<size_t>(WorldSize));
		MPI_Allgather(&LocalEdgeValueCount, 1, MPI_INT, Counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
		std::vector<int> Displacements(static_cast<size_t>(WorldSize), 0);
		for (int OtherRank = 1; OtherRank < WorldSize; ++OtherRank)
			Displacements[static_cast<size_t>(OtherRank)] = Displacements[static_cast<size_t>(OtherRank - 1)] +
				Counts[static_cast<size_t>(OtherRank - 1)];
		const int EdgeValueCount = Displacements.back() + Counts.back();
		std::vector<unsigned long long> EdgeValues(static_cast<size_t>(EdgeValueCount));
		MPI_Allgatherv(LocalEdges.data(), LocalEdgeValueCount, MPI_UNSIGNED_LONG_LONG, EdgeValues.data(),
			Counts.data(), Displacements.data(), MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		std::vector<std::vector<size_t> > Graph(PointCount);
		for (size_t Offset = 0; Offset < EdgeValues.size(); Offset += 2)
		{
			const size_t First = static_cast<size_t>(EdgeValues[Offset]);
			const size_t Second = static_cast<size_t>(EdgeValues[Offset + 1]);
			Graph[First].push_back(Second);
			Graph[Second].push_back(First);
		}
		for (std::vector<size_t> &Adjacent : Graph)
		{
			std::sort(Adjacent.begin(), Adjacent.end());
			Adjacent.erase(std::unique(Adjacent.begin(), Adjacent.end()), Adjacent.end());
		}

		auto K2Flags = [&Graph, PointCount](size_t Source)
		{
			std::vector<char> Flags(PointCount, 0);
			Flags[Source] = 1;
			for (size_t FirstNeighbor : Graph[Source])
			{
				Flags[FirstNeighbor] = 1;
				for (size_t SecondNeighbor : Graph[FirstNeighbor])
					Flags[SecondNeighbor] = 1;
			}
			return Flags;
		};

		for (size_t First = 0; First < PointCount; ++First)
		{
			if (!Eligible[First])
				continue;
			for (size_t Second : Graph[First])
			{
				if (Second <= First || !Eligible[Second] || Owners[First] == Owners[Second])
					continue;
				const std::vector<char> FirstTargets = K2Flags(First);
				const std::vector<char> SecondTargets = K2Flags(Second);
				bool FitsGrowthLimit = true;
				for (size_t Target = 0; Target < PointCount; ++Target)
				{
					if (Target == First || Target == Second)
						continue;
					double ClaimedVolume = FirstTargets[Target] ? Volumes[First] : 0.0;
					if (SecondTargets[Target])
						ClaimedVolume += Volumes[Second];
					if (ClaimedVolume > 2.0 * Volumes[Target] * (1.0 + 1e-12))
					{
						FitsGrowthLimit = false;
						break;
					}
				}
				if (FitsGrowthLimit)
					return std::make_pair(First, Second);
			}
		}
		throw UniversalError("Could not find an adjacent cross-rank pair satisfying the AMR volume limit");
#endif
	}

	struct Totals
	{
		double Mass = 0;
		double Energy = 0;
		Vector3D Momentum;
	};

	Totals ComputeTotals(Tessellation3D const& Tess, std::vector<ComputationalCell3D> const& Cells)
	{
		Totals Result;
		for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
		{
			const double Mass = Cells[Index].density * Tess.GetVolume(Index);
			Result.Mass += Mass;
			Result.Momentum += Mass * Cells[Index].velocity;
			Result.Energy += Mass * (Cells[Index].internal_energy +
				0.5 * ScalarProd(Cells[Index].velocity, Cells[Index].velocity));
		}
		return Result;
	}

	double RelativeDifference(double First, double Second)
	{
		return std::abs(First - Second) / std::max(std::abs(Second), 1e-30);
	}
}

int main()
{
	int Rank = 0;
	int WorldSize = 1;
#ifdef RICH_MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
	MPI_Comm_size(MPI_COMM_WORLD, &WorldSize);
#endif

	try
	{
		// The production failure occurred at |x| ~ 3e3 with sub-unit cells. A
		// translated, jittered mesh gives the same poorly conditioned oblique planes
		// without exceeding the coordinate range supported by the tessellation builder.
		const double Origin = 3000.0;
		const Vector3D Lower(Origin, Origin, Origin);
		const Vector3D Upper(Origin + 1.0, Origin + 1.0, Origin + 1.0);
		Face UnorderedFace;
		UnorderedFace.vertices = {
			Vector3D(Origin + 0.1, Origin, Origin), Vector3D(Origin, Origin, Origin),
			Vector3D(Origin, Origin + 0.1, Origin), Vector3D(Origin - 0.1, Origin, Origin),
			Vector3D(Origin, Origin - 0.1, Origin)};
		const Face OrderedFace = ConvexHullFace(UnorderedFace);
		const double HullAreaError = std::abs(polygonArea(OrderedFace) - 0.02) / 0.02;
		const bool HullPassed = OrderedFace.vertices.size() == 4 && HullAreaError <= 1e-10;
		std::vector<Vector3D> Points;
		if (Rank == 0)
		{
			Points = CartesianMesh(12, 12, 12, Lower, Upper);
			const double Jitter = 0.012;
			for (size_t Index = 0; Index < Points.size(); ++Index)
			{
				Points[Index].x += Jitter * std::sin(0.73 * static_cast<double>(Index) + 0.2);
				Points[Index].y += Jitter * std::sin(1.13 * static_cast<double>(Index) + 0.7);
				Points[Index].z += Jitter * std::sin(1.57 * static_cast<double>(Index) + 1.1);
			}
		}
#ifdef RICH_MPI
		Points = MPI_Spread(Points, 0, MPI_COMM_WORLD);
#endif

		Voronoi3D Tess(Lower, Upper);
#ifdef RICH_MPI
		Tess.BuildParallel(Points);
#else
		Tess.Build(Points);
#endif

		IdealGas Eos(5.0 / 3.0);
		std::vector<ComputationalCell3D> InitialCells(Tess.GetPointNo());
		for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
		{
			ComputationalCell3D &Cell = InitialCells[Index];
			const Vector3D RelativePoint = Tess.GetMeshPoint(Index) - Lower;
			Cell.density = 1.0 + 0.1 * RelativePoint.x + 0.03 * RelativePoint.y;
			Cell.internal_energy = 2.0 + 0.05 * RelativePoint.z;
			Cell.velocity = Vector3D(0.02 * RelativePoint.y, -0.01 * RelativePoint.x,
				0.015 * RelativePoint.z);
			Cell.pressure = Eos.de2p(Cell.density, Cell.internal_energy,
				Cell.tracers, ComputationalCell3D::tracerNames);
		}

		Simulation Sim(Tess, InitialCells, Eos);
		std::vector<ComputationalCell3D> &Cells = Sim.getCells();
		const std::pair<size_t, size_t> RemovalPair =
			FindSafeCrossRankPair(Tess, Cells, Lower, Upper, Rank, WorldSize);
		std::vector<size_t> InitialIds;
		std::vector<std::pair<size_t, size_t> > InitialNeighborPairs;
		std::unordered_map<size_t, double> OldVolumes;
		std::vector<size_t> Neighbors;
		for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
		{
			InitialIds.push_back(Cells[Index].ID);
			OldVolumes[Cells[Index].ID] = Tess.GetVolume(Index);
			Tess.GetNeighbors(Index, Neighbors);
			for (size_t Neighbor : Neighbors)
			{
				if (Tess.IsPointOutsideBox(Neighbor))
					continue;
				const size_t First = std::min(Cells[Index].ID, Cells[Neighbor].ID);
				const size_t Second = std::max(Cells[Index].ID, Cells[Neighbor].ID);
				if (First == Cells[Index].ID)
					InitialNeighborPairs.emplace_back(First, Second);
			}
		}

		size_t PointsBefore = Tess.GetPointNo();
		Totals Before = ComputeTotals(Tess, Cells);
		NoRefine3D Refine;
		RemoveIdPair3D Remove(RemovalPair.first, RemovalPair.second);
		RigidWallGenerator3D Ghost;
		LinearGauss3D Interp(Eos, Ghost);
		AMR3D Amr(Eos, Refine, Remove, Interp);
		Amr(Sim);

		Totals After = ComputeTotals(Tess, Sim.getCells());
		size_t PointsAfter = Tess.GetPointNo();
		double MaxVolumeGrowth = 0.0;
		for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
		{
			const size_t Id = Sim.getCells()[Index].ID;
			auto Found = OldVolumes.find(Id);
			if (Found == OldVolumes.end())
				throw UniversalError("Surviving cell ID missing from old-volume map");
			MaxVolumeGrowth = std::max(MaxVolumeGrowth, Tess.GetVolume(Index) / Found->second);
		}

		size_t GlobalPointCount = PointsBefore;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &GlobalPointCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
		std::vector<int> RemovedById(GlobalPointCount, 0);
		for (size_t Id : InitialIds)
			RemovedById[Id] = 1;
		for (size_t Index = 0; Index < Tess.GetPointNo(); ++Index)
			RemovedById[Sim.getCells()[Index].ID] = 0;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, RemovedById.data(), static_cast<int>(RemovedById.size()),
			MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
		size_t AdjacentRemovedPairs = 0;
		for (const std::pair<size_t, size_t> &Pair : InitialNeighborPairs)
			if (RemovedById[Pair.first] && RemovedById[Pair.second])
				++AdjacentRemovedPairs;

#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &Before.Mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &Before.Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &Before.Momentum.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &After.Mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &After.Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &After.Momentum.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &PointsBefore, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &PointsAfter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &AdjacentRemovedPairs, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &MaxVolumeGrowth, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

		const double MassError = RelativeDifference(After.Mass, Before.Mass);
		const double EnergyError = RelativeDifference(After.Energy, Before.Energy);
		const double MomentumError = fastabs(After.Momentum - Before.Momentum) /
			std::max(fastabs(Before.Momentum), 1e-30);
		const size_t RemovedCount = PointsBefore - PointsAfter;
		const double Tolerance = 1e-6;
		const bool Passed = HullPassed && RemovedCount >= 2 && AdjacentRemovedPairs >= 1 &&
			MassError <= Tolerance && EnergyError <= Tolerance && MomentumError <= Tolerance &&
			MaxVolumeGrowth <= 3.0 * (1.0 + 1e-8);

		if (Rank == 0)
		{
			std::cout << "Convex face hull vertices: " << OrderedFace.vertices.size() << "\n";
			std::cout << "Convex face hull area error: " << HullAreaError << "\n";
			std::cout << "Adjacent pair IDs: " << RemovalPair.first << " " << RemovalPair.second << "\n";
			std::cout << "Removed count: " << RemovedCount << "\n";
			std::cout << "Adjacent removed pairs: " << AdjacentRemovedPairs << "\n";
			std::cout << "Mass relative error: " << MassError << "\n";
			std::cout << "Energy relative error: " << EnergyError << "\n";
			std::cout << "Momentum relative error: " << MomentumError << "\n";
			std::cout << "Maximum volume growth: " << MaxVolumeGrowth << "\n";
			std::cout << (Passed ? "PASSED" : "FAILED") << std::endl;

			std::ofstream Metrics("amr_neighbor_remove_high_coordinate_metrics.txt");
			Metrics.setf(std::ios::scientific);
			Metrics.precision(16);
			Metrics << "convex_hull_pass " << (HullPassed ? 1 : 0) << "\n";
			Metrics << "convex_hull_area_error " << HullAreaError << "\n";
			Metrics << "removed_count " << RemovedCount << "\n";
			Metrics << "adjacent_removed_pairs " << AdjacentRemovedPairs << "\n";
			Metrics << "mass_error " << MassError << "\n";
			Metrics << "energy_error " << EnergyError << "\n";
			Metrics << "momentum_error " << MomentumError << "\n";
			Metrics << "max_volume_growth " << MaxVolumeGrowth << "\n";
			Metrics << "volume_growth_limit 3.0\n";
			Metrics << "coordinate_origin " << Origin << "\n";
			Metrics << "pass " << (Passed ? 1 : 0) << "\n";
		}

#ifdef RICH_MPI
		MPI_Finalize();
#endif
		return Passed ? 0 : 1;
	}
	catch (UniversalError const& Error)
	{
		reportError(Error);
#ifdef RICH_MPI
		MPI_Abort(MPI_COMM_WORLD, 2);
#endif
		return 2;
	}
}
