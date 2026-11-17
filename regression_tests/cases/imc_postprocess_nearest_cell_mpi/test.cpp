// Regression check for NearestLocalCells (source/3D/radiation/postprocess/
// NearestCellOwner.hpp), which hands a point to the rank owning the cell that
// contains it. The post-process probes start their rays with it and the FLD
// quadrature samples the flux with it; before it, both used the owner of the
// point's domain region and that rank's nearest local generator, which is the
// wrong cell near rank boundaries (a probe ray then failed with "no face
// intersection found" and hung the job).
//
// On a 4-rank mesh of a fixed random generator set (dense core, sparse
// envelope, like a post-process snapshot):
//   - every point inside the box is taken by exactly one rank;
//   - the taken cell's generator is the nearest of all generators (brute
//     force over the global set);
//   - points outside the box are taken by nobody by default and by exactly
//     one rank with includeOutsideBox;
//   - the old lookup's misses are reported (not a pass condition: how many
//     there are depends on the decomposition, not on NearestLocalCells).

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/3D/radiation/postprocess/NearestCellOwner.hpp"
#include "source/misc/universal_error.hpp"

namespace
{

constexpr std::size_t kGenerators = 6000;
constexpr std::size_t kInsideQueries = 20000;
constexpr std::size_t kSphereQueries = 4096;
constexpr std::size_t kOutsideQueries = 500;
constexpr double kSphereRadius = 0.7;

double NearestGlobalDistance(std::vector<Vector3D> const& generators, Vector3D const& point)
{
    double best = std::numeric_limits<double>::infinity();
    for (Vector3D const& g : generators)
        best = std::min(best, fastabs(g - point));
    return best;
}

bool SameDistance(double a, double b)
{
    return std::abs(a - b) <= 1e-12 * std::max(1.0, std::max(a, b));
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exitCode = 0;
    try {
        Vector3D const ll(-1.0, -1.0, -1.0);
        Vector3D const ur(1.0, 1.0, 1.0);

        // The same generator set on every rank; each rank contributes every
        // size-th point and BuildParallel repartitions them.
        std::mt19937_64 generatorRng(20261117);
        std::uniform_real_distribution<double> unit(-1.0, 1.0);
        std::vector<Vector3D> generators(kGenerators);
        for (std::size_t i = 0; i < kGenerators; ++i) {
            Vector3D p(unit(generatorRng), unit(generatorRng), unit(generatorRng));
            if (i % 2 == 0)
                p = p * 0.3;
            generators[i] = p * 0.999;
        }
        std::vector<Vector3D> localGenerators;
        for (std::size_t i = static_cast<std::size_t>(rank); i < kGenerators;
             i += static_cast<std::size_t>(size))
            localGenerators.push_back(generators[i]);

        Voronoi3D tess(ll, ur);
        tess.BuildParallel(localGenerators);
        std::size_t const nCells = tess.GetPointNo();

        // Queries: uniform in the box, on a sphere (like the probe start
        // points), and outside the box.
        std::mt19937_64 queryRng(777);
        std::vector<Vector3D> queries;
        std::vector<char> outside;
        for (std::size_t i = 0; i < kInsideQueries; ++i) {
            queries.emplace_back(unit(queryRng) * 0.999, unit(queryRng) * 0.999, unit(queryRng) * 0.999);
            outside.push_back(0);
        }
        double const golden = std::acos(-1.0) * (3.0 - std::sqrt(5.0));
        for (std::size_t i = 0; i < kSphereQueries; ++i) {
            double const z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(kSphereQueries);
            double const rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
            double const phi = golden * static_cast<double>(i);
            queries.emplace_back(kSphereRadius * rxy * std::cos(phi), kSphereRadius * rxy * std::sin(phi),
                                 kSphereRadius * z);
            outside.push_back(0);
        }
        for (std::size_t i = 0; i < kOutsideQueries; ++i) {
            Vector3D p(unit(queryRng), unit(queryRng), unit(queryRng));
            double const scale = 1.2 / std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z), 1e-3});
            queries.push_back(p * scale);
            outside.push_back(1);
        }
        std::size_t const nQueries = queries.size();
        for (std::size_t i = 0; i < nQueries; ++i)
            if ((tess.IsPointOutsideBox(queries[i]) ? 1 : 0) != outside[i])
                throw UniversalError("nearest-cell test generated a query on the wrong side of the box");

        std::vector<std::size_t> const cells =
            imc_postprocess_tde::NearestLocalCells(tess, queries, rank);
        std::vector<std::size_t> const cellsAll =
            imc_postprocess_tde::NearestLocalCells(tess, queries, rank, true);

        std::vector<int> taken(nQueries, 0);
        std::vector<int> takenAll(nQueries, 0);
        unsigned long long nearestMismatch = 0;
        unsigned long long oldWrong = 0;
        for (std::size_t i = 0; i < nQueries; ++i) {
            if (cells[i] < nCells)
                taken[i] = 1;
            if (cellsAll[i] < nCells)
                takenAll[i] = 1;
            double const best = (cellsAll[i] < nCells || (!outside[i] && tess.GetOwner(queries[i]) == rank))
                ? NearestGlobalDistance(generators, queries[i]) : 0.0;
            if (cellsAll[i] < nCells &&
                !SameDistance(fastabs(tess.GetMeshPoint(cellsAll[i]) - queries[i]), best))
                ++nearestMismatch;
            if (cells[i] < nCells && cells[i] != cellsAll[i])
                ++nearestMismatch;
            // The old lookup: the owner of the point's region, nearest local generator.
            if (!outside[i] && nCells > 0 && tess.GetOwner(queries[i]) == rank) {
                std::size_t const oldCell = tess.GetContainingCell(queries[i]);
                if (!SameDistance(fastabs(tess.GetMeshPoint(oldCell) - queries[i]), best))
                    ++oldWrong;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, taken.data(), static_cast<int>(nQueries), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, takenAll.data(), static_cast<int>(nQueries), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &nearestMismatch, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &oldWrong, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        unsigned long long localCells = nCells;
        unsigned long long minCells = 0;
        MPI_Allreduce(&localCells, &minCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

        unsigned long long insideNotOnce = 0;
        unsigned long long outsideTaken = 0;
        unsigned long long allNotOnce = 0;
        for (std::size_t i = 0; i < nQueries; ++i) {
            if (!outside[i] && taken[i] != 1)
                ++insideNotOnce;
            if (outside[i] && taken[i] != 0)
                ++outsideTaken;
            if (takenAll[i] != 1)
                ++allNotOnce;
        }

        bool const pass = size > 1 && minCells > 0 && insideNotOnce == 0 && outsideTaken == 0 &&
                          allNotOnce == 0 && nearestMismatch == 0;
        if (rank == 0) {
            auto report = [](bool ok, char const* what) {
                std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
            };
            report(size > 1, "runs on more than one rank");
            report(minCells > 0, "every rank owns cells");
            report(insideNotOnce == 0, "every point inside the box is taken by exactly one rank");
            report(outsideTaken == 0, "points outside the box are not taken by default");
            report(allNotOnce == 0, "with includeOutsideBox every point is taken by exactly one rank");
            report(nearestMismatch == 0, "the taken cell's generator is the global nearest");
            std::cout << "INFO the old owner-region lookup misses " << oldWrong
                      << " points on this decomposition" << std::endl;

            std::ofstream metrics("imc_postprocess_nearest_cell_mpi_metrics.txt");
            metrics << std::setprecision(17)
                    << "mpi_size " << size << "\n"
                    << "queries " << nQueries << "\n"
                    << "min_rank_cells " << minCells << "\n"
                    << "inside_not_taken_once " << insideNotOnce << "\n"
                    << "outside_taken " << outsideTaken << "\n"
                    << "include_outside_not_taken_once " << allNotOnce << "\n"
                    << "nearest_mismatch " << nearestMismatch << "\n"
                    << "old_lookup_wrong " << oldWrong << "\n"
                    << "pass " << (pass ? 1 : 0) << "\n";
            std::cout << (pass ? "imc_postprocess_nearest_cell_mpi PASSED" : "imc_postprocess_nearest_cell_mpi FAILED")
                      << std::endl;
        }
        exitCode = pass ? 0 : 1;
    } catch (UniversalError const& eo) {
        std::cerr << "UniversalError on rank " << rank << ": " << eo.getErrorMessage() << std::endl;
        exitCode = 2;
        MPI_Abort(MPI_COMM_WORLD, exitCode);
    } catch (std::exception const& e) {
        std::cerr << "Exception on rank " << rank << ": " << e.what() << std::endl;
        exitCode = 3;
        MPI_Abort(MPI_COMM_WORLD, exitCode);
    }

    MPI_Finalize();
    return exitCode;
}
