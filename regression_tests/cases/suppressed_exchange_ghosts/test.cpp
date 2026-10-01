// Regression: distributed Voronoi builds with the point exchange suppressed.
//
// The individual-timestep hydro path never exchanges mesh points between MPI
// ranks (every BuildParallel/BuildPartiallyParallel call passes
// suppressRebalancing=true, suppressExchange=true).  As the moving mesh
// drifts, a rank's points leave the Hilbert ranges the load balancer assigned
// to it.  Ghost range queries must therefore be routed by where points really
// are, not by nominal Hilbert ownership; otherwise ranks tessellate boundary
// cells with an incomplete neighbour set and produce phantom faces.
//
// This case exchanges points once (nominal ownership), then applies a rigid
// periodic shift of several rank-domain widths to every owned point and
// rebuilds with the exchange suppressed.  Every owned cell volume must agree
// with an exchanged (normally routed) tessellation of the same generators.
// The oracle is built in parallel: a serial build inside an MPI binary would
// itself issue collectives on rank 0 alone.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include <mpi_utils/mpi_collectives.hpp>
#endif

namespace
{
struct CellComparison
{
    double max_volume_rel_error = 0;
    std::size_t volume_mismatches = 0;
    std::size_t face_count_mismatches = 0;
    double total_volume_rel_error = 0;
    std::size_t reference_cells = 0;
    std::size_t compared_cells = 0;
};

std::vector<double> gather_doubles(std::vector<double> const& local, int root)
{
#ifdef RICH_MPI
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int const count = static_cast<int>(local.size());
    std::vector<int> counts(rank == root ? size : 0);
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
    std::vector<int> displacements(rank == root ? size : 0, 0);
    std::vector<double> result;
    if(rank == root)
    {
        for(int i = 1; i < size; ++i)
            displacements[i] = displacements[i - 1] + counts[i - 1];
        result.resize(static_cast<std::size_t>(displacements[size - 1] + counts[size - 1]));
    }
    MPI_Gatherv(local.data(), count, MPI_DOUBLE, result.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, root, MPI_COMM_WORLD);
    return result;
#else
    (void)root;
    return local;
#endif
}

// One owned cell: generator position, volume, face count.
struct CellRecord
{
    double x, y, z, volume, faces;
};

bool by_position(CellRecord const& a, CellRecord const& b)
{
    if(a.x != b.x) return a.x < b.x;
    if(a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

// Gathers (position, volume, face count) of every owned cell to rank 0.
std::vector<CellRecord> gather_cells(Voronoi3D const& tess, int rank)
{
    std::size_t const n_owned = tess.GetPointNo();
    std::vector<double> flat;
    flat.reserve(5 * n_owned);
    std::vector<Vector3D> const& points = tess.getMeshPoints();
    for(std::size_t i = 0; i < n_owned; ++i)
    {
        flat.push_back(points[i].x);
        flat.push_back(points[i].y);
        flat.push_back(points[i].z);
        flat.push_back(tess.GetVolume(i));
        flat.push_back(static_cast<double>(tess.GetCellFaces(i).size()));
    }
    std::vector<double> const all = gather_doubles(flat, 0);
    std::vector<CellRecord> records;
    if(rank == 0)
    {
        if(all.size() % 5 != 0)
            throw UniversalError("suppressed_exchange_ghosts: gathered cell array is malformed");
        records.resize(all.size() / 5);
        for(std::size_t i = 0; i < records.size(); ++i)
            records[i] = CellRecord{all[5 * i], all[5 * i + 1], all[5 * i + 2], all[5 * i + 3], all[5 * i + 4]};
        std::sort(records.begin(), records.end(), by_position);
    }
    return records;
}

// Compares the owned cells of `candidate` against `reference`. Both hold the
// same global generator set (bit-identical coordinates) under different rank
// ownership, so cells are matched by sorted position. Collective.
CellComparison compare_tessellations(Voronoi3D const& candidate, Voronoi3D const& reference,
                                     Vector3D const& ll, Vector3D const& ur, int rank)
{
    std::vector<CellRecord> const cand = gather_cells(candidate, rank);
    std::vector<CellRecord> const ref = gather_cells(reference, rank);
    CellComparison result;
    if(rank == 0)
    {
        if(cand.size() != ref.size())
            throw UniversalError("suppressed_exchange_ghosts: candidate and reference cell counts differ");
        result.reference_cells = ref.size();
        double total_volume = 0;
        for(std::size_t g = 0; g < ref.size(); ++g)
        {
            if(cand[g].x != ref[g].x || cand[g].y != ref[g].y || cand[g].z != ref[g].z)
                throw UniversalError("suppressed_exchange_ghosts: generator sets differ between builds");
            double const rel = std::abs(cand[g].volume - ref[g].volume) / ref[g].volume;
            result.max_volume_rel_error = std::max(result.max_volume_rel_error, rel);
            if(rel > 1e-8)
                ++result.volume_mismatches;
            if(cand[g].faces != ref[g].faces)
                ++result.face_count_mismatches;
            total_volume += cand[g].volume;
            ++result.compared_cells;
        }
        double const box_volume = (ur.x - ll.x) * (ur.y - ll.y) * (ur.z - ll.z);
        result.total_volume_rel_error = std::abs(total_volume - box_volume) / box_volume;
    }
    return result;
}

double wrap_unit(double x)
{
    x = std::fmod(x, 1.0);
    if(x < 0)
        x += 1.0;
    // Keep strictly inside the box; generators on the boundary are invalid.
    return std::min(std::max(x, 1e-9), 1.0 - 1e-9);
}

std::vector<Vector3D> shifted_points(std::vector<Vector3D> const& points, Vector3D const& shift)
{
    std::vector<Vector3D> result(points.size());
    for(std::size_t i = 0; i < points.size(); ++i)
        result[i] = Vector3D(wrap_unit(points[i].x + shift.x),
                             wrap_unit(points[i].y + shift.y),
                             wrap_unit(points[i].z + shift.z));
    return result;
}

std::vector<Vector3D> owned_points(Voronoi3D const& tess)
{
    return std::vector<Vector3D>(tess.getMeshPoints().begin(),
                                 tess.getMeshPoints().begin() + tess.GetPointNo());
}

// Reference tessellation of the same global generator set with the exchange
// allowed. Points migrate to their nominal owner, so the Hilbert-range routing
// is exact for this build and it serves as the oracle.
void build_reference(Voronoi3D& reference, std::vector<Vector3D> const& points)
{
#ifdef RICH_MPI
    reference.BuildParallel(points);
#else
    reference.Build(points);
#endif
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
        Vector3D const ll(0.0, 0.0, 0.0);
        Vector3D const ur(1.0, 1.0, 1.0);
        std::size_t const Np = (world_size > 1)
            ? static_cast<std::size_t>(2e5)
            : static_cast<std::size_t>(2e4);
        // Several rank-domain widths for 64 ranks (domain width ~ 0.25).
        Vector3D const shift(0.23, -0.17, 0.31);

        std::vector<Vector3D> points;
        if(rank == 0)
            points = RandRectangular(Np, ll, ur);
#ifdef RICH_MPI
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

        // Stage 1: a normal build. Points migrate to their nominal Hilbert
        // owner, so this stage establishes the ownership the routing tree
        // describes. Only the box volume is checked here.
        Voronoi3D tess(ll, ur);
        build_reference(tess, points);
        std::vector<Vector3D> owned = owned_points(tess);
        CellComparison exchanged;
        {
            std::vector<CellRecord> const cells = gather_cells(tess, rank);
            if(rank == 0)
            {
                double total_volume = 0;
                for(CellRecord const& cell : cells)
                    total_volume += cell.volume;
                exchanged.compared_cells = cells.size();
                exchanged.reference_cells = cells.size();
                double const box_volume = (ur.x - ll.x) * (ur.y - ll.y) * (ur.z - ll.z);
                exchanged.total_volume_rel_error = std::abs(total_volume - box_volume) / box_volume;
            }
        }

        // Stage 2: drift every owned point out of its nominal Hilbert range and
        // rebuild without exchanging. This is the individual-timestep build mode.
        // The oracle is an exchanged build of the same generators.
        std::vector<Vector3D> drifted = shifted_points(owned, shift);
#ifdef RICH_MPI
        tess.BuildParallel(drifted, true /* no rebalance */, true /* no exchange */);
#else
        tess.Build(drifted);
#endif
        if(tess.GetPointNo() != drifted.size())
            throw UniversalError("suppressed_exchange_ghosts: suppressed build changed the owned point count");
        Voronoi3D reference(ll, ur);
        build_reference(reference, drifted);
        CellComparison const suppressed = compare_tessellations(tess, reference, ll, ur, rank);

        // Stage 3: a second suppressed build with a further drift exercises the
        // refresh of the routing tree on an already switched agent.
        drifted = shifted_points(drifted, Vector3D(-shift.y, shift.z, -shift.x));
#ifdef RICH_MPI
        tess.BuildParallel(drifted, true, true);
#else
        tess.Build(drifted);
#endif
        build_reference(reference, drifted);
        CellComparison const suppressed_again = compare_tessellations(tess, reference, ll, ur, rank);

        int passed = 0;
        if(rank == 0)
        {
            bool const ok =
                exchanged.compared_cells == Np &&
                suppressed.compared_cells == Np &&
                suppressed_again.compared_cells == Np &&
                exchanged.total_volume_rel_error < 1e-10 &&
                suppressed.volume_mismatches == 0 &&
                suppressed_again.volume_mismatches == 0 &&
                suppressed.total_volume_rel_error < 1e-10 &&
                suppressed_again.total_volume_rel_error < 1e-10;
            passed = ok ? 1 : 0;
            std::cout << "ranks = " << world_size << "\n"
                      << "num_points = " << Np << "\n"
                      << "exchanged: total_volume_rel_error = " << exchanged.total_volume_rel_error << "\n"
                      << "suppressed: max_volume_rel_error = " << suppressed.max_volume_rel_error
                      << " volume_mismatches = " << suppressed.volume_mismatches
                      << " face_count_mismatches = " << suppressed.face_count_mismatches << "\n"
                      << "suppressed_again: max_volume_rel_error = " << suppressed_again.max_volume_rel_error
                      << " volume_mismatches = " << suppressed_again.volume_mismatches
                      << " face_count_mismatches = " << suppressed_again.face_count_mismatches << "\n"
                      << "pass = " << passed << std::endl;

            std::ofstream out("suppressed_exchange_ghosts_metrics.txt");
            out.setf(std::ios::scientific);
            out.precision(16);
            out << "mode " << (world_size > 1 ? "mpi" : "serial") << "\n";
            out << "ranks " << world_size << "\n";
            out << "num_points " << Np << "\n";
            out << "shift " << shift.x << " " << shift.y << " " << shift.z << "\n";
            out << "exchanged_total_volume_rel_error " << exchanged.total_volume_rel_error << "\n";
            out << "exchanged_volume_mismatches " << exchanged.volume_mismatches << "\n";
            out << "suppressed_max_volume_rel_error " << suppressed.max_volume_rel_error << "\n";
            out << "suppressed_volume_mismatches " << suppressed.volume_mismatches << "\n";
            out << "suppressed_face_count_mismatches " << suppressed.face_count_mismatches << "\n";
            out << "suppressed_total_volume_rel_error " << suppressed.total_volume_rel_error << "\n";
            out << "suppressed_again_max_volume_rel_error " << suppressed_again.max_volume_rel_error << "\n";
            out << "suppressed_again_volume_mismatches " << suppressed_again.volume_mismatches << "\n";
            out << "suppressed_again_face_count_mismatches " << suppressed_again.face_count_mismatches << "\n";
            out << "suppressed_again_total_volume_rel_error " << suppressed_again.total_volume_rel_error << "\n";
            out << "pass " << passed << "\n";
        }
#ifdef RICH_MPI
        MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
#endif
        return passed ? 0 : 1;
    }
    catch(UniversalError const& e)
    {
        reportError(e);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        return 2;
    }
}
