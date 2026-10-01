// Regression: segmented Hilbert ownership (several curve ranges per rank).
//
// Individual time steps spread clustered active cells by letting one rank own
// several disjoint Hilbert-curve segments (CurveLoadBalancer::segmentOwner,
// HilbertLoadBalancer::setSegments / rebalanceInterleaved).  Every consumer of
// ownership must follow the segment owners: the point exchange (getOwner), the
// ghost routing tree (HilbertRectangularTree3D leaf owners), box changes (the
// (boundary, owner) pairs sort together) and restart IO.
//
// Stages, each compared cell for cell (by generator position) with a
// positional reference tessellation of the same generators:
//   1. interleaved segments (4 per rank, piece i to rank i mod P);
//   2. scrambled owners over 3 pieces per rank;
//   3. the scrambled balancer after a box change;
// plus: every owned point belongs to its rank, the rank tree routes every owned
// point to its rank, and positional and segmented balancers survive an HDF5
// round trip.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include <mpi_utils/mpi_collectives.hpp>
#include <MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp>
#include <MeshDecomposer3D/hilbert/rectangular/HilbertRectangularTree3D.hpp>
#include "source/3D/tessellation/io/load_balancing/LoadBalancerIOHandlerFactory.hpp"
#include "source/utils/hdf5/HDF5Writer.hpp"
#include "source/utils/hdf5/HDF5Reader.hpp"
#endif

namespace
{
struct CellComparison
{
    double max_volume_rel_error = 0;
    std::size_t volume_mismatches = 0;
    std::size_t face_count_mismatches = 0;
    std::size_t compared_cells = 0;
};

#ifdef RICH_MPI
std::vector<double> gather_doubles(std::vector<double> const& local, int root)
{
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
}

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
        records.resize(all.size() / 5);
        for(std::size_t i = 0; i < records.size(); ++i)
            records[i] = CellRecord{all[5 * i], all[5 * i + 1], all[5 * i + 2], all[5 * i + 3], all[5 * i + 4]};
        std::sort(records.begin(), records.end(), by_position);
    }
    return records;
}

// Collective; the verdict is meaningful on rank 0.
CellComparison compare_tessellations(Voronoi3D const& candidate, std::vector<CellRecord> const& ref, int rank)
{
    std::vector<CellRecord> const cand = gather_cells(candidate, rank);
    CellComparison result;
    if(rank == 0)
    {
        if(cand.size() != ref.size())
            throw UniversalError("segmented_hilbert_ownership: candidate and reference cell counts differ");
        for(std::size_t g = 0; g < ref.size(); ++g)
        {
            if(cand[g].x != ref[g].x || cand[g].y != ref[g].y || cand[g].z != ref[g].z)
                throw UniversalError("segmented_hilbert_ownership: generator sets differ between builds");
            double const rel = std::abs(cand[g].volume - ref[g].volume) / ref[g].volume;
            result.max_volume_rel_error = std::max(result.max_volume_rel_error, rel);
            if(rel > 1e-8)
                ++result.volume_mismatches;
            if(cand[g].faces != ref[g].faces)
                ++result.face_count_mismatches;
            ++result.compared_cells;
        }
    }
    return result;
}

std::vector<Vector3D> owned_points(Voronoi3D const& tess)
{
    return std::vector<Vector3D>(tess.getMeshPoints().begin(),
                                 tess.getMeshPoints().begin() + tess.GetPointNo());
}

// Owned points that the balancer assigns elsewhere, ranks without points, and
// owned points the rank tree does not route to this rank (all global sums).
struct OwnershipCheck
{
    unsigned long long foreign_points = 0;
    unsigned long long empty_ranks = 0;
    unsigned long long unrouted_points = 0;
};

OwnershipCheck check_ownership(Voronoi3D const& tess, HilbertLoadBalancer<Vector3D> const& lb, int rank)
{
    std::vector<Vector3D> const points = owned_points(tess);
    auto const convertor = std::dynamic_pointer_cast<HilbertRectangularConvertor3D<Vector3D>>(lb.getConvertor());
    if(!convertor)
        throw UniversalError("segmented_hilbert_ownership: balancer has no rectangular convertor");
    HilbertRectangularTree3D<Vector3D> tree(convertor, lb.getBoundaries(), MPI_COMM_WORLD, lb.getSegmentOwner());
    unsigned long long local[3] = {0, points.empty() ? 1ull : 0ull, 0};
    for(Vector3D const& point : points)
    {
        if(lb.getOwner(point) != rank)
            ++local[0];
        auto const ranks = tree.getIntersectingRanks(point, 1e-9);
        if(std::find(ranks.begin(), ranks.end(), rank) == ranks.end())
            ++local[2];
    }
    unsigned long long global[3] = {0, 0, 0};
    MPI_Allreduce(local, global, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return OwnershipCheck{global[0], global[1], global[2]};
}

// Writes the balancer to a per-rank HDF5 file, reads it back and compares.
// Returns the number of ranks whose round trip differed.  Collective.
unsigned long long round_trip(HilbertLoadBalancer<Vector3D> const& lb, std::vector<Vector3D> const& points,
                              int rank, std::string const& tag)
{
    std::string const file = "segmented_lb_" + tag + "_" + std::to_string(rank) + ".h5";
    {
        HDF5Writer writer(file);
        LoadBalancerIO::writeLoadBalancer(writer, "/lb", lb);
        writer.Dump();
        writer.Close();
    }
    HDF5Reader reader(file);
    std::shared_ptr<LoadBalancer<Vector3D>> const loaded = LoadBalancerIO::readLoadBalancer(reader, "/lb");
    auto const hilbert = std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(loaded);
    unsigned long long bad = 0;
    if(!hilbert || hilbert->getBoundaries() != lb.getBoundaries() ||
       hilbert->getSegmentOwner() != lb.getSegmentOwner())
        bad = 1;
    else
        for(Vector3D const& point : points)
            if(hilbert->getOwner(point) != lb.getOwner(point))
            {
                bad = 1;
                break;
            }
    std::remove(file.c_str());
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return bad;
}
#endif
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

    int passed = 0;
    try
    {
#ifdef RICH_MPI
        Vector3D const ll(0.0, 0.0, 0.0);
        Vector3D const ur(1.0, 1.0, 1.0);
        std::size_t const Np = static_cast<std::size_t>(2e5);
        std::vector<Vector3D> points;
        if(rank == 0)
            points = RandRectangular(Np, ll, ur);
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);

        // Positional reference.
        Voronoi3D reference(ll, ur);
        reference.BuildParallel(points);
        std::vector<CellRecord> const reference_cells = gather_cells(reference, rank);

        Voronoi3D tess(ll, ur);
        tess.BuildParallel(points);
        auto const positional = std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(tess.GetLoadBalancer());
        if(!positional || !positional->positionalOwnership())
            throw UniversalError("segmented_hilbert_ownership: expected a positional Hilbert balancer");
        OwnershipCheck const positional_ownership = check_ownership(tess, *positional, rank);
        unsigned long long const positional_io = round_trip(*positional, owned_points(tess), rank, "positional");

        // Stage 1: interleaved segments.
        std::shared_ptr<HilbertLoadBalancer<Vector3D>> interleaved = positional->clone();
        interleaved->rebalanceInterleaved(owned_points(tess), std::vector<double>(), 4);
        tess.SetLoadBalancer(interleaved);
        tess.BuildParallel(owned_points(tess), true /* keep the partition */, false);
        CellComparison const interleaved_cells = compare_tessellations(tess, reference_cells, rank);
        OwnershipCheck const interleaved_ownership = check_ownership(tess, *interleaved, rank);
        unsigned long long const interleaved_io = round_trip(*interleaved, owned_points(tess), rank, "interleaved");

        // Stage 2: scrambled owners over 3 pieces per rank (a stride coprime
        // with the rank count keeps every rank non-empty).
        std::shared_ptr<HilbertLoadBalancer<Vector3D>> scrambled = positional->clone();
        scrambled->rebalanceInterleaved(owned_points(tess), std::vector<double>(), 3);
        std::vector<int> owners(scrambled->getSegmentOwner().size());
        int stride = 7;
        while(std::gcd(stride, world_size) != 1)
            stride += 2;
        for(std::size_t i = 0; i < owners.size(); ++i)
            owners[i] = static_cast<int>((i * static_cast<std::size_t>(stride) + 3) %
                                         static_cast<std::size_t>(world_size));
        scrambled->setSegments(scrambled->getBoundaries(), owners);
        tess.SetLoadBalancer(scrambled);
        tess.BuildParallel(owned_points(tess), true, false);
        CellComparison const scrambled_cells = compare_tessellations(tess, reference_cells, rank);
        OwnershipCheck const scrambled_ownership = check_ownership(tess, *scrambled, rank);

        // Stage 3: the scrambled balancer after a box change.  The owner
        // multiset must survive and owners must stay attached to their
        // (remapped) boundaries.
        std::shared_ptr<HilbertLoadBalancer<Vector3D>> boxed = scrambled->clone();
        std::vector<int> owners_before = boxed->getSegmentOwner();
        Vector3D const new_ll(-0.05, -0.02, -0.03);
        Vector3D const new_ur(1.04, 1.03, 1.06);
        // Expected (boundary, owner) pairs: every boundary mapped as
        // HilbertLoadBalancer::changeBox maps it (through the old curve's
        // position, affinely into the padded new box, onto the new curve),
        // each keeping its owner.
        std::vector<std::pair<curve_index_t, int>> expected_pairs;
        {
            auto const old_convertor = boxed->getConvertor();
            Vector3D const old_ll = old_convertor->getLL();
            Vector3D const old_size = old_convertor->getUR() - old_ll;
            Vector3D padded_ll = new_ll;
            Vector3D padded_ur = new_ur;
            Vector3D const len = new_ur - new_ll;
            padded_ll.x -= std::abs(SPACE_FACTOR * len.x);
            padded_ll.y -= std::abs(SPACE_FACTOR * len.y);
            padded_ll.z -= std::abs(SPACE_FACTOR * len.z);
            padded_ur.x += std::abs(SPACE_FACTOR * len.x);
            padded_ur.y += std::abs(SPACE_FACTOR * len.y);
            padded_ur.z += std::abs(SPACE_FACTOR * len.z);
            Vector3D const new_size = padded_ur - padded_ll;
            HilbertRectangularConvertor3D<Vector3D> const new_convertor(padded_ll, padded_ur, old_convertor->getOrder());
            for(std::size_t i = 0; i < owners_before.size(); ++i)
            {
                Vector3D const p = old_convertor->d2xyz(boxed->getBoundaries()[i]);
                Vector3D const q(padded_ll.x + (p.x - old_ll.x) / old_size.x * new_size.x,
                                 padded_ll.y + (p.y - old_ll.y) / old_size.y * new_size.y,
                                 padded_ll.z + (p.z - old_ll.z) / old_size.z * new_size.z);
                expected_pairs.emplace_back(new_convertor.xyz2d(q.x, q.y, q.z), owners_before[i]);
            }
        }
        boxed->changeBox({new_ll, new_ur});
        std::vector<std::pair<curve_index_t, int>> actual_pairs;
        for(std::size_t i = 0; i < boxed->getBoundaries().size(); ++i)
            actual_pairs.emplace_back(boxed->getBoundaries()[i], boxed->getSegmentOwner()[i]);
        std::sort(expected_pairs.begin(), expected_pairs.end());
        std::sort(actual_pairs.begin(), actual_pairs.end());
        unsigned long long const owner_multiset_changed = expected_pairs != actual_pairs ? 1 : 0;
        tess.SetLoadBalancer(boxed);
        tess.BuildParallel(owned_points(tess), true, false);
        CellComparison const boxed_cells = compare_tessellations(tess, reference_cells, rank);
        OwnershipCheck const boxed_ownership = check_ownership(tess, *boxed, rank);
        unsigned long long const boxed_io = round_trip(*boxed, owned_points(tess), rank, "boxed");

        auto ownership_ok = [](OwnershipCheck const& check)
        {
            return check.foreign_points == 0 && check.empty_ranks == 0 && check.unrouted_points == 0;
        };
        auto cells_ok = [Np](CellComparison const& cmp)
        {
            return cmp.compared_cells == Np && cmp.volume_mismatches == 0 && cmp.face_count_mismatches == 0;
        };
        if(rank == 0)
        {
            bool const ok = ownership_ok(positional_ownership) && positional_io == 0 &&
                cells_ok(interleaved_cells) && ownership_ok(interleaved_ownership) && interleaved_io == 0 &&
                cells_ok(scrambled_cells) && ownership_ok(scrambled_ownership) &&
                owner_multiset_changed == 0 && cells_ok(boxed_cells) && ownership_ok(boxed_ownership) &&
                boxed_io == 0;
            passed = ok ? 1 : 0;
            std::ofstream out("segmented_hilbert_ownership_metrics.txt");
            out.setf(std::ios::scientific);
            out.precision(16);
            auto report = [&out](std::string const& name, CellComparison const& cmp, OwnershipCheck const& own)
            {
                out << name << "_compared_cells " << cmp.compared_cells << "\n"
                    << name << "_max_volume_rel_error " << cmp.max_volume_rel_error << "\n"
                    << name << "_volume_mismatches " << cmp.volume_mismatches << "\n"
                    << name << "_face_count_mismatches " << cmp.face_count_mismatches << "\n"
                    << name << "_foreign_points " << own.foreign_points << "\n"
                    << name << "_empty_ranks " << own.empty_ranks << "\n"
                    << name << "_unrouted_points " << own.unrouted_points << "\n";
            };
            out << "mode mpi\n" << "ranks " << world_size << "\n" << "num_points " << Np << "\n";
            out << "positional_foreign_points " << positional_ownership.foreign_points << "\n"
                << "positional_unrouted_points " << positional_ownership.unrouted_points << "\n"
                << "positional_io_failures " << positional_io << "\n";
            report("interleaved", interleaved_cells, interleaved_ownership);
            out << "interleaved_segments " << interleaved->getSegmentOwner().size() << "\n"
                << "interleaved_io_failures " << interleaved_io << "\n";
            report("scrambled", scrambled_cells, scrambled_ownership);
            out << "box_change_owner_pairs_changed " << owner_multiset_changed << "\n";
            report("boxed", boxed_cells, boxed_ownership);
            out << "boxed_io_failures " << boxed_io << "\n";
            out << "pass " << passed << "\n";
            std::cout << "segmented_hilbert_ownership pass = " << passed << std::endl;
        }
        MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
#else
        std::ofstream out("segmented_hilbert_ownership_metrics.txt");
        out << "mode serial\npass 1\n";
        passed = 1;
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
