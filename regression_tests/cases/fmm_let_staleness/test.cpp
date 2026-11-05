#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef RICH_MPI
#error "fmm_let_staleness needs an MPI build"
#endif
#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"
#include "source/misc/universal_error.hpp"

// The letStalenessDiagnostics counters on a retained compat LET plan, for a
// hand-built two-rank geometry in which the class of every LET interaction and
// every stale verdict follows from distances alone.  A counter that never
// fires, fires on the wrong interaction, counts an inactive interaction, or
// counts once per wave instead of once per solve fails one of the checks.
//
// Rank 0 holds eight clusters, one per octant, centred at (+-5/3)^3.  Rank 1
// holds one cluster at c1 = (3.6, 3.5, 2.5).  Every cluster is a centre
// particle and six axis anchors at distance 1.  leafCapacity 8 makes rank 1's
// root a leaf and splits rank 0's root (56 particles) once.  Rank 0's root cube
// has half size 1.25 * 8/3 = 10/3 about the origin, so each octant leaf is
// centred on its cluster with half size 5/3.  Radii carry the persistent
// envelope factor 1.05: every leaf 1.05, rank 0's root
// 1.05 * (sqrt(3) * 5/3 + 1) = 4.081.  theta = 0.5.
//
// The rank roots fail the opening test (1.05 + 4.081 > 0.5 * |c1| = 2.80), so
// every interaction below is a LET interaction.  The leaf-to-c1 distances d are
//   leaf  0     1     2     3     4     5     6     7
//   d     8.47  6.91  6.96  4.95  7.42  5.58  5.64  2.79
// M2L needs 2.1 <= theta d: leaves 0-6, from both sides.  Leaf 7 is M2P from
// both sides (closest particle 2.22 from the other centre, 1.05 < 1.11).
//
// Bound (cube radius) staleness, which already holds on the rebuild solve:
//   rank 0 targets, source cube radius sqrt(3) * 1.25 = 2.165:
//     M2L stale when 1.05 + 2.165 > theta d: leaves 3, 5, 6; M2P leaf 7 stale.
//   rank 1 target, source cube radius sqrt(3) * 5/3 = 2.887:
//     M2L stale when 1.05 + 2.887 > theta d: sources 1-6; M2P source 7 stale.
// Optimistic staleness is zero until solve 3.
//
// Solve 2 repeats solve 1.  Solve 3 moves, under the retained plan, the centre
// particle of
//   leaf 7 by 1.5 toward c1 (now 1.29 from c1, 1.05 > 0.645: M2P stale);
//   leaf 3 by 1.5 along +z (envelope 1.575, 2.625 > theta 4.95: M2L stale);
//   leaf 0 by 1.5 along +x (envelope 1.575, 2.625 < theta 8.47: not stale).
// The leaf envelopes grow by 1.5 (beyond 0, 0.1 and 0.3, not 0.6) on rank 0;
// its root envelope grows to 1.05 * (sqrt(3) * 5/3 + 1.5), ratio 1.1286.
// Rank 1 sees no optimistic staleness: it uses the sources' build radii.
// Solve 4 moves leaf 3's particles into leaf 1, inside leaf 1's radius: leaf 3's
// interactions become inactive on both ranks (an empty target on rank 0, an
// omitted payload on rank 1) and leave every count.
//
// Every margin above is at least 5%; the lattice rounding of the cubes is below
// 1e-6.  The whole sequence runs with one LET wave and with a 200-byte wave
// budget, which puts each of rank 1's eight subscriptions in its own wave.

namespace
{
const double kLeafOffset = 5.0 / 3.0;
const Vector3D kRank1Center(3.6, 3.5, 2.5);
const int kSolves = 4;

Vector3D leafCenter(int octant)
{
    return Vector3D((octant & 1) ? kLeafOffset : -kLeafOffset,
                    (octant & 2) ? kLeafOffset : -kLeafOffset,
                    (octant & 4) ? kLeafOffset : -kLeafOffset);
}

void addCluster(std::vector<Vector3D>& points, const Vector3D& center)
{
    points.push_back(center);
    points.push_back(center + Vector3D(1, 0, 0));
    points.push_back(center + Vector3D(-1, 0, 0));
    points.push_back(center + Vector3D(0, 1, 0));
    points.push_back(center + Vector3D(0, -1, 0));
    points.push_back(center + Vector3D(0, 0, 1));
    points.push_back(center + Vector3D(0, 0, -1));
}

// points[7 * o] is the centre particle of rank 0's leaf o.
std::vector<Vector3D> positionsFor(int rank, int solve)
{
    std::vector<Vector3D> points;
    if(rank == 1)
    {
        addCluster(points, kRank1Center);
        return points;
    }
    for(int octant = 0; octant < 8; ++octant)
        addCluster(points, leafCenter(octant));
    if(solve >= 3)
    {
        const Vector3D toward = kRank1Center - leafCenter(7);
        points[7 * 7] = leafCenter(7) + toward * (1.5 / abs(toward));
        points[7 * 3] = leafCenter(3) + Vector3D(0, 0, 1.5);
        points[7 * 0] = leafCenter(0) + Vector3D(1.5, 0, 0);
    }
    if(solve >= 4)
    {
        for(int k = 0; k < 7; ++k)
            points[7 * 3 + k] = leafCenter(1) + Vector3D(
                (k & 1) ? 0.4 : -0.4, (k & 2) ? 0.4 : -0.4,
                (k & 4) ? 0.4 : -0.4);
    }
    return points;
}

// Per rank and solve: rebuilt, plan age, active M2L interactions, M2P
// evaluations (7 target particles), inactive M2L interactions, the four stale
// counts, and the four growth flags.
const int kFields = 13;
const char* const kFieldNames[kFields] = {
    "let_rebuilt", "let_plan_age", "let_m2l", "let_m2p_evaluations",
    "let_inactive_m2l", "m2l_optimistic", "m2l_bound", "m2p_optimistic",
    "m2p_bound", "growth_0", "growth_0p1", "growth_0p3", "growth_0p6"};
const unsigned long long kExpected[2][kSolves][kFields] = {
    {{1, 0, 7, 7, 0, 0, 3, 0, 1, 0, 0, 0, 0},
     {0, 1, 7, 7, 0, 0, 3, 0, 1, 0, 0, 0, 0},
     {0, 2, 7, 7, 0, 1, 3, 1, 1, 1, 1, 1, 0},
     {0, 3, 6, 7, 1, 0, 2, 1, 1, 1, 1, 1, 0}},
    {{1, 0, 7, 7, 0, 0, 6, 0, 1, 0, 0, 0, 0},
     {0, 1, 7, 7, 0, 0, 6, 0, 1, 0, 0, 0, 0},
     {0, 2, 7, 7, 0, 0, 6, 0, 1, 0, 0, 0, 0},
     {0, 3, 6, 7, 1, 0, 5, 0, 1, 0, 0, 0, 0}}};
// Rank 0's root growth ratio per solve; rank 1's stays 1.
const double kRootGrowth[kSolves] = {1.0, 1.0, 1.1286, 1.1286};

int runSequence(const char* name, std::size_t maxLetWaveBytes, int rank,
                std::ofstream& out)
{
    FmmGravityOptions gravity;
    gravity.expansionOrder = 2;
    gravity.thetaCritical = 0.5;
    gravity.leafCapacity = 8;
    FmmDistributedOptions distributed;
    distributed.spatiallyRedistributeForGravity = false;
    distributed.persistentLocalTreeTopology = true;
    distributed.letStalenessDiagnostics = true;
    distributed.maxLetWaveBytes = maxLetWaveBytes;
    DistributedFmmGravityCalculator calculator(gravity, distributed);

    const Vector3D domainLower(-64, -64, -64), domainUpper(64, 64, 64);
    int failures = 0;
    for(int solve = 1; solve <= kSolves; ++solve)
    {
        const std::vector<Vector3D> positions = positionsFor(rank, solve);
        const std::vector<double> masses(positions.size(), 1.0);
        std::vector<std::uint64_t> ids(positions.size());
        for(std::size_t i = 0; i < ids.size(); ++i)
            ids[i] = static_cast<std::uint64_t>(rank) * 1000 + i + 1;
        std::vector<Vector3D> acceleration;
        calculator.solve(positions, masses, ids, domainLower, domainUpper,
                         acceleration);
        const FmmSolveStats& st = calculator.stats();

        const unsigned long long local[kFields] = {
            st.letTopologyRebuilt ? 1ull : 0ull,
            static_cast<unsigned long long>(st.letPlanAgeSolves),
            static_cast<unsigned long long>(st.letM2LCount),
            static_cast<unsigned long long>(st.letM2PCount),
            static_cast<unsigned long long>(st.letInactiveM2LCount),
            static_cast<unsigned long long>(st.letM2LOptimisticStaleCount),
            static_cast<unsigned long long>(st.letM2LBoundStaleCount),
            static_cast<unsigned long long>(st.letM2POptimisticStaleCount),
            static_cast<unsigned long long>(st.letM2PBoundStaleCount),
            st.planningRadiusGrowthBeyond[0] ? 1ull : 0ull,
            st.planningRadiusGrowthBeyond[1] ? 1ull : 0ull,
            st.planningRadiusGrowthBeyond[2] ? 1ull : 0ull,
            st.planningRadiusGrowthBeyond[3] ? 1ull : 0ull};
        unsigned long long all[2][kFields] = {};
        MPI_Gather(local, kFields, MPI_UNSIGNED_LONG_LONG, all, kFields,
                   MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        double growth[2] = {};
        MPI_Gather(&st.rootRadiusGrowthRatio, 1, MPI_DOUBLE, growth, 1,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        const unsigned long long localWaves =
            static_cast<unsigned long long>(st.letWaveCount);
        unsigned long long waves[2] = {};
        MPI_Gather(&localWaves, 1, MPI_UNSIGNED_LONG_LONG, waves, 1,
                   MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        if(rank != 0)
            continue;

        for(int r = 0; r < 2; ++r)
        {
            std::string line = std::string("staleness ") + name +
                " solve=" + std::to_string(solve) + " rank=" +
                std::to_string(r);
            for(int f = 0; f < kFields; ++f)
            {
                line += " " + std::string(kFieldNames[f]) + "=" +
                    std::to_string(all[r][f]);
                if(all[r][f] != kExpected[r][solve - 1][f])
                {
                    ++failures;
                    out << "mismatch " << name << " solve " << solve
                        << " rank " << r << " " << kFieldNames[f]
                        << " got " << all[r][f] << " expected "
                        << kExpected[r][solve - 1][f] << "\n";
                }
            }
            const double expectedGrowth = r == 0 ? kRootGrowth[solve - 1] : 1.0;
            line += " root_growth=" + std::to_string(growth[r]);
            if(!(std::fabs(growth[r] - expectedGrowth) < 0.005))
            {
                ++failures;
                out << "mismatch " << name << " solve " << solve << " rank "
                    << r << " root_growth got " << growth[r] << " expected "
                    << expectedGrowth << "\n";
            }
            // One wave without a budget; with it, each of rank 1's eight
            // multipole subscriptions (48 + 9 * 8 bytes) fills a wave alone,
            // and every rank runs the global count.
            const unsigned long long expectedWaves =
                maxLetWaveBytes == 0 ? 1 : 8;
            line += " wave_count=" + std::to_string(waves[r]);
            if(waves[r] != expectedWaves)
            {
                ++failures;
                out << "mismatch " << name << " solve " << solve << " rank "
                    << r << " wave_count got " << waves[r] << " expected "
                    << expectedWaves << "\n";
            }
            std::printf("%s\n", line.c_str());
        }
    }
    std::fflush(stdout);
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return failures;
}
}

int main(void)
{
    MPI_Init(NULL, NULL);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2)
    {
        if(rank == 0)
            std::fprintf(stderr, "fmm_let_staleness needs exactly 2 MPI ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    std::ofstream out;
    if(rank == 0)
        out.open("fmm_let_staleness_metrics.txt");
    int failures = 0;
    try
    {
        failures += runSequence("one_wave", 0, rank, out);
        failures += runSequence("waves", 200, rank, out);
    }
    catch(const UniversalError& error)
    {
        reportError(error);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if(rank == 0)
    {
        out << "failures " << failures << "\n";
        out << "pass " << (failures == 0 ? 1 : 0) << "\n";
    }
    MPI_Finalize();
    return 0;
}
