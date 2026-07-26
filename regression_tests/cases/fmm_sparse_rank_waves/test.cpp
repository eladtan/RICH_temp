// Miniature reproduction of the TDE sparse-rank LET pathology.
//
// On the production case (37.2M cells, 1152 ranks) Hilbert decomposition
// balances cell counts but not volume, so a handful of ranks receive a sparse
// dusting of cells spanning the whole domain. Their FMM root then encloses every
// other rank's domain, their near field becomes global, and the LET tried to
// receive ~99.9% of all particles in one exchange -- about 2.1 GB, past the MPI
// int limit.
//
// This test recreates the geometry in miniature so the failure mode and its fix
// can be exercised in seconds on a few ranks instead of a 6-node job:
//
//   * ranks 1..size-1 each own a compact cluster inside a dense core;
//   * rank 0 owns a sparse set scattered through the entire domain, giving it a
//     root that encloses every other rank;
//   * cell counts are deliberately balanced, as they are in production.
//
// It asserts three things:
//   1. accuracy against direct summation, so the wave machinery cannot silently
//      drop or double-count an interaction;
//   2. the pathology is actually present, i.e. rank 0's root is far larger than
//      the median, otherwise the test would pass vacuously;
//   3. a small maxLetWaveBytes forces several waves, all results still match,
//      and the observed per-wave payload respects the budget.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"

namespace
{
struct Body
{
    Vector3D position;
    double mass = 0.0;
    std::uint64_t id = 0;
    int ownerRank = -1;
    std::uint64_t ownerLocalIndex = 0;
};

constexpr int kBodiesPerRank = 12;
constexpr double kDomainHalfSize = 1.0;
// The dense core occupies a small fraction of the domain, as the TDE core does.
constexpr double kCoreHalfSize = 0.05;

// Deterministic low-discrepancy sequence; avoids depending on any RNG.
double radicalInverse(unsigned int index, unsigned int base)
{
    double result = 0.0;
    double weight = 1.0 / static_cast<double>(base);
    while(index > 0)
    {
        result += weight * static_cast<double>(index % base);
        index /= base;
        weight /= static_cast<double>(base);
    }
    return result;
}

std::vector<Body> bodiesForRank(int rank, int size)
{
    std::vector<Body> result;
    for(int i = 0; i < kBodiesPerRank; ++i)
    {
        const unsigned int sample =
            static_cast<unsigned int>(rank * kBodiesPerRank + i + 1);
        Body body;
        if(rank == 0)
        {
            // Sparse dusting across the full domain. This is the pathological
            // rank: its bounding cube encloses every dense cluster below.
            body.position = Vector3D(
                -0.9 + 1.8 * radicalInverse(sample, 2),
                -0.9 + 1.8 * radicalInverse(sample, 3),
                -0.9 + 1.8 * radicalInverse(sample, 5));
        }
        else
        {
            // Compact cluster inside the dense core, one per rank, so each of
            // these ranks looks like an ordinary well-behaved rank.
            const double angle = 2.0 * 3.14159265358979323846 *
                static_cast<double>(rank) / static_cast<double>(std::max(1, size));
            const Vector3D centre(0.6 * kCoreHalfSize * std::cos(angle),
                                  0.6 * kCoreHalfSize * std::sin(angle),
                                  0.2 * kCoreHalfSize *
                                      std::cos(2.0 * angle));
            const double spread = 0.12 * kCoreHalfSize;
            body.position = Vector3D(
                centre.x + spread * (radicalInverse(sample, 2) - 0.5),
                centre.y + spread * (radicalInverse(sample, 3) - 0.5),
                centre.z + spread * (radicalInverse(sample, 5) - 0.5));
        }
        body.mass = 0.5 + 0.03 * static_cast<double>(rank) +
                    0.011 * static_cast<double>(i);
        // Duplicate application IDs on purpose: physical identity must come
        // from the owner token, not this field.
        body.id = static_cast<std::uint64_t>(i % 3);
        body.ownerRank = rank;
        body.ownerLocalIndex = static_cast<std::uint64_t>(i);
        result.push_back(body);
    }
    return result;
}

std::vector<Body> allBodies(int size)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local = bodiesForRank(rank, size);
        result.insert(result.end(), local.begin(), local.end());
    }
    return result;
}

bool sameBody(const Body& first, const Body& second)
{
    return first.ownerRank == second.ownerRank &&
           first.ownerLocalIndex == second.ownerLocalIndex;
}

Vector3D directAcceleration(const Body& target, const std::vector<Body>& all)
{
    Vector3D result;
    for(const Body& source : all)
    {
        if(sameBody(source, target))
            continue;
        const Vector3D delta = target.position - source.position;
        const double r2 = delta.x * delta.x + delta.y * delta.y +
                          delta.z * delta.z;
        const double invR = 1.0 / std::sqrt(r2);
        result -= source.mass * delta * (invR * invR * invR);
    }
    return result;
}

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

void unpack(const std::vector<Body>& bodies,
            std::vector<Vector3D>& positions,
            std::vector<double>& masses,
            std::vector<std::uint64_t>& ids)
{
    positions.clear();
    masses.clear();
    ids.clear();
    for(const Body& body : bodies)
    {
        positions.push_back(body.position);
        masses.push_back(body.mass);
        ids.push_back(body.id);
    }
}

double worstError(const std::vector<Body>& localBodies,
                  const std::vector<Body>& globalBodies,
                  const std::vector<Vector3D>& acceleration)
{
    double maximum = 0.0;
    for(std::size_t i = 0; i < localBodies.size(); ++i)
    {
        const Vector3D reference =
            directAcceleration(localBodies[i], globalBodies);
        maximum = std::max(maximum,
            norm(acceleration[i] - reference) /
            std::max(1.0, norm(reference)));
    }
    return maximum;
}

// Largest componentwise difference between two acceleration sets. Wave count
// must not change the answer, so this should be at or near zero.
double worstDifference(const std::vector<Vector3D>& first,
                       const std::vector<Vector3D>& second)
{
    double maximum = 0.0;
    const std::size_t count = std::min(first.size(), second.size());
    for(std::size_t i = 0; i < count; ++i)
        maximum = std::max(maximum,
            norm(first[i] - second[i]) / std::max(1.0, norm(first[i])));
    return maximum;
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // The scattered sparse rank makes this geometry demanding, so use an order
    // and opening angle tight enough that the accuracy tolerance below is a
    // real constraint rather than a formality.
    FmmGravityOptions options;
    options.expansionOrder = 6;
    options.thetaCritical = 0.3;
    options.leafCapacity = 2;
    options.computePotential = false;
    options.validateFinite = true;

    const std::vector<Body> localBodies = bodiesForRank(rank, size);
    const std::vector<Body> globalBodies = allBodies(size);
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    unpack(localBodies, positions, masses, ids);
    const Vector3D lower(-kDomainHalfSize, -kDomainHalfSize, -kDomainHalfSize);
    const Vector3D upper(kDomainHalfSize, kDomainHalfSize, kDomainHalfSize);

    // Reference: one wave, i.e. the pre-wave behaviour.
    std::vector<Vector3D> singleWaveAcceleration;
    double singleWaveError = 0.0;
    std::size_t singleWaveCount = 0;
    {
        FmmDistributedOptions distributed;
        // This is deliberately the historical rank-root pathology. Keep the
        // compatibility geometry here; patch-mode sparse coverage lives in
        // fmm_patch_let_mpi and fmm_patch_moving_mesh.
        distributed.enablePatchForest = false;
        distributed.maxRemoteBytes = 64u * 1024u * 1024u;
        distributed.maxLetWaveBytes = 0; // splitting disabled
        DistributedFmmGravityCalculator solver(options, distributed);
        solver.solve(positions, masses, ids, lower, upper,
                     singleWaveAcceleration);
        singleWaveError =
            worstError(localBodies, globalBodies, singleWaveAcceleration);
        singleWaveCount = solver.stats().letWaveCount;
    }

    // A budget of a few records forces the planner to split aggressively. It
    // must stay above one multipole record (40 + 84 * 8 bytes at order 6) or the
    // planner correctly refuses an indivisible payload.
    std::vector<Vector3D> manyWaveAcceleration;
    double manyWaveError = 0.0;
    std::size_t manyWaveCount = 0;
    std::size_t manyWavePeakBytes = 0;
    {
        FmmDistributedOptions distributed;
        distributed.enablePatchForest = false;
        distributed.maxRemoteBytes = 64u * 1024u * 1024u;
        distributed.maxLetWaveBytes = 2048;
        DistributedFmmGravityCalculator solver(options, distributed);
        solver.solve(positions, masses, ids, lower, upper,
                     manyWaveAcceleration);
        manyWaveError =
            worstError(localBodies, globalBodies, manyWaveAcceleration);
        manyWaveCount = solver.stats().letWaveCount;
        manyWavePeakBytes = solver.stats().letMaxWavePayloadBytes;
    }

    const double waveDifference =
        worstDifference(singleWaveAcceleration, manyWaveAcceleration);

    // Confirm the pathology is present: rank 0's root must dwarf the others,
    // otherwise this test would pass without exercising anything.
    double localSpan = 0.0;
    if(!positions.empty())
    {
        Vector3D low = positions.front();
        Vector3D high = positions.front();
        for(const Vector3D& point : positions)
        {
            low.x = std::min(low.x, point.x);
            low.y = std::min(low.y, point.y);
            low.z = std::min(low.z, point.z);
            high.x = std::max(high.x, point.x);
            high.y = std::max(high.y, point.y);
            high.z = std::max(high.z, point.z);
        }
        localSpan = std::max(high.x - low.x,
                             std::max(high.y - low.y, high.z - low.z));
    }
    double sparseSpan = rank == 0 ? localSpan : 0.0;
    double denseSpan = rank == 0 ? 0.0 : localSpan;
    double globalSparseSpan = 0.0;
    double globalDenseSpanMax = 0.0;
    MPI_Allreduce(&sparseSpan, &globalSparseSpan, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&denseSpan, &globalDenseSpanMax, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    const double spanRatio = globalDenseSpanMax > 0.0 ?
        globalSparseSpan / globalDenseSpanMax : 0.0;

    const double errorTolerance = 1e-3;
    const double agreementTolerance = 1e-10;
    const int localChecks[6] = {
        singleWaveError <= errorTolerance ? 1 : 0,
        manyWaveError <= errorTolerance ? 1 : 0,
        waveDifference <= agreementTolerance ? 1 : 0,
        // The tiny budget must actually have produced more than one wave.
        manyWaveCount > singleWaveCount ? 1 : 0,
        // The pathology must be present for this test to mean anything.
        (size < 3 || spanRatio >= 5.0) ? 1 : 0,
        std::isfinite(singleWaveError) && std::isfinite(manyWaveError) ? 1 : 0};
    int globalChecks[6] = {};
    MPI_Allreduce(localChecks, globalChecks, 6, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    double globalSingleWaveError = 0.0;
    double globalManyWaveError = 0.0;
    double globalWaveDifference = 0.0;
    MPI_Allreduce(&singleWaveError, &globalSingleWaveError, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&manyWaveError, &globalManyWaveError, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&waveDifference, &globalWaveDifference, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    unsigned long long localPeak =
        static_cast<unsigned long long>(manyWavePeakBytes);
    unsigned long long globalPeak = 0;
    MPI_Allreduce(&localPeak, &globalPeak, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);

    const bool globalPass = globalChecks[0] && globalChecks[1] &&
                            globalChecks[2] && globalChecks[3] &&
                            globalChecks[4] && globalChecks[5];

    if(rank == 0)
    {
        std::ofstream output("fmm_sparse_rank_waves_metrics.txt");
        output.setf(std::ios::scientific);
        output.precision(16);
        output << "ranks " << size << "\n";
        output << "single_wave_error " << globalSingleWaveError << "\n";
        output << "many_wave_error " << globalManyWaveError << "\n";
        output << "wave_difference " << globalWaveDifference << "\n";
        output << "single_wave_count " << singleWaveCount << "\n";
        output << "many_wave_count " << manyWaveCount << "\n";
        output << "many_wave_peak_bytes " << globalPeak << "\n";
        output << "sparse_span " << globalSparseSpan << "\n";
        output << "dense_span_max " << globalDenseSpanMax << "\n";
        output << "span_ratio " << spanRatio << "\n";
        output << "single_wave_accurate " << globalChecks[0] << "\n";
        output << "many_wave_accurate " << globalChecks[1] << "\n";
        output << "waves_agree " << globalChecks[2] << "\n";
        output << "waves_split " << globalChecks[3] << "\n";
        output << "pathology_present " << globalChecks[4] << "\n";
        output << "finite " << globalChecks[5] << "\n";
        output << "pass " << globalPass << "\n";
        std::cout << "fmm_sparse_rank_waves ranks=" << size
                  << " single_wave_error=" << globalSingleWaveError
                  << " many_wave_error=" << globalManyWaveError
                  << " wave_difference=" << globalWaveDifference
                  << " single_wave_count=" << singleWaveCount
                  << " many_wave_count=" << manyWaveCount
                  << " span_ratio=" << spanRatio
                  << " pass=" << globalPass << std::endl;
    }

    MPI_Finalize();
    return globalPass ? 0 : 1;
}
