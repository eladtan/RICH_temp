#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"
#include "source/3D/gravity/fmm/mpi/FmmRecordExchange.hpp"

// Lifecycle of the temporary gravity-owner redistribution
// (spatiallyRedistributeForGravity): the splitters are frozen across warm
// solves, so a domain change (UpdateBox) or drift of the particles relative to
// them must resample them before the owner exchange collapses onto one rank.
// The fixture is a TDE caricature: 95% of the mass in a tight cluster, every
// input rank holding the same mix, one input rank empty.

namespace
{
constexpr std::size_t kBodiesPerActiveRank = 600;
constexpr double kClusterFraction = 0.95;
constexpr double kClusterHalfWidth = 2e-3;
// Direct-sum bound for the clustered sets: the same 1e-2 that fmm_gravity_mpi
// allows its clustered case.  At theta 0.3 the baseline solve, before any
// lifecycle event, already sits near 2.4e-3; the exact contract is the
// warm-versus-fresh comparison.
constexpr double kClusteredDirectTolerance = 1e-2;

struct Body
{
    Vector3D position;
    double mass = 0.0;
};

std::uint64_t splitMix(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

double unitSample(std::uint64_t seed)
{
    return static_cast<double>(splitMix(seed) >> 11) * 0x1.0p-53;
}

// Every rank generates the same global list, so the direct reference needs no
// communication.  Body k is in the cluster when k mod 20 != 0.
std::vector<Body> makeBodies(std::size_t count, const Vector3D& clusterCenter)
{
    std::vector<Body> bodies(count);
    for(std::size_t k = 0; k < count; ++k)
    {
        const std::uint64_t seed = 3 * static_cast<std::uint64_t>(k);
        const double u = unitSample(seed);
        const double v = unitSample(seed + 1);
        const double w = unitSample(seed + 2);
        const bool clustered = (k % 20) != 0;
        bodies[k].position = clustered ?
            Vector3D(clusterCenter.x + kClusterHalfWidth * (2.0 * u - 1.0),
                     clusterCenter.y + kClusterHalfWidth * (2.0 * v - 1.0),
                     clusterCenter.z + kClusterHalfWidth * (2.0 * w - 1.0)) :
            Vector3D(-0.95 + 1.9 * u, -0.95 + 1.9 * v, -0.95 + 1.9 * w);
        bodies[k].mass = 1.0 / static_cast<double>(count);
    }
    return bodies;
}

// Indices of the bodies an input rank holds.  shares[r] is rank r's weight;
// assignment is by a per-body hash, so every rank sees the same spatial mix.
std::vector<std::size_t> heldBy(int rank, std::size_t count,
                                const std::vector<double>& shares)
{
    double total = 0.0;
    for(double share : shares)
        total += share;
    std::vector<std::size_t> result;
    for(std::size_t k = 0; k < count; ++k)
    {
        const double u = total * unitSample(0x51ed27ull + 7 * k);
        double cumulative = 0.0;
        int owner = static_cast<int>(shares.size()) - 1;
        for(std::size_t r = 0; r < shares.size(); ++r)
        {
            cumulative += shares[r];
            if(u < cumulative)
            {
                owner = static_cast<int>(r);
                break;
            }
        }
        if(owner == rank)
            result.push_back(k);
    }
    return result;
}

std::vector<double> equalShares(int size)
{
    // The last rank holds nothing, as in the sibling MPI regressions.
    std::vector<double> shares(static_cast<std::size_t>(size), 1.0);
    shares.back() = 0.0;
    return shares;
}

struct LocalInput
{
    std::vector<std::size_t> indices;
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
};

LocalInput localInput(const std::vector<Body>& bodies,
                      const std::vector<std::size_t>& indices)
{
    LocalInput input;
    input.indices = indices;
    for(std::size_t index : indices)
    {
        input.positions.push_back(bodies[index].position);
        input.masses.push_back(bodies[index].mass);
        input.ids.push_back(static_cast<std::uint64_t>(index));
    }
    return input;
}

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

double directError(const std::vector<Body>& bodies, const LocalInput& input,
                   const std::vector<Vector3D>& acceleration)
{
    double maximum = 0.0;
    for(std::size_t i = 0; i < input.indices.size(); ++i)
    {
        const std::size_t target = input.indices[i];
        Vector3D reference;
        for(std::size_t source = 0; source < bodies.size(); ++source)
        {
            if(source == target)
                continue;
            const Vector3D delta =
                bodies[target].position - bodies[source].position;
            const double r = norm(delta);
            reference -= bodies[source].mass * delta * (1.0 / (r * r * r));
        }
        maximum = std::max(maximum, norm(acceleration[i] - reference) /
                                    std::max(1.0, norm(reference)));
    }
    double global = 0.0;
    MPI_Allreduce(&maximum, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

double compareSolutions(const std::vector<Vector3D>& first,
                        const std::vector<Vector3D>& second)
{
    double maximum = 0.0;
    for(std::size_t i = 0; i < first.size(); ++i)
        maximum = std::max(maximum, norm(first[i] - second[i]) /
                                    std::max(1.0, norm(second[i])));
    double global = 0.0;
    MPI_Allreduce(&maximum, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

FmmGravityOptions gravityOptions()
{
    FmmGravityOptions options;
    options.expansionOrder = 5;
    options.thetaCritical = 0.3;
    options.leafCapacity = 16;
    options.computePotential = false;
    options.validateFinite = true;
    return options;
}

FmmDistributedOptions redistributedOptions()
{
    FmmDistributedOptions options;
    options.enablePatchForest = false;
    options.spatiallyRedistributeForGravity = true;
    // Keep the direct-force accuracy contract independent of the lossy LET
    // wire encodings, as in fmm_gravity_mpi.
    options.compactLetParticlePayload = false;
    options.quantizedLetParticlePayload = false;
    return options;
}

double ownerImbalance(const FmmSolveStats& stats)
{
    return stats.gravityOwnerMeanCount > 0.0 ?
        static_cast<double>(stats.gravityOwnerMaxCount) /
            stats.gravityOwnerMeanCount : 0.0;
}

double staleOwnerFraction(const FmmSolveStats& stats)
{
    return stats.gravityParticleTotal > 0 ?
        static_cast<double>(stats.gravityOwnerMaxCountBeforeResample) /
            static_cast<double>(stats.gravityParticleTotal) : 0.0;
}

// Value printed after "name " in a reportError dump, or "" when absent.
std::string reportField(const std::string& report, const std::string& name)
{
    std::istringstream lines(report);
    std::string line;
    const std::string key = " " + name + " ";
    while(std::getline(lines, line))
    {
        const std::size_t at = (" " + line).find(key);
        if(at != std::string::npos)
            return line.substr(at + key.size() - 1);
    }
    return "";
}

struct Metrics
{
    std::ostringstream text;
    bool pass = true;

    void value(const std::string& name, double value)
    {
        text << name << ' ' << value << '\n';
    }

    void require(const std::string& name, bool ok)
    {
        text << name << "_ok " << (ok ? 1 : 0) << '\n';
        pass = pass && ok;
    }
};

// Every rank must reach the same verdict; a disagreement is itself a failure.
bool allRanks(bool local)
{
    int value = local ? 1 : 0;
    int minimum = 0;
    int maximum = 0;
    MPI_Allreduce(&value, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return minimum == 1 && maximum == 1;
}

bool constructionRejected(const FmmDistributedOptions& options)
{
    try
    {
        DistributedFmmGravityCalculator calculator(gravityOptions(), options);
        (void) calculator;
    }
    catch(const UniversalError&)
    {
        return true;
    }
    return false;
}

struct Record
{
    double payload = 0.0;
    std::uint64_t origin = 0;
};

std::vector<std::vector<Record>> recordsToSend(int rank, int size)
{
    // Rank r sends peer p exactly p + 1 records, so rank p receives
    // size * (p + 1) records in total.
    std::vector<std::vector<Record>> send(static_cast<std::size_t>(size));
    for(int peer = 0; peer < size; ++peer)
        for(int i = 0; i <= peer; ++i)
        {
            Record record;
            record.payload = 1000.0 * rank + i;
            record.origin = static_cast<std::uint64_t>(rank);
            send[static_cast<std::size_t>(peer)].push_back(record);
        }
    return send;
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Metrics metrics;
    metrics.value("ranks", size);
    if(size < 3)
    {
        if(rank == 0)
            std::cerr << "fmm_gravity_mpi_redistribute needs at least 3 ranks\n";
        MPI_Finalize();
        return 1;
    }

    const std::size_t bodyCount =
        kBodiesPerActiveRank * static_cast<std::size_t>(size - 1);
    const Vector3D firstCluster(0.3, 0.1, -0.2);
    const Vector3D movedCluster(-0.45, -0.3, 0.35);
    const Vector3D lower0(-1.0, -1.0, -1.0);
    const Vector3D upper(1.0, 1.0, 1.0);
    // UpdateBox-style growth: one face moves out by 17% of the extent.
    const Vector3D lower1(lower0.x - 0.17 * (upper.x - lower0.x),
                          lower0.y, lower0.z);
    const std::vector<Body> bodies = makeBodies(bodyCount, firstCluster);
    const LocalInput input =
        localInput(bodies, heldBy(rank, bodyCount, equalShares(size)));
    const double balanceBound = 1.5;

    // 1. Domain change on a warm calculator.
    {
        DistributedFmmGravityCalculator warm(gravityOptions(),
                                             redistributedOptions());
        std::vector<Vector3D> baseline;
        warm.solve(input.positions, input.masses, input.ids, lower0, upper,
                   baseline);
        const FmmSolveStats first = warm.stats();
        metrics.value("baseline_owner_imbalance", ownerImbalance(first));
        metrics.require("baseline_balanced",
                        ownerImbalance(first) <= balanceBound);
        metrics.require("baseline_initial_sample_only",
                        first.gravitySplittersResampled &&
                        first.gravityResampleCount == 0 &&
                        first.gravitySplitterEpoch == 1 &&
                        !first.gravityDomainChanged);
        metrics.value("baseline_direct_error",
                      directError(bodies, input, baseline));

        std::vector<Vector3D> grown;
        bool grownThrew = false;
        try
        {
            warm.solve(input.positions, input.masses, input.ids, lower1, upper,
                       grown);
        }
        catch(const UniversalError& error)
        {
            grownThrew = true;
            if(rank == 0)
                std::cerr << "domain_grow failed: " << error.getErrorMessage()
                          << '\n';
        }
        const bool grownOk = allRanks(!grownThrew);
        metrics.require("domain_grow_no_throw", grownOk);
        if(grownOk)
        {
            const FmmSolveStats second = warm.stats();
            metrics.value("domain_grow_stale_owner_fraction",
                          staleOwnerFraction(second));
            metrics.value("domain_grow_owner_imbalance", ownerImbalance(second));
            metrics.value("domain_grow_root_change_ranks",
                          static_cast<double>(second.ranksWithRootGeometryChange));
            metrics.require("domain_grow_detected", second.gravityDomainChanged);
            // The stale splitters really collapsed the partition.
            metrics.require("domain_grow_stale_partition_collapsed",
                            staleOwnerFraction(second) >= 0.5);
            metrics.require("domain_grow_rebalanced",
                            ownerImbalance(second) <= balanceBound);
            metrics.require("domain_grow_one_resample",
                            second.gravitySplittersResampled &&
                            second.gravityResampleCount == 1 &&
                            second.gravitySplitterEpoch == 2);
            metrics.require("domain_grow_all_roots_rebuilt",
                            second.ranksWithRootGeometryChange ==
                                static_cast<std::size_t>(size) &&
                            second.processTopologyRebuilt);
            const double grownError = directError(bodies, input, grown);
            metrics.value("domain_grow_direct_error", grownError);
            metrics.require("domain_grow_direct",
                            grownError < kClusteredDirectTolerance);

            DistributedFmmGravityCalculator fresh(gravityOptions(),
                                                  redistributedOptions());
            std::vector<Vector3D> freshAcceleration;
            fresh.solve(input.positions, input.masses, input.ids, lower1,
                        upper, freshAcceleration);
            const double warmVsFresh =
                compareSolutions(grown, freshAcceleration);
            metrics.value("domain_grow_vs_fresh", warmVsFresh);
            // Same bound as the warm-versus-fresh contract in fmm_gravity_mpi.
            metrics.require("domain_grow_matches_fresh", warmVsFresh < 2e-4);

            std::vector<Vector3D> steady;
            warm.solve(input.positions, input.masses, input.ids, lower1, upper,
                       steady);
            const FmmSolveStats third = warm.stats();
            metrics.require("steady_domain_keeps_splitters",
                            !third.gravityDomainChanged &&
                            !third.gravitySplittersResampled &&
                            third.gravityResampleCount == 1 &&
                            third.gravitySplitterEpoch == 2);
        }
    }

    // 2. Drift: the cluster moves while the domain stays fixed, so only the
    // load trigger can notice the stale splitters.
    {
        FmmDistributedOptions options = redistributedOptions();
        options.gravityOwnerImbalanceTrigger = 2.5;
        DistributedFmmGravityCalculator drift(gravityOptions(), options);
        std::vector<Vector3D> acceleration;
        drift.solve(input.positions, input.masses, input.ids, lower0, upper,
                    acceleration);
        const std::vector<Body> moved = makeBodies(bodyCount, movedCluster);
        const LocalInput movedInput = localInput(moved, input.indices);
        bool driftThrew = false;
        try
        {
            drift.solve(movedInput.positions, movedInput.masses, movedInput.ids,
                        lower0, upper, acceleration);
        }
        catch(const UniversalError& error)
        {
            driftThrew = true;
            if(rank == 0)
                std::cerr << "drift failed: " << error.getErrorMessage() << '\n';
        }
        const bool driftOk = allRanks(!driftThrew);
        metrics.require("drift_no_throw", driftOk);
        if(driftOk)
        {
            const FmmSolveStats stats = drift.stats();
            metrics.value("drift_stale_owner_fraction", staleOwnerFraction(stats));
            metrics.value("drift_owner_imbalance", ownerImbalance(stats));
            metrics.require("drift_without_domain_change",
                            !stats.gravityDomainChanged);
            metrics.require("drift_stale_partition_collapsed",
                            staleOwnerFraction(stats) >= 0.5);
            metrics.require("drift_resampled_once",
                            stats.gravitySplittersResampled &&
                            stats.gravityResampleCount == 1);
            metrics.require("drift_rebalanced",
                            ownerImbalance(stats) <= balanceBound);
            const double driftError = directError(moved, movedInput,
                                                  acceleration);
            metrics.value("drift_direct_error", driftError);
            metrics.require("drift_direct",
                            driftError < kClusteredDirectTolerance);
        }
    }

    // 3. An owner that stays over its byte limit after resampling is a
    // collective error carrying the numbers needed to act on it.
    {
        FmmDistributedOptions options = redistributedOptions();
        const double mean = static_cast<double>(bodyCount) / size;
        options.maxGravityOwnerBytesPerRank =
            static_cast<std::size_t>(0.5 * mean) * 64;
        bool threw = false;
        bool actionable = false;
        try
        {
            DistributedFmmGravityCalculator limited(gravityOptions(), options);
            std::vector<Vector3D> acceleration;
            limited.solve(input.positions, input.masses, input.ids, lower0,
                          upper, acceleration);
        }
        catch(const UniversalError& error)
        {
            threw = true;
            std::ostringstream report;
            reportError(error, report);
            const std::string text = report.str();
            actionable =
                error.getErrorMessage().find("exceeds its byte limit") !=
                    std::string::npos &&
                !reportField(text, "worst_rank").empty() &&
                !reportField(text, "worst_owner_records").empty() &&
                !reportField(text, "mean_owner_records").empty() &&
                !reportField(text, "imbalance_max_over_mean").empty() &&
                reportField(text, "global_particle_total") ==
                    std::to_string(bodyCount) &&
                reportField(text, "record_bytes") == "64";
        }
        metrics.require("owner_limit_error_collective", allRanks(threw));
        metrics.require("owner_limit_error_actionable", allRanks(actionable));
    }

    // 4. The owner limit bounds gravity ownership only.  Rank 0 holds half of
    // the hydro particles, so its returned results exceed the owner limit even
    // though every gravity owner stays below it.
    {
        std::vector<double> shares(static_cast<std::size_t>(size), 0.0);
        shares[0] = 0.5 * (size - 1);
        for(int r = 1; r < size - 1; ++r)
            shares[static_cast<std::size_t>(r)] =
                0.5 * (size - 1) / static_cast<double>(size - 2);
        const LocalInput skewed =
            localInput(bodies, heldBy(rank, bodyCount, shares));
        FmmDistributedOptions options = redistributedOptions();
        const double mean = static_cast<double>(bodyCount) / size;
        options.maxGravityOwnerBytesPerRank =
            static_cast<std::size_t>(2.0 * mean) * 64;
        unsigned long long rankZeroHeld = rank == 0 ? skewed.indices.size() : 0;
        MPI_Bcast(&rankZeroHeld, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        const bool returnExceedsOwnerLimit = rankZeroHeld * 48 >
            options.maxGravityOwnerBytesPerRank;
        metrics.require("skewed_return_exceeds_owner_limit",
                        returnExceedsOwnerLimit);
        bool threw = false;
        std::vector<Vector3D> acceleration;
        try
        {
            DistributedFmmGravityCalculator skewedCalculator(gravityOptions(),
                                                             options);
            skewedCalculator.solve(skewed.positions, skewed.masses, skewed.ids,
                                   lower0, upper, acceleration);
        }
        catch(const UniversalError& error)
        {
            threw = true;
            if(rank == 0)
                std::cerr << "skewed solve failed: " << error.getErrorMessage()
                          << '\n';
        }
        const bool skewedOk = allRanks(!threw);
        metrics.require("owner_limit_spares_result_return", skewedOk);
        if(skewedOk)
        {
            const double skewedError = directError(bodies, skewed, acceleration);
            metrics.value("skewed_direct_error", skewedError);
            metrics.require("skewed_direct",
                            skewedError < kClusteredDirectTolerance);
        }
    }

    // 5. The record exchange itself: a correct prediction passes, a wrong one
    // or an over-limit receive throws on every rank with the right diagnosis.
    {
        const std::vector<std::vector<Record>> send = recordsToSend(rank, size);
        const std::uint64_t expected =
            static_cast<std::uint64_t>(size) * static_cast<std::uint64_t>(rank + 1);
        const std::vector<Record> received = FmmRecordExchange::exchangeByRank(
            send, MPI_COMM_WORLD, "record exchange test", expected);
        // Every origin must deliver exactly records 0..rank, once each.
        bool contentOk = received.size() == expected;
        std::vector<std::vector<int>> seen(
            static_cast<std::size_t>(size),
            std::vector<int>(static_cast<std::size_t>(rank + 1), 0));
        for(const Record& record : received)
        {
            const double index = record.payload -
                1000.0 * static_cast<double>(record.origin);
            const bool valid = record.origin < static_cast<std::uint64_t>(size) &&
                index >= 0.0 && index <= rank && index == std::floor(index);
            contentOk = contentOk && valid;
            if(valid)
                ++seen[static_cast<std::size_t>(record.origin)]
                      [static_cast<std::size_t>(index)];
        }
        for(const std::vector<int>& origin : seen)
            for(int count : origin)
                contentOk = contentOk && count == 1;
        metrics.require("record_exchange_delivers", allRanks(contentOk));

        std::string mismatchReport;
        bool mismatchThrew = false;
        try
        {
            (void) FmmRecordExchange::exchangeByRank(
                send, MPI_COMM_WORLD, "record exchange test",
                rank == 1 ? expected + 1 : expected);
        }
        catch(const UniversalError& error)
        {
            mismatchThrew = true;
            std::ostringstream report;
            reportError(error, report);
            mismatchReport = report.str();
        }
        metrics.require("record_exchange_mismatch_detected",
                        allRanks(mismatchThrew &&
                                 reportField(mismatchReport,
                                     "ranks_with_count_mismatch") == "1" &&
                                 reportField(mismatchReport,
                                     "ranks_with_malformed_counts") == "0"));

        std::string limitReport;
        bool limitThrew = false;
        const std::size_t largest =
            static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
        try
        {
            (void) FmmRecordExchange::exchangeByRank(
                send, MPI_COMM_WORLD, "record exchange test", expected,
                sizeof(Record) * (largest - 1));
        }
        catch(const UniversalError& error)
        {
            limitThrew = true;
            std::ostringstream report;
            reportError(error, report);
            limitReport = report.str();
        }
        metrics.require("record_exchange_limit_detected",
                        allRanks(limitThrew &&
                                 reportField(limitReport, "ranks_over_limit") == "1" &&
                                 reportField(limitReport,
                                     "largest_receiver_rank") ==
                                     std::to_string(size - 1) &&
                                 reportField(limitReport,
                                     "ranks_with_count_mismatch") == "0"));
    }

    // 6. Stale dyadic lattice without redistribution: after a domain change
    // one rank's bodies leave its retained root while the others stay inside
    // theirs.  Every rank must move to the new lattice, or the LET plan sees
    // two lattice IDs and rejects the source roots.
    {
        FmmDistributedOptions options = redistributedOptions();
        options.spatiallyRedistributeForGravity = false;
        std::vector<Body> spread(bodyCount);
        const std::size_t perRank = kBodiesPerActiveRank;
        for(std::size_t k = 0; k < bodyCount; ++k)
        {
            const double owner = static_cast<double>(k / perRank);
            spread[k].position = Vector3D(
                -0.8 + 1.6 * owner / std::max(1, size - 2) +
                    0.05 * (2.0 * unitSample(11 * k) - 1.0),
                0.3 * (2.0 * unitSample(11 * k + 1) - 1.0),
                0.3 * (2.0 * unitSample(11 * k + 2) - 1.0));
            spread[k].mass = 1.0 / static_cast<double>(bodyCount);
        }
        std::vector<std::size_t> own;
        if(rank < size - 1)
            for(std::size_t k = rank * perRank; k < (rank + 1) * perRank; ++k)
                own.push_back(k);
        const LocalInput before = localInput(spread, own);
        DistributedFmmGravityCalculator owned(gravityOptions(), options);
        std::vector<Vector3D> acceleration;
        owned.solve(before.positions, before.masses, before.ids, lower0, upper,
                    acceleration);
        // Rank 0's bodies move into the newly added slab, outside the old
        // domain and therefore outside its old root.
        std::vector<Body> shifted = spread;
        for(std::size_t k = 0; k < perRank; ++k)
            shifted[k].position.x = -1.2 + 0.05 * unitSample(13 * k);
        const LocalInput after = localInput(shifted, own);
        bool threw = false;
        try
        {
            owned.solve(after.positions, after.masses, after.ids, lower1,
                        upper, acceleration);
        }
        catch(const UniversalError& error)
        {
            threw = true;
            if(rank == 0)
                std::cerr << "lattice scenario failed: "
                          << error.getErrorMessage() << '\n';
        }
        const bool latticeOk = allRanks(!threw);
        metrics.require("lattice_partial_rebuild_no_throw", latticeOk);
        if(latticeOk)
        {
            const FmmSolveStats stats = owned.stats();
            metrics.require("lattice_all_roots_rebuilt",
                            stats.gravityDomainChanged &&
                            stats.ranksWithRootGeometryChange ==
                                static_cast<std::size_t>(size));
            const double latticeError = directError(shifted, after, acceleration);
            metrics.value("lattice_direct_error", latticeError);
            metrics.require("lattice_direct", latticeError < 1e-3);
        }
    }

    // 7. Option validation for the new controls, including cross-rank
    // consistency.
    {
        FmmDistributedOptions options = redistributedOptions();
        bool ok = !constructionRejected(options);
        options.gravityOwnerImbalanceTrigger = 1.0;
        ok = ok && constructionRejected(options);
        options.gravityOwnerImbalanceTrigger =
            std::numeric_limits<double>::quiet_NaN();
        ok = ok && constructionRejected(options);
        options = redistributedOptions();
        options.maxGravityOwnerBytesPerRank = 63;
        ok = ok && constructionRejected(options);
        options.maxGravityOwnerBytesPerRank =
            static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
        ok = ok && constructionRejected(options);
        options = redistributedOptions();
        options.gravityOwnerImbalanceTrigger = rank == 0 ? 9.0 : 8.0;
        ok = ok && constructionRejected(options);
        options = redistributedOptions();
        options.maxGravityOwnerBytesPerRank =
            rank == 0 ? 1024 * 1024 : 2048 * 1024;
        ok = ok && constructionRejected(options);
        metrics.require("options_validated", allRanks(ok));
    }

    metrics.text << "pass " << (metrics.pass ? 1 : 0) << '\n';
    if(rank == 0)
    {
        std::ofstream output("fmm_gravity_mpi_redistribute_metrics.txt");
        output << metrics.text.str();
        std::cout << metrics.text.str();
    }
    MPI_Finalize();
    return metrics.pass ? 0 : 1;
}
