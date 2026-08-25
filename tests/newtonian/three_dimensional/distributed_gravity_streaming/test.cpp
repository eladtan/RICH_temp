#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef RICH_MPI
#include <mpi.h>

#include "source/3D/gravity/DistributedGravityCalculator.hpp"
#endif

#ifdef RICH_MPI
namespace
{
constexpr int required_ranks = 4;
constexpr int sources_per_rank = 256;
constexpr double direct_tolerance = 1e-11;
constexpr double mode_tolerance = 1e-14;

std::vector<Vector3D> sourcesForRank(int const rank)
{
    std::vector<Vector3D> sources;
    sources.reserve(sources_per_rank);
    for(int index = 0; index < sources_per_rank; ++index)
    {
        int const ix = index % 8;
        int const iy = (index / 8) % 8;
        int const iz = index / 64;
        double const rank_offset = 0.04 * static_cast<double>(rank);
        sources.emplace_back(
            -1.4 + (static_cast<double>(ix) + 0.31 + rank_offset) * 2.8 / 8.0,
            -1.4 + (static_cast<double>(iy) + 0.43 + 0.5 * rank_offset) * 2.8 / 8.0,
            -1.2 + (static_cast<double>(iz) + 0.37 + 0.25 * rank_offset) * 2.4 / 4.0);
    }
    return sources;
}

std::vector<gravity_result_t> massesForRank(int const rank)
{
    std::vector<gravity_result_t> masses(sources_per_rank);
    for(int index = 0; index < sources_per_rank; ++index)
        masses[index] = 1.0 + 1e-3 *
            static_cast<double>(rank * sources_per_rank + index);
    return masses;
}

std::vector<Vector3D> targetsForRank(
    int const rank, std::vector<Vector3D> const& local_sources)
{
    if(rank == 0)
        return {local_sources[73], local_sources[182]};
    if(rank == 1)
        return {local_sources[91]};
    if(rank == 2)
        return {local_sources[137]};
    // Rank 3 deliberately owns sources but has no active gravity targets.
    return {};
}

double norm(Vector3D const& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

bool finite(Vector3D const& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool sameBits(double const first, double const second)
{
    std::uint64_t first_bits = 0;
    std::uint64_t second_bits = 0;
    static_assert(sizeof(first_bits) == sizeof(first),
                  "Unexpected double width");
    std::memcpy(&first_bits, &first, sizeof(first));
    std::memcpy(&second_bits, &second, sizeof(second));
    return first_bits == second_bits;
}

bool sameBits(Vector3D const& first, Vector3D const& second)
{
    return sameBits(first.x, second.x) &&
           sameBits(first.y, second.y) &&
           sameBits(first.z, second.z);
}

bool sameFlatValue(MassedValue<Vector3D> const& first,
                   MassedValue<Vector3D> const& second)
{
    char first_bytes[MassedValue<Vector3D>::FLAT_BYTE_SIZE];
    char second_bytes[MassedValue<Vector3D>::FLAT_BYTE_SIZE];
    first.dumpFlat(first_bytes);
    second.dumpFlat(second_bytes);
    return std::memcmp(first_bytes, second_bytes, sizeof(first_bytes)) == 0;
}

int exerciseReceiveOnlyFlatExchange(bool const peer_payload, int const rank,
                                    int& is_receive_only_rank)
{
    constexpr int sender_rank = 0;
    constexpr int receiver_rank = 2;
    MassedValue<Vector3D> const expected(
        Vector3D(0.25, -0.5, 0.75), 3.5);

    std::vector<std::vector<MassedValue<Vector3D>>> send_values(
        required_ranks);
    std::vector<int> neighbors;
    if(rank == sender_rank)
    {
        send_values[receiver_rank].push_back(expected);
        neighbors.push_back(receiver_rank);
    }
    else if(rank == receiver_rank)
        neighbors.push_back(sender_rank);

    std::size_t sent_values = 0;
    for(std::vector<MassedValue<Vector3D>> const& destination : send_values)
        sent_values += destination.size();

    FlatSparseHandle handle = MPI_flat_sparse_pack_and_post_counts(
        send_values, neighbors, MPI_COMM_WORLD);
    int received_values = 0;
    int correct = 1;
    auto acceptValue = [&](rank_t const source_rank,
                           MassedValue<Vector3D> const& value)
    {
        ++received_values;
        correct = correct && source_rank == sender_rank &&
                  sameFlatValue(value, expected);
    };

    if(peer_payload)
    {
        MPI_flat_sparse_post_peer_payload(handle, MPI_COMM_WORLD);
        MPI_flat_sparse_wait_payload(handle);
        MPI_flat_sparse_visit_received_by_peer<MassedValue<Vector3D>>(
            handle, acceptValue);
    }
    else
    {
        MPI_flat_sparse_post_payload(handle, MPI_COMM_WORLD);
        std::vector<std::vector<MassedValue<Vector3D>>> received_by_rank =
            MPI_flat_sparse_wait<MassedValue<Vector3D>>(handle);
        for(int source_rank = 0; source_rank < required_ranks; ++source_rank)
            for(MassedValue<Vector3D> const& value :
                received_by_rank[source_rank])
                acceptValue(source_rank, value);
    }

    is_receive_only_rank = sent_values == 0 && received_values > 0 ? 1 : 0;
    int const expected_received = rank == receiver_rank ? 1 : 0;
    int const expected_receive_only = rank == receiver_rank ? 1 : 0;
    return correct && received_values == expected_received &&
           is_receive_only_rank == expected_receive_only;
}

Vector3D directAcceleration(Vector3D const& target)
{
    long double acceleration[3] = {0, 0, 0};
    for(int source_rank = 0; source_rank < required_ranks; ++source_rank)
    {
        std::vector<Vector3D> const sources = sourcesForRank(source_rank);
        std::vector<gravity_result_t> const masses =
            massesForRank(source_rank);
        for(std::size_t source = 0; source < sources.size(); ++source)
        {
            long double const dx =
                static_cast<long double>(sources[source].x) - target.x;
            long double const dy =
                static_cast<long double>(sources[source].y) - target.y;
            long double const dz =
                static_cast<long double>(sources[source].z) - target.z;
            long double const distance_squared = dx * dx + dy * dy + dz * dz;
            if(distance_squared == 0)
                continue;
            long double const inverse_distance_cubed = 1.0L /
                (distance_squared * std::sqrt(distance_squared));
            long double const factor =
                static_cast<long double>(masses[source]) *
                inverse_distance_cubed;
            acceleration[0] += dx * factor;
            acceleration[1] += dy * factor;
            acceleration[2] += dz * factor;
        }
    }
    return Vector3D(static_cast<double>(acceleration[0]),
                    static_cast<double>(acceleration[1]),
                    static_cast<double>(acceleration[2]));
}

double scaledDifference(Vector3D const& first, Vector3D const& second)
{
    return norm(first - second) / std::max(1.0, norm(second));
}

bool sameCounters(DistributedGravityCalculator::SolveTiming const& legacy,
                  DistributedGravityCalculator::SolveTiming const& streamed)
{
    return legacy.sentValues == streamed.sentValues &&
           legacy.receivedValues == streamed.receivedValues &&
           legacy.remoteFlatNodes == streamed.remoteFlatNodes &&
           legacy.activeExchangeNeighbors ==
               streamed.activeExchangeNeighbors;
}

bool emptyTargetTimingAccounts(
    DistributedGravityCalculator::SolveTiming const& timing)
{
    double const accounted = timing.exchangePlan +
        timing.countPostAndRelease + timing.payloadWaitAndUnpack;
    double const tolerance = 1e-10 * std::max(1.0, std::abs(timing.total));
    return timing.total > 0.0 &&
           std::abs(accounted - timing.total) <= tolerance &&
           timing.localFlatBuild == 0.0 && timing.localWalk == 0.0 &&
           timing.remoteTreeBuild == 0.0 &&
           timing.remoteFlatBuild == 0.0 && timing.remoteWalk == 0.0 &&
           timing.remoteFlatNodes == 0.0;
}

void writeRankCountFailure(int const actual_ranks)
{
    std::ofstream output("distributed_gravity_streaming_metrics.txt");
    output << "ranks " << actual_ranks << '\n';
    output << "required_ranks " << required_ranks << '\n';
    output << "pass 0\n";
}
}
#endif

int main(int argc, char** argv)
{
#ifndef RICH_MPI
    (void) argc;
    (void) argv;
    std::cerr << "distributed_gravity_streaming requires an MPI build\n";
    return 1;
#else
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != required_ranks)
    {
        if(rank == 0)
            writeRankCountFailure(size);
        MPI_Finalize();
        return 1;
    }

    try
    {
        int receive_only_rank_legacy = 0;
        int receive_only_rank_streamed = 0;
        int flat_receive_only_legacy = exerciseReceiveOnlyFlatExchange(
            false, rank, receive_only_rank_legacy);
        int flat_receive_only_streamed = exerciseReceiveOnlyFlatExchange(
            true, rank, receive_only_rank_streamed);

        std::vector<Vector3D> const local_sources = sourcesForRank(rank);
        std::vector<gravity_result_t> const local_masses = massesForRank(rank);
        std::vector<Vector3D> const local_targets =
            targetsForRank(rank, local_sources);

        Vector3D const domain_lower(-2.0, -2.0, -2.0);
        Vector3D const domain_upper(2.0, 2.0, 2.0);
        // A strict opening angle makes this small tree a direct-force oracle
        // while still exercising the flat sparse remote payload exchange.
        DistributedGravityCalculator calculator(
            local_sources, local_masses, domain_lower, domain_upper,
            1e-6, false, MPI_COMM_WORLD);

        int environment_ok =
            (setenv("RICH_INDIVIDUAL_PERF_TRACE", "1", 1) == 0 &&
             setenv("RICH_GRAVITY_STREAM_REMOTE_PAYLOAD", "0", 1) == 0) ?
                1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &environment_ok, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(environment_ok == 0)
            throw std::runtime_error("Failed to select legacy payload mode");
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<Vector3D> const legacy =
            calculator.getAcceleration(local_targets);
        DistributedGravityCalculator::SolveTiming const legacy_counters =
            calculator.getLastSolveTiming();

        environment_ok =
            setenv("RICH_GRAVITY_STREAM_REMOTE_PAYLOAD", "1", 1) == 0 ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &environment_ok, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(environment_ok == 0)
            throw std::runtime_error("Failed to select streaming payload mode");
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<Vector3D> const streamed =
            calculator.getAcceleration(local_targets);
        DistributedGravityCalculator::SolveTiming const streamed_counters =
            calculator.getLastSolveTiming();
        unsetenv("RICH_GRAVITY_STREAM_REMOTE_PAYLOAD");
        unsetenv("RICH_INDIVIDUAL_PERF_TRACE");

        int local_finite = legacy.size() == local_targets.size() &&
                           streamed.size() == local_targets.size() ? 1 : 0;
        int local_exact = local_finite;
        double local_mode_error = 0;
        double local_direct_error = 0;
        if(local_finite != 0)
            for(std::size_t target = 0; target < local_targets.size(); ++target)
            {
                Vector3D const direct = directAcceleration(local_targets[target]);
                local_finite = local_finite && finite(legacy[target]) &&
                    finite(streamed[target]) && finite(direct);
                local_exact = local_exact &&
                    sameBits(legacy[target], streamed[target]);
                local_mode_error = std::max(local_mode_error,
                    scaledDifference(streamed[target], legacy[target]));
                local_direct_error = std::max(local_direct_error,
                    std::max(scaledDifference(legacy[target], direct),
                             scaledDifference(streamed[target], direct)));
            }

        int global_finite = local_finite;
        int global_exact = local_exact;
        int global_counter_match =
            sameCounters(legacy_counters, streamed_counters) ? 1 : 0;
        int global_empty_target_timing =
            (!local_targets.empty() ||
             (emptyTargetTimingAccounts(legacy_counters) &&
              emptyTargetTimingAccounts(streamed_counters))) ? 1 : 0;
        int zero_target_ranks = local_targets.empty() ? 1 : 0;
        int zero_source_ranks = local_sources.empty() ? 1 : 0;
        double global_mode_error = local_mode_error;
        double global_direct_error = local_direct_error;
        double received_values = streamed_counters.receivedValues;

        MPI_Allreduce(MPI_IN_PLACE, &global_finite, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &global_exact, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &global_counter_match, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &global_empty_target_timing, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &flat_receive_only_legacy, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &flat_receive_only_streamed, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &receive_only_rank_legacy, 1, MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &receive_only_rank_streamed, 1, MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zero_target_ranks, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zero_source_ranks, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &global_mode_error, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &global_direct_error, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &received_values, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        int const payload_exercised = received_values > 0 ? 1 : 0;
        int const passed = global_finite && global_exact &&
            global_counter_match && global_empty_target_timing &&
            flat_receive_only_legacy && flat_receive_only_streamed &&
            receive_only_rank_legacy == 1 &&
            receive_only_rank_streamed == 1 && payload_exercised &&
            zero_target_ranks == 1 && zero_source_ranks == 0 &&
            global_mode_error <= mode_tolerance &&
            global_direct_error <= direct_tolerance;

        if(rank == 0)
        {
            std::ofstream output("distributed_gravity_streaming_metrics.txt");
            output.setf(std::ios::scientific);
            output << std::setprecision(17);
            output << "ranks " << size << '\n';
            output << "zero_source_ranks " << zero_source_ranks << '\n';
            output << "zero_target_ranks " << zero_target_ranks << '\n';
            output << "receive_only_ranks_legacy "
                   << receive_only_rank_legacy << '\n';
            output << "receive_only_ranks_streamed "
                   << receive_only_rank_streamed << '\n';
            output << "flat_receive_only_legacy "
                   << flat_receive_only_legacy << '\n';
            output << "flat_receive_only_streamed "
                   << flat_receive_only_streamed << '\n';
            output << "empty_target_timing_accounted "
                   << global_empty_target_timing << '\n';
            output << "finite " << global_finite << '\n';
            output << "mode_bitwise_exact " << global_exact << '\n';
            output << "counter_match " << global_counter_match << '\n';
            output << "payload_exercised " << payload_exercised << '\n';
            output << "received_values " << received_values << '\n';
            output << "mode_max_scaled_error " << global_mode_error << '\n';
            output << "direct_max_scaled_error " << global_direct_error << '\n';
            output << "pass " << passed << '\n';
            std::cout << "distributed_gravity_streaming ranks=" << size
                      << " zero_target_ranks=" << zero_target_ranks
                      << " receive_only_ranks="
                      << receive_only_rank_streamed
                      << " received_values=" << received_values
                      << " mode_exact=" << global_exact
                      << " direct_error=" << global_direct_error
                      << " pass=" << passed << std::endl;
        }

        MPI_Finalize();
        return passed ? 0 : 1;
    }
    catch(std::exception const& error)
    {
        std::cerr << "distributed_gravity_streaming rank=" << rank
                  << " exception=" << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
#endif
}
