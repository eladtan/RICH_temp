#ifndef FMM_RECORD_EXCHANGE_HPP
#define FMM_RECORD_EXCHANGE_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "misc/universal_error.hpp"

namespace FmmRecordExchange
{
// Marks an exchange whose receive total has no independent prediction.
constexpr std::uint64_t kUnknownRecordCount =
    std::numeric_limits<std::uint64_t>::max();

// Byte counts and displacements travel as MPI int, so no rank may receive more.
constexpr std::size_t kMaxTransportBytes =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

// Collective MPI_Alltoallv of trivially copyable records, one vector per peer.
// Each rank checks its receive counts and the verdict is reduced, so every rank
// throws the same error or none does:
//  - every per-peer byte count is non-negative and a whole number of records;
//  - the receive total fits maxReceiveBytes (at most kMaxTransportBytes);
//  - the received record count equals expectedRecords, when it is known.
// The last check is what separates a genuine load imbalance from corrupted
// counts, which can be non-negative and record-aligned.
template<typename T>
std::vector<T> exchangeByRank(
    const std::vector<std::vector<T>>& sendByRank,
    const MPI_Comm& comm,
    const char* context,
    std::uint64_t expectedRecords = kUnknownRecordCount,
    std::size_t maxReceiveBytes = kMaxTransportBytes)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "FmmRecordExchange records must be wire-copyable");
    int size = 1;
    int rank = 0;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if(sendByRank.size() != static_cast<std::size_t>(size))
        throw UniversalError(std::string(context) + ": invalid peer table");
    if(maxReceiveBytes < sizeof(T) || maxReceiveBytes > kMaxTransportBytes)
    {
        UniversalError error(std::string(context) +
                             ": receive limit outside [record size, INT_MAX]");
        error.addEntry("receive_limit_bytes", maxReceiveBytes);
        error.addEntry("record_bytes", sizeof(T));
        throw error;
    }
    std::vector<int> sendCounts(static_cast<std::size_t>(size), 0);
    std::vector<int> receiveCounts(static_cast<std::size_t>(size), 0);
    std::vector<int> sendDisplacements(static_cast<std::size_t>(size), 0);
    std::vector<int> receiveDisplacements(static_cast<std::size_t>(size), 0);
    std::size_t totalSendRecords = 0;
    bool localValid = true;
    for(int peer = 0; peer < size; ++peer)
    {
        const std::size_t count = sendByRank[static_cast<std::size_t>(peer)].size();
        localValid = localValid && count <= kMaxTransportBytes / sizeof(T) &&
            totalSendRecords <= kMaxTransportBytes / sizeof(T) - count;
        if(!localValid)
            break;
        sendCounts[static_cast<std::size_t>(peer)] = static_cast<int>(
            count * sizeof(T));
        sendDisplacements[static_cast<std::size_t>(peer)] = static_cast<int>(
            totalSendRecords * sizeof(T));
        totalSendRecords += count;
    }
    int localValidInt = localValid ? 1 : 0;
    int globalValidInt = 0;
    MPI_Allreduce(&localValidInt, &globalValidInt, 1, MPI_INT, MPI_LAND, comm);
    if(globalValidInt == 0)
    {
        UniversalError error(std::string(context) +
                             ": send byte counts exceed MPI int");
        std::size_t localSendRecords = 0;
        for(const std::vector<T>& peer : sendByRank)
            localSendRecords += peer.size();
        error.addEntry("record_bytes", sizeof(T));
        error.addEntry("local_rank", rank);
        error.addEntry("local_send_records", localSendRecords);
        error.addEntry("local_send_valid", localValid ? 1 : 0);
        throw error;
    }

    std::vector<T> sendFlat;
    sendFlat.reserve(totalSendRecords);
    for(const std::vector<T>& peer : sendByRank)
        sendFlat.insert(sendFlat.end(), peer.begin(), peer.end());
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                 receiveCounts.data(), 1, MPI_INT, comm);

    // Sum every well-formed count before applying any limit, so a rejected
    // exchange still reports its true size.
    unsigned long long receiveBytes = 0;
    bool malformed = false;
    for(int peer = 0; peer < size; ++peer)
    {
        const int bytes = receiveCounts[static_cast<std::size_t>(peer)];
        if(bytes < 0 || bytes % static_cast<int>(sizeof(T)) != 0)
        {
            malformed = true;
            continue;
        }
        receiveBytes += static_cast<unsigned long long>(bytes);
    }
    const bool overLimit = receiveBytes > maxReceiveBytes;
    const bool mismatch = !malformed &&
        expectedRecords != kUnknownRecordCount &&
        receiveBytes / sizeof(T) != expectedRecords;
    const int localFlags[3] = {malformed ? 1 : 0, mismatch ? 1 : 0,
                               overLimit ? 1 : 0};
    int globalFlags[3] = {0, 0, 0};
    MPI_Allreduce(localFlags, globalFlags, 3, MPI_INT, MPI_SUM, comm);
    if(globalFlags[0] != 0 || globalFlags[1] != 0 || globalFlags[2] != 0)
    {
        struct
        {
            double value;
            int rank;
        } localWorst{static_cast<double>(receiveBytes), rank}, worst{0.0, 0};
        MPI_Allreduce(&localWorst, &worst, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
        // Corrupted counts can also overflow the limit, so a count mismatch is
        // reported ahead of the overflow it may have caused.
        const char* reason = globalFlags[0] != 0 ?
            "invalid receive byte counts" : globalFlags[1] != 0 ?
            "received record count differs from the sender prediction" :
            "receive total exceeds the per-rank limit";
        UniversalError error(std::string(context) + ": " + reason);
        // The largest receiver is the culprit of an over-limit exchange; for a
        // malformed or mismatched count, each rank's local entries name it.
        error.addEntry("record_bytes", sizeof(T));
        error.addEntry("largest_receiver_rank", worst.rank);
        error.addEntry("largest_receive_bytes", worst.value);
        error.addEntry("receive_limit_bytes", maxReceiveBytes);
        error.addEntry("ranks_with_malformed_counts", globalFlags[0]);
        error.addEntry("ranks_with_count_mismatch", globalFlags[1]);
        error.addEntry("ranks_over_limit", globalFlags[2]);
        error.addEntry("local_rank", rank);
        error.addEntry("local_receive_bytes", receiveBytes);
        error.addEntry("local_counts_malformed", malformed ? 1 : 0);
        if(expectedRecords != kUnknownRecordCount)
            error.addEntry("local_expected_records", expectedRecords);
        throw error;
    }

    std::size_t totalReceiveBytes = 0;
    for(int peer = 0; peer < size; ++peer)
    {
        receiveDisplacements[static_cast<std::size_t>(peer)] =
            static_cast<int>(totalReceiveBytes);
        totalReceiveBytes += static_cast<std::size_t>(
            receiveCounts[static_cast<std::size_t>(peer)]);
    }
    std::vector<T> received(totalReceiveBytes / sizeof(T));
    MPI_Alltoallv(
        sendFlat.empty() ? nullptr : sendFlat.data(),
        sendCounts.data(), sendDisplacements.data(), MPI_BYTE,
        received.empty() ? nullptr : received.data(),
        receiveCounts.data(), receiveDisplacements.data(), MPI_BYTE, comm);
    return received;
}
}

#endif // RICH_MPI

#endif // FMM_RECORD_EXCHANGE_HPP
