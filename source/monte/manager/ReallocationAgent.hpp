#ifndef REALLOCATION_AGENT_HPP
#define REALLOCATION_AGENT_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include "mpi/mpi_commands.hpp"

class ReallocationAgent
{
public:
    ReallocationAgent(const MPI_Comm &comm, const std::function<void(rank_t)> &reallocationFunction);

    ~ReallocationAgent() = default;

    rank_t ShouldReallocate(void);

    void RequestReallocation(rank_t fromRank);

    void HandleWaitingReallocations(void);

    void HandleAllWaitingReallocations(void);

private:
    MPI_Comm comm;
    rank_t rank;
    rank_t size; // todo: unnecessary
    std::vector<std::pair<rank_t, double>> incoming;
    std::function<void(rank_t)> reallocationFunction;
};

#endif // RICH_MPI
#endif // REALLOCATION_AGENT_HPP