#ifndef REALLOCATION_AGENT_HPP
#define REALLOCATION_AGENT_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include "mpi/mpi_commands.hpp"

class ReallocationAgent
{
public:
    ReallocationAgent(const MPI_Comm &comm, const std::function<void(rank_t)> &reallocationFunction);

    ~ReallocationAgent();

    rank_t ShouldReallocate(void) const;

    void RequestReallocation(rank_t fromRank);

    void HandleWaitingReallocations(void);

private:
    MPI_Comm comm;
    rank_t rank;
    rank_t size; // todo: unnecessary
    std::function<void(rank_t)> reallocationFunction;
    
    // rank_t *who_to_reallocate;
    // rank_t *wants_for_reallocation;
    // MPI_Win win;
    // MPI_Win wants_win;

    void RequestReallocation(rank_t fromRank, rank_t toRank);

    void FreeReallocation(rank_t rank);

    void SetMyWant(rank_t rank);

    void LockForChanges(void);

    void UnlockForChanges(void);
};

#endif // RICH_MPI
#endif // REALLOCATION_AGENT_HPP