#ifdef RICH_MPI

#include "RankSync.hpp"

std::vector<rank_t> GetRanksOrder(const MPI_Comm &comm)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    rank_t even_comm_size = (size / 2) * 2;

    bool sizeIsEven = (even_comm_size == size);
    std::vector<rank_t> ranks(even_comm_size);
    std::iota(ranks.begin(), ranks.end(), 0);
        
    rank_t lastRank = size - 1;

    std::vector<rank_t> myOrder;
    if(not sizeIsEven)
    {
        if(rank != lastRank)
        {
            myOrder.push_back(rank);
        }
    }
    else
    {
        myOrder.push_back(rank);
    }
    
    rank_t P = even_comm_size / 2;

    for(rank_t i = 0; i < (even_comm_size - 1); i++)
    {
        for(rank_t j = 0; j < P; j++)
        {
            rank_t a = ranks[j];
            rank_t b = ranks[even_comm_size - 1 - j];
            if(rank == a)
            {
                myOrder.push_back(b);
            }
            if(rank == b)
            {
                myOrder.push_back(a);
            }
        }

        std::vector<rank_t> new_ranks = {ranks[0], ranks.back()};
        new_ranks.insert(new_ranks.end(), ranks.begin() + 1, ranks.end() - 1);
        ranks = std::move(new_ranks);
    }

    if(sizeIsEven or (not sizeIsEven and rank != lastRank))
    {
        assert(myOrder.size() == even_comm_size);
    }
    return myOrder;
}

void ForEachRankSync(const MPI_Comm &comm, const std::vector<rank_t> &order, const std::function<void(rank_t)> &func)
{
    rank_t rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for(const rank_t &_rank : order)
    {
        MPI_Barrier(comm);
        func(_rank);
    }

    rank_t lastRank = size - 1;
    rank_t even_comm_size = (size / 2) * 2;
    bool sizeIsEven = (even_comm_size == size);

    if(not sizeIsEven)
    {
        if(rank == lastRank)
        {
            for(rank_t i = 0; i < even_comm_size; i++)
            {
                MPI_Barrier(comm); // match with their barrier
            }
        }
        // now everybody need to synchronize with rank `N-1`
        for(rank_t _rank = 0; _rank < size; _rank++)
        {
            MPI_Barrier(comm);
            if(rank == lastRank or rank == _rank)
            {
                rank_t otherRank = (rank == _rank)? lastRank : _rank;
                func(otherRank);
            }
        }
    }
    MPI_Barrier(comm);
}

#endif // RICH_MPI