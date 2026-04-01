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

void ForEachRankSync(const MPI_Comm &comm, const std::vector<rank_t> &order, const std::function<void(rank_t)> &func, bool use_barrier)
{
    rank_t rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for(const rank_t &_rank : order)
    {
        if(use_barrier) MPI_Barrier(comm);
        func(_rank);
    }

    rank_t lastRank = size - 1;
    rank_t even_comm_size = (size / 2) * 2;
    bool sizeIsEven = (even_comm_size == size);

    if(not sizeIsEven)
    {
        if(use_barrier)
        {
            if(rank == lastRank)
            {
                for(rank_t i = 0; i < even_comm_size; i++)
                {
                    MPI_Barrier(comm); // match with their barrier
                }
            }
        }
        // now everybody need to synchronize with rank `N-1`
        for(rank_t _rank = 0; _rank < size; _rank++)
        {
            if(use_barrier) MPI_Barrier(comm);
            if(rank == lastRank or rank == _rank)
            {
                rank_t otherRank = (rank == _rank)? lastRank : _rank;
                func(otherRank);
            }
        }
    }
    MPI_Barrier(comm);
}

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors, const std::function<void(rank_t)> &func)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all new-neighbor sets (lightweight: total data = sum of |N_p|)
    int my_count = static_cast<int>(new_neighbors.size());
    std::vector<int> all_counts(size);
    MPI_Allgather(&my_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(size, 0);
    for(int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + all_counts[i - 1];
    }
    int total = displs[size - 1] + all_counts[size - 1];

    std::vector<rank_t> all_neighbors(total);
    MPI_Allgatherv(new_neighbors.data(), my_count, MPI_INT,
                   all_neighbors.data(), all_counts.data(), displs.data(),
                   MPI_INT, comm);

    // Build edge set (canonical: u < v, deduplicated)
    struct Edge { rank_t u, v; int color; };
    std::set<std::pair<rank_t, rank_t>> edge_set;
    for(rank_t r = 0; r < size; r++)
    {
        for(int j = displs[r]; j < displs[r] + all_counts[r]; j++)
        {
            rank_t nbr = all_neighbors[j];
            if(r != nbr)
            {
                edge_set.insert({std::min(r, nbr), std::max(r, nbr)});
            }
        }
    }
    std::vector<Edge> edges;
    edges.reserve(edge_set.size());
    for(const auto &[u, v] : edge_set)
    {
        edges.push_back({u, v, -1});
    }

    // Greedy edge coloring (only touches vertices that appear in edges)
    std::unordered_map<rank_t, std::vector<bool>> used_colors;
    int num_colors = 0;

    for(auto &edge : edges)
    {
        auto &cu = used_colors[edge.u];
        auto &cv = used_colors[edge.v];
        int c = 0;
        while((c < static_cast<int>(cu.size()) and cu[c]) or
               (c < static_cast<int>(cv.size()) and cv[c]))
        {
            c++;
        }

        edge.color = c;
        if(c >= static_cast<int>(cu.size()))
        {
            cu.resize(c + 1, false);
        }
        if(c >= static_cast<int>(cv.size()))
        {
            cv.resize(c + 1, false);
        }
        cu[c] = true;
        cv[c] = true;
        num_colors = std::max(num_colors, c + 1);
    }

    // Build my schedule: (color → partner)
    std::vector<std::pair<int, rank_t>> my_schedule;
    for(const auto &edge : edges)
    {
        if(edge.u == rank)
        {
            my_schedule.push_back({edge.color, edge.v});
        }
        else if(edge.v == rank)
        {
            my_schedule.push_back({edge.color, edge.u});
        }
    }
    std::sort(my_schedule.begin(), my_schedule.end(), [](const std::pair<int, rank_t> &a, const std::pair<int, rank_t> &b)
    {
        return a.first < b.first;
    });

    // Execute: one round per color
    MPI_Allreduce(MPI_IN_PLACE, &num_colors, 1, MPI_INT, MPI_MAX, comm);
    for(int c = 0; c < num_colors; c++)
    {
        MPI_Barrier(comm);
        for(const auto &[color, partner] : my_schedule)
        {
            if(color == c)
            {
                func(partner);
                break;
            }
        }
    }
}
#endif // RICH_MPI