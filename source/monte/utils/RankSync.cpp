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

struct RoundInfo { rank_t partner = -1; int split_color = MPI_UNDEFINED; int key = 0; };

static std::pair<std::vector<RoundInfo>, int> BuildEdgeSchedule(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int send_count = static_cast<int>(new_neighbors.size());
    std::vector<int> all_counts(static_cast<size_t>(size));
    MPI_Allgather(&send_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<size_t>(size));
    displs[0] = 0;
    for(rank_t i = 1; i < size; i++)
        displs[static_cast<size_t>(i)] = displs[static_cast<size_t>(i - 1)] + all_counts[static_cast<size_t>(i - 1)];
    int total = displs[static_cast<size_t>(size - 1)] + all_counts[static_cast<size_t>(size - 1)];

    std::vector<rank_t> all_neighbors(static_cast<size_t>(total));
    MPI_Allgatherv(new_neighbors.data(), send_count, MPI_INT,
                   all_neighbors.data(), all_counts.data(), displs.data(), MPI_INT, comm);

    std::set<std::pair<rank_t, rank_t>> edge_set;
    for(rank_t r = 0; r < size; r++)
    {
        int offset = displs[static_cast<size_t>(r)];
        int count = all_counts[static_cast<size_t>(r)];
        for(int j = 0; j < count; j++)
        {
            rank_t nbr = all_neighbors[static_cast<size_t>(offset + j)];
            if(r != nbr)
            {
                edge_set.insert({std::min(r, nbr), std::max(r, nbr)});
            }
        }
    }

    std::vector<std::pair<rank_t, rank_t>> edges(edge_set.begin(), edge_set.end());
    size_t num_edges = edges.size();
    if(num_edges == 0)
        return {{}, 0};

    // Greedy edge coloring: at most 2*Delta - 1 colors
    std::vector<std::vector<bool>> vertex_used(static_cast<size_t>(size));
    std::vector<int> edge_color(num_edges);
    int num_colors = 0;

    for(size_t i = 0; i < num_edges; i++)
    {
        auto [u, v] = edges[i];
        auto &used_u = vertex_used[static_cast<size_t>(u)];
        auto &used_v = vertex_used[static_cast<size_t>(v)];
        size_t max_len = std::max(used_u.size(), used_v.size());

        int c = 0;
        for(; static_cast<size_t>(c) < max_len; c++)
        {
            bool taken_u = static_cast<size_t>(c) < used_u.size() && used_u[static_cast<size_t>(c)];
            bool taken_v = static_cast<size_t>(c) < used_v.size() && used_v[static_cast<size_t>(c)];
            if(!taken_u && !taken_v)
                break;
        }

        edge_color[i] = c;
        if(static_cast<size_t>(c) >= used_u.size()) used_u.resize(static_cast<size_t>(c + 1), false);
        if(static_cast<size_t>(c) >= used_v.size()) used_v.resize(static_cast<size_t>(c + 1), false);
        used_u[static_cast<size_t>(c)] = true;
        used_v[static_cast<size_t>(c)] = true;
        num_colors = std::max(num_colors, c + 1);
    }

    std::vector<RoundInfo> my_rounds(static_cast<size_t>(num_colors));
    for(size_t i = 0; i < num_edges; i++)
    {
        auto [u, v] = edges[i];
        int c = edge_color[i];
        if(rank == u)
        {
            my_rounds[static_cast<size_t>(c)] = {v, static_cast<int>(i), 0};
        }    
        else if(rank == v)
        {
            my_rounds[static_cast<size_t>(c)] = {u, static_cast<int>(i), 1};
        }
    }

    return {std::move(my_rounds), num_colors};
}

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors,
                           const std::function<void(rank_t, MPI_Comm)> &func, bool withBarrier)
{
    auto [my_rounds, num_colors] = BuildEdgeSchedule(comm, new_neighbors);

    for(int c = 0; c < num_colors; c++)
    {
        if(withBarrier)
        {
            MPI_Barrier(comm);
        }
        const RoundInfo &info = my_rounds[static_cast<size_t>(c)];

        MPI_Comm pair_comm;
        MPI_Comm_split(comm, info.split_color, info.key, &pair_comm);

        if(info.partner >= 0)
        {
            func(info.partner, pair_comm);
        }
    }
}

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors,
                           const std::function<void(rank_t)> &func, bool withBarrier)
{
    auto [my_rounds, num_colors] = BuildEdgeSchedule(comm, new_neighbors);

    for(int c = 0; c < num_colors; c++)
    {
        if(withBarrier) MPI_Barrier(comm);

        const RoundInfo &info = my_rounds[static_cast<size_t>(c)];
        if(info.partner >= 0)
        {
            func(info.partner);
        }
    }
}
#endif // RICH_MPI