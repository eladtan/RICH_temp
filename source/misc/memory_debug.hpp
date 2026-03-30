#ifndef MEMORY_DEBUG_HPP
#define MEMORY_DEBUG_HPP

#ifdef MEMORY_DEBUG

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace memory_debug {
inline void print_memory(const std::string& label)
{
    double rss_gb = 0.0;
    std::ifstream status("/proc/self/status");
    std::string line;
    while(std::getline(status, line))
    {
        if(line.compare(0, 6, "VmRSS:") == 0)
        {
            std::size_t pos = line.find_first_of("0123456789");
            if(pos != std::string::npos)
            {
                rss_gb = std::stod(line.substr(pos)) / 1048576.0;
            }
            break;
        }
    }
    double max_rss = rss_gb, sum_rss = rss_gb;
    int rank = 0;
#ifdef RICH_MPI
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Allreduce(MPI_IN_PLACE, &max_rss, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_rss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Per-node memory: split communicator by shared-memory domain (= physical node)
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    double node_rss = rss_gb;
    MPI_Allreduce(MPI_IN_PLACE, &node_rss, 1, MPI_DOUBLE, MPI_SUM, node_comm);

    std::vector<int> node_ranks(node_size);
    MPI_Gather(&rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT, 0, node_comm);

    int is_leader = (node_rank == 0) ? 1 : 0;
    int num_nodes = 0;
    MPI_Reduce(&is_leader, &num_nodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    struct { double val; int idx; } local_node, max_node;
    local_node.val = (node_rank == 0) ? node_rss : -1.0;
    local_node.idx = rank;
    MPI_Allreduce(&local_node, &max_node, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(proc_name, &name_len);
    std::string hostname(proc_name, static_cast<std::size_t>(name_len));

    if(rank == 0)
    {
        std::string node_info;
        if(max_node.idx == 0)
        {
            node_info = hostname + " (ranks:";
            for(int i = 0; i < node_size; ++i)
            {
                if(i > 0)
                {
                    node_info += ",";
                }
                node_info += std::to_string(node_ranks[i]);
            }
            node_info += ")";
        }
        else
        {
            int info_len;
            MPI_Recv(&info_len, 1, MPI_INT, max_node.idx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            node_info.resize(static_cast<std::size_t>(info_len));
            MPI_Recv(&node_info[0], info_len, MPI_CHAR, max_node.idx, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::cerr << "[MEMORY] " << label
                  << ": max_rank=" << std::fixed << std::setprecision(3) << max_rss
                  << " GB, sum_all=" << sum_rss << " GB"
                  << ", max_node=" << max_node.val << " GB on " << node_info
                  << ", num_nodes=" << num_nodes
                  << std::endl;
    }
    else if(rank == max_node.idx)
    {
        std::string node_info = hostname + " (ranks:";
        for(int i = 0; i < node_size; ++i)
        {
            if(i > 0)
            {
                node_info += ",";
            }
            node_info += std::to_string(node_ranks[i]);
        }
        node_info += ")";
        int info_len = static_cast<int>(node_info.size());
        MPI_Send(&info_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(node_info.data(), info_len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&node_comm);
#else
    if(rank == 0)
    {
        std::cerr << "[MEMORY] " << label
                  << ": max_rank=" << std::fixed << std::setprecision(3) << max_rss
                  << " GB, sum_all=" << sum_rss << " GB" << std::endl;
    }
#endif
}
} // namespace memory_debug

#define MEMORY_DEBUG_PRINT(label) memory_debug::print_memory(label)

#else
#define MEMORY_DEBUG_PRINT(label) ((void)0)
#endif // MEMORY_DEBUG

#endif // MEMORY_DEBUG_HPP
