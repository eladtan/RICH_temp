#ifndef MEMORY_DEBUG_HPP
#define MEMORY_DEBUG_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <malloc.h>
#include <unistd.h>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace memory_debug {

    inline void check_system_memory(const char *context)
    {
        #ifdef MEMORY_DEBUG
        std::ifstream meminfo("/proc/meminfo");
        if(!meminfo.is_open()) return;

        long long mem_free_kb = -1, buffers_kb = 0, cached_kb = 0;
        std::string line;
        while(std::getline(meminfo, line))
        {
            auto parse_value = [&](const char *prefix, long long &out) -> bool {
                std::string pfx(prefix);
                if(line.compare(0, pfx.size(), pfx) == 0)
                {
                    std::size_t pos = line.find_first_of("0123456789", pfx.size());
                    if(pos != std::string::npos)
                        out = std::stoll(line.substr(pos));
                    return true;
                }
                return false;
            };
            parse_value("MemFree:", mem_free_kb);
            parse_value("Buffers:", buffers_kb);
            parse_value("Cached:", cached_kb);
        }

        long long buff_cache_kb = buffers_kb + cached_kb;
        constexpr long long threshold_free_kb = 1024LL * 1024;
        constexpr long long threshold_cache_kb = 30LL * 1024 * 1024;

        bool low_free = (mem_free_kb >= 0 && mem_free_kb < threshold_free_kb);
        bool high_cache = (buff_cache_kb > threshold_cache_kb);

        if(low_free || high_cache)
        {
            int rank = 0;
            char hostname[256] = "unknown";
    #ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            int name_len;
            MPI_Get_processor_name(hostname, &name_len);
    #else
            gethostname(hostname, sizeof(hostname));
    #endif
            double free_mb = (mem_free_kb >= 0) ? mem_free_kb / 1024.0 : -1;
            double cache_gb = buff_cache_kb / (1024.0 * 1024.0);
            std::cerr << "[MEMORY WARNING] rank " << rank << " on " << hostname
                    << " (" << context << "): "
                    << "MemFree=" << std::fixed << std::setprecision(0) << free_mb << " MB, "
                    << "buff/cache=" << std::setprecision(1) << cache_gb << " GB";
            if(low_free)
                std::cerr << " [FREE BELOW 1024 MB]";
            if(high_cache)
                std::cerr << " [CACHE ABOVE 30 GB]";
            std::cerr << std::endl;
        }
        #endif // MEMORY_DEBUG
    }

} // namespace memory_debug

#ifdef MEMORY_DEBUG

#include <vector>

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
    // Cache the node communicator to avoid MPI resource leak from repeated split/free
    static MPI_Comm node_comm = MPI_COMM_NULL;
    static int node_rank = -1, node_size = -1;
    if(node_comm == MPI_COMM_NULL)
    {
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_size);
    }

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

#else
    if(rank == 0)
    {
        std::cerr << "[MEMORY] " << label
                  << ": max_rank=" << std::fixed << std::setprecision(3) << max_rss
                  << " GB, sum_all=" << sum_rss << " GB" << std::endl;
    }
#endif
}
#include <cstdio>
inline void dump_malloc_info(const std::string &output_dir, int cycle)
{
    int rank = 0, world_size = 1;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    // Rank 0: write detailed XML
    if(rank == 0)
    {
        std::string path = output_dir + "/malloc_info_" + std::to_string(cycle) + ".xml";
        FILE *f = fopen(path.c_str(), "w");
        if(f) { malloc_info(0, f); fclose(f); }
    }

    // All ranks: gather mallinfo2 stats via MPI_Reduce
    struct mallinfo2 mi = mallinfo2();
    double local[4];
    local[0] = static_cast<double>(mi.arena);
    local[1] = static_cast<double>(mi.uordblks);
    local[2] = static_cast<double>(mi.fordblks);
    local[3] = static_cast<double>(mi.hblkhd);

    double sum_vals[4], max_vals[4];
#ifdef RICH_MPI
    MPI_Reduce(local, sum_vals, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, max_vals, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    for(int i = 0; i < 4; ++i) { sum_vals[i] = local[i]; max_vals[i] = local[i]; }
#endif

    if(rank == 0)
    {
        std::string csv_path = output_dir + "/malloc_stats.csv";
        bool need_header = true;
        {
            std::ifstream probe(csv_path, std::ios::ate);
            if(probe.is_open() && probe.tellg() > 0)
                need_header = false;
        }
        std::ofstream csv(csv_path, std::ios::app);
        if(need_header)
            csv << "cycle,nranks,"
                << "sum_arena,sum_inuse,sum_free,sum_mmap,"
                << "max_arena,max_inuse,max_free,max_mmap,"
                << "r0_arena,r0_inuse,r0_free,r0_mmap\n";
        csv << cycle << "," << world_size
            << "," << static_cast<long long>(sum_vals[0])
            << "," << static_cast<long long>(sum_vals[1])
            << "," << static_cast<long long>(sum_vals[2])
            << "," << static_cast<long long>(sum_vals[3])
            << "," << static_cast<long long>(max_vals[0])
            << "," << static_cast<long long>(max_vals[1])
            << "," << static_cast<long long>(max_vals[2])
            << "," << static_cast<long long>(max_vals[3])
            << "," << static_cast<long long>(local[0])
            << "," << static_cast<long long>(local[1])
            << "," << static_cast<long long>(local[2])
            << "," << static_cast<long long>(local[3])
            << "\n";
    }
}

} // namespace memory_debug

#define MEMORY_DEBUG_PRINT(label) memory_debug::print_memory(label)
#define MEMORY_DEBUG_MALLOC_INFO(dir, cycle) memory_debug::dump_malloc_info(dir, cycle)

#else
#define MEMORY_DEBUG_PRINT(label) ((void)0)
#define MEMORY_DEBUG_MALLOC_INFO(dir, cycle) ((void)0)
#endif // MEMORY_DEBUG

#endif // MEMORY_DEBUG_HPP
