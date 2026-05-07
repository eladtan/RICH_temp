#ifdef RICH_MPI

#include "utils/debug/cleanNode.hpp"
#include "misc/universal_error.hpp"

#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct RogueProcess
{
    int pid;
    std::string user;
    double cpuPercent;
    std::string stat;
    std::string command;
};

std::vector<RogueProcess> findRogueProcesses(const std::set<int> &knownPids,
                                              double cpuThreshold)
{
    std::vector<RogueProcess> rogues;
    FILE *fp = popen("ps -eo pid,user,pcpu,stat,comm --no-headers", "r");
    if (!fp)
        return rogues;

    char line[1024];
    while (fgets(line, sizeof(line), fp))
    {
        std::istringstream iss(line);
        RogueProcess rp;
        if (!(iss >> rp.pid >> rp.user >> rp.cpuPercent >> rp.stat))
            continue;
        std::getline(iss >> std::ws, rp.command);
        if (rp.command.empty())
            continue;

        if (knownPids.count(rp.pid))
            continue;

        if (rp.cpuPercent > cpuThreshold)
            rogues.push_back(std::move(rp));
    }
    pclose(fp);
    return rogues;
}

} // anonymous namespace

void ensureCleanNode(MPI_Comm comm, int checkCount, int sleepSeconds)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);

    int nodeRank, nodeSize;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int globalRank;
    MPI_Comm_rank(comm, &globalRank);

    // key=0 means MPI_Comm_split_type orders by original rank in comm,
    // so nodeRank 0 is the process with the smallest global rank on this node.
    const bool isMaster = (nodeRank == 0);

    int pidInt = static_cast<int>(getpid());
    std::vector<int> allPids(isMaster ? nodeSize : 0);
    MPI_Gather(&pidInt, 1, MPI_INT, allPids.data(), 1, MPI_INT, 0, nodeComm);

    if (isMaster)
    {
        std::set<int> knownPids(allPids.begin(), allPids.end());

        char hostname[MPI_MAX_PROCESSOR_NAME];
        int nameLen;
        MPI_Get_processor_name(hostname, &nameLen);

        for (int check = 0; check < checkCount; ++check)
        {
            auto rogues = findRogueProcesses(knownPids, 20.0);
            if (!rogues.empty())
            {
                std::string host(hostname, static_cast<size_t>(nameLen));
                UniversalError eo("Rogue process(es) detected on node " + host);
                eo.addEntry("reporting_global_rank", globalRank);
                eo.addEntry("ranks_on_node", nodeSize);
                for (const auto &rp : rogues)
                {
                    eo.addEntry("rogue_pid", rp.pid);
                    eo.addEntry("rogue_user", rp.user);
                    eo.addEntry("rogue_cpu%", rp.cpuPercent);
                    eo.addEntry("rogue_stat", rp.stat);
                    eo.addEntry("rogue_command", rp.command);
                }
                MPI_Comm_free(&nodeComm);
                throw eo;
            }
            sleep(static_cast<unsigned>(sleepSeconds));
        }
    }
    else
    {
        sleep(static_cast<unsigned>(checkCount * sleepSeconds));
    }

    MPI_Barrier(nodeComm);
    MPI_Comm_free(&nodeComm);
}

#endif // RICH_MPI
