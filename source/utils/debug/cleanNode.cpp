#ifdef RICH_MPI

#include "utils/debug/cleanNode.hpp"
#include "misc/universal_error.hpp"

#include <mpi.h>
#include <unistd.h>
#include <array>
#include <cstdio>
#include <fstream>
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

// Daemons that launch and manage the ranks (Open MPI's PRRTE, MPICH's Hydra, Slurm).
// ps reports CPU averaged over a process's lifetime, so a launcher that has just
// started the job often shows more than the threshold.
const std::set<std::string> launcherCommands = {
    "prte", "prted", "prterun", "orted", "mpirun", "mpiexec", "hydra_pmi_proxy", "srun", "slurmstepd"};

// This process and the processes that started it (e.g. perf, timeout, a binding
// script), nearest first; unused entries are 0.
constexpr int maxLineage = 16;

std::array<int, maxLineage> processLineage()
{
    std::array<int, maxLineage> lineage{};
    int pid = static_cast<int>(getpid());
    for (int i = 0; i < maxLineage and pid > 1; ++i)
    {
        lineage[i] = pid;
        // /proc/<pid>/stat: "pid (comm) state ppid ...", comm may contain spaces
        std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
        std::string content((std::istreambuf_iterator<char>(stat)), std::istreambuf_iterator<char>());
        const size_t close = content.rfind(')');
        if (close == std::string::npos)
            break;
        std::istringstream rest(content.substr(close + 1));
        char state;
        int parent = 0;
        if (!(rest >> state >> parent))
            break;
        pid = parent;
    }
    return lineage;
}

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

        if (knownPids.count(rp.pid) or launcherCommands.count(rp.command))
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

    const std::array<int, maxLineage> lineage = processLineage();
    std::vector<int> allPids(isMaster ? nodeSize * maxLineage : 0);
    MPI_Gather(lineage.data(), maxLineage, MPI_INT, allPids.data(), maxLineage, MPI_INT, 0, nodeComm);

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
