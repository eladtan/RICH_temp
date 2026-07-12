#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/FmmPeerExchange.hpp"

namespace
{
struct WireRecord
{
    int source = -1;
    int round = -1;
};

std::vector<int> peersFor(int rank, int size, int pattern)
{
    std::vector<int> peers;
    if(size <= 1)
        return peers;

    const int next = (rank + 1) % size;
    const int previous = (rank + size - 1) % size;
    switch(pattern)
    {
    case 0:
        if(rank % 2 == 0)
            peers.push_back(next);
        break;
    case 1:
        if(rank % 2 != 0)
            peers.push_back(next);
        break;
    case 2:
        peers.push_back(next);
        break;
    case 3:
        break;
    case 4:
        peers.push_back(previous);
        peers.push_back(next);
        break;
    case 5:
        if(rank == 0)
            for(int peer = 1; peer < size; ++peer)
                peers.push_back(peer);
        break;
    case 6:
        if(rank != 0)
            peers.push_back(0);
        break;
    default:
        break;
    }
    std::sort(peers.begin(), peers.end());
    peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    peers.erase(std::remove(peers.begin(), peers.end(), rank), peers.end());
    return peers;
}

std::vector<int> expectedSources(int target, int size, int pattern)
{
    std::vector<int> result;
    for(int source = 0; source < size; ++source)
    {
        const std::vector<int> peers = peersFor(source, size, pattern);
        if(std::find(peers.begin(), peers.end(), target) != peers.end())
            result.push_back(source);
    }
    return result;
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int patternCount = 7;
    const int cycles = 6;
    const int rounds = patternCount * cycles;
    bool localPass = size > 1;
    FmmPeerExchange exchange;

    for(int round = 0; round < rounds; ++round)
    {
        const int pattern = round % patternCount;
        const std::vector<int> peers = peersFor(rank, size, pattern);
        exchange.reset(MPI_COMM_WORLD, peers);

        std::unordered_map<int, std::vector<char>> sendByRank;
        for(int peer : peers)
        {
            WireRecord record;
            record.source = rank;
            record.round = round;
            std::vector<char>& buffer = sendByRank[peer];
            buffer.resize(sizeof(record));
            std::memcpy(buffer.data(), &record, sizeof(record));
        }

        const FmmPeerExchangeResult received = exchange.exchangeBytes(sendByRank);
        std::vector<int> actualSources;
        for(const FmmReceivedMessage& message : received.messages())
        {
            const FmmByteView view = received.view(message);
            if(view.size != sizeof(WireRecord))
            {
                localPass = false;
                continue;
            }
            WireRecord record;
            std::memcpy(&record, view.data, sizeof(record));
            if(record.source != message.source || record.round != round)
                localPass = false;
            actualSources.push_back(message.source);
        }
        std::sort(actualSources.begin(), actualSources.end());
        if(actualSources != expectedSources(rank, size, pattern))
            localPass = false;
    }

    exchange.clear();
    int local = localPass ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if(rank == 0)
    {
        std::ofstream output("fmm_peer_exchange_rebuild_metrics.txt");
        output << "ranks " << size << "\n";
        output << "patterns " << patternCount << "\n";
        output << "cycles " << cycles << "\n";
        output << "rounds " << rounds << "\n";
        output << "pass " << global << "\n";
        std::cout << "fmm_peer_exchange_rebuild ranks=" << size
                  << " rounds=" << rounds
                  << " pass=" << global << std::endl;
    }

    MPI_Finalize();
    return global ? 0 : 1;
}
