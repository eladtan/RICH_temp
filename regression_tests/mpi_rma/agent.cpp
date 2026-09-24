#include <rma/RMAFactory.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message)
{
    if(!condition) throw std::runtime_error(message);
}

void check_agent(MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    const int peer = 1 - rank;
    auto agent = RMAFactory::Create<std::uint64_t>(RDMA_Type::MPI_RMA, 16, comm);
    std::fill_n(agent->GetLocalPointer(), 16, 777);
    agent->SyncLocal();
    MPI_Barrier(comm);

    // A deferred operation keeps caller-owned buffers alive until Flush.
    const std::uint64_t values[] = {10u + rank, 20u + rank, 30u + rank};
    agent->Put(values, 3, peer, 4, false);
    agent->Flush(peer);
    MPI_Barrier(comm);
    agent->SyncLocal();
    require(agent->GetLocalPointer()[3] == 777 &&
            agent->GetLocalPointer()[7] == 777, "Put overwrote a guard element");
    for(int i = 0; i < 3; ++i)
        require(agent->GetLocalPointer()[4 + i] == 10u * (i + 1) + peer,
                "Put used an incorrect displacement or value");
    std::uint64_t fetched[3] = {};
    agent->Get(fetched, 3, peer, 4, false);
    agent->Flush(peer);
    require(std::equal(fetched, fetched + 3, values), "deferred Get mismatch");

    const std::uint32_t slots[] = {0, 2, 15};
    agent->PutScatter(values, slots, 3, peer);
    using Entry = RemoteMemoryAgent<std::uint64_t>::PutBatchEntry;
    const Entry entries[] = {{0, 8, 2}, {2, 12, 1}};
    agent->PutBatch(values, 3, entries, 2, peer);
    MPI_Barrier(comm);
    agent->SyncLocal();
    require(agent->GetLocalPointer()[15] == 30u + peer &&
            agent->GetLocalPointer()[8] == 10u + peer &&
            agent->GetLocalPointer()[12] == 30u + peer, "scatter/batch mismatch");

    std::vector<std::uint64_t> saved(agent->GetLocalPointer(), agent->GetLocalPointer() + 16);
    agent->Resize(31);
    require(std::equal(saved.begin(), saved.end(), agent->GetLocalPointer()),
            "Resize lost received data");
    agent->Resize(9);
    require(std::equal(saved.begin(), saved.begin() + 9, agent->GetLocalPointer()),
            "shrinking Resize lost the retained prefix");
    agent->Replace(0);
    require(agent->GetCount() == 0, "zero-sized Replace has wrong count");
    agent->Resize(2);
    std::fill_n(agent->GetLocalPointer(), 2, 123456);
    agent->SyncLocal();
    MPI_Barrier(comm);

    // Before the fix, false returned an unfinished stack result. Its subsequent
    // completion could overwrite a different stack frame and segfault.
    for(std::uint64_t i = 0; i < 128; ++i)
    {
        auto old = agent->FetchAndAdd(1, peer, 0, false);
        require(old == 123456 + i, "FetchAndAdd(false) returned an unfinished result");
    }
    agent->Flush(peer);
    require(agent->FetchAndAdd(2, peer, 0) == 123584, "FetchAndAdd(true) mismatch");
    std::uint64_t old = 0;
    const std::uint64_t desired = 99, expected = 123456;
    agent->CompareAndSwap(desired, expected, old, peer, 1, false);
    agent->Flush(peer);
    require(old == expected, "deferred CompareAndSwap mismatch");
    MPI_Barrier(comm);
    agent->SyncLocal();
    require(agent->GetLocalPointer()[0] == 123586 &&
            agent->GetLocalPointer()[1] == 99, "atomic target values mismatch");
    agent->Free();
    agent->Free(); // Explicit Free and destruction must be idempotent.

    std::uint64_t buffer[] = {0, 0, 0};
    auto borrowed = RMAFactory::CreateOver<std::uint64_t>(RDMA_Type::MPI_RMA, buffer, 3, comm);
    borrowed->Put(values, 3, peer, 0);
    MPI_Barrier(comm);
    borrowed->SyncLocal();
    require(buffer[1] == 20u + peer, "user-buffer transfer mismatch");
    bool rejected = false;
    try { borrowed->Resize(4); } catch(const std::runtime_error &) { rejected = true; }
    require(rejected, "Resize accepted borrowed storage");
    borrowed->Free();
    require(buffer[1] == 20u + peer, "Free damaged borrowed storage");
}
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try
    {
        require(size == 2, "run this regression with exactly two ranks");
        // Reverse rank order to expose accidental use of world ranks.
        MPI_Comm comm;
        MPI_Comm_split(MPI_COMM_WORLD, 0, 1 - rank, &comm);
        check_agent(comm);
        MPI_Comm_free(&comm);
        if(rank == 0) std::cout << "PASS: MPI RMA transfers, resize, borrowed storage, and deferred atomics\n";
    }
    catch(const std::exception &error)
    {
        std::cerr << "Rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
