// Check that destruction retires MPI receive requests before their buffers die.
#include <manager/parallel/ReallocationAgent.hpp>
#include <algorithm>
#include <iostream>
#include <vector>

namespace {
std::vector<MPI_Request> receives;
}

extern "C" int MPI_Irecv(void *buffer, int count, MPI_Datatype type,
                         int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    int error = PMPI_Irecv(buffer, count, type, source, tag, comm, request);
    if(error == MPI_SUCCESS) receives.push_back(*request);
    return error;
}

extern "C" int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
    MPI_Request original = *request;
    int error = PMPI_Wait(request, status);
    if(error == MPI_SUCCESS)
        receives.erase(std::remove(receives.begin(), receives.end(), original), receives.end());
    return error;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    {
        STORM::ReallocationAgent agent(MPI_COMM_WORLD, [](rank_t) {},
            [](rank_t, double) { return STORM::ReallocationMetadata{}; },
            [](rank_t, const STORM::ReallocationMetadata &) {});
    }
    const int outstanding = static_cast<int>(receives.size());
    // No messages were sent. Retire the already-cancelled requests so this
    // diagnostic itself can finalize cleanly even with the original destructor.
    for(auto &request : receives) PMPI_Wait(&request, MPI_STATUS_IGNORE);
    int total = 0;
    MPI_Allreduce(&outstanding, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0)
        std::cout << "Unretired receives after ReallocationAgent destruction: " << total << '\n';
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
