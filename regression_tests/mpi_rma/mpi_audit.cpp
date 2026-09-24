// Standard PMPI interception: observe the unmodified benchmark's RMA traffic.
#include <mpi.h>
#include <cstdio>

namespace {
unsigned long long put_calls = 0, put_bytes = 0, get_calls = 0, windows = 0;
}

extern "C" int MPI_Put(const void *origin, int count, MPI_Datatype type,
                       int rank, MPI_Aint disp, int target_count,
                       MPI_Datatype target_type, MPI_Win win)
{
    const int error = PMPI_Put(origin, count, type, rank, disp, target_count, target_type, win);
    if(error == MPI_SUCCESS)
    {
        int bytes = 0;
        PMPI_Type_size(type, &bytes);
        ++put_calls;
        put_bytes += static_cast<unsigned long long>(count) * bytes;
    }
    return error;
}

extern "C" int MPI_Get(void *result, int count, MPI_Datatype type,
                       int rank, MPI_Aint disp, int target_count,
                       MPI_Datatype target_type, MPI_Win win)
{
    const int error = PMPI_Get(result, count, type, rank, disp, target_count, target_type, win);
    if(error == MPI_SUCCESS) ++get_calls;
    return error;
}

extern "C" int MPI_Win_create(void *base, MPI_Aint size, int disp_unit,
                              MPI_Info info, MPI_Comm comm, MPI_Win *win)
{
    const int error = PMPI_Win_create(base, size, disp_unit, info, comm, win);
    if(error == MPI_SUCCESS) ++windows;
    return error;
}

extern "C" int MPI_Finalize()
{
    unsigned long long local[] = {put_calls, put_bytes, get_calls, windows}, totals[4] = {};
    PMPI_Reduce(local, totals, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    int rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
        std::printf("MPI RMA audit: puts=%llu bytes=%llu gets=%llu windows=%llu\n",
                    totals[0], totals[1], totals[2], totals[3]);
    return PMPI_Finalize();
}
