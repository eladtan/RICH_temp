#include "out.h"

#ifdef RICH_MPI

void printToHDF5(const Voronoi3D &voronoi, const std::string &fileName)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dummy = 0;
    if(rank != 0)
    {
        MPI_Recv(&dummy, 1, MPI_INT, rank - 1, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    WriteVoronoi(voronoi, fileName);
    if(rank != size - 1)
    {
        MPI_Send(&dummy, 1, MPI_INT, rank + 1, 999, MPI_COMM_WORLD);        
    }
}

#endif // RICH_MPI