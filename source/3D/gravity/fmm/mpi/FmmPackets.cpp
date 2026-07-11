#include "3D/gravity/fmm/mpi/FmmPackets.hpp"

#ifdef RICH_MPI

#include <cmath>

double FmmRemoteNodeDescriptor::geometricRadius() const
{
    return std::sqrt(3.0) * halfSize;
}

#endif // RICH_MPI
