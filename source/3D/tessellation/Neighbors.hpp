#ifndef TESSELLATION3D_NEIGHBORS_HPP
#define TESSELLATION3D_NEIGHBORS_HPP

#include <vector>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include "Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include <mpi_utils/serialize/Serializer.hpp>
    #include "mpi/mpi_commands.hpp"
#endif // RICH_MPI

struct RemotePoint
{
    #ifdef RICH_MPI
        rank_t rank;
        size_t indexOnRank;
    #else
        size_t index;
    #endif // RICH_MPI
    size_t distance;

    #ifdef RICH_MPI
        RemotePoint(rank_t _rank = -1, size_t _indexOnRank = std::numeric_limits<size_t>::max(), size_t _distance = std::numeric_limits<size_t>::max())
        : rank(_rank), indexOnRank(_indexOnRank), distance(_distance){};

        bool operator<(const RemotePoint &other) const
        {
            return this->rank < other.rank or (this->rank == other.rank and this->indexOnRank < other.indexOnRank);
        };

        bool operator==(const RemotePoint &other) const
        {
            return this->rank == other.rank and this->indexOnRank == other.indexOnRank;
        };
    #else // RICH_MPI
        RemotePoint(size_t _index = std::numeric_limits<size_t>::max(), size_t _distance = std::numeric_limits<size_t>::max()): index(_index), distance(_distance){};

        bool operator<(const RemotePoint &other) const
        {
            return this->index < other.index;
        };

        bool operator==(const RemotePoint &other) const
        {
            return this->index == other.index;
        };

    #endif // RICH_MPI
};

using NeighborsList = boost::container::flat_set<RemotePoint>;
using PointsToNeighborsMap = boost::container::flat_map<size_t, NeighborsList>;

struct ComputationalCell3DVector3D 
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    ComputationalCell3D cell;
    Vector3D point;
    size_t local_index;

    ComputationalCell3DVector3D(const ComputationalCell3D &_cell = ComputationalCell3D(), const Vector3D &_point = Vector3D(), size_t _local_index = 0): cell(_cell), point(_point), local_index(_local_index){};

	#ifdef RICH_MPI
		force_inline size_t dump(Serializer *serializer) const override
		{
			size_t bytes = 0;
			bytes += serializer->insert(this->cell);
			bytes += serializer->insert(this->point);
			return bytes;
		}

		force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
		{
			size_t bytes = 0;
			bytes += serializer->extract(this->cell, byteOffset);
			bytes += serializer->extract(this->point, byteOffset + bytes);
			return bytes;
		}
    #endif // RICH_MPI
};

#ifdef RICH_MPI
    PointsToNeighborsMap GetKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost = true, const MPI_Comm &comm = MPI_COMM_WORLD);
#else // RICH_MPI
    PointsToNeighborsMap GetKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost = true);
#endif // RICH_MPI

#ifdef RICH_MPI
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost = true, const MPI_Comm &comm = MPI_COMM_WORLD);
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, size_t order, bool atMost = true, const MPI_Comm &comm = MPI_COMM_WORLD);
#else // RICH_MPI
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order);
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, size_t order);
#endif // RICH_MPI

#ifdef RICH_MPI

/*!
\brief Returns the k neighbors (including all below) from other cpus and their data
\param tess The tessellation
\param points The local points for which to get their neighbors on other cpus
\param cells The computational cells
\param order the order of neighbors (k)
\return An array for each point in points, containing neighbors and their data
*/
std::vector<std::vector<std::pair<ComputationalCell3D, Vector3D>>> GetNeighborCellsPoints(const Tessellation3D &tess, const std::vector<size_t> &points, const std::vector<ComputationalCell3D> &cells, size_t order);

#endif // RICH_MPI

#endif // TESSELLATION3D_NEIGHBORS_HPP