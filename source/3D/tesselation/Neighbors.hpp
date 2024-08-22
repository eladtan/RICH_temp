#ifndef TESSELLATION3D_NEIGHBORS_HPP
#define TESSELLATION3D_NEIGHBORS_HPP

#include <vector>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include "Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#include <iostream> // TODO REMOVE
#include "utils/printing/print.hpp" // TODO REMOVE

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
#endif // RICH_MPI

struct RemotePoint
{
    #ifdef RICH_MPI
        int rank;
        size_t indexOnRank;
    #else
        size_t index;
    #endif // RICH_MPI
    size_t distance;

    #ifdef RICH_MPI
        RemotePoint(int _rank = -1, size_t _indexOnRank = std::numeric_limits<size_t>::max(), size_t _distance = std::numeric_limits<size_t>::max())
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
        RemotePoint(size_t _index = std::numeric_limits<size_t>::max()): index(_index){};

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

using PointsToNeighborsMap = boost::container::flat_map<size_t, boost::container::flat_set<RemotePoint>>;

struct ComputationalCell3DVector3D 
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    static size_t CELL_CHUNK_SIZE;

    ComputationalCell3D cell;
    Vector3D point;
    size_t local_index;

    ComputationalCell3DVector3D(const ComputationalCell3D &_cell = ComputationalCell3D(), const Vector3D &_point = Vector3D(), size_t _local_index = 0): cell(_cell), point(_point), local_index(_local_index){};

    #ifdef RICH_MPI
        inline size_t getChunkSize() const override
        {
            if(CELL_CHUNK_SIZE == std::numeric_limits<size_t>::max())
            {
                CELL_CHUNK_SIZE = this->cell.getChunkSize();
            }
            return CELL_CHUNK_SIZE + 4;
        }

        inline std::vector<double> serialize() const override
        {
            std::vector<double> res = this->cell.serialize();
            res.push_back(this->point.x);
            res.push_back(this->point.y);
            res.push_back(this->point.z);
            res.push_back(static_cast<double>(this->local_index));
            return res;
        }

        inline void unserialize(const std::vector<double> &data) override
        {
            this->cell.unserialize(std::vector<double>(data.cbegin(), data.cbegin() + CELL_CHUNK_SIZE));
            this->point.x = data[CELL_CHUNK_SIZE];
            this->point.y = data[CELL_CHUNK_SIZE + 1];
            this->point.z = data[CELL_CHUNK_SIZE + 2];
            this->local_index = static_cast<size_t>(data[CELL_CHUNK_SIZE + 3]);
        }
    #endif // RICH_MPI
};

#ifdef RICH_MPI
    PointsToNeighborsMap GetKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost = false, const MPI_Comm &comm = MPI_COMM_WORLD);
#else // RICH_MPI
    PointsToNeighborsMap GetKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost = false);
#endif // RICH_MPI

/*!
\brief Returns the k neighbors (including all below) from other cpus and their data
\param tess The tessellation
\param points The local points for which to get their neighbors on other cpus
\param cells The computational cells
\param order the order of neighbors (k)
\return An array for each point in points, containing neighbors and their data
*/
std::vector<std::vector<std::pair<ComputationalCell3D, Vector3D>>> GetNeighborCellsPoints(const Tessellation3D &tess, const std::vector<size_t> &points, const std::vector<ComputationalCell3D> &cells, size_t order);
#endif // TESSELLATION3D_NEIGHBORS_HPP