#ifndef _HASH_BRUTE_FORCE_RANGE_HPP
#define _HASH_BRUTE_FORCE_RANGE_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include "3D/environment/HilbertEnvAgent.hpp"
#include "RangeFinder.hpp"

#define HASH_SIZE 128

class HashBruteForceFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    HashBruteForceFinder(const EnvironmentAgent *envAgent, const HilbertConvertor3D *convertor, const Kernelization3D::IndexingKernel3D *indexing, RandomAccessIterator first, RandomAccessIterator last);

    template<typename Container>
    inline HashBruteForceFinder(const EnvironmentAgent *envAgent, const Kernelization3D::IndexingKernel3D *indexing, Container points):
         HashBruteForceFinder(envAgent, indexing, points.begin(), points.end()){};
    
    inline ~HashBruteForceFinder() override = default;

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        throw UniversalError("HashBruteForceFinder::closestPointInSphere not implemented");
    }

    std::vector<size_t> range(const Vector3D &center, double radius, size_t N, const _set<size_t> &ignore) const override
    {
        typename HilbertEnvironmentAgent::CellsSet intersectingCells = this->envAgent->getIntersectingCells(Vector3D(center.x, center.y, center.z), radius);
        std::vector<size_t> result;
        for(hilbert_index_t cell : intersectingCells)
        {
            if(result.size() >= N)
            {
                break;
            }

            if(this->envAgent->getCellOwner(cell) == this->rank)
            {
                const size_t *_points = this->cellsPoints[cell % this->cellsPointsSize].data();
                size_t cellPointsSize = this->cellsPoints[cell % this->cellsPointsSize].size();
                for(size_t i = 0; i < cellPointsSize; i++)
                {
                    //__builtin_prefetch(&this->myPoints[_points[i]]);
                    const Vector3D &point = this->myPoints[_points[i]];
                    double distanceSquared = (point.x - center.x) * (point.x - center.x) + (point.y - center.y) * (point.y - center.y) + (point.z - center.z) * (point.z - center.z);
                    if(distanceSquared <= ((radius * radius) + EPSILON))
                    {
                        if(ignore.find(i) == ignore.cend())
                        {
                            // do not ignore
                            result.push_back(i);
                        }
                    }
                }
            }
        }
        return result;
    }

    inline size_t size() const override{return this->pointsSize;};

private:
    size_t pointsSize;
    int rank;
    std::vector<std::vector<size_t>> cellsPoints;
    size_t cellsPointsSize;
    std::vector<Vector3D> myPoints;
    const HilbertEnvironmentAgent *envAgent;
    const HilbertConvertor3D *convertor;
    const Kernelization3D::IndexingKernel3D *indexing;
};

template<typename RandomAccessIterator>
inline HashBruteForceFinder::HashBruteForceFinder(const EnvironmentAgent *envAgent, const HilbertConvertor3D *convertor, const Kernelization3D::IndexingKernel3D *indexing, RandomAccessIterator first, RandomAccessIterator last):
    envAgent(dynamic_cast<const HilbertEnvironmentAgent*>(envAgent)), convertor(convertor), indexing(indexing)
{
    this->cellsPointsSize = static_cast<size_t>(std::pow(2, 1.4 * this->envAgent->getOrder())); // heuristic
    this->cellsPoints.resize(this->cellsPointsSize);

    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    size_t index = 0;
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &point = *it;
        this->myPoints.push_back(point);
        hilbert_index_t cell = this->convertor->xyz2d((*this->indexing)(point));
        this->cellsPoints[cell % this->cellsPointsSize].push_back(index);
        index++;
    }
    this->pointsSize = index;
};

#endif // RICH_MPI

#endif // _HASH_BRUTE_FORCE_RANGE_HPP