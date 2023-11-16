#ifndef _SMART_BRUTE_FORCE_RANGE_HPP
#define _SMART_BRUTE_FORCE_RANGE_HPP

#ifdef RICH_MPI

#include <mpi.h>

// for maps:
#include <map>
#include <boost/container/flat_map.hpp>

#include "3D/environment/HilbertEnvAgent.hpp"
#include "RangeFinder.hpp"

class SmartBruteForceFinder : public RangeFinder
{
public:
    template<typename K, typename V>
    using _map = boost::container::flat_map<K, V>;

    template<typename RandomAccessIterator>
    SmartBruteForceFinder(const EnvironmentAgent *envAgent, const HilbertConvertor3D *convertor, const IndexingKernel3D *indexing, RandomAccessIterator first, RandomAccessIterator last):
        envAgent(dynamic_cast<const HilbertEnvironmentAgent*>(envAgent)), convertor(convertor), indexing(indexing)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        size_t index = 0;
        for(RandomAccessIterator it = first; it != last; it++)
        {
            const Vector3D &point = *it;
            this->myPoints.push_back(point);
            hilbert_index_t cell = this->convertor->xyz2d((*this->indexing)(point));
            if(this->cellsPoints.find(cell) == this->cellsPoints.end())
            {
                this->cellsPoints[cell] = std::vector<size_t>();
            }
            this->cellsPoints[cell].push_back(index);
            index++;
        }
        this->pointsSize = index;
    };

    template<typename Container>
    inline SmartBruteForceFinder(const EnvironmentAgent *envAgent, const IndexingKernel3D *indexing, Container points):
         SmartBruteForceFinder(envAgent, indexing, points.begin(), points.end()){};
    inline ~SmartBruteForceFinder() = default;

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        throw UniversalError("SmartBruteForceFinder::closestPointInSphere not implemented");
    }

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

    std::vector<size_t> range(const Vector3D &center, double radius, size_t N, const _set<size_t> &ignore) const override
    {
        throw UniversalError("SmartBruteForceFinder::range not implemented correctly"); // `ignore` isn't addressed

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
                auto it = this->cellsPoints.find(cell);
                if(it == this->cellsPoints.end())
                {
                    continue;
                }
                size_t cellPointsSize = (*it).second.size();
                const size_t *_points = (*it).second.data();
                for(size_t i = 0; i < cellPointsSize; i++)
                {
                    __builtin_prefetch(&this->myPoints[_points[i]]);
                    const Vector3D &point = this->myPoints[_points[i]];
                    double distanceSquared = (point.x - center.x) * (point.x - center.x) + (point.y - center.y) * (point.y - center.y) + (point.z - center.z) * (point.z - center.z);
                    if(distanceSquared <= (radius * radius))
                    {
                        result.push_back(i);
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
    _map<hilbert_index_t, std::vector<size_t>> cellsPoints;
    std::vector<Vector3D> myPoints;
    const HilbertEnvironmentAgent *envAgent;
    const HilbertConvertor3D *convertor;
    const IndexingKernel3D *indexing;
};

#endif // RICH_MPI

#endif // _SMART_BRUTE_FORCE_RANGE_HPP