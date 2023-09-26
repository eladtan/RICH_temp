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
    HashBruteForceFinder(const EnvironmentAgent *envAgent, RandomAccessIterator first, RandomAccessIterator last): envAgent(envAgent)
    {
        this->cellsPointsSize = static_cast<size_t>(std::pow(2, 1.4 * this->envAgent->getOrder()));
        this->cellsPoints.resize(this->cellsPointsSize);

        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        size_t index = 0;
        for(RandomAccessIterator it = first; it != last; it++)
        {
            const Vector3D &point = *it;
            this->myPoints.push_back(point);
            hilbert_index_t cell = this->envAgent->xyz2d(point);
            this->cellsPoints[cell % this->cellsPointsSize].push_back(index);
            index++;
        }
        this->pointsSize = index;
    };

    template<typename Container>
    inline HashBruteForceFinder(const EnvironmentAgent *envAgent, Container points): HashBruteForceFinder(dynamic_cast<const HilbertEnvironmentAgent*>(envAgent), points.begin(), points.end()){};
    inline ~HashBruteForceFinder() = default;

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        return std::vector<size_t>();
    }

    std::vector<size_t> range(const Vector3D &center, double radius) const override
    {
        boost::container::flat_set<size_t> intersectingCells = this->envAgent->getIntersectingCells(center, radius);
        std::vector<size_t> result;
        for(hilbert_index_t cell : intersectingCells)
        {
            if(this->envAgent->getCellOwner(cell) == this->rank)
            {
                const size_t *_points = this->cellsPoints[cell % this->cellsPointsSize].data();
                size_t cellPointsSize = this->cellsPoints[cell % this->cellsPointsSize].size();
                for(size_t i = 0; i < cellPointsSize; i++)
                {
                    //__builtin_prefetch(&this->myPoints[_points[i]]);
                    const Vector3D &point = this->myPoints[_points[i]];
                    double distanceSquared = (point.x - center.x) * (point.x - center.x) + (point.y - center.y) * (point.y - center.y) + (point.z - center.z) * (point.z - center.z);
                    if(distanceSquared <= radius * radius)
                    {
                        result.push_back(_points[i]);
                    }
                }
            }
        }
        return result;
    }

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

    inline size_t size() const override{return this->pointsSize;};

private:
    size_t pointsSize;
    int rank;
    std::vector<std::vector<size_t>> cellsPoints;
    size_t cellsPointsSize;
    std::vector<Vector3D> myPoints;
    const HilbertEnvironmentAgent *envAgent;
};

#endif // RICH_MPI

#endif // _HASH_BRUTE_FORCE_RANGE_HPP