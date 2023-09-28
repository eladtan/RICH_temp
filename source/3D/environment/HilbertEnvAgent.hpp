#ifndef _HILBERT_ENVIRONMENT_AGENT_HPP
#define _HILBERT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "3D/hilbert/hilbertTypes.h"
#include "3D/hilbert/HilbertOrder3D.hpp"

#define AVERAGE_INTERSECT 128
#define MAX_HILBERT_ORDER 18
#define NULL_ORDER -1

class HilbertEnvironmentAgent : public EnvironmentAgent
{
public:
    inline HilbertEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, int order, const MPI_Comm &comm = MPI_COMM_WORLD): EnvironmentAgent(ll, ur, comm)
    {
        this->dx = std::max(std::max((ur.x - ll.x), (ur.y - ll.y)), (ur.z - ll.z));
        this->setOrder(order);
    };
    inline HilbertEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD): EnvironmentAgent(ll, ur, comm), order(NULL_ORDER){};
    _set<int> getIntersectingRanks(const Vector3D &center, double radius) const override;
    _set<hilbert_index_t> getIntersectingCells(const Vector3D &center, double radius) const;
    inline int getOwner(const Vector3D &point) const override{return this->getCellOwner(this->xyz2d(point));};
    inline int getCellOwner(hilbert_index_t d) const
    {
        return std::min<int>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)), this->size - 1);
    };
    inline int getOrder() const{return this->order;};
    inline hilbert_index_t xyz2d(const Vector3D &point) const{
        return EnvironmentAgent::curve.Hilbert3D_xyz2d(Vector3D((point.x - this->ll.x) / dx, (point.y - this->ll.y) / dx, (point.z - this->ll.z) / dx), this->order);
    };
    inline void update(const std::vector<Vector3D> &points) override
    {
        return; // nothing to do
    }
    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder) override
    {
        this->range = newRange;
        this->setOrder(newOrder);
    }

private:
    Vector3D myll, myur, sidesLengths;
    double dx;
    int order;
    int rank, size;
    std::vector<hilbert_index_t> range;

    inline void setOrder(int order)
    {
        if(order == NULL_ORDER)
        {
            return;
        }
        this->order = std::min<int>(order, MAX_HILBERT_ORDER);
        this->sidesLengths = (this->ur - this->ll) / pow(2, this->order);
    }
};

/**
 * Helper functions for getIntersectingCircle. 
*/
namespace
{
    /**
     * rounding up to the nearest hilbert corner in a specific axis.
    */
    inline coord_t getClosestCornerAbove(coord_t val, coord_t minCord, coord_t sideLength)
    {
        return ceil((val - minCord) / sideLength) * sideLength + minCord;
    }

    /**
     * rounding down to the nearest hilbert corner in a specific axis.
    */
    inline coord_t getClosestCornerBelow(coord_t val, coord_t minCord, coord_t sideLength)
    {
        return floor((val - minCord) / sideLength) * sideLength + minCord;
    }
}

/**
 * The old method for calculating intersecting circle. Inefficient for too-large hilbert accuracies.
*/
typename EnvironmentAgent::_set<hilbert_index_t> HilbertEnvironmentAgent::getIntersectingCells(const Vector3D &center, double radius) const
{
    typename EnvironmentAgent::_set<hilbert_index_t> hilbertCells;
    hilbertCells.reserve(AVERAGE_INTERSECT);

    coord_t _minX, _maxX;
    _minX = std::max(std::min(center.x - radius, this->ur.x), this->ll.x);
    _maxX = std::max(std::min(center.x + radius, this->ur.x), this->ll.x);
    _minX = getClosestCornerBelow(_minX, this->ll.x, this->sidesLengths.x);
    _maxX = getClosestCornerAbove(_maxX, this->ll.x, this->sidesLengths.x);

    for(coord_t _x = _minX; _x <= _maxX; _x += this->sidesLengths.x)
    {
        coord_t closestX =  (center.x < _x)? _x : ((center.x > _x + this->sidesLengths.x)? _x + this->sidesLengths.x : center.x);
        coord_t distanceXSquared = radius * radius - (closestX - center.x) * (closestX - center.x);
        distanceXSquared = std::max(static_cast<double>(0), distanceXSquared);

        coord_t _minY, _maxY;
        _minY = std::max(std::min(center.y - sqrt(distanceXSquared), this->ur.y), this->ll.y);
        _maxY = std::max(std::min(center.y + sqrt(distanceXSquared), this->ur.y), this->ll.y);
        _minY = getClosestCornerBelow(_minY, this->ll.y, this->sidesLengths.y);
        _maxY = getClosestCornerAbove(_maxY, this->ll.y, this->sidesLengths.y);

        for(coord_t _y = _minY; _y <= _maxY; _y += this->sidesLengths.y)
        {
            coord_t closestY =  (center.y < _y)? _y : ((center.y > _y + this->sidesLengths.y)? _y + this->sidesLengths.y : center.y);
            coord_t distanceYSquared = distanceXSquared - (closestY - center.y) * (closestY - center.y);
            distanceYSquared = std::max(static_cast<double>(0), distanceYSquared);
            
            coord_t _minZ, _maxZ;
            _minZ = std::max(std::min(center.z - sqrt(distanceYSquared), this->ur.z), this->ll.z);
            _maxZ = std::max(std::min(center.z + sqrt(distanceYSquared), this->ur.z), this->ll.z);
            _minZ = getClosestCornerBelow(_minZ, this->ll.z, this->sidesLengths.z);
            _maxZ = getClosestCornerAbove(_maxZ, this->ll.z, this->sidesLengths.z);
            
            for(coord_t _z = _minZ; _z <= _maxZ; _z += this->sidesLengths.z)
            {
                coord_t closestZ =  (center.z < _z)? _z : ((center.z > _z + this->sidesLengths.z)? _z + this->sidesLengths.z : center.z);

                if(std::abs(((closestX - center.x) * (closestX - center.x) + (closestY - center.y) * (closestY - center.y) + (closestZ - center.z) * (closestZ - center.z)) - (radius * radius)) <= EPSILON)
                {
                    // the testing point is inside the circle iff the whole cube intersects the circle
                    hilbertCells.insert(this->xyz2d(Vector3D(_x + (this->sidesLengths.x) / 2, _y + (this->sidesLengths.y) / 2, _z + (this->sidesLengths.z) / 2)));
                }
            }
        }
    }
    
    return hilbertCells;
}

/**
 * The old method for calculating intersecting circle. Inefficient for too-large hilbert accuracies.
*/
typename EnvironmentAgent::_set<int> HilbertEnvironmentAgent::getIntersectingRanks(const Vector3D &center, double radius) const
{
    EnvironmentAgent::_set<int> ranks;
    for(const hilbert_index_t &cellIdx : this->getIntersectingCells(center, radius))
    {
        ranks.insert(this->getCellOwner(cellIdx));
    }
    return ranks;
}

#endif // RICH_MPI

#endif // _HILBERT_ENVIRONMENT_AGENT_HPP