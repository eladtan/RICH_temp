#ifndef _POINTS_MANAGER_HPP
#define _POINTS_MANAGER_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <vector>
#include <mpi.h>
#include <assert.h>

// #include "utils/balance/balance.hpp"
#include "utils/exchange/exchange.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/environment/EnvironmentAgent.h"
#include "3D/tessellation/loadBalancing/LoadBalancer.hpp"

#define IMBALANCE_FACTOR 1.15

/**
 * \author Maor Mizrachi
 * \brief A result for points exchange running.
*/
struct PointsExchangeResult
{
    std::vector<Vector3D> newPoints;
    std::vector<double> newRadiuses;
    std::vector<int> sentProcessors;
    std::vector<std::vector<size_t>> sentIndicesToProcessors;
    std::vector<size_t> indicesToSelf;
    std::vector<Vector3D> newCMs;
    std::vector<double> newWeights;
    std::vector<bool> participatingIndices;
};

struct PointData : public Serializable
{
    size_t indexInAllPoints;
    Vector3D point;
    double radius;
    Vector3D CM;
    double weight;
    bool participating;

    PointData() = default;

    size_t dump(Serializer *serializer) const override
    {
        size_t bytes = 0;
        bytes += serializer->insert(this->indexInAllPoints);
        bytes += serializer->insert(this->point);
        bytes += serializer->insert(this->radius);
        bytes += serializer->insert(this->CM);
        bytes += serializer->insert(this->weight);
        bytes += serializer->insert(this->participating);
        return bytes;
    }

    size_t load(const Serializer *serializer, size_t byteOffset) override
    {
        size_t bytes = 0;
        bytes += serializer->extract(this->indexInAllPoints, byteOffset);
        bytes += serializer->extract(this->point, byteOffset + bytes);
        bytes += serializer->extract(this->radius, byteOffset + bytes);
        bytes += serializer->extract(this->CM, byteOffset + bytes);
        bytes += serializer->extract(this->weight, byteOffset + bytes);
        bytes += serializer->extract(this->participating, byteOffset + bytes);
        return bytes;
    }
};

/**
 * \author Maor Mizrachi
 * \brief A point manager performs data movement between ranks (borders determination and points exchange according to borders).
*/
class PointsManager
{
public:
    inline PointsManager(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD): ll(ll), ur(ur), comm(comm), totalWeight(0), hadRebalance(false)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
    };

    virtual ~PointsManager() = default;

    virtual std::shared_ptr<PointsManager> clone(void) const = 0;

    virtual std::string getTypeName() const = 0;

    PointsManager &operator=(const PointsManager &other) = delete;

    virtual PointsExchangeResult exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange) = 0;

    virtual void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) = 0;

    virtual const std::shared_ptr<EnvironmentAgent> getEnvironmentAgent() const = 0;

    virtual void setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer) = 0;

    virtual std::shared_ptr<LoadBalancer> getLoadBalancer(void) = 0;
    
    virtual const std::shared_ptr<LoadBalancer> getLoadBalancer(void) const = 0;

    inline bool didRebalance(void) const{return this->hadRebalance;};

    void setImbalanceTolerance(double tolerance);

    void reportImbalance(size_t localPointCount) const;

    bool checkForRebalance(double myWeight) const;

    bool shouldRebalance(const std::vector<double> &weights) const;

    inline bool shouldRebalance(void) const{return this->checkForRebalance(this->totalWeight);};

    PointsExchangeResult update(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool doRebalance = true, bool doExchange = true);

protected:
    Vector3D ll, ur;
    MPI_Comm comm;
    int rank, size;
    double totalWeight;
    double imbalanceTolerance = IMBALANCE_FACTOR;
    bool hadRebalance;

    /**
     * performs a point exchange, according to a given determination function (point -> rank)
    */
    template<typename DetermineFunc>
    PointsExchangeResult pointsExchange(const DetermineFunc &func, const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM) const;
};

template<typename DetermineFunc>
PointsExchangeResult PointsManager::pointsExchange(const DetermineFunc &func, const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM) const
{
    std::vector<PointData> data;
    data.reserve(allPoints.size());
    for(size_t pointIdx = 0; pointIdx < allPoints.size(); pointIdx++)
    {
        const Vector3D &point = allPoints[pointIdx];
        data.emplace_back();
        PointData &pointRadius = data.back();
        pointRadius.indexInAllPoints = pointIdx;
        pointRadius.point = point;
        pointRadius.radius = radiuses[pointIdx];
        pointRadius.weight = allWeights[pointIdx];
        pointRadius.CM = previous_CM[pointIdx];
        pointRadius.participating = false;
    }
    
    for(const size_t &pointIdx : indicesToWorkWith)
    {
        data[pointIdx].participating = true;
    }

    // // re-build the function so that it maintains the points that are not participating
    ExchangeAnswer<PointData> answer = dataExchange(data, func, this->comm);

    // arrange the return value data structure
    PointsExchangeResult toReturn;
    
    toReturn.indicesToSelf = std::move(answer.indicesToMe);
    toReturn.sentProcessors = std::move(answer.processesSend);
    toReturn.sentIndicesToProcessors = std::move(answer.indicesToProcesses);

    std::vector<PointData> &ans = answer.output;
    toReturn.newPoints.reserve(ans.size());
    toReturn.newRadiuses.reserve(ans.size());
    toReturn.newCMs.reserve(ans.size());
    toReturn.newWeights.reserve(ans.size());
    toReturn.participatingIndices.resize(ans.size(), false);

    for(const PointData &_point : ans)
    {
        size_t pointIdx = toReturn.newPoints.size();
        toReturn.newPoints.emplace_back(_point.point.x, _point.point.y, _point.point.z);
        if(_point.participating)
        {
            toReturn.participatingIndices[pointIdx] = true;
        }
        toReturn.newRadiuses.push_back(_point.radius);
        toReturn.newCMs.emplace_back(_point.CM.x, _point.CM.y, _point.CM.z);
        toReturn.newWeights.push_back(_point.weight);
    }

    assert(toReturn.newPoints.size() == toReturn.newWeights.size());
    return toReturn;
};

#endif // RICH_MPI

#endif // _POINTS_MANAGER_HPP