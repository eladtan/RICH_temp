#ifndef _POINTS_MANAGER_HPP
#define _POINTS_MANAGER_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <vector>
#include <mpi.h>
#include <assert.h>

// #include "utils/balance/balance.hpp"
#include "utils/balance/weightedBalance.hpp"
#include "utils/exchange/exchange.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/environment/EnvironmentAgent.h"

#define BALANCE_FACTOR 1.15

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
    std::vector<size_t> participatingIndices;
};

typedef struct _3DPointData
{
    size_t indexInAllPoints;
    _3DPoint point;
    double radius;
    _3DPoint CM;
    double weight;
    bool participating;
} _3DPointData;

/**
 * \author Maor Mizrachi
 * \brief A point manager performs data movement between ranks (borders determination and points exchange according to borders).
*/
class PointsManager
{
public:
    inline PointsManager(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD): ll(ll), ur(ur), comm(comm), totalWeight(0)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
    };

    virtual ~PointsManager() = default;

    PointsManager &operator=(const PointsManager &other) = delete;

    virtual PointsExchangeResult exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM) = 0;

    virtual void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) = 0;

    virtual const EnvironmentAgent *getEnvironmentAgent() const = 0;

    bool checkForRebalance(double myWeight) const
    {
        // checks if I have too many weight, and notify other ranks
        double totalWeight;
        MPI_Allreduce(&myWeight, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double idealWeight = totalWeight / this->size;
        int I_say = (myWeight >= (BALANCE_FACTOR * idealWeight))? 1 : 0; // if I say 'rebalance' or not
        if(I_say)
        {
            std::cout << "my weight is " << myWeight << " and the ideal weight is " << idealWeight << std::endl;
        }
        int rebalance = 0; // if someone says 'rebalance' or not
        MPI_Allreduce(&I_say, &rebalance, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if((rebalance > 0) and (this->rank == 0))
        {
            std::cout << "doing rebalance" << std::endl;
        }
        return (rebalance > 0);
    };

    PointsExchangeResult update(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool doRebalance = true)
    {
        // if envAgent is null, the `exchange` will perform an initialization as well.
        // `rebalance` is used only when the environment agent is initialized.
        
        std::chrono::high_resolution_clock::time_point start, end;

        start = std::chrono::high_resolution_clock::now();
        PointsExchangeResult result = this->exchange(allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM);
        this->totalWeight = std::accumulate(result.newWeights.cbegin(), result.newWeights.cend(), 0.0);
        end = std::chrono::high_resolution_clock::now();
        if(this->rank == 0)
        {
            std::cout << "Time for exchange: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
        }

        if(doRebalance and this->checkForRebalance(this->totalWeight))
        {
            start = std::chrono::high_resolution_clock::now();
            assert(this->getEnvironmentAgent() != nullptr);
            this->rebalance(allPoints, allWeights);
            result = this->exchange(allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM);
            this->totalWeight = std::accumulate(result.newWeights.cbegin(), result.newWeights.cend(), 0.0);
            end = std::chrono::high_resolution_clock::now();
            if(this->rank == 0)
            {
                std::cout << "Time for load balancing: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
            }

        }
        // std::cout << "total weight of rank " << this->rank << " is " << this->totalWeight << " with " << result.newPoints.size() << " points" << std::endl;
        return result;
    }

protected:
    Vector3D ll, ur;
    MPI_Comm comm;
    int rank, size;
    double totalWeight;

    /**
     * performs a point exchange, according to a given determination function (point -> rank)
    */
    template<typename DetermineFunc>
    PointsExchangeResult pointsExchange(const DetermineFunc &func, const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM) const
    {
        std::vector<_3DPointData> data;
        data.reserve(allPoints.size());
        for(size_t pointIdx = 0; pointIdx < allPoints.size(); pointIdx++)
        {
            const Vector3D &point = allPoints[pointIdx];
            data.emplace_back();
            _3DPointData &pointRadius = data.back();
            pointRadius.indexInAllPoints = pointIdx;
            pointRadius.point = _3DPoint(point.x, point.y, point.z);
            pointRadius.radius = radiuses[pointIdx];
            pointRadius.weight = allWeights[pointIdx];
            pointRadius.CM = _3DPoint(previous_CM[pointIdx].x, previous_CM[pointIdx].y, previous_CM[pointIdx].z);
            pointRadius.participating = false;
        }
        
        for(const size_t &pointIdx : indicesToWorkWith)
        {
            data[pointIdx].participating = true;
        }

        // // re-build the function so that it maintains the points that are not participating
        // auto new_func = [&func, this, &participating](const _3DPointData &point){return ((not participating[point.indexInAllPoints])? this->rank : func(point));};
        ExchangeAnswer<_3DPointData> answer = dataExchange(data, func, this->comm);

        // arrange the return value data structure
        PointsExchangeResult toReturn;
        
        toReturn.indicesToSelf = std::move(answer.indicesToMe);
        toReturn.sentProcessors = std::move(answer.processesSend);
        toReturn.sentIndicesToProcessors = std::move(answer.indicesToProcesses);

        std::vector<_3DPointData> &ans = answer.output;
        toReturn.newPoints.reserve(ans.size());
        toReturn.newRadiuses.reserve(ans.size());
        toReturn.newCMs.reserve(ans.size());
        toReturn.newWeights.reserve(ans.size());
        toReturn.participatingIndices.resize(ans.size(), false);

        for(const _3DPointData &_point : ans)
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
};

#endif // RICH_MPI

#endif // _POINTS_MANAGER_HPP