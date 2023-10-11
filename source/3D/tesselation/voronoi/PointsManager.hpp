#ifndef _POINTS_MANAGER_HPP
#define _POINTS_MANAGER_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <vector>
#include <mpi.h>
#include <assert.h>

#include "utils/balance/balance.hpp"
#include "utils/exchange/exchange.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/environment/EnvironmentAgent.h"

#define BALANCE_FACTOR 1.1

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
};

/**
 * \author Maor Mizrachi
 * \brief A point manager performs data movement between ranks (borders determination and points exchange according to borders).
*/
class PointsManager
{
public:
    inline PointsManager(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD): ll(ll), ur(ur), comm(comm)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
    };

    bool checkForRebalance(const std::vector<Vector3D> &points) const
    {
        // checks if I have too many points, and notify other ranks
        size_t mySize = points.size();
        size_t N;
        MPI_Allreduce(&mySize, &N, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        size_t ideal = N / this->size;
        int I_say = (mySize >= (BALANCE_FACTOR * static_cast<double>(ideal)))? 1 : 0; // if I say 'rebalance' or not
        int rebalance = 0; // if someone says 'rebalance' or not
        MPI_Allreduce(&I_say, &rebalance, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(rebalance > 0 and this->rank == 0)
        {
            std::cout << "doing rebalance" << std::endl;
        }
        return (rebalance > 0);
    };

    /**
     * performs a point exchange, according to a given determination function (point -> rank)
    */
    template<typename DetermineFunc>
    PointsExchangeResult pointsExchange(const DetermineFunc &func, const std::vector<Vector3D> &points, const std::vector<double> &radiuses) const
    {
        std::vector<_3DPointRadius> data;
        data.reserve(points.size());
        for(size_t i = 0; i < points.size(); i++)
        {
            const Vector3D &point = points[i];
            const double radius = radiuses[i];
            data.push_back({_3DPoint(point.x, point.y, point.z), radius});
        }

        ExchangeAnswer<_3DPointRadius> answer = dataExchange(data, func, this->comm);

        // arrange the return value data structure
        PointsExchangeResult toReturn;

        toReturn.indicesToSelf = std::move(answer.indicesToMe);
        toReturn.sentProcessors = std::move(answer.processesSend);
        toReturn.sentIndicesToProcessors = std::move(answer.indicesToProcesses);

        std::vector<_3DPointRadius> &ans = answer.output;
        std::vector<Vector3D> pointAns;
        std::vector<double> radiusesAns;
        toReturn.newPoints.reserve(ans.size());
        toReturn.newRadiuses.reserve(ans.size());

        for(const _3DPointRadius &_point : ans)
        {
            toReturn.newPoints.emplace_back(Vector3D(_point.point.x, _point.point.y, _point.point.z));
            toReturn.newRadiuses.push_back(_point.radius);
        }
        return toReturn;
    };

    /**
     * re-calculates the borders, to be equally-divided
    */
   /*
    std::vector<hilbert_index_t> redetermineBorders(const std::vector<Vector3D> &points, const IndexingKernel3D *indexing) const
    {
        std::vector<hilbert_index_t> indices;
        for(const Vector3D &point : points)
        {
            indices.push_back(Hilbert3DConvertor::xyz2d(indexing(point), ));
        }
        return getBorders(indices);
    };
*/
    inline PointsExchangeResult pointsExchangeByEnvAgent(const EnvironmentAgent *envAgent, const std::vector<Vector3D> &points, const std::vector<double> &radiuses) const
    {
        return this->pointsExchange([envAgent](const _3DPointRadius &_point){return envAgent->getOwner(Vector3D(_point.point.x, _point.point.y, _point.point.z));}, points, radiuses);
    };

    /*
    inline PointsExchangeResult pointsExchange(const std::vector<hilbert_index_t> &ranges, const IndexingKernel3D *indexing, const std::vector<Vector3D> &points, const std::vector<double> &radiuses) const
    {
        return this->pointsExchange([this, ranges, indexing](const _3DPointRadius &_point){
            return std::min<hilbert_index_t>(std::distance(ranges.cbegin(), std::upper_bound(ranges.cbegin(), ranges.cend(), Hilbert3DConvertor::xyz2d((*this->indexing)(Vector3D(_point.point.x, _point.point.y, _point.point.z))))), (this->size - 1));
            }, points, radiuses);
    };
    */

private:
    MPI_Comm comm;
    int size;
    int rank;
    Vector3D ll, ur;
};

#endif // RICH_MPI

#endif // _POINTS_MANAGER_HPP