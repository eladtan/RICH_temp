#ifdef RICH_MPI

#include "PointsManager.hpp"

void PointsManager::setImbalanceTolerance(double tolerance)
{
    this->imbalanceTolerance = tolerance;
}

void PointsManager::reportImbalance(void) const
{
    struct
    {
        double weight;
        int rank;
    } myWeightedRank, maxWeighted, minWeighted;
    myWeightedRank.weight = this->totalWeight;
    myWeightedRank.rank = this->rank;
    MPI_Allreduce(&myWeightedRank, &maxWeighted, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Allreduce(&myWeightedRank, &minWeighted, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    double avgWeight;
    MPI_Allreduce(&this->totalWeight, &avgWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avgWeight /= static_cast<double>(this->size);
    if(this->rank == 0)
    {
        std::cout << "Imbalance report: max weight is " << maxWeighted.weight << " in rank " << maxWeighted.rank << ", min weight is " << minWeighted.weight << " in rank " << minWeighted.rank << ", average weight is " << avgWeight << std::endl;
    }
}

bool PointsManager::checkForRebalance(double myWeight) const
{
    struct
    {
        double weight;
        int rank;
    } myWeightRanked, maxWeight;

    // checks if I have too many weight, and notify other ranks
    double totalWeight;
    MPI_Allreduce(&myWeight, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double idealWeight = totalWeight / this->size;
    myWeightRanked.weight = myWeight;
    myWeightRanked.rank = this->rank;
    MPI_Allreduce(&myWeightRanked, &maxWeight, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    if(this->rank == 0)
    {
        std::cout << "Max weight in rank " << maxWeight.rank << ", its weight is " << maxWeight.weight << ", ideal is " << idealWeight << std::endl;
    }
    if(maxWeight.weight >= (this->imbalanceTolerance * idealWeight))
    {
        if(this->rank == 0)
        {
            std::cout << "Doing rebalance!" << std::endl;
        }
        return true;
    }
    return false;
};

bool PointsManager::shouldRebalance(const std::vector<double> &weights) const
{
    double totalWeight = std::accumulate(weights.cbegin(), weights.cend(), 0.0);
    return this->checkForRebalance(totalWeight);
};

PointsExchangeResult PointsManager::update(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool doRebalance, bool doExchange)
{
    // if envAgent is null, the `exchange` will perform an initialization as well.
    // `rebalance` is used only when the environment agent is initialized.
    
    std::chrono::high_resolution_clock::time_point start, end;

    start = std::chrono::high_resolution_clock::now();
    PointsExchangeResult result = this->exchange(allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM, not doExchange);
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
        result = this->exchange(allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM, not doExchange);
        this->totalWeight = std::accumulate(result.newWeights.cbegin(), result.newWeights.cend(), 0.0);
        end = std::chrono::high_resolution_clock::now();
        if(this->rank == 0)
        {
            std::cout << "Time for load balancing: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
        }
        this->reportImbalance();
    }
    // std::cout << "total weight of rank " << this->rank << " is " << this->totalWeight << " with " << result.newPoints.size() << " points" << std::endl;
    return result;
}

#endif // RICH_MPI