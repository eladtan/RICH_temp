#include "HilbertPointsManager.hpp"

#ifdef RICH_MPI

PointsExchangeResult HilbertPointsManager::exchange(const std::vector<Vector3D> &points, const std::vector<double> &radiuses)
{
    PointsExchangeResult exchangeResult;
    if(this->envAgent != nullptr)
    {
        exchangeResult = this->pointsExchangeByEnvAgent(points, radiuses);
        this->envAgent->updatePoints(exchangeResult.newPoints);
    }
    else
    {
        exchangeResult = this->initialize(points, radiuses);
    }
    return exchangeResult;
}

void HilbertPointsManager::rebalance(const std::vector<Vector3D> &points)
{
    std::vector<hilbert_index_t> indices;
    for(const Vector3D &point : points)
    {
        indices.push_back(Hilbert3DConvertor::xyz2d((*this->indexing)(point), this->hilbertOrder));
    }
    this->responsibilityRange = getBorders(indices);
    
    if(this->envAgent != nullptr)
    {
        this->envAgent->updateBorders(this->responsibilityRange, this->hilbertOrder);
    }
}

PointsExchangeResult HilbertPointsManager::initialize(const std::vector<Vector3D> &points, const std::vector<double> &radiuses)
{
    // calculate the first and initial order, and set it to the deepest hilbert order we have
    std::vector<Vector3D> kerneledVectors;
    Vector3D kerneledLL(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D kerneledUR(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
    kerneledVectors.reserve(points.size());
    for(const Vector3D &point : points)
    {
        Vector3D kerneledPoint = (*this->indexing)(point);
        kerneledVectors.push_back(kerneledPoint);
        kerneledLL.x = std::min<double>(kerneledLL.x, kerneledPoint.x);
        kerneledLL.y = std::min<double>(kerneledLL.y, kerneledPoint.y);
        kerneledLL.z = std::min<double>(kerneledLL.z, kerneledPoint.z);
        kerneledUR.x = std::max<double>(kerneledUR.x, kerneledPoint.x);
        kerneledUR.y = std::max<double>(kerneledUR.y, kerneledPoint.y);
        kerneledUR.z = std::max<double>(kerneledUR.z, kerneledPoint.z);
    }
    OctTree<Vector3D> tree(kerneledLL, kerneledUR, kerneledVectors);
    int depth = tree.getDepth(); // my own depth
    MPI_Allreduce(&depth, &this->hilbertOrder, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // calculates maximal depth
    this->hilbertOrder = std::min<size_t>(MAX_HILBERT_ORDER, this->hilbertOrder);

    this->rebalance(points); // determine initial borders
    PointsExchangeResult exchangeResult = this->pointsExchange([this](const _3DPointRadius &_point){
        return std::min<hilbert_index_t>(std::distance(this->responsibilityRange.cbegin(), std::upper_bound(this->responsibilityRange.cbegin(), this->responsibilityRange.cend(), Hilbert3DConvertor::xyz2d((*this->indexing)(Vector3D(_point.point.x, _point.point.y, _point.point.z)), this->hilbertOrder))), (this->size - 1));
        }, points, radiuses); // exchange

    // initialize environment agent
    this->envAgent = new DistributedOctEnvironmentAgent(this->indexing, this->ll, this->ur, exchangeResult.newPoints, this->responsibilityRange, this->hilbertOrder);

    return exchangeResult;
}

#endif // RICH_MPI