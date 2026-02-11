#include "HilbertPointsManager.hpp"

#ifdef RICH_MPI

PointsExchangeResult HilbertPointsManager::exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange)
{
    PointsExchangeResult exchangeResult;
    const std::vector<size_t> &responsibilityRange = dynamic_cast<const PartitionLoadBalancer*>(this->loadBalancer.get())->boundaries;

    if(this->envAgent != nullptr)
    {
        if(noExchange)
        {
            exchangeResult = this->pointsExchange([this, &responsibilityRange](const PointData &_point)
            {
                return this->rank;
            },
            allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM); // exchange
        }
        else
        {
            exchangeResult = this->pointsExchange([this, &responsibilityRange](const PointData &_point)
            {
                hilbert_index_t d = this->convertor->xyz2d((*this->indexing)(_point.point.x, _point.point.y, _point.point.z));
                size_t index = std::distance(responsibilityRange.cbegin(), std::upper_bound(responsibilityRange.cbegin(), responsibilityRange.cend(), d));
                return std::min<hilbert_index_t>(index, (this->size - 1));
            },
            allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM); // exchange
        }
        this->envAgent->updatePoints(exchangeResult.newPoints);
    }
    else
    {
        if(allPoints.size() != indicesToWorkWith.size())
        {
            UniversalError eo("Error in HilbertPointsManager::exchange: in the first build, a mesh with all the points should be built. Currently, points and indicesToWorkWith have different sizes");
            eo.addEntry("allPoints.size()", allPoints.size());
            eo.addEntry("indicesToWorkWith.size()", indicesToWorkWith.size());
            throw eo;
        }
        exchangeResult = this->initialize(allPoints, allWeights, radiuses, previous_CM);
    }

    return exchangeResult;
}

void HilbertPointsManager::setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer)
{
    if(dynamic_cast<PartitionLoadBalancer*>(loadBalancer.get()) == nullptr)
    {
        throw UniversalError("HilbertPointsManager::setLoadBalancer: given load balancer is not a PartitionLoadBalancer");
    }
    if(this->rank == 0)
    {
        std::cout << "Restoring Load Balancer" << std::endl;
    }
    this->loadBalancer = loadBalancer;
    if(this->envAgent != nullptr)
    {
        this->envAgent->updateBorders(dynamic_cast<const PartitionLoadBalancer*>(this->loadBalancer.get())->boundaries, this->convertor->getOrder());
    }
}

std::shared_ptr<LoadBalancer> HilbertPointsManager::getLoadBalancer(void)
{
    return this->loadBalancer;
}

void HilbertPointsManager::rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights)
{
    if(this->convertor == nullptr)
    {
        throw UniversalError("HilbertPointsManager::rebalance: convertor was not initialized yet");
    }

    std::vector<hilbert_index_t> indices;
    for(const Vector3D &point : points)
    {
        indices.push_back(this->convertor->xyz2d((*this->indexing)(point)));
    }

    std::vector<size_t> responsibilityRange;
    int dont_do_weights = (weights.empty() and std::all_of(weights.cbegin(), weights.cend(), [&weights](const double &x){return x == weights[0];}))? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &dont_do_weights, 1, MPI_INT, MPI_MAX, this->comm);
    if(this->rank == 0)
    {
        std::cout << "Running rebalancing" << std::endl;
    }
    if(dont_do_weights)
    {
        // responsibilityRange = getBorders(indices);
        responsibilityRange = getWeightedBorders2(indices, std::vector<double>(points.size(), 1.0));
    }
    else
    {
        // responsibilityRange = getWeightedBorders(indices, weights);
        responsibilityRange = getWeightedBorders2(indices, weights);
    }
    
    if(this->envAgent != nullptr)
    {
        this->envAgent->updateBorders(responsibilityRange, this->convertor->getOrder());
    }
    
    this->loadBalancer = std::make_shared<PartitionLoadBalancer>(responsibilityRange);
}

/*
heuristic to determine the hilbert order.
Also initializes the convertor.
*/
void HilbertPointsManager::initializeHilbertParameters(const std::vector<Vector3D> &points)
{
    std::vector<Vector3D> kerneledVectors;
    Vector3D kerneledLL(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D kerneledUR(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
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

    // consider the ll and ur as well
    for(const Vector3D &point : std::vector<Vector3D>({this->ll, this->ur}))
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
    int hilbertOrder;
    MPI_Allreduce(&depth, &hilbertOrder, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // calculates maximal depth

    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.x, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.y, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.z, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.z, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // make a little bit space
    double x_length = kerneledUR.x - kerneledLL.x, y_length = kerneledUR.y - kerneledLL.y, z_length = kerneledUR.z - kerneledLL.z;
    kerneledLL.x -= std::abs(SPACE_FACTOR * x_length);
    kerneledLL.y -= std::abs(SPACE_FACTOR * y_length);
    kerneledLL.z -= std::abs(SPACE_FACTOR * z_length);
    kerneledUR.x += std::abs(SPACE_FACTOR * x_length);
    kerneledUR.y += std::abs(SPACE_FACTOR * y_length);
    kerneledUR.z += std::abs(SPACE_FACTOR * z_length);
    
    hilbertOrder = std::min<size_t>(MAX_HILBERT_ORDER, hilbertOrder);
    this->convertor = new HilbertRectangularConvertor3D(kerneledLL, kerneledUR, hilbertOrder);
    // this->convertor = new HilbertOrdinaryConvertor3D(kerneledLL, kerneledUR, hilbertOrder);
}

PointsExchangeResult HilbertPointsManager::initialize(const std::vector<Vector3D> &points, const std::vector<double> &weights, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM)
{
    // if(this->rank == 0)
    // {
    //     std::cout << "initializes the points manager, and the environment agent" << std::endl;
    // }

    // calculate the first and initial order, and set it to the deepest hilbert order we have

    std::vector<size_t> allIndices(points.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);

    this->initializeHilbertParameters(points); // also initializes the convertor

    this->rebalance(points, weights); // determines initial borders

    const std::vector<size_t> &responsibilityRange = dynamic_cast<const PartitionLoadBalancer*>(this->loadBalancer.get())->boundaries;

    // making exchange according to these borders
    PointsExchangeResult exchangeResult = this->pointsExchange([this, &responsibilityRange](const PointData &_point)
    {
        hilbert_index_t d = this->convertor->xyz2d((*this->indexing)(_point.point));
        size_t index = std::distance(responsibilityRange.cbegin(), std::upper_bound(responsibilityRange.cbegin(), responsibilityRange.cend(), d));
        return std::min<hilbert_index_t>(index, (this->size - 1));
    },
    points, weights, allIndices, radiuses, previous_CM); // exchange
        
    // initialize environment agent
    if(this->customIndexingIsSet)
    {
        this->envAgent = new DistributedOctEnvironmentAgent(this->ll, this->ur, exchangeResult.newPoints, responsibilityRange, this->convertor, this->indexing.get());
    }
    else
    {
        // use hilbert tree, as it is better
        this->envAgent = new HilbertTreeEnvironmentAgent(this->ll, this->ur, exchangeResult.newPoints, responsibilityRange, this->convertor);
    }

    return exchangeResult;
}

#endif // RICH_MPI