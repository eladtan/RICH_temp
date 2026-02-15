#include "HilbertPointsManager.hpp"
#include "3D/tessellation/loadBalancing/CurveLoadBalancer.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"
#include <memory>

#ifdef RICH_MPI

HilbertPointsManager::HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> &indexing, const MPI_Comm &comm)
    : PointsManager(ll, ur, comm)
{
    if(indexing.get() == nullptr)
    {
        this->indexing = std::shared_ptr<const Kernelization3D::Identity>(new Kernelization3D::Identity()); // default kernel
        this->customIndexingIsSet = false;
    }
    else
    {
        this->indexing = indexing;
        this->customIndexingIsSet = true;
    }
}

std::shared_ptr<PointsManager> HilbertPointsManager::clone(void) const
{

    std::shared_ptr<HilbertPointsManager> clone = std::make_shared<HilbertPointsManager>(
        this->ll, this->ur,
        this->customIndexingIsSet ? this->indexing : std::shared_ptr<const Kernelization3D::IndexingKernel3D>(),
        this->comm);
    
    // Deep-copy convertor (mutable state)
    clone->convertor = std::dynamic_pointer_cast<HilbertConvertor3D>(this->convertor->clone());

    clone->loadBalancer = std::dynamic_pointer_cast<HilbertLoadBalancer>(this->loadBalancer->clone(clone->convertor, clone->indexing));

    clone->envAgent = this->envAgent->clone(clone->loadBalancer);
    return clone;
}

PointsExchangeResult HilbertPointsManager::exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange)
{
    PointsExchangeResult exchangeResult;
    const std::vector<curve_index_t> &responsibilityRange = this->loadBalancer->boundaries;

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
        this->envAgent->onExchange(exchangeResult.newPoints);
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

void HilbertPointsManager::setLoadBalancer(std::shared_ptr<LoadBalancer> newLoadBalancer)
{
    HilbertLoadBalancer *hilbertLoadBalancer = dynamic_cast<HilbertLoadBalancer*>(newLoadBalancer.get());
    if(hilbertLoadBalancer == nullptr)
    {
        throw UniversalError("HilbertPointsManager::setLoadBalancer: given load balancer is not a HilbertLoadBalancer");
    }
    if(this->rank == 0)
    {
        std::cout << "Restoring Load Balancer" << std::endl;
    }

    this->loadBalancer = std::dynamic_pointer_cast<HilbertLoadBalancer>(newLoadBalancer);
    this->loadBalancer->convertor = this->convertor;
    this->loadBalancer->indexing = this->indexing;
    this->envAgent->setLoadBalancer(this->loadBalancer);
}

std::shared_ptr<LoadBalancer> HilbertPointsManager::getLoadBalancer(void)
{
    return this->loadBalancer;
}

void HilbertPointsManager::rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights)
{
    this->loadBalancer = std::dynamic_pointer_cast<HilbertLoadBalancer>(this->loadBalancer->clone(this->convertor, this->indexing));
    this->loadBalancer->rebalance(points, weights);
    if(this->envAgent != nullptr)
    {
        this->envAgent->setLoadBalancer(this->loadBalancer);
    }
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
    this->convertor = std::make_shared<HilbertRectangularConvertor3D>(kerneledLL, kerneledUR, hilbertOrder);
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
    
    this->loadBalancer = std::make_shared<HilbertLoadBalancer>(this->convertor, this->indexing);

    this->rebalance(points, weights); // determines initial borders

    const std::vector<curve_index_t> &responsibilityRange = this->loadBalancer->boundaries;

    // making exchange according to these borders
    PointsExchangeResult exchangeResult = this->pointsExchange([this, &responsibilityRange](const PointData &_point)
    {
        curve_index_t d = this->convertor->xyz2d((*this->indexing)(_point.point));
        size_t index = std::distance(responsibilityRange.cbegin(), std::upper_bound(responsibilityRange.cbegin(), responsibilityRange.cend(), d));
        return std::min<size_t>(index, (this->size - 1));
    },
    points, weights, allIndices, radiuses, previous_CM); // exchange
        
    // initialize environment agent
    if(this->customIndexingIsSet)
    {
        this->envAgent = std::make_shared<DistributedOctEnvironmentAgent>(this->ll, this->ur, exchangeResult.newPoints, this->loadBalancer, this->comm);
    }
    else
    {
        // use hilbert tree, as it is better
        this->envAgent = std::make_shared<HilbertTreeEnvironmentAgent>(this->ll, this->ur, this->loadBalancer, this->comm);
    }

    return exchangeResult;
}

#endif // RICH_MPI