#include "HilbertPointsManager.hpp"
#include "3D/tessellation/loadBalancing/CurveLoadBalancer.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"
#include "misc/universal_error.hpp"
#include <memory>

#ifdef RICH_MPI

HilbertPointsManager::HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm)
    : PointsManager(ll, ur, comm)
{
    this->customIndexingIsSet = false;
}

std::shared_ptr<PointsManager> HilbertPointsManager::clone(void) const
{
    std::shared_ptr<HilbertPointsManager> clone = std::make_shared<HilbertPointsManager>(this->ll, this->ur,  this->comm);
    
    clone->loadBalancer = std::dynamic_pointer_cast<HilbertLoadBalancer>(this->loadBalancer->clone());

    clone->envAgent = this->envAgent->clone(clone->loadBalancer);
    return clone;
}

PointsExchangeResult HilbertPointsManager::exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange)
{
    PointsExchangeResult exchangeResult;

    if(this->envAgent != nullptr)
    {
        if(noExchange)
        {
            exchangeResult = this->pointsExchange([this](const PointData &_point)
            {
                (void) _point; // unused parameter
                return this->rank;
            },
            allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM); // exchange
        }
        else
        {
            exchangeResult = this->pointsExchange([this](const PointData &_point)
            {
                return this->loadBalancer->getOwner(_point.point);
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
        exchangeResult = this->initialize(allPoints, allWeights, radiuses, previous_CM, noExchange);
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

    auto indexing = this->loadBalancer->getIndexing();
    if (indexing && dynamic_cast<const Kernelization3D::Identity *>(indexing.get()) == nullptr)
    {
        this->customIndexingIsSet = true;
    }

    if(this->envAgent != nullptr)
    {
        this->envAgent->setLoadBalancer(this->loadBalancer);
    }
}

std::shared_ptr<LoadBalancer> HilbertPointsManager::getLoadBalancer(void)
{
    return this->loadBalancer->clone();
}

const std::shared_ptr<LoadBalancer> HilbertPointsManager::getLoadBalancer(void) const
{
    return this->loadBalancer->clone();
}

void HilbertPointsManager::rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights)
{
    this->loadBalancer->rebalance(points, weights);
    if(this->envAgent != nullptr)
    {
        this->envAgent->setLoadBalancer(this->loadBalancer);
    }
}

void HilbertPointsManager::setIndexing(std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing)
{
    this->customIndexingIsSet = true;
    if (this->loadBalancer != nullptr)
    {
        this->loadBalancer->setIndexing(indexing);
    }
    else
    {
        this->pendingIndexing_ = indexing;
    }
    this->envAgent = nullptr;
}

PointsExchangeResult HilbertPointsManager::initialize(const std::vector<Vector3D> &points, const std::vector<double> &weights, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange)
{
    // if(this->rank == 0)
    // {
    //     std::cout << "initializes the points manager, and the environment agent" << std::endl;
    // }

    // calculate the first and initial order, and set it to the deepest hilbert order we have

    std::vector<size_t> allIndices(points.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);
    
    if(not noExchange)
    {
        auto indexing = (this->pendingIndexing_ != nullptr)
            ? this->pendingIndexing_
            : std::make_shared<const Kernelization3D::Identity>();
        this->pendingIndexing_ = nullptr;
        this->loadBalancer = std::make_shared<HilbertLoadBalancer>(this->ll, this->ur, points, indexing);
        this->rebalance(points, weights);
    }

    // making exchange according to these borders
    PointsExchangeResult exchangeResult;
    if(noExchange)
    {
        exchangeResult = this->pointsExchange([this](const PointData &_point)
        {
            (void) _point; // unused parameter
            return this->rank;
        },
        points, weights, allIndices, radiuses, previous_CM);
    }
    else
    {
        exchangeResult = this->pointsExchange([this](const PointData &_point)
        {
            return this->loadBalancer->getOwner(_point.point);
        },
        points, weights, allIndices, radiuses, previous_CM);
    }    
            
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