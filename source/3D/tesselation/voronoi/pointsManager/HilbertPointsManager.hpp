#ifndef HILBERT_POINTS_MANAGER_HPP
#define HILBERT_POINTS_MANAGER_HPP

#ifdef RICH_MPI

#include <numeric>
#include <memory>
#include "PointsManager.hpp"
#include "3D/environment/kernels/Identity.hpp" // for default kernelization
#include "3D/environment/hilbert/DistributedOctEnvAgent.hpp"
#include "3D/environment/hilbert/HilbertTreeEnvAgent.hpp"
#include "3D/hilbert/rectangular/HilbertRectangularConvertor3D.hpp"
#include "3D/hilbert/ordinary/HilbertOrdinaryConvertor3D.hpp"
#include "3D/tesselation/loadBalancing/PartitionLoadBalancer.hpp"
#include "utils/balance/weightedBalance2.hpp"

#define SPACE_FACTOR 1e-5

class HilbertPointsManager : public PointsManager
{
public:
    HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> &indexing = std::shared_ptr<const Kernelization3D::IndexingKernel3D>(), const MPI_Comm &comm = MPI_COMM_WORLD)
        : PointsManager(ll, ur, comm), envAgent(nullptr), convertor(nullptr)
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

    inline ~HilbertPointsManager() override
    {
        delete this->envAgent;
        delete this->convertor;
    };

    inline const EnvironmentAgent *getEnvironmentAgent() const override
    {
        return this->envAgent;
    }

    HilbertPointsManager &operator=(const HilbertPointsManager &other) = delete;

    PointsExchangeResult exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM) override;

    void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) override;

    const Kernelization3D::IndexingKernel3D *getIndexingKernel() const{return this->indexing.get();};
    
    void setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer) override;

    std::shared_ptr<LoadBalancer> getLoadBalancer(void) override;

private:
    void initializeHilbertParameters(const std::vector<Vector3D> &points);

    PointsExchangeResult initialize(const std::vector<Vector3D> &points, const std::vector<double> &weights, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM);

    HilbertCurveEnvironmentAgent *envAgent;
    HilbertConvertor3D *convertor;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing;
    std::shared_ptr<LoadBalancer> loadBalancer;
    bool customIndexingIsSet;
};

#endif // RICH_MPI

#endif // HILBERT_POINTS_MANAGER_HPP