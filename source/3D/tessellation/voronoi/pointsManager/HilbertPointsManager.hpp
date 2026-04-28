#ifndef HILBERT_POINTS_MANAGER_HPP
#define HILBERT_POINTS_MANAGER_HPP

#include "3D/environment/hilbert/HilbertCurveEnvAgent.hpp"
#ifdef RICH_MPI

#include <numeric>
#include <memory>
#include "PointsManager.hpp"
#include "3D/environment/kernels/Identity.hpp"
#include "3D/environment/hilbert/DistributedOctEnvAgent.hpp"
#include "3D/environment/hilbert/HilbertTreeEnvAgent.hpp"
#include "3D/environment/hilbert/HilbertCurveEnvAgent.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"

class HilbertPointsManager : public PointsManager
{
public:
    static constexpr const char *type_name = "hilbert";
    
    HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD);

    inline ~HilbertPointsManager() override = default;

    std::string getTypeName() const override { return type_name; }

    std::shared_ptr<PointsManager> clone(void) const override;

    inline const std::shared_ptr<EnvironmentAgent> getEnvironmentAgent() const override{return this->envAgent;}

    HilbertPointsManager &operator=(const HilbertPointsManager &other) = delete;

    PointsExchangeResult exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange) override;

    void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) override;

    void setIndexing(std::shared_ptr<const Kernelization3D::IndexingKernel3D> const& indexing);

    std::shared_ptr<const Kernelization3D::IndexingKernel3D> getIndexing() const
    {
        if (this->loadBalancer != nullptr) return this->loadBalancer->getIndexing();
        return this->pendingIndexing_;
    }
    
    void setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer) override;

    std::shared_ptr<LoadBalancer> getLoadBalancer(void) override;

    const std::shared_ptr<LoadBalancer> getLoadBalancer(void) const override;

private:
    PointsExchangeResult initialize(const std::vector<Vector3D> &points, const std::vector<double> &weights, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange);

    std::shared_ptr<HilbertLoadBalancer> loadBalancer = nullptr;
    std::shared_ptr<HilbertCurveEnvironmentAgent> envAgent = nullptr;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D> pendingIndexing_ = nullptr;
    bool customIndexingIsSet;
};

#endif // RICH_MPI

#endif // HILBERT_POINTS_MANAGER_HPP