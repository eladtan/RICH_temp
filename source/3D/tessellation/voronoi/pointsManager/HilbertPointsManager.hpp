#ifndef HILBERT_POINTS_MANAGER_HPP
#define HILBERT_POINTS_MANAGER_HPP

#include "3D/environment/hilbert/HilbertCurveEnvAgent.hpp"
#ifdef RICH_MPI

#include <numeric>
#include <memory>
#include "PointsManager.hpp"
#include "3D/environment/kernels/Identity.hpp" // for default kernelization
#include "3D/environment/hilbert/DistributedOctEnvAgent.hpp"
#include "3D/environment/hilbert/HilbertTreeEnvAgent.hpp"
#include "3D/environment/hilbert/HilbertCurveEnvAgent.hpp"
#include "3D/hilbert/rectangular/HilbertRectangularConvertor3D.hpp"
#include "3D/hilbert/ordinary/HilbertOrdinaryConvertor3D.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"

#define SPACE_FACTOR 1e-5

class HilbertPointsManager : public PointsManager
{
public:
    HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> &indexing = std::shared_ptr<const Kernelization3D::IndexingKernel3D>(), const MPI_Comm &comm = MPI_COMM_WORLD);

    inline ~HilbertPointsManager() override = default;

    std::shared_ptr<PointsManager> clone(void) const override;

    inline const std::shared_ptr<EnvironmentAgent> getEnvironmentAgent() const override{return this->envAgent;}

    HilbertPointsManager &operator=(const HilbertPointsManager &other) = delete;

    PointsExchangeResult exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange) override;

    void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) override;

    const Kernelization3D::IndexingKernel3D *getIndexingKernel() const{return this->indexing.get();};
    
    void setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer) override;

    std::shared_ptr<LoadBalancer> getLoadBalancer(void) override;

private:
    void initializeHilbertParameters(const std::vector<Vector3D> &points);

    PointsExchangeResult initialize(const std::vector<Vector3D> &points, const std::vector<double> &weights, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM);

    std::shared_ptr<HilbertLoadBalancer> loadBalancer = nullptr;
    std::shared_ptr<HilbertCurveEnvironmentAgent> envAgent = nullptr;
    std::shared_ptr<HilbertConvertor3D> convertor = nullptr;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing = nullptr;
    bool customIndexingIsSet;
};

#endif // RICH_MPI

#endif // HILBERT_POINTS_MANAGER_HPP