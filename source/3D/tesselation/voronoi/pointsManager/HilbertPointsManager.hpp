#ifndef HILBERT_POINTS_MANAGER_HPP
#define HILBERT_POINTS_MANAGER_HPP

#include "3D/environment/DistributedOctEnvAgent.hpp"
#include "PointsManager.hpp"

class HilbertPointsManager : public PointsManager
{
public:
    HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *indexing, const MPI_Comm &comm = MPI_COMM_WORLD)
        : PointsManager(ll, ur, comm), envAgent(nullptr), indexing(indexing), hilbertOrder(0){};

    inline ~HilbertPointsManager() override{delete this->envAgent;};

    inline const EnvironmentAgent *getEnvironmentAgent() const override
    {
        return this->envAgent;
    }

    PointsExchangeResult exchange(const std::vector<Vector3D> &points, const std::vector<double> &radiuses) override;

    void rebalance(const std::vector<Vector3D> &points) override;

private:
    PointsExchangeResult initialize(const std::vector<Vector3D> &points, const std::vector<double> &radiuses);

    int hilbertOrder;
    DistributedOctEnvironmentAgent *envAgent;
    const IndexingKernel3D *indexing;
    std::vector<hilbert_index_t> responsibilityRange;
};

#endif // HILBERT_POINTS_MANAGER_HPP