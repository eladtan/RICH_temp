#ifndef HILBERT_POINTS_MANAGER_HPP
#define HILBERT_POINTS_MANAGER_HPP

#ifdef RICH_MPI

#include <memory>
#include "3D/environment/DistributedOctEnvAgent.hpp"
#include "PointsManager.hpp"
#include "3D/environment/kernels/Identity.hpp"

#define SPACE_FACTOR 1e-10

class HilbertPointsManager : public PointsManager
{
public:
    HilbertPointsManager(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<const IndexingKernel3D> &indexing = std::shared_ptr<const IndexingKernel3D>(), const MPI_Comm &comm = MPI_COMM_WORLD)
        : PointsManager(ll, ur, comm), envAgent(nullptr), hilbertOrder(0), convertor(nullptr)
    {
        if(indexing.get() == nullptr)
        {
            this->indexing = std::make_shared<const Identity>(Identity()); // default kernel
        }
        else
        {
            this->indexing = indexing;
        }
    }

    inline ~HilbertPointsManager() override{delete this->envAgent; delete this->convertor;};

    inline const EnvironmentAgent *getEnvironmentAgent() const override
    {
        return this->envAgent;
    }

    PointsExchangeResult exchange(const std::vector<Vector3D> &points, const std::vector<double> &radiuses) override;

    void rebalance(const std::vector<Vector3D> &points) override;

    const IndexingKernel3D *getIndexingKernel() const{return this->indexing.get();};
    
private:
    void determineHilbertOrder(const std::vector<Vector3D> &points);

    PointsExchangeResult initialize(const std::vector<Vector3D> &points, const std::vector<double> &radiuses);

    int hilbertOrder;
    HilbertConvertor3D *convertor;
    DistributedOctEnvironmentAgent *envAgent;
    std::shared_ptr<const IndexingKernel3D> indexing;
    std::vector<hilbert_index_t> responsibilityRange;
};

#endif // RICH_MPI

#endif // HILBERT_POINTS_MANAGER_HPP