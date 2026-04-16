#ifndef REMESH_STEP_HPP
#define REMESH_STEP_HPP

#include "PhysicsStep.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include <functional>

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
    #include "newtonian/three_dimensional/CostCalculator3D.hpp"
#endif // RICH_MPI

class RemeshStep : public PhysicsStep
{
public:
    static constexpr const char *step_name = "remesh";

    using PointGenerator = std::function<std::vector<Vector3D>(
        const Tessellation3D &tess, double time)>;

    using PostRebuildCallback = std::function<void(void)>;

    RemeshStep(Tessellation3D &tess,
               std::vector<ComputationalCell3D> &cells,
               std::vector<Conserved3D> &extensives,
               PointGenerator generator
               #ifdef RICH_MPI
               , std::shared_ptr<CostCalculator3D> cost = nullptr
               #endif // RICH_MPI
               );

    void setPostRebuild(PostRebuildCallback cb) { postRebuild = std::move(cb); }

    void step(double dt) override;

    double suggestTimeStep(void) const override;

    std::string getName(void) const override { return step_name; }

    #ifdef RICH_MPI
        bool allowRebalance(void) override;

        std::string getRequiredLB(void) const override;

        std::vector<double> getLoadBalanceWeights(void) override;

        ExchangeChain GetExchangeChain(void) override;
    #endif // RICH_MPI

private:
    Tessellation3D &tess;
    std::vector<ComputationalCell3D> &cells;
    std::vector<Conserved3D> &extensives;
    PointGenerator generator;
    PostRebuildCallback postRebuild;

    #ifdef RICH_MPI
        ExchangeChain exchangeChain;
        std::shared_ptr<CostCalculator3D> cost;
    #endif // RICH_MPI
};

#endif // REMESH_STEP_HPP
