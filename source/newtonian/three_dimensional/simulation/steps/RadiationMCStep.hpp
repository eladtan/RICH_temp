#ifndef RADIATION_MC_STEP_HPP
#define RADIATION_MC_STEP_HPP

#include "PhysicsStep.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
    #include "utils/rma/RMAFactory.hpp"
#endif // RICH_MPI

class RadiationMCStep : public PhysicsStep
{
public:
    static constexpr const char *step_name = "radiation-mc";
    #ifdef RICH_MPI
    enum ManagerType
    {
        MPI_RMA,
        IBV_RDMA,
        P2P,
        AUTO_RDMA
    };
    #endif // RICH_MPI

    RadiationMCStep(const Tessellation3D &tess,
                    std::vector<ComputationalCell3D> &cells,
                    std::vector<Conserved3D> &extensives,
                    std::shared_ptr<MonteCarloRadiationPhysics3D> physics,
                    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl,
                    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond,
                    const std::vector<Particle3D> &particles = std::vector<Particle3D>(),
                    size_t initialParticlesPerCell = 50,
                    bool withHydro = false
                    #ifdef RICH_MPI
                        , ManagerType managerType = ManagerType::AUTO_RDMA
                        , std::shared_ptr<CostCalculator3D> cost = nullptr
                        , const MonteCarloConfig &monteCarloConfig = MonteCarloConfig()
                    #endif // RICH_MPI
                );

    void step(double dt) override;

    double suggestTimeStep(void) const override;

    std::string getName(void) const override { return step_name; }

    std::vector<Particle3D> &getParticles(void);

    inline const Tessellation3D &getTessellation(void) const{return this->tess;};

    inline const std::vector<ComputationalCell3D> &getCells(void) const{return this->cells;};

    inline const std::vector<Conserved3D> &getExtensives(void) const{return this->extensives;};

    const std::vector<Particle3D> &getParticles(void) const;

    inline std::shared_ptr<MonteCarloManager3D> getManager(void) const{return this->manager;};

    inline const std::vector<double> &getEradTimeAvg(void) const{return this->physics->getEradTimeAvg();};
    
    inline std::vector<double> &getEradTimeAvg(void){return this->physics->getEradTimeAvg();};

    inline const auto &getEgTimeAvg(void) const{return this->physics->getEgTimeAvg();};

    inline auto &getEgTimeAvg(void){return this->physics->getEgTimeAvg();};

    #ifdef RICH_MPI
        inline void setCost(std::shared_ptr<CostCalculator3D> newCost){this->cost = newCost;};

        inline std::shared_ptr<CostCalculator3D> getCost(void) const{return this->cost;};
    
        ExchangeChain GetExchangeChain(void) override;

        bool allowRebalance(void) override;

        std::string getRequiredLB(void) const override;

        std::vector<double> getLoadBalanceWeights(void) override;


        void beforeLB(void) override;

        void afterLB(void) override;
    #endif // RICH_MPI

private:
    const Tessellation3D &tess;
    std::vector<ComputationalCell3D> &cells;
    std::vector<Conserved3D> &extensives;
    std::vector<Particle3D> particles;
    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl;
    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond;
    std::shared_ptr<MonteCarloRadiationPhysics3D> physics;
    std::shared_ptr<MonteCarloManager3D> manager;
    bool withHydro;
    size_t stepCounter;
    double suggested_dt;
    #ifdef RICH_MPI
        ManagerType managerType;
        std::shared_ptr<CostCalculator3D> cost;
    #endif // RICH_MPI
};


#endif // RADIATION_MC_STEP_HPP