#ifndef HYDRO_STEP_HPP
#define HYDRO_STEP_HPP

#include "PhysicsStep.hpp"
#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"

class HydroStep : public PhysicsStep
{
public:
    static constexpr const char *step_name = "hydro";

    enum StepType
    {
        TIMEADVANCE_2,
        TIMEADVANCE_LAGRANGIAN_1D
    };

    HydroStep(HDSim3D &sim, StepType stepType,
              const ComputationalCell3D* left_ext = nullptr,
              const ComputationalCell3D* right_ext = nullptr);

    void step(double dt) override;

    double suggestTimeStep(void) const override;

    std::string getName(void) const override { return step_name; }

    inline const Tessellation3D &getTessellation(void) const{return sim.getTessellation();};
    inline Tessellation3D &getTessellation(void){return sim.getTessellation();};
    inline const std::vector<ComputationalCell3D> &getCells(void) const{return sim.getCells();};
    
    inline std::vector<ComputationalCell3D> &getCells(void){return sim.getCells();};

    #ifdef RICH_MPI
        bool allowRebalance(void) override;

        std::string getRequiredLB(void) const override;

        std::vector<double> getLoadBalanceWeights(void) override;

        void beforeLB(void) override;

        void afterLB(void) override;

        std::shared_ptr<CostCalculator3D> getCost(void);

        void setCost(std::shared_ptr<CostCalculator3D> newCost);

        ExchangeChain GetExchangeChain(void) override;
    #endif // RICH_MPI

private:
    HDSim3D &sim;
    StepType stepType;
    const ComputationalCell3D* left_external_;
    const ComputationalCell3D* right_external_;
};

#endif // HYDRO_STEP_HPP