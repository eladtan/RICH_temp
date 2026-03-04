#ifndef HYDRO_STEP_HPP
#define HYDRO_STEP_HPP

#include "PhysicsStep.hpp"
#include "hdsim_3d.hpp"
#include "CostCalculator3D.hpp"

class HydroStep : public PhysicsStep
{
public:
    enum StepType
    {
        TIMEADVANCE_2
    };

    HydroStep(HDSim3D &sim, StepType stepType);

    void step(double dt) override;

    double suggestTimeStep(void) const override;

    std::string getName (void) const override;

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
};

#endif // HYDRO_STEP_HPP