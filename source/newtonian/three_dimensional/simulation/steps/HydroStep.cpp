#include "HydroStep.hpp"

HydroStep::HydroStep(HDSim3D &sim, StepType stepType) : sim(sim), stepType(stepType)
{}

void HydroStep::step(double dt)
{
    switch(this->stepType)
    {
        case StepType::TIMEADVANCE_2:
            this->sim.timeAdvance2();
            break;
        default:
            throw std::runtime_error("Invalid step type");
    }
}

double HydroStep::suggestTimeStep(void) const
{
    return this->sim.suggestTimeStep();
}

#ifdef RICH_MPI
    bool HydroStep::allowRebalance(void)
    {
        return true;
    }

    std::string HydroStep::getRequiredLB(void) const
    {
        return "hydro";
    }

    std::vector<double> HydroStep::getLoadBalanceWeights(void)
    {
        return this->sim.cost_calc_->CalculateCost(this->sim.getTessellation(), this->sim.getCells());
    }

    void HydroStep::beforeLB(void)
    {
        return;
    }

    void HydroStep::afterLB(void)
    {
        return;
    }

    std::shared_ptr<CostCalculator3D> HydroStep::getCost(void)
    {
        return this->sim.cost_calc_;
    }

    void HydroStep::setCost(std::shared_ptr<CostCalculator3D> newCost)
    {
        this->sim.cost_calc_ = newCost;
    }

    ExchangeChain HydroStep::GetExchangeChain(void)
    {
        return this->sim.GetExchangeChain();
    }
#endif // RICH_MPI
