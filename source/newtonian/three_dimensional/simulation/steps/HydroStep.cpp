#include "HydroStep.hpp"

void InvalidateIndividualEventMeshReuseValidation(HDSim3D const& simulation);

HydroStep::HydroStep(HDSim3D &sim, StepType stepType,
                     const ComputationalCell3D* left_ext,
                     const ComputationalCell3D* right_ext)
    : sim(sim), stepType(stepType),
      left_external_(left_ext), right_external_(right_ext)
{}

void HydroStep::step(double dt)
{
    switch(this->stepType)
    {
        case StepType::TIMEADVANCE_2:
            this->sim.timeAdvance2();
            break;
        case StepType::TIMEADVANCE_LAGRANGIAN_1D:
            this->sim.timeAdvanceLagrangian1D(left_external_, right_external_);
            break;
        default:
            throw std::runtime_error("Invalid step type");
    }
}

double HydroStep::suggestTimeStep(void) const
{
    return this->sim.suggestTimeStep();
}

void HydroStep::stepIndividual(const IndividualStepContext &context)
{
    if(this->stepType != StepType::TIMEADVANCE_2)
        throw std::logic_error("Only TIMEADVANCE_2 supports individual timesteps");
    this->sim.timeAdvanceIndividual(context);
}

void HydroStep::suggestIndividualTimeSteps(
    const IndividualStepContext &context,
    std::vector<double> &time_step_limits) const
{
    this->sim.suggestIndividualTimeSteps(context, time_step_limits);
}

bool HydroStep::getIndividualGeneratorPoints(
    std::vector<Vector3D>& points) const
{
    std::vector<Vector3D> const& canonical =
        this->sim.GetIndividualGeneratorPoints();
    if(canonical.empty())
        return false;
    points = canonical;
    return true;
}

void HydroStep::afterIndividualAMR(void)
{
    InvalidateIndividualEventMeshReuseValidation(this->sim);
    this->sim.ResetIndividualMeshState();
}

void HydroStep::onIndividualForceAllActiveLatch(void) noexcept
{
    this->sim.ReleaseIndividualPartialMeshScratch();
}

void HydroStep::beforeIndividualRebalance(void) noexcept
{
    this->sim.ReleaseIndividualRebalanceScratch();
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
        InvalidateIndividualEventMeshReuseValidation(this->sim);
        this->sim.ResetIndividualMeshState();
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
