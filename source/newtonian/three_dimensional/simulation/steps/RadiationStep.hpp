#ifndef RADIATION_STEP_HPP
#define RADIATION_STEP_HPP

#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "Radiation/RadiationDriver.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/ProgressTracker.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "PhysicsStep.hpp"
#include <tuple>
#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
#endif // RICH_MPI

class RadiationStep : public PhysicsStep
{
public:
    RadiationStep(Tessellation3D &tess, std::vector<ComputationalCell3D> &cells,
                    std::vector<Conserved3D> &extensives,
                    ProgressTracker &pt,
                    #ifdef RICH_MPI
                        std::shared_ptr<CostCalculator3D> cost,
                    #endif // RICH_MPI
                    const RadiationDriver &matrix_builder, bool no_hydro);

    void step(double dt) override;

    bool supportsIndividualTimeSteps(void) const override;

    void stepIndividual(IndividualStepContext const& context) override;

    void suggestIndividualTimeSteps(
        IndividualStepContext const& context,
        std::vector<double>& time_step_limits) const override;

    void suggestIndividualWakeDeadlines(
        IndividualStepContext const& context,
        std::vector<double>& wake_deadlines) const override;


    void afterIndividualAMR(void) noexcept override;

    void onIndividualForceAllActiveLatch(void) noexcept override;

    void beforeIndividualRebalance(void) noexcept override;

    double suggestTimeStep(void) const override;

    // The driver's per-cell form of the last global step's limit, matched to
    // the current owned cells by ID (the mesh has moved and cells migrated
    // since); unmatched cells set no limit.
    bool collectCellTimeStepLimits(std::vector<double>& limits) const override;

    bool cellTimeStepLimitsCached(void) const override {return true;}

    std::size_t cellTimeStepLimitFallbacks(void) const override
    {return cell_limit_fallbacks;}

    double cellTimeStepLimitMinimum(void) const override
    {return cell_limit_minimum;}

    std::string getName(void) const override;

    std::map<std::string, double>
    getIndividualPerformanceCounters(void) const override
    {return last_individual_performance;}

    std::size_t GetCumulativeIndividualRejectedCandidates(void) const
    {return cumulative_individual_rejected_candidates;}

    double GetSmallestIndividualCandidateFraction(void) const
    {return smallest_individual_candidate_fraction;}

    #ifdef RICH_MPI
        bool allowRebalance(void) override;

        std::string getRequiredLB(void) const override;

        std::vector<double> getLoadBalanceWeights(void) override;
    #endif // RICH_MPI

private:
    Tessellation3D &tess;
    std::vector<ComputationalCell3D> &cells;
    std::vector<Conserved3D> &extensives;
    ProgressTracker &pt;
    const RadiationDriver &matrix_builder;
    double suggested_dt;
    std::vector<std::size_t> cell_limit_ids;
    std::vector<double> cell_limit_values;
    mutable std::size_t cell_limit_fallbacks = 0;
    double cell_limit_minimum = std::numeric_limits<double>::infinity();
    std::vector<double> suggested_individual_dt;
    std::vector<double> suggested_individual_wake_deadline;
    std::size_t cumulative_individual_rejected_candidates = 0;
    double smallest_individual_candidate_fraction = 1.0;
    std::map<std::string, double> last_individual_performance;
    #ifdef RICH_MPI
        std::shared_ptr<CostCalculator3D> cost;
    #endif // RICH_MPI
};

#endif // RADIATION_STEP_HPP
