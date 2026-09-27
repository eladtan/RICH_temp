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

    bool supportsIndividualTimeSteps(void) const override
    {return stepType == StepType::TIMEADVANCE_2 && sim.supportsIndividualTimeSteps();}

    void stepIndividual(const IndividualStepContext &context) override;

    void suggestIndividualTimeSteps(const IndividualStepContext &context,
                                    std::vector<double> &time_step_limits) const override;
    void suggestIndividualChangeWakes(const IndividualStepContext &context,
                                      std::vector<double> &change_ratios) const override;

    void onIndividualSchedulerStart(double global_time_step) override
    {this->sim.SetIndividualGlobalStepReference(global_time_step);}

    bool getIndividualGeneratorPoints(
        std::vector<Vector3D>& points) const override;

    bool getIndividualCellCentroids(
        std::vector<Vector3D>& centroids) const override;

    bool contributesIndividualHydrodynamicSignal(void) const override
    {return true;}

    // 1 CFL/source, 2 mesh drift, 3 mass loss, 4 thermal loss (HDSim3D).
    bool getIndividualLimitReasons(
        std::vector<unsigned char>& reasons) const override;

    bool collectCellTimeStepLimits(std::vector<double>& limits) const override
    {return sim.CollectCellTimeStepLimits(limits);}

    bool cellTimeStepLimitsStale(void) const override
    {return sim.CellTimeStepLimitsStale();}

    bool synchronizedCellTimeStepLimits(
        std::vector<Vector3D> const& point_velocities,
        std::vector<double>& limits,
        std::vector<Vector3D>& accelerations) const override
    {return sim.SynchronizedTimeStepLimits(point_velocities, limits,
                                           accelerations);}

    bool refreshIndividualAccelerations(
        std::vector<Vector3D>& accelerations) const override
    {return sim.RefreshIndividualAccelerations(accelerations);}

    const std::vector<size_t>& getIndividualMeshTargetIDs(void) const
    {return sim.GetIndividualMeshTargetIDs();}

    void restoreIndividualMeshTargetIDs(
        const std::vector<size_t>& target_ids)
    {sim.RestoreIndividualMeshTargetIDs(target_ids);}

    void afterIndividualAMR(void) override;

    void onIndividualForceAllActiveLatch(void) noexcept override;

    void beforeIndividualRebalance(void) noexcept override;

    std::string getName(void) const override { return step_name; }

    SourceStepTiming getSourceStepTiming(void) const override
    {return sim.GetLastSourceStepTiming();}

    MeshBuildTiming getMeshBuildTiming(void) const override
    {return sim.GetLastMeshBuildTiming();}

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
