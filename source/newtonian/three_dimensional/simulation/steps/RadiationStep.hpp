#ifndef RADIATION_STEP_HPP
#define RADIATION_STEP_HPP

#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "Radiation/RadiationDriver.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/ProgressTracker.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "PhysicsStep.hpp"
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

    double suggestTimeStep(void) const override;

    std::string getName(void) const override;

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
    #ifdef RICH_MPI
        std::shared_ptr<CostCalculator3D> cost;
    #endif // RICH_MPI
};

#endif // RADIATION_STEP_HPP