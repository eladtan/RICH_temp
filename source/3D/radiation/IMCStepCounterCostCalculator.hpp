#ifndef IMC_STEP_COUNTER_COST_CALCULATOR_HPP
#define IMC_STEP_COUNTER_COST_CALCULATOR_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"

class IMCStepCounterCostCalculator : public CostCalculator3D
{
public:
    struct Parameters {
        double floorCost;
        double missingCellCost;
        Parameters() : floorCost(1.0), missingCellCost(1.0) {}
    };

    IMCStepCounterCostCalculator(
        std::unordered_map<size_t, double> costByCellID,
        Parameters params = Parameters());

    std::vector<double> CalculateCost(const Tessellation3D& tess,
                                      const vector<ComputationalCell3D>& cells) const override;

private:
    std::unordered_map<size_t, double> costByCellID_;
    Parameters params_;
};

#endif // RICH_MPI

#endif // IMC_STEP_COUNTER_COST_CALCULATOR_HPP
