#ifndef RMTV_COST_CALCULATOR_HPP
#define RMTV_COST_CALCULATOR_HPP
#ifdef RICH_MPI
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "RMTVOpacity.hpp"
#include <cmath>
#include <memory>

// Load-balance weights computed from the cell state handed in, instead of from
// the previous step's per-cell counters. The counter arrays are migrated with
// the cells, but a weight built in place cannot be mis-indexed at all, which is
// what this is here to test.
//
// Cost of a cell over one step is (packets it holds) x (steps each one takes):
//   packets      ~ radiation energy in the cell / packet energy ~ T^4 * V
//                  (population control targets a share of the global budget by
//                  cell energy, so it is the energy, not the emission rate)
//   steps/packet ~ dt * c / (kappa * dx^2)     (diffusion hop rate)
// giving T^4 * V / (kappa * dx^2) = T^4 * V^(1/3) / kappa. The opacity does not
// cancel, and it is what carries the dynamic range: the hot rarefied centre has
// kappa ~ 3e2 against ~5e4 in the shell, so its cells cost a few hundred times
// more. A second term covers the floor of `photons` packets that every cell
// emits regardless of its energy.
class RMTVCostCalculator final : public CostCalculator3D
{
    std::shared_ptr<RMTVOpacity> opacity_;
    double emissionScale_, floorScale_;

  public:
    RMTVCostCalculator(std::shared_ptr<RMTVOpacity> opacity, double emissionScale,
                       double floorScale)
        : opacity_(std::move(opacity)), emissionScale_(emissionScale), floorScale_(floorScale)
    {
    }

    std::vector<double> CalculateCost(const Tessellation3D &tess,
                                      const vector<ComputationalCell3D> &cells) const override
    {
        const size_t N = tess.GetPointNo();
        std::vector<double> weights(N, 1.0);
        for(size_t i = 0; i < N && i < cells.size(); ++i)
        {
            const double V = tess.GetVolume(i);
            if(!(V > 0) || !(cells[i].temperature > 0) || !(cells[i].density > 0))
            {
                continue;
            }
            const double kappa = opacity_->CalcPlanckOpacity(cells[i]);
            if(!(kappa > 0))
            {
                continue;
            }
            const double T2 = cells[i].temperature * cells[i].temperature;
            const double energyTerm = T2 * T2 * std::cbrt(V) / kappa;
            const double floorTerm = 1.0 / (kappa * std::cbrt(V * V));
            weights[i] += emissionScale_ * energyTerm + floorScale_ * floorTerm;
        }
        return weights;
    }
};
#endif // RICH_MPI
#endif // RMTV_COST_CALCULATOR_HPP
