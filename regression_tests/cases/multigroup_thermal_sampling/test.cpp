// Thermal frequency sampling of the RICH opacity adapter
// (RICHRadiationOpacityAdapter::GetThermalEnergy and
// SampleThermalEnergyInGroup), which places emitted and re-emitted
// multigroup packets in frequency.
//
// The cumulative the adapter inverts holds kappa_g b_g in cm^-1, far below
// 1e-12 in low-density gas. STORM::LinearInterpolation used to treat any
// point within an absolute 1e-12 of a node as the node, so draws landed on
// the next group's lower edge (snap_151: emission in 15-100 eV relabelled to
// exactly 100 eV, the edge of a transparent group, which then escaped).
// At absorption scales 1e-40, 1e-20, 1e-12 and 1 cm^-1:
//  1. SampleThermalEnergyInGroup(g) returns a frequency inside group g,
//     below its upper boundary;
//  2. GetThermalEnergy reproduces GetThermalGroupPdf within 5 sigma per
//     group and never returns a group boundary;
//  3. the samples do not depend on the scale (the same random numbers give
//     the same frequency to 1e-9 relative, rounding only);
//  4. inside a group the mean frequency is the group centre within 5 sigma
//     (the cumulative is linear inside a group);
// and STORM::LinearInterpolation gives the same answer for x and 1e-30 x.

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/Radiation/conj_grad_solve.hpp"
#include "source/misc/universal_error.hpp"

static_assert(ENERGY_GROUPS_NUM == 4,
              "multigroup_thermal_sampling is built with --energy_groups_num=4");

namespace
{

constexpr double kEv = 1.602176634e-12;

// Absorption scale * a frequency shape, so the groups' weights differ.
class ScaledOpacity final : public OpacityCalculator
{
public:
    explicit ScaledOpacity(double scale) : scale_(scale) {}

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override { return scale_ * cell.density; }
    double CalcPlanckOpacityAtTemperature(ComputationalCell3D const& cell, double) const override { return scale_ * cell.density; }
    double CalcScatteringOpacity(ComputationalCell3D const&) const override { return 0.0; }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const&, double) const override { return 0.0; }
    double CalcScatteringOpacity(ComputationalCell3D const&, double) const override { return 0.0; }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const&, double, double) const override { return 0.0; }
    double CalcAbsorptionOpacity(ComputationalCell3D const&, double frequency) const override
    {
        return scale_ * std::pow(frequency / kEv + 1.0, -1.5);
    }
    double CalcAbsorptionOpacityAtTemperature(ComputationalCell3D const& cell, double frequency, double) const override
    {
        return CalcAbsorptionOpacity(cell, frequency);
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const&) const override { return 1.0; }
    double CalcDiffusionCoefficient(ComputationalCell3D const&, double) const override { return 1.0; }

private:
    double scale_;
};

using Adapter = RICHRadiationOpacityAdapter;
using Boundaries = Adapter::GroupBoundaries;

Boundaries MakeBoundaries()
{
    Boundaries b{};
    std::array<double, ENERGY_GROUPS_NUM + 1> const eV = {1e-2, 3.0, 10.0, 30.0, 1e3};
    for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
        b[g] = eV[g] * kEv;
    return b;
}

ComputationalCell3D MakeCell()
{
    ComputationalCell3D cell;
    cell.density = 1e-12;
    cell.temperature = 6e4;
    cell.ID = 7;
    return cell;
}

} // namespace

int main()
{
    std::ofstream metrics("multigroup_thermal_sampling_metrics.txt");
    metrics << std::setprecision(10);
    bool pass = true;
    auto check = [&](bool ok, std::string const& what) {
        std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
        pass = pass && ok;
    };

    try {
        Boundaries const boundaries = MakeBoundaries();
        ComputationalCell3D const cell = MakeCell();
        std::vector<double> const scales = {1e-40, 1e-20, 1e-12, 1.0};
        std::vector<std::vector<double>> reference;  // samples at each scale, same random numbers
        for (double const scale : scales) {
            Adapter adapter(std::make_shared<ScaledOpacity>(scale));
            std::mt19937_64 rng(20261011);
            std::uniform_real_distribution<double> unit(0.0, 1.0);

            // 1. and 4. within-group sampling
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                size_t const n = 20000;
                size_t outside = 0;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    double const f = adapter.SampleThermalEnergyInGroup(cell, g, unit(rng), boundaries);
                    if (!(f >= boundaries[g] && f < boundaries[g + 1]) || adapter.findGroup(f, boundaries) != g)
                        ++outside;
                    sum += f;
                }
                double const width = boundaries[g + 1] - boundaries[g];
                double const mean = sum / static_cast<double>(n);
                double const centre = 0.5 * (boundaries[g] + boundaries[g + 1]);
                double const sigma = width / std::sqrt(12.0 * static_cast<double>(n));
                std::string const tag = "scale " + std::to_string(scale) + " group " + std::to_string(g);
                metrics << "in_group_outside_scale" << scale << "_g" << g << " " << outside << "\n";
                check(outside == 0, tag + ": every in-group sample lies inside the group (" +
                      std::to_string(outside) + " outside)");
                check(std::abs(mean - centre) <= 5.0 * sigma,
                      tag + ": in-group mean at the centre ((mean-centre)/sigma=" +
                      std::to_string((mean - centre) / sigma) + ")");
            }

            // 2. and 3. full thermal sampling
            Adapter::GroupArray const pdf = adapter.GetThermalGroupPdf(cell, boundaries);
            size_t const n = 200000;
            std::array<double, ENERGY_GROUPS_NUM> counts{};
            size_t onBoundary = 0;
            std::vector<double> samples;
            samples.reserve(n);
            for (size_t k = 0; k < n; ++k) {
                double const f = adapter.GetThermalEnergy(cell, unit(rng), boundaries);
                for (size_t g = 1; g <= ENERGY_GROUPS_NUM; ++g)
                    if (f == boundaries[g])
                        ++onBoundary;
                counts[adapter.findGroup(f, boundaries)] += 1.0;
                samples.push_back(f);
            }
            reference.push_back(std::move(samples));
            std::string const tag = "scale " + std::to_string(scale);
            check(onBoundary == 0, tag + ": no thermal sample on a group boundary (" + std::to_string(onBoundary) + ")");
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                double const expected = static_cast<double>(n) * pdf[g];
                double const sigma = std::sqrt(std::max(expected * (1.0 - pdf[g]), 1.0));
                metrics << "thermal_pull_scale" << scale << "_g" << g << " " << (counts[g] - expected) / sigma << "\n";
                check(std::abs(counts[g] - expected) <= 5.0 * sigma,
                      tag + " group " + std::to_string(g) + ": thermal sampling follows the group pdf (pull=" +
                      std::to_string((counts[g] - expected) / sigma) + ")");
            }
        }
        // 3. scale independence
        double worst = 0.0;
        for (size_t s = 1; s < reference.size(); ++s)
            for (size_t k = 0; k < reference[s].size(); ++k)
                worst = std::max(worst, std::abs(reference[s][k] - reference[0][k]) / reference[0][k]);
        metrics << "scale_independence_max_relative " << worst << "\n";
        // Rounding of (target - c_g) / width in narrow groups gives ~1e-11;
        // the old edge snapping gave differences of order 1.
        check(worst <= 1e-9, "thermal samples do not depend on the opacity scale (max relative difference " +
              std::to_string(worst * 1e12) + "e-12)");

        // STORM::LinearInterpolation is scale invariant
        std::vector<double> const x = {0.0, 1.0, 3.0, 6.0};
        std::vector<double> const y = {0.0, 10.0, 20.0, 30.0};
        std::vector<double> xs;
        for (double v : x)
            xs.push_back(v * 1e-30);
        double worstInterp = 0.0;
        for (double xi : {0.0, 0.5, 0.999999999, 1.0, 2.0, 5.999999, 6.0}) {
            double const a = STORM::LinearInterpolation(x, y, xi);
            double const b = STORM::LinearInterpolation(xs, y, xi * 1e-30);
            worstInterp = std::max(worstInterp, std::abs(a - b));
        }
        metrics << "linear_interpolation_scale_difference " << worstInterp << "\n";
        check(worstInterp <= 1e-9, "LinearInterpolation gives the same result for x and 1e-30 x (max difference " +
              std::to_string(worstInterp) + ")");
    } catch (UniversalError const& eo) {
        std::cout << "FAIL exception: " << eo.getErrorMessage() << std::endl;
        pass = false;
    } catch (std::exception const& e) {
        std::cout << "FAIL exception: " << e.what() << std::endl;
        pass = false;
    }

    metrics << "pass " << (pass ? 1 : 0) << "\n";
    std::cout << (pass ? "multigroup_thermal_sampling PASSED" : "multigroup_thermal_sampling FAILED") << std::endl;
    return pass ? 0 : 1;
}
