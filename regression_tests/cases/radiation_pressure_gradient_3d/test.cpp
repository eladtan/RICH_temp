#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/universal_error.hpp"
#include "source/monte/boundary/RigidBoundary.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"

namespace
{
constexpr double timestep = 0.4;
constexpr double maximumAllowedError = 1e-10;
const Vector3D exactGradient(0.23, -0.31, 0.17);

class ZeroOpacity final : public OpacityCalculator
{
public:
    double CalcPlanckOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &, double) const override
    {
        return 0.0;
    }
};

struct Mode
{
    const char *name;
    bool mmc;
    bool ddmc;
};

struct ModeResult
{
    double maxAbsError = 0.0;
    std::size_t cellsChecked = 0;
};

double RadiationEnergy(const Vector3D &position)
{
    return 4.0 + exactGradient.x * position.x
        + exactGradient.y * position.y
        + exactGradient.z * position.z;
}

std::vector<ComputationalCell3D> MakeCells(const Tessellation3D &tess,
                                           const IdealGas &eos)
{
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        ComputationalCell3D &cell = cells[i];
        cell.ID = i;
        cell.density = 1.0;
        cell.temperature = 1.0;
        cell.velocity = Vector3D(0.0, 0.0, 0.0);
        cell.internal_energy = eos.dT2e(
            cell.density, cell.temperature, cell.tracers,
            ComputationalCell3D::tracerNames);
        cell.pressure = eos.de2p(
            cell.density, cell.internal_energy, cell.tracers,
            ComputationalCell3D::tracerNames);
        cell.Erad = 0.0;
    }
    return cells;
}

ModeResult RunMode(Tessellation3D &tess, const Mode &mode)
{
    IdealGas eos(5.0 / 3.0, 1.0, 1.0, 0.0);
    std::vector<ComputationalCell3D> cells = MakeCells(tess, eos);
    std::vector<Conserved3D> extensives(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    auto eosPtr = std::make_shared<IdealGas>(eos);
    auto opacity = std::make_shared<ZeroOpacity>();
    auto boundary =
        std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

    RadiationIMCParameters parameters;
    parameters.newPhotonsPerCell = 1;
    parameters.withHydro = true;
    parameters.diffusionPressureGradient = true;
    parameters.MMC = mode.mmc;
    parameters.withDDMC = mode.ddmc;
    parameters.noHydroFeedback = false;
    parameters.withMultigroupOpacity = false;
    parameters.withRandomWalk = false;
    for(std::size_t g = 0; g < parameters.energyBoundaries.size(); ++g)
        parameters.energyBoundaries[g] = static_cast<double>(g + 1);
    parameters.energyBoundariesProvided = true;

    RadiationIMC physics(
        tess, boundary, cells, extensives, eosPtr, opacity, parameters);
    std::vector<double> &radiationTally = physics.getEradTimeAvg();
    for(std::size_t i = 0; i < radiationTally.size(); ++i)
    {
        radiationTally[i] = RadiationEnergy(tess.GetCellCM(i))
            * timestep * tess.GetVolume(i);
    }

    const std::vector<RadiationIMC::Particle> particles;
    physics.postStep(particles, timestep);

    ModeResult result;
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        const Vector3D center = tess.GetCellCM(i);
        if(std::abs(center.x) >= 0.26 ||
           std::abs(center.y) >= 0.26 ||
           std::abs(center.z) >= 0.26)
        {
            continue;
        }

        const double scale = -3.0 / (timestep * tess.GetVolume(i));
        const Vector3D recoveredGradient = extensives[i].momentum * scale;
        const double cellError = std::max({
            std::abs(recoveredGradient.x - exactGradient.x),
            std::abs(recoveredGradient.y - exactGradient.y),
            std::abs(recoveredGradient.z - exactGradient.z)});
        if(!std::isfinite(cellError))
            result.maxAbsError = std::numeric_limits<double>::infinity();
        else
            result.maxAbsError = std::max(result.maxAbsError, cellError);
        ++result.cellsChecked;
    }
    return result;
}
}

int main()
{
    try
    {
        const Vector3D lower(-0.5, -0.5, -0.5);
        const Vector3D upper(0.5, 0.5, 0.5);
        std::vector<Vector3D> points = CartesianMesh(5, 5, 5, lower, upper);
        for(std::size_t i = 0; i < points.size(); ++i)
        {
            const double n = static_cast<double>(i + 1);
            points[i].x += 0.018 * std::sin(0.73 * n);
            points[i].y += 0.018 * std::sin(1.11 * n + 0.4);
            points[i].z += 0.018 * std::sin(1.57 * n + 0.9);
        }

        Voronoi3D tess(lower, upper);
        tess.Build(points);

        const std::array<Mode, 3> modes{{
            {"imc", false, false},
            {"mmc", true, false},
            {"ddmc", false, true}}};
        std::array<ModeResult, 3> results{};
        bool pass = true;
        for(std::size_t i = 0; i < modes.size(); ++i)
        {
            results[i] = RunMode(tess, modes[i]);
            pass = pass && results[i].cellsChecked > 0
                && results[i].maxAbsError < maximumAllowedError;
        }

        std::ofstream out("radiation_pressure_gradient_3d_metrics.txt");
        out << std::scientific << std::setprecision(16);
        for(std::size_t i = 0; i < modes.size(); ++i)
        {
            out << modes[i].name << "_max_abs "
                << results[i].maxAbsError << '\n';
            out << modes[i].name << "_cells "
                << results[i].cellsChecked << '\n';
        }
        out << "pass " << (pass ? 1 : 0) << '\n';

        for(std::size_t i = 0; i < modes.size(); ++i)
        {
            std::cout << modes[i].name << ": max_abs="
                      << results[i].maxAbsError << ", cells="
                      << results[i].cellsChecked << '\n';
        }
        std::cout << (pass ? "PASS" : "FAIL") << std::endl;
        return pass ? 0 : 1;
    }
    catch(const UniversalError &error)
    {
        reportError(error);
        return 1;
    }
    catch(const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
