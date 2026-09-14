#include "Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "Radiation/STAgreyOpacity.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
void require(bool condition, const char *message)
{
    if(!condition) throw std::runtime_error(message);
}

void near(double actual, double expected)
{
    require(std::isfinite(actual) && std::isfinite(expected), "nonfinite opacity");
    require(std::abs(actual - expected) <= 1e-11 * std::max(std::abs(expected), 1e-100),
            "explicit-temperature opacity mismatch");
}

// Exercise the production table loaders/interpolators with known log-linear data.
struct Tables
{
    std::filesystem::path path;
    Tables()
    {
        char pattern[] = "/tmp/rich-opacity-temperature-XXXXXX";
        const char *directory = mkdtemp(pattern);
        require(directory != nullptr, "cannot create temporary opacity tables");
        path = directory;
        std::ofstream temperatures(path / "T.txt"), densities(path / "rho.txt");
        temperatures.precision(17);
        densities.precision(17);
        for(int i = 0; i < 128; ++i)
        {
            temperatures << 1.0 + 0.1 * i << '\n';
            densities << -5.0 + 0.1 * i << '\n';
        }
        for(const char *name : {"ross.txt", "planck.txt", "scatter.txt"})
        {
            std::ofstream table(path / name);
            table.precision(17);
            for(int i = 0; i < 128; ++i)
                for(int j = 0; j < 128; ++j)
                    table << 2.0 * (1.0 + 0.1 * i) + 0.5 * (-5.0 + 0.1 * j) << '\n';
        }
    }
    ~Tables() { std::filesystem::remove_all(path); }
};
}

int main(int argc, char **argv)
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
#endif
    int result = 0;
    try
    {
        ComputationalCell3D cell;
        cell.density = std::exp(0.4);
        cell.temperature = std::exp(4.2);
        cell.pressure = 3.0;
        cell.tracers[0] = 0.7;
        cell.ID = 42;
        cell.Eg.assign(ENERGY_GROUPS_NUM, 7.0);
        const double *groupStorage = cell.Eg.data();
        const double originalTemperature = cell.temperature;
        const double faceTemperature = std::exp(5.3);
        const double energy = units::k_boltz * faceTemperature;
        // The reference deliberately changes a separate cell, outside evaluation.
        ComputationalCell3D reference = cell;
        reference.temperature = faceTemperature;

        auto power = std::make_shared<MCPowerLawOpacity>(2.0, 3.0, 1.2, -2.0, 0.8, 1.0);
        RICHRadiationOpacityAdapter adapter(power);
        near(adapter.CalcPlanckOpacityAtTemperature(cell, faceTemperature),
             2.0 * std::pow(cell.density, 1.2) * std::pow(faceTemperature, -2.0));
        near(adapter.CalcScatteringOpacityAtTemperature(cell, energy, faceTemperature),
             3.0 * std::pow(cell.density, 0.8) * faceTemperature);
        near(power->CalcPlanckOpacity(cell), power->CalcPlanckOpacityAtTemperature(cell, originalTemperature));

        GrayPowerLawOpacity grayPower(1.0, 0.0, 0.0, 2.0, 1.2, -2.0);
        near(grayPower.CalcAbsorptionOpacityAtTemperature(cell, energy, faceTemperature),
             grayPower.CalcAbsorptionOpacity(reference, energy));
        FreeFreeAbsorptionOpacityMultigroup freeFree(1.0, {energy}, {0.1 * energy, 10.0 * energy});
        near(freeFree.CalcAbsorptionOpacityAtTemperature(cell, energy, faceTemperature),
             freeFree.CalcAbsorptionOpacity(reference, energy));

        Tables tables;
        STAgreyOpacity grayTable(tables.path.string() + "/");
        GraySTAopacity groupTable(tables.path.string() + "/");
        for(double temperature : {std::exp(3.1), faceTemperature, std::exp(7.1)})
        {
            const double expected = temperature * temperature * std::sqrt(cell.density);
            near(grayTable.CalcPlanckOpacityAtTemperature(cell, temperature), expected);
            near(grayTable.CalcScatteringOpacityAtTemperature(cell, temperature), expected);
            near(groupTable.CalcAbsorptionOpacityAtTemperature(cell, energy, temperature), expected);
            near(groupTable.CalcScatteringOpacityAtTemperature(cell, energy, temperature), expected);
        }

        auto legacy = [](const ComputationalCell3D &c, double e) { return c.density * c.temperature + e; };
        auto explicitTemperature = [&cell, originalTemperature](const ComputationalCell3D &c, double e, double t)
        {
            require(&c == &cell, "opacity callback received a copied cell");
            require(c.temperature == originalTemperature, "cell temperature was modified");
            return c.density * t + e;
        };
        AnalyticOpacity analytic(legacy, legacy, legacy, {energy}, {0.1 * energy, 10.0 * energy},
                                 explicitTemperature, explicitTemperature);
        near(analytic.CalcAbsorptionOpacityAtTemperature(cell, energy, faceTemperature),
             cell.density * faceTemperature + energy);
        near(analytic.CalcScatteringOpacityAtTemperature(cell, energy, faceTemperature),
             cell.density * faceTemperature + energy);
        near(analytic.CalcAbsorptionOpacity(cell, energy), legacy(cell, energy));
        AnalyticOpacity oldCallbacks(legacy, legacy, legacy, {energy}, {0.1 * energy, 10.0 * energy});
        near(oldCallbacks.CalcAbsorptionOpacity(cell, energy), legacy(cell, energy));
        bool rejected = false;
        try { oldCallbacks.CalcAbsorptionOpacityAtTemperature(cell, energy, faceTemperature); }
        catch(const UniversalError &) { rejected = true; }
        require(rejected, "legacy callback silently ignored the evaluation temperature");

        require(cell.temperature == originalTemperature && cell.density == std::exp(0.4) &&
                cell.pressure == 3.0 && cell.tracers[0] == 0.7 && cell.ID == 42 &&
                cell.Eg.data() == groupStorage && cell.Eg.front() == 7.0,
                "opacity evaluation modified the computational cell");
        std::cout << "OPACITY_TEMPERATURE_PASS\n";
    }
    catch(const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    catch(const UniversalError &error)
    {
        std::cerr << error.getErrorMessage() << '\n';
        result = 1;
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return result;
}
