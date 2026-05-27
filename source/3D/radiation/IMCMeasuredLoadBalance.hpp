#ifndef IMC_MEASURED_LOAD_BALANCE_HPP
#define IMC_MEASURED_LOAD_BALANCE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace imc_measured_lb {

struct Parameters {
    double floorCost = 1.0;
    double stepWeight = 1.0;
    double particleWeight = 0.0;
    double medianClampFactor = 30.0;
    double missingCellCost = 1.0;
    double predictiveWeight = 0.0;
    double grayZeroStepInflation = 2.0;
    double multigroupZeroStepInflation = 5.0;
    bool useMedianClamp = true;
};

struct LocalCellMeasurement {
    size_t globalCellID;
    size_t localCellIndex;
    size_t stepCount;
    size_t particleCount;
    double predictiveCost;
};

inline double MedianPositive(std::vector<double> values)
{
    values.erase(
        std::remove_if(values.begin(), values.end(),
                       [](double x) { return !(x > 0.0) || !std::isfinite(x); }),
        values.end());

    if (values.empty())
        return 1.0;

    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<LocalCellMeasurement> BuildLocalMeasurements(
    std::vector<size_t> const& cellIDs,
    std::vector<size_t> const& steps,
    std::vector<size_t> const& particlesPerCell);

std::unordered_map<size_t, double> BuildMeasuredCosts(
    std::vector<LocalCellMeasurement> const& globalMeasurements,
    Parameters const& params,
    bool multigroup);

#ifdef RICH_MPI
std::vector<LocalCellMeasurement> GatherMeasurementsAllRanks(
    std::vector<LocalCellMeasurement> const& localMeasurements,
    MPI_Comm comm);

void PrintMeasuredLBDiagnostics(
    std::vector<LocalCellMeasurement> const& localMeasurements,
    std::unordered_map<size_t, double> const& costByCellID,
    bool multigroup,
    MPI_Comm comm);

void PrintPostRepartitionDiagnostics(
    std::unordered_map<size_t, double> const& costByCellID,
    std::vector<size_t> const& newCellIDs,
    double missingCellCost,
    bool multigroup,
    MPI_Comm comm);
#endif

} // namespace imc_measured_lb

#endif // IMC_MEASURED_LOAD_BALANCE_HPP
