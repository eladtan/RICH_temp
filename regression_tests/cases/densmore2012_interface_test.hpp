#ifndef RICH_REGRESSION_TESTS_DENSMORE2012_INTERFACE_TEST_HPP
#define RICH_REGRESSION_TESTS_DENSMORE2012_INTERFACE_TEST_HPP

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/mpi/mpi_commands.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "densmore2012_interface_mesh.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/RadiationOpacity.hpp"
#include "source/monte/population/Comb.hpp"
#include "source/monte/boundary/SideTemperature.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/3D/radiation/IMCCostCalculator.hpp"

namespace densmore2012_interface_test
{
    namespace fs = std::filesystem;

    struct RunOptions
    {
        bool useCenteredInterfaceMesh = false;
        std::size_t ddmcMaxGroupCutoff = ENERGY_GROUPS_NUM;
        bool ddmcInterfaceDiagnostics = false;
    };

    class InterfaceOpacity : public OpacityCalculator
    {
    public:
        InterfaceOpacity(double sigma0Left, double sigma0Right,
                         const std::vector<double> &groupCenters,
                         const std::vector<double> &groupBoundaries)
            : sigma0Left_(sigma0Left), sigma0Right_(sigma0Right),
              groupCenters_(groupCenters), groupBoundaries_(groupBoundaries)
        {
            this->energy_groups_center = groupCenters_;
            this->energy_groups_boundary = groupBoundaries_;
        }

        double CalcPlanckOpacity(
            const ComputationalCell3D &cell) const override
        {
            double const sigma0 = getSigma0(cell);
            double const kT = units::k_boltz * cell.temperature;
            double const sqrtKT = std::sqrt(kT);

            double weightedSum = 0.0;
            double totalWeight = 0.0;
            for(std::size_t g = 0; g < groupCenters_.size(); ++g)
            {
                double const a = groupBoundaries_[g] / kT;
                double const b = groupBoundaries_[g + 1] / kT;
                double const Bg = planck_integral::planck_integral(a, b);
                double const Eg = groupCenters_[g];
                double const sigmaG = sigma0 / (sqrtKT * Eg * Eg * Eg);
                weightedSum += sigmaG * Bg;
                totalWeight += Bg;
            }
            return weightedSum / totalWeight;
        }

        double CalcScatteringOpacity(
            const ComputationalCell3D &) const override
        {
            return 0.0;
        }

        double CalcAbsorptionOpacity(const ComputationalCell3D &cell,
                                     double energy) const override
        {
            double const sigma0 = getSigma0(cell);
            double const kT = units::k_boltz * cell.temperature;
            energy = std::clamp(energy, groupBoundaries_.front(),
                                groupBoundaries_.back());
            auto const it = std::upper_bound(groupBoundaries_.begin(),
                                             groupBoundaries_.end(), energy);
            std::size_t const idx = static_cast<std::size_t>(
                std::distance(groupBoundaries_.begin(), it));
            std::size_t const g = idx == 0 ? 0 :
                std::min(idx - 1, groupCenters_.size() - 1);
            double const Eg = groupCenters_[g];
            return sigma0 / (std::sqrt(kT) * Eg * Eg * Eg);
        }

    private:
        double getSigma0(const ComputationalCell3D &cell) const
        {
            return cell.tracers[0] > 0.5 ? sigma0Left_ : sigma0Right_;
        }

        double sigma0Left_;
        double sigma0Right_;
        std::vector<double> groupCenters_;
        std::vector<double> groupBoundaries_;
    };

    enum CellField : std::size_t
    {
        X = 0,
        GENERATOR_X,
        X_LEFT,
        X_RIGHT,
        VOLUME,
        TEMPERATURE,
        DENSITY,
        INTERNAL_ENERGY_SPECIFIC,
        INTERNAL_ENERGY_EXTENSIVE,
        ERAD_SPECIFIC,
        ERAD_TIME_AVG_RAW,
        FLECK,
        PLANCK_OPACITY,
        MATERIAL_REGION,
        DDMC_ELIGIBLE,
        DDMC_BOUNDARY_EXCLUDED,
        DDMC_UNSUPPORTED_BOUNDARY_FACES,
        DDMC_GROUP_CUTOFF,
        DDMC_SIGMA_T,
        DDMC_SIGMA_A,
        DDMC_SIGMA_ENERGY_ABS,
        DDMC_SIGMA_DIFFUSION,
        DDMC_SIGMA_PARTICLE_GATE,
        DDMC_SIGMA_GROUP_EXIT,
        DDMC_GAMMA,
        DDMC_DIFFUSION_COEFFICIENT,
        DDMC_TOTAL_LEAK_RATE,
        DDMC_INTERNAL_LEAK_RATE_SUM,
        DDMC_INTERNAL_CONDUCTANCE_SUM,
        DDMC_CHANNEL_RATE_SUM,
        DDMC_TRANSPORT_CHANNEL_RATE_SUM,
        DDMC_BOUNDARY_RATE_SUM,
        DDMC_MIXED_FACE_COUNT,
        DDMC_FACE_COUNT,
        CELL_FIELD_COUNT
    };

    static const std::array<const char *, CELL_FIELD_COUNT> cellFieldNames = {{
        "x_cm",
        "generator_x_cm",
        "x_left_cm",
        "x_right_cm",
        "volume_cm3",
        "temperature_K",
        "density",
        "internal_energy_specific",
        "internal_energy_extensive",
        "Erad_specific",
        "Erad_time_avg_raw",
        "fleck",
        "planck_opacity",
        "material_region",
        "ddmc_eligible",
        "ddmc_boundary_excluded",
        "ddmc_unsupported_boundary_faces",
        "ddmc_group_cutoff",
        "ddmc_sigmaT",
        "ddmc_sigmaA",
        "ddmc_sigmaEnergyAbs",
        "ddmc_sigmaDiffusion",
        "ddmc_sigmaParticleGate",
        "ddmc_sigmaGroupExit",
        "ddmc_gamma",
        "ddmc_D",
        "ddmc_total_leak_rate",
        "ddmc_internal_leak_rate_sum",
        "ddmc_internal_conductance_sum",
        "ddmc_channel_rate_sum",
        "ddmc_transport_channel_rate_sum",
        "ddmc_boundary_rate_sum",
        "ddmc_mixed_face_count",
        "ddmc_face_count"
    }};

    struct CellRecord
#ifdef RICH_MPI
        : public Serializable
#endif
    {
        std::array<double, CELL_FIELD_COUNT> value{};

#ifdef RICH_MPI
        std::size_t dump(Serializer *serializer) const override
        {
            std::size_t offset = 0;
            for(double const x : value)
                offset += serializer->insert(x);
            return offset;
        }

        std::size_t load(const Serializer *serializer,
                         std::size_t byteOffset) override
        {
            std::size_t read = 0;
            for(double &x : value)
                read += serializer->extract(x, byteOffset + read);
            return read;
        }
#endif

        bool operator<(CellRecord const &other) const
        {
            return value[X] < other.value[X];
        }
    };

    inline double ExtractDebugScalar(std::string const &debug,
                                     std::string const &key,
                                     double fallback = 0.0)
    {
        std::string const needle = " " + key + "=";
        std::size_t const keyPos = debug.find(needle);
        if(keyPos == std::string::npos)
            return fallback;

        std::size_t const begin = keyPos + needle.size();
        std::size_t end = debug.find(' ', begin);
        if(end == std::string::npos)
            end = debug.size();

        try
        {
            return std::stod(debug.substr(begin, end - begin));
        }
        catch(std::exception const &)
        {
            return fallback;
        }
    }

    inline std::pair<double, double> CellXBounds(Tessellation3D const &grid,
                                                  std::size_t cellIndex)
    {
        double left = std::numeric_limits<double>::infinity();
        double right = -std::numeric_limits<double>::infinity();
        for(std::size_t const faceIndex : grid.GetCellFaces(cellIndex))
        {
            double const faceX = grid.FaceCM(faceIndex).x;
            left = std::min(left, faceX);
            right = std::max(right, faceX);
        }
        return std::make_pair(left, right);
    }

    inline CellRecord MakeCellRecord(
        std::size_t cellIndex,
        Tessellation3D const &grid,
        std::vector<ComputationalCell3D> const &cells,
        std::vector<Conserved3D> const &conserved,
        RadiationIMC const &physics)
    {
        CellRecord record;
        ComputationalCell3D const &cell = cells[cellIndex];
        std::pair<double, double> const bounds =
            CellXBounds(grid, cellIndex);

        record.value[X] = grid.GetCellCM(cellIndex).x;
        record.value[GENERATOR_X] = grid.GetMeshPoint(cellIndex).x;
        record.value[X_LEFT] = bounds.first;
        record.value[X_RIGHT] = bounds.second;
        record.value[VOLUME] = grid.GetVolume(cellIndex);
        record.value[TEMPERATURE] = cell.temperature;
        record.value[DENSITY] = cell.density;
        record.value[INTERNAL_ENERGY_SPECIFIC] = cell.internal_energy;
        record.value[INTERNAL_ENERGY_EXTENSIVE] =
            conserved[cellIndex].internal_energy;
        record.value[ERAD_SPECIFIC] = cell.Erad;
        record.value[MATERIAL_REGION] = cell.tracers[0] > 0.5 ? 0.0 : 1.0;

        std::vector<double> const &eradTime = physics.getEradTimeAvg();
        if(cellIndex < eradTime.size())
            record.value[ERAD_TIME_AVG_RAW] = eradTime[cellIndex];

        std::vector<double> const &fleck = physics.getFactorFleck();
        if(cellIndex < fleck.size())
            record.value[FLECK] = fleck[cellIndex];

        std::vector<double> const &planck = physics.getPlanckOpacities();
        if(cellIndex < planck.size())
            record.value[PLANCK_OPACITY] = planck[cellIndex];

        double const diagnosticFrequency =
            ComputationalCell3D::energyBoundaries[0];
        std::string const debug = physics.getAccelerationDebugInfo(
            cellIndex, diagnosticFrequency);

        record.value[DDMC_ELIGIBLE] =
            ExtractDebugScalar(debug, "eligible");
        record.value[DDMC_BOUNDARY_EXCLUDED] =
            ExtractDebugScalar(debug, "boundary_excluded");
        record.value[DDMC_UNSUPPORTED_BOUNDARY_FACES] =
            ExtractDebugScalar(debug, "unsupported_boundary_faces");
        record.value[DDMC_GROUP_CUTOFF] =
            ExtractDebugScalar(debug, "group_cutoff");
        record.value[DDMC_SIGMA_T] =
            ExtractDebugScalar(debug, "sigmaT");
        record.value[DDMC_SIGMA_A] =
            ExtractDebugScalar(debug, "sigmaA");
        record.value[DDMC_SIGMA_ENERGY_ABS] =
            ExtractDebugScalar(debug, "sigmaEnergyAbs");
        record.value[DDMC_SIGMA_DIFFUSION] =
            ExtractDebugScalar(debug, "sigmaDiffusion");
        record.value[DDMC_SIGMA_PARTICLE_GATE] =
            ExtractDebugScalar(debug, "sigmaParticleGate");
        record.value[DDMC_SIGMA_GROUP_EXIT] =
            ExtractDebugScalar(debug, "sigmaGroupExit");
        record.value[DDMC_GAMMA] =
            ExtractDebugScalar(debug, "ddmc_gamma");
        record.value[DDMC_DIFFUSION_COEFFICIENT] =
            ExtractDebugScalar(debug, "D");
        record.value[DDMC_TOTAL_LEAK_RATE] =
            ExtractDebugScalar(debug, "leak_rate");
        record.value[DDMC_INTERNAL_LEAK_RATE_SUM] =
            ExtractDebugScalar(debug, "ddmc_internal_leak_rate_sum");
        record.value[DDMC_INTERNAL_CONDUCTANCE_SUM] =
            ExtractDebugScalar(debug, "ddmc_internal_conductance_sum");
        record.value[DDMC_CHANNEL_RATE_SUM] =
            ExtractDebugScalar(debug, "ddmc_channel_rate_sum");
        record.value[DDMC_TRANSPORT_CHANNEL_RATE_SUM] =
            ExtractDebugScalar(debug, "ddmc_transport_channel_rate_sum");
        record.value[DDMC_BOUNDARY_RATE_SUM] =
            ExtractDebugScalar(debug, "ddmc_boundary_rate_sum");
        record.value[DDMC_MIXED_FACE_COUNT] =
            ExtractDebugScalar(debug, "ddmc_mixed_face_count");
        record.value[DDMC_FACE_COUNT] =
            ExtractDebugScalar(debug, "faces");

        return record;
    }

    inline std::vector<CellRecord> GatherCellRecords(
        Tessellation3D const &grid,
        std::vector<ComputationalCell3D> const &cells,
        std::vector<Conserved3D> const &conserved,
        RadiationIMC const &physics,
        double xMin,
        double xMax,
        int rank)
    {
        std::vector<CellRecord> local;
        for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        {
            double const x = grid.GetCellCM(i).x;
            if(x >= xMin && x <= xMax)
                local.push_back(MakeCellRecord(
                    i, grid, cells, conserved, physics));
        }

#ifdef RICH_MPI
        std::vector<CellRecord> gathered =
            MPI_Gatherv_serializable(local, 0, MPI_COMM_WORLD);
#else
        std::vector<CellRecord> gathered = local;
#endif
        if(rank == 0)
            std::sort(gathered.begin(), gathered.end());
        return gathered;
    }

    inline void WriteCellHeader(std::ostream &out,
                                bool includeStepAndTime)
    {
        if(includeStepAndTime)
            out << "step\ttime_s\t";
        for(std::size_t i = 0; i < cellFieldNames.size(); ++i)
        {
            if(i > 0)
                out << '\t';
            out << cellFieldNames[i];
        }
        out << '\n';
    }

    inline void WriteCellRecord(std::ostream &out,
                                CellRecord const &record)
    {
        for(std::size_t i = 0; i < record.value.size(); ++i)
        {
            if(i > 0)
                out << '\t';
            out << record.value[i];
        }
        out << '\n';
    }

    enum CounterIndex : std::size_t
    {
        DDMC_STEPS = 0,
        DDMC_LEAKS,
        DDMC_CENSUS,
        DDMC_UPSCATTER,
        DDMC_FALLBACK,
        DDMC_RESIDENT_LEAKS,
        DDMC_TRANSPORT_LEAKS,
        DDMC_REMOTE_RESIDENT_LEAKS,
        DDMC_INTERFACE_INCIDENT,
        DDMC_INTERFACE_ADMITTED,
        DDMC_INTERFACE_REFLECTED,
        DDMC_INTERFACE_BYPASS,
        DDMC_INVALID_GEOMETRY,
        COUNTER_COUNT
    };

    inline void WriteGlobalHeader(std::ostream &out)
    {
        out << "step\ttime_s\tmaterial_energy\tradiation_cell_energy"
            << "\terad_time_avg_raw_sum\tvolume_averaged_temperature"
            << "\tddmc_eligible_cells\tfirst_ddmc_x_cm"
            << "\tfirst_thick_ddmc_x_cm"
            << "\tfirst_thick_cell_ddmc_eligible"
            << "\tddmc_steps\tddmc_leaks\tddmc_census\tddmc_upscatter"
            << "\tddmc_fallback\tddmc_resident_leaks"
            << "\tddmc_transport_leaks\tddmc_remote_resident_leaks"
            << "\tddmc_interface_incident\tddmc_interface_admitted"
            << "\tddmc_interface_reflected\tddmc_interface_bypass"
            << "\tddmc_invalid_geometry\n";
    }

    inline void WriteGlobalDiagnostics(
        std::ostream &out,
        std::size_t step,
        double time,
        Tessellation3D const &grid,
        std::vector<ComputationalCell3D> const &cells,
        std::vector<Conserved3D> const &conserved,
        RadiationIMC const &physics,
        int rank)
    {
        std::array<double, 4> localSums{{0.0, 0.0, 0.0, 0.0}};
        double localVolume = 0.0;
        double localFirstDDMCX = std::numeric_limits<double>::infinity();
        double localFirstThickDDMCX = std::numeric_limits<double>::infinity();
        unsigned long long localEligibleCount = 0;
        unsigned long long localFirstThickEligible = 0;

        std::vector<double> const &eradTime = physics.getEradTimeAvg();
        for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        {
            double const volume = grid.GetVolume(i);
            double const x = grid.GetCellCM(i).x;
            localSums[0] += conserved[i].internal_energy;
            localSums[1] += cells[i].density * volume * cells[i].Erad;
            if(i < eradTime.size())
                localSums[2] += eradTime[i];
            localSums[3] += cells[i].temperature * volume;
            localVolume += volume;

            std::string const debug = physics.getAccelerationDebugInfo(
                i, ComputationalCell3D::energyBoundaries[0]);
            bool const eligible =
                ExtractDebugScalar(debug, "eligible") > 0.5;
            if(eligible)
            {
                ++localEligibleCount;
                localFirstDDMCX = std::min(localFirstDDMCX, x);
                if(x >= densmore2012_interface_mesh::interfacePosition)
                    localFirstThickDDMCX =
                        std::min(localFirstThickDDMCX, x);
            }
            if(x >= densmore2012_interface_mesh::interfacePosition &&
               x < densmore2012_interface_mesh::interfacePosition +
                   densmore2012_interface_mesh::thickCellWidth && eligible)
            {
                localFirstThickEligible = 1;
            }
        }

        std::array<unsigned long long, COUNTER_COUNT> localCounters{};
        localCounters[DDMC_STEPS] =
            static_cast<unsigned long long>(physics.getDDMCStepCount());
        localCounters[DDMC_LEAKS] =
            static_cast<unsigned long long>(physics.getDDMCLeakCount());
        localCounters[DDMC_CENSUS] =
            static_cast<unsigned long long>(physics.getDDMCCensusCount());
        localCounters[DDMC_UPSCATTER] =
            static_cast<unsigned long long>(physics.getDDMCUpscatterCount());
        localCounters[DDMC_FALLBACK] =
            static_cast<unsigned long long>(physics.getDDMCFallbackCount());

        std::string debug;
        if(grid.GetPointNo() > 0)
        {
            debug = physics.getAccelerationDebugInfo(
                0, ComputationalCell3D::energyBoundaries[0]);
        }
        localCounters[DDMC_RESIDENT_LEAKS] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_resident_leaks"));
        localCounters[DDMC_TRANSPORT_LEAKS] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_transport_leaks"));
        localCounters[DDMC_REMOTE_RESIDENT_LEAKS] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_remote_resident_leaks"));
        localCounters[DDMC_INTERFACE_INCIDENT] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_interface_incident"));
        localCounters[DDMC_INTERFACE_ADMITTED] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_interface_admitted"));
        localCounters[DDMC_INTERFACE_REFLECTED] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_interface_reflected"));
        localCounters[DDMC_INTERFACE_BYPASS] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_interface_bypass"));
        localCounters[DDMC_INVALID_GEOMETRY] =
            static_cast<unsigned long long>(ExtractDebugScalar(
                debug, "ddmc_leak_invalid_geometry"));

        std::array<double, 4> globalSums{};
        double globalVolume = 0.0;
        double globalFirstDDMCX = std::numeric_limits<double>::infinity();
        double globalFirstThickDDMCX =
            std::numeric_limits<double>::infinity();
        unsigned long long globalEligibleCount = 0;
        unsigned long long globalFirstThickEligible = 0;
        std::array<unsigned long long, COUNTER_COUNT> globalCounters{};

#ifdef RICH_MPI
        MPI_Reduce(localSums.data(), globalSums.data(),
                   static_cast<int>(localSums.size()), MPI_DOUBLE, MPI_SUM,
                   0, MPI_COMM_WORLD);
        MPI_Reduce(&localVolume, &globalVolume, 1, MPI_DOUBLE, MPI_SUM,
                   0, MPI_COMM_WORLD);
        MPI_Reduce(&localFirstDDMCX, &globalFirstDDMCX, 1, MPI_DOUBLE,
                   MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFirstThickDDMCX, &globalFirstThickDDMCX, 1,
                   MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localEligibleCount, &globalEligibleCount, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFirstThickEligible, &globalFirstThickEligible, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(localCounters.data(), globalCounters.data(),
                   static_cast<int>(localCounters.size()),
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#else
        globalSums = localSums;
        globalVolume = localVolume;
        globalFirstDDMCX = localFirstDDMCX;
        globalFirstThickDDMCX = localFirstThickDDMCX;
        globalEligibleCount = localEligibleCount;
        globalFirstThickEligible = localFirstThickEligible;
        globalCounters = localCounters;
#endif

        if(rank != 0)
            return;

        double const firstDDMCX = std::isfinite(globalFirstDDMCX)
            ? globalFirstDDMCX
            : std::numeric_limits<double>::quiet_NaN();
        double const firstThickDDMCX = std::isfinite(globalFirstThickDDMCX)
            ? globalFirstThickDDMCX
            : std::numeric_limits<double>::quiet_NaN();
        double const volumeAverageTemperature = globalVolume > 0.0
            ? globalSums[3] / globalVolume
            : std::numeric_limits<double>::quiet_NaN();

        out << step << '\t' << time
            << '\t' << globalSums[0]
            << '\t' << globalSums[1]
            << '\t' << globalSums[2]
            << '\t' << volumeAverageTemperature
            << '\t' << globalEligibleCount
            << '\t' << firstDDMCX
            << '\t' << firstThickDDMCX
            << '\t' << globalFirstThickEligible;
        for(unsigned long long const counter : globalCounters)
            out << '\t' << counter;
        out << '\n';
        out.flush();
    }

    inline void WriteRawRankDebug(
        std::string const &outputDirectory,
        std::string const &prefix,
        Tessellation3D const &grid,
        RadiationIMC const &physics,
        int rank)
    {
        std::ostringstream name;
        name << prefix << "_rank" << std::setfill('0') << std::setw(4)
             << rank << "_acceleration_debug.txt";
        std::ofstream out(fs::path(outputDirectory) / name.str());
        out << std::setprecision(17);
        out << "# One line per local cell in 1.90 <= x <= 2.10 cm.\n";
        out << "# The text after x is RadiationIMC::getAccelerationDebugInfo.\n";
        for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        {
            double const x = grid.GetCellCM(i).x;
            if(x < 1.90 || x > 2.10)
                continue;
            out << "x=" << x
                << physics.getAccelerationDebugInfo(
                    i, ComputationalCell3D::energyBoundaries[0])
                << '\n';
        }
    }

    inline void AppendPrefixedDiagnostics(std::ostream &out,
                                          std::size_t step,
                                          double time,
                                          std::string const &rows)
    {
        std::istringstream input(rows);
        std::string line;
        while(std::getline(input, line))
        {
            if(!line.empty())
                out << step << '\t' << time << '\t' << line << '\n';
        }
    }

    inline void ValidateInterfaceFace(Tessellation3D const &grid)
    {
        // In an MPI tessellation each rank stores only its local/ghost subset
        // of the global faces. Therefore most ranks are not expected to own
        // the material interface at x=2. Validate the global minimum face
        // displacement rather than requiring every rank to contain that face.
        double localClosest = std::numeric_limits<double>::infinity();
        for(std::size_t faceIndex = 0;
            faceIndex < grid.GetTotalFacesNumber(); ++faceIndex)
        {
            localClosest = std::min(
                localClosest,
                std::abs(grid.FaceCM(faceIndex).x -
                         densmore2012_interface_mesh::interfacePosition));
        }

        double globalClosest = std::numeric_limits<double>::infinity();
        MPI_Allreduce(&localClosest, &globalClosest, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);

        if(globalClosest > 1e-12)
        {
            UniversalError error(
                "Densmore interface diagnostic mesh missed x=2 face");
            error.addEntry("Global closest face displacement", globalClosest);
            throw error;
        }
    }

    template<bool EnableDDMC>
    int Run(int argc,
            char **argv,
            std::string const &outputDirectory,
            std::string const &prefix,
            RunOptions const &options = RunOptions{})
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

        int rank = 0;
        int worldSize = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        try
        {
            std::size_t const Nx = options.useCenteredInterfaceMesh
                ? densmore2012_interface_mesh::centeredInterfaceCellCount
                : densmore2012_interface_mesh::cellCount;
            // This is a mechanism-isolation matrix, not the final variance
            // study.  Halve packet statistics so the six paired jobs are about
            // twice as fast while preserving identical statistics in each pair.
            constexpr std::size_t newPhotonsPerCell = 25;
            constexpr std::size_t maxPhotonsPerCell = 100;
            constexpr bool useRandomWalk = false;

            std::size_t const groupCount = ENERGY_GROUPS_NUM;
            std::vector<double> groupCenters(groupCount);
            std::vector<double> groupBoundaries(groupCount + 1);
            double const Emin = units::kev * 1e-4;
            double const Emax = units::kev * 1e2;
            groupBoundaries[0] = Emin;
            for(std::size_t g = 0; g < groupCount; ++g)
            {
                groupBoundaries[g + 1] =
                    std::pow(Emax / Emin, 1.0 / groupCount) *
                    groupBoundaries[g];
                groupCenters[g] = 0.5 *
                    (groupBoundaries[g + 1] + groupBoundaries[g]);
            }
            for(std::size_t g = 0; g <= groupCount; ++g)
                ComputationalCell3D::energyBoundaries[g] = groupBoundaries[g];

            ComputationalCell3D::stickerNames.push_back("Left");
            ComputationalCell3D::stickerNames.push_back("Right");

            double const cv = 1e15 / units::kev_kelvin;
            IdealGas eos(1.4, cv, 1, 0);
            double const sigma0Left = 10.0 * std::pow(units::kev, 3.5);
            double const sigma0Right = 1000.0 * std::pow(units::kev, 3.5);
            double const initialTemperature = units::ev_kelvin;
            double const boundaryTemperature = units::kev_kelvin;
            double const finalTime = 1e-9;
            double const dt = 5e-12;
            std::size_t const iterations =
                static_cast<std::size_t>(finalTime / dt);

            constexpr double transverseWidth = 3.0 / 256.0;
            Vector3D const lowerLeft(
                0.0, -0.5 * transverseWidth, -0.5 * transverseWidth);
            Vector3D const upperRight(
                densmore2012_interface_mesh::domainLength,
                0.5 * transverseWidth, 0.5 * transverseWidth);

            std::vector<Vector3D> points;
            if(rank == 0)
            {
                points = options.useCenteredInterfaceMesh
                    ? densmore2012_interface_mesh::BuildCenteredInterfaceVoronoiSites()
                    : densmore2012_interface_mesh::BuildVoronoiSites();
            }
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            Voronoi3D tessellation(lowerLeft, upperRight);
            tessellation.BuildParallel(points);
            ValidateInterfaceFace(tessellation);

            ComputationalCell3D initialCell;
            initialCell.density = 1.0;
            initialCell.temperature = initialTemperature;
            initialCell.velocity = Vector3D(0.0, 0.0, 0.0);
            initialCell.internal_energy = eos.dT2e(
                initialCell.density, initialCell.temperature,
                initialCell.tracers, ComputationalCell3D::tracerNames);
            initialCell.pressure = eos.de2p(
                initialCell.density, initialCell.internal_energy,
                initialCell.tracers, ComputationalCell3D::tracerNames);
            initialCell.Erad = units::arad *
                std::pow(initialTemperature, 4) / initialCell.density;

            std::vector<ComputationalCell3D> initialCells(
                tessellation.GetPointNo(), initialCell);
            for(std::size_t i = 0; i < initialCells.size(); ++i)
            {
                if(tessellation.GetCellCM(i).x <
                   densmore2012_interface_mesh::interfacePosition)
                    initialCells[i].tracers[0] = 1.0;
                else
                    initialCells[i].tracers[1] = 1.0;
            }

            Simulation simulation(tessellation, initialCells, eos);
            std::shared_ptr<TimeStepFunction3D> timeStep =
                std::make_shared<ManualTimeStep>();
            simulation.SetTimeStepFunction(timeStep);

            std::vector<ComputationalCell3D> &cells =
                simulation.getCells();
            std::vector<Conserved3D> &conserved =
                simulation.getExtensives();
            conserved.resize(cells.size());
            for(std::size_t i = 0; i < cells.size(); ++i)
                PrimitiveToConserved(cells[i], tessellation.GetVolume(i),
                                     conserved[i]);

            auto eosPointer = std::make_shared<IdealGas>(eos);
            auto opacity = std::make_shared<InterfaceOpacity>(
                sigma0Left, sigma0Right, groupCenters, groupBoundaries);

            constexpr bool withHydro = false;
            constexpr std::size_t boundaryPhotonsPerCell = 50;
            std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>>
                boundaryCondition =
                    std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
                        tessellation, cells, boundaryTemperature,
                        boundaryPhotonsPerCell, true);

            RadiationIMCParameters parameters = {
                .newPhotonsPerCell = newPhotonsPerCell,
                .withHydro = withHydro,
                .diffusionPressureGradient = false,
                .MMC = false,
                .withMultigroupOpacity = true,
                .withRandomWalk = useRandomWalk,
                .withDDMC = EnableDDMC,
                .ddmcUseMultigroupPGRW = EnableDDMC,
                .ddmcMaxGroupCutoff = options.ddmcMaxGroupCutoff,
                .ddmcInterfaceDiagnostics = options.ddmcInterfaceDiagnostics,
                .noHydroFeedback = false,
                .withEgTimeAvg = true
            };

            auto physics = std::make_shared<RadiationIMC>(
                tessellation, boundaryCondition, cells, conserved,
                eosPointer, opacity, parameters);
            physics->reseedRNG(
                UINT64_C(0xD3120000) + static_cast<std::uint64_t>(rank));

            std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>>
                populationControl =
                    std::make_shared<CombPopulationControl<
                        Vector3D, Tessellation3D>>(
                            tessellation, maxPhotonsPerCell, 5);

            std::vector<Particle3D> initialParticles;
            std::size_t const initialParticlesPerCell = 0;
            auto radiationStep = std::make_shared<RadiationMCStep>(
                tessellation, cells, conserved, physics,
                populationControl, boundaryCondition, initialParticles,
                initialParticlesPerCell, withHydro
#ifdef RICH_MPI
                , RadiationMCStep::ManagerType::AUTO_RDMA
#endif
            );
            simulation.addPhysics(radiationStep);
#ifdef RICH_MPI
            radiationStep->setCost(
                std::make_shared<IMCCostCalculator>(
                    radiationStep->getManager()));
            simulation.setForceRebalanceSteps(4);
            simulation.addMigrationBuffer(
                radiationStep->getManager()->GetCellsStepsCounters());
#endif
            simulation.SetTimeStep(dt);

            fs::create_directories(outputDirectory);
            std::ofstream faceHistoryOutput;
            std::ofstream eventHistoryOutput;
            if constexpr(EnableDDMC)
            {
                if(options.ddmcInterfaceDiagnostics)
                {
                    std::ostringstream rankText;
                    rankText << std::setfill('0') << std::setw(4) << rank;
                    faceHistoryOutput.open(
                        fs::path(outputDirectory) /
                        (prefix + "_rank" + rankText.str() +
                         "_ddmc_face_history.tsv"));
                    eventHistoryOutput.open(
                        fs::path(outputDirectory) /
                        (prefix + "_rank" + rankText.str() +
                         "_ddmc_interface_events.tsv"));
                    faceHistoryOutput << std::setprecision(17);
                    eventHistoryOutput << std::setprecision(17);
                    faceHistoryOutput
                        << "step\ttime_s\tsource_cell_id\ttarget_cell_id"
                        << "\tface_index\tsource_generator_x_cm"
                        << "\tsource_cell_cm_x_cm\ttarget_generator_x_cm"
                        << "\tface_x_cm\tsource_volume_cm3"
                        << "\tsource_cutoff\ttarget_cutoff"
                        << "\tsource_eligible\ttarget_eligible"
                        << "\tsource_sigma_diffusion\ttarget_sigma_diffusion"
                        << "\tsource_D\ttarget_D\tsource_distance_to_face"
                        << "\ttarget_distance_to_face\tarea\tconductance"
                        << "\tinternal_rate\tboundary_rate\tsource_band_mass"
                        << "\tcommon_band_mass\tddmc_fraction\tddmc_rate"
                        << "\ttransport_rate\ttotal_rate\n";
                    eventHistoryOutput
                        << "step\ttime_s\tkind\tsource_cell_id\ttarget_cell_id"
                        << "\tface_index\tsource_generator_x_cm"
                        << "\ttarget_generator_x_cm\tface_x_cm\tgroup"
                        << "\tsource_cutoff\ttarget_cutoff\tcount"
                        << "\tsigned_energy\tabsolute_energy\tmu_sum\tmu_count"
                        << "\tadmission_probability_sum"
                        << "\tadmission_probability_count\n";
                }
            }
            std::ofstream globalOutput;
            std::ofstream historyOutput;
            if(rank == 0)
            {
                globalOutput.open(fs::path(outputDirectory) /
                                  (prefix + "_global_diagnostics.tsv"));
                historyOutput.open(fs::path(outputDirectory) /
                                   (prefix + "_interface_history.tsv"));
                globalOutput << std::setprecision(17);
                historyOutput << std::setprecision(17);
                WriteGlobalHeader(globalOutput);
                WriteCellHeader(historyOutput, true);

                std::cout
                    << "Densmore exposed-interface diagnostic ("
                    << (EnableDDMC ? "DDMC" : "pure IMC") << ")"
                    << "\n  cells=" << Nx
                    << (options.useCenteredInterfaceMesh
                        ? " (centered 0.005-cm interface cells), ranks="
                        : " (100 nominal thin + 200 thick), ranks=")
                    << worldSize
                    << "\n  material face x=2 cm; first thick cell "
                    << "[2,2.005] cm"
                    << "\n  G=" << groupCount
                    << ", max DDMC cutoff="
                    << options.ddmcMaxGroupCutoff
                    << ", interface event diagnostics="
                    << options.ddmcInterfaceDiagnostics
                    << ", new/cell=" << newPhotonsPerCell
                    << ", max/cell=" << maxPhotonsPerCell
                    << "\n  dt=" << dt
                    << " s, t_final=" << finalTime
                    << " s, iterations=" << iterations
                    << std::endl;
            }

            double simulationTime = 0.0;
            auto const startWall =
                std::chrono::high_resolution_clock::now();

            for(std::size_t iteration = 0;
                iteration < iterations; ++iteration)
            {
                auto const stepStart =
                    std::chrono::high_resolution_clock::now();
                simulation.step();
                simulationTime += dt;
                simulation.SetTimeStep(dt);

                std::size_t const completedStep = iteration + 1;

                if constexpr(EnableDDMC)
                {
                    if(options.ddmcInterfaceDiagnostics)
                    {
                        AppendPrefixedDiagnostics(
                            faceHistoryOutput, completedStep, simulationTime,
                            physics->getDDMCFaceDiagnosticsTSV(1.90, 2.10));
                        AppendPrefixedDiagnostics(
                            eventHistoryOutput, completedStep, simulationTime,
                            physics->getDDMCInterfaceEventDiagnosticsTSV(
                                1.90, 2.10));
                        faceHistoryOutput.flush();
                        eventHistoryOutput.flush();
                    }
                }

                bool const diagnosticStep = completedStep == 1 ||
                    completedStep % 10 == 0 ||
                    completedStep == iterations;
                if(diagnosticStep)
                {
                    WriteGlobalDiagnostics(
                        globalOutput, completedStep, simulationTime,
                        tessellation, cells, conserved, *physics, rank);
                    std::vector<CellRecord> interfaceRecords =
                        GatherCellRecords(
                            tessellation, cells, conserved, *physics,
                            1.90, 2.10, rank);
                    if(rank == 0)
                    {
                        for(CellRecord const &record : interfaceRecords)
                        {
                            historyOutput << completedStep << '\t'
                                          << simulationTime << '\t';
                            WriteCellRecord(historyOutput, record);
                        }
                        historyOutput.flush();
                    }
                }

                auto const stepEnd =
                    std::chrono::high_resolution_clock::now();
                double const stepSeconds =
                    std::chrono::duration<double>(
                        stepEnd - stepStart).count();
                double const elapsed =
                    std::chrono::duration<double>(
                        stepEnd - startWall).count();
                double const fraction =
                    static_cast<double>(completedStep) /
                    static_cast<double>(iterations);
                double const eta = fraction > 0.0
                    ? elapsed * (1.0 - fraction) / fraction
                    : 0.0;

                if(rank == 0 &&
                   (iteration % 10 == 0 || completedStep == iterations))
                {
                    std::cout << "Cycle " << completedStep << "/"
                              << iterations << " ("
                              << static_cast<int>(100.0 * fraction)
                              << "%) t=" << simulationTime
                              << " step=" << stepSeconds
                              << "s ETA=" << static_cast<int>(eta) / 60
                              << "m" << static_cast<int>(eta) % 60
                              << "s" << std::endl;
                }
            }

            std::vector<CellRecord> finalRecords = GatherCellRecords(
                tessellation, cells, conserved, *physics,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(), rank);

            if(rank == 0)
            {
                std::ofstream cellsOutput(
                    fs::path(outputDirectory) /
                    (prefix + "_cells.tsv"));
                cellsOutput << std::setprecision(17);
                WriteCellHeader(cellsOutput, false);
                for(CellRecord const &record : finalRecords)
                    WriteCellRecord(cellsOutput, record);

                std::ofstream profileOutput(
                    fs::path(outputDirectory) /
                    (prefix + "_profile.txt"));
                profileOutput << std::setprecision(17);
                profileOutput << "# Densmore exposed-interface "
                              << (EnableDDMC ? "DDMC" : "MC")
                              << " diagnostic t=" << simulationTime
                              << " Nx=" << Nx << "\n";
                profileOutput << "# x(cm) T(K)\n";
                for(CellRecord const &record : finalRecords)
                    profileOutput << record.value[X] << ' '
                                  << record.value[TEMPERATURE] << '\n';
            }

            WriteRawRankDebug(outputDirectory, prefix,
                              tessellation, *physics, rank);

            if(rank == 0)
            {
                auto const endWall =
                    std::chrono::high_resolution_clock::now();
                std::cout << "Total wall time: "
                          << std::chrono::duration<double>(
                                 endWall - startWall).count()
                          << "s\nWrote diagnostics under "
                          << outputDirectory << std::endl;
            }
        }
        catch(UniversalError const &error)
        {
            std::cerr << "=== UniversalError on rank " << rank
                      << " ===" << std::endl;
            reportError(error);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        catch(std::exception const &error)
        {
            std::cerr << "=== std::exception on rank " << rank
                      << ": " << error.what() << " ===" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Finalize();
        return 0;
    }
}

#endif // RICH_REGRESSION_TESTS_DENSMORE2012_INTERFACE_TEST_HPP
