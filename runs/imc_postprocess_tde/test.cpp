#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "source/3D/output/read3D.hpp"
#include "source/3D/output/Snapshot3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/monte/MonteCarloManager3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/SphericalObserver.hpp"
#include "source/3D/radiation/IMCMeasuredLoadBalance.hpp"
#include "source/3D/radiation/IMCStepCounterCostCalculator.hpp"
#include "source/monte/boundary/Vacuum.hpp"
#include "source/monte/population/NoControl.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/STAgreyOpacity.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/misc/universal_error.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/utils.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

namespace {

// ============================================================
// STA multigroup opacity from BaseTDECompton
// ============================================================
class STAMGopacityMC : public OpacityCalculator
{
private:
    std::vector<double> rho_, T_;
    std::vector<std::vector<std::vector<double>>> planck_, scatter_;
    std::unordered_map<size_t, double> rosselandScale_;

public:
    void SetRosselandScaleFactors(std::unordered_map<size_t, double> factors)
    {
        rosselandScale_ = std::move(factors);
    }

    STAMGopacityMC(std::string const& file_directory)
    {
        energy_groups_boundary = read_vector(file_directory + "frequency_edges.txt");
        for (double& Egb : energy_groups_boundary)
            Egb *= 11604.5 * CG::boltzmann_constant;
        energy_groups_center.resize(energy_groups_boundary.size() - 1);
        for (size_t i = 0; i < energy_groups_boundary.size() - 1; ++i)
            energy_groups_center[i] = std::sqrt(energy_groups_boundary[i] * energy_groups_boundary[i + 1]);
        size_t const Ng = energy_groups_boundary.size() - 1;

        T_ = read_vector(file_directory + "T.txt");
        for (size_t i = 0; i < T_.size(); ++i)
        {
            T_[i] *= 11604.5;
            T_[i] = std::log(T_[i]);
        }
        size_t const Nt = T_.size();

        rho_ = read_vector(file_directory + "rho.txt");
        size_t const Nrho = rho_.size();
        for (size_t i = 0; i < Nrho; ++i)
            rho_[i] = std::log(rho_[i]);

        planck_.resize(Ng);
        scatter_.resize(Ng);
        for (size_t i = 0; i < Ng; ++i)
        {
            auto temp_planck = read_vector(file_directory + "sigma_absorption_rossland_" + std::to_string(i + 1) + ".txt");
            auto temp_scatter = read_vector(file_directory + "sigma_scattering_planck_" + std::to_string(i + 1) + ".txt");
            planck_[i].resize(Nrho);
            scatter_[i].resize(Nrho);
            for (size_t j = 0; j < Nrho; ++j)
            {
                planck_[i][j].resize(Nt);
                scatter_[i][j].resize(Nt);
                for (size_t k = 0; k < Nt; ++k)
                {
                    planck_[i][j][k] = std::log(temp_planck[j * Nt + k]) + rho_[j];
                    scatter_[i][j][k] = std::log(temp_scatter[j * Nt + k]) + rho_[j];
                }
            }
        }
    }

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
    {
        size_t const group = findGroup(energy);
        double T = std::log(cell.temperature);
        double d = std::log(cell.density);
        double d_ratio = 1;
        double T_ratio = 1;
        if (d < rho_[0])
        {
            d_ratio = cell.density / std::exp(rho_[0]);
            d = rho_[0];
        }
        if (d > rho_.back())
        {
            d_ratio = cell.density / std::exp(rho_.back());
            d = rho_.back();
        }
        if (T < T_[0])
            T = T_[0];
        if (T > T_.back())
        {
            T_ratio = std::pow(cell.temperature / std::exp(T_.back()), -1.5);
            T = T_.back();
        }
        double result = std::exp(BiLinearInterpolation(rho_, T_, planck_[group], d, T)) * d_ratio * T_ratio;
        if (!rosselandScale_.empty()) {
            auto it = rosselandScale_.find(cell.ID);
            if (it != rosselandScale_.end())
                result *= it->second;
        }
        return result;
    }

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override
    {
        size_t const group = findGroup(energy);
        double T = std::log(cell.temperature);
        double d = std::log(cell.density);
        double d_ratio = 1;
        if (d < rho_[0])
        {
            d_ratio = cell.density / std::exp(rho_[0]);
            d = rho_[0];
        }
        if (d > rho_.back())
        {
            d_ratio = cell.density / std::exp(rho_.back());
            d = rho_.back();
        }
        if (T < T_[0])
            T = T_[0];
        if (T > T_.back())
            T = T_.back();
        return std::exp(BiLinearInterpolation(rho_, T_, scatter_[group], d, T)) * d_ratio;
    }

    double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
    {
        double avg = 0.0;
        for (size_t g = 0; g < energy_groups_center.size(); ++g)
            avg += CalcScatteringOpacity(cell, energy_groups_center[g]);
        return avg / static_cast<double>(energy_groups_center.size());
    }

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
    {
        double kT = CG::boltzmann_constant * cell.temperature;
        double weightedSum = 0.0;
        double totalWeight = 0.0;
        for (size_t g = 0; g < energy_groups_center.size(); ++g)
        {
            double nu = energy_groups_center[g];
            double x = nu / kT;
            double planckWeight = (x > 0.0 && x < 500.0) ? x * x * x / std::expm1(x) : 0.0;
            weightedSum += CalcAbsorptionOpacity(cell, nu) * planckWeight;
            totalWeight += planckWeight;
        }
        return (totalWeight > 0.0) ? weightedSum / totalWeight : 0.0;
    }
};

// ============================================================
// CLI Config
// ============================================================
struct Config
{
    std::string inputPath = "/home/elads/TDEMG/R0.47M0.5BH1e+06beta1S50n1.5Compton/snap_full_136.h5";
    std::string outputPath = "tde_postprocess_output.h5";
    std::string vtkOutput;
    std::string opacityDir = "/home/elads/RICH/data/STA/MG/";
    std::string greyOpacityDir;
    std::string eosDir = "/home/elads/RICH/data/EOS/";
    double radius = 5e14;
    size_t nObservers = 256;
    double sourceDt = 1.0;
    double transportTime = 0.0;
    Vector3D center = Vector3D(0, 0, 0);
    size_t photonsPerCell = 100;
    bool compton = false;
    size_t comptonSamples = 200000;
    bool comptonAngleDependent = true;
    size_t nGenerations = 1;
    bool ddmc = true;
    bool useCellVelocities = true;
    bool polarization = false;
    int polarizationManualScatterings = 4;
    double polarizationDepolarizationScatterings = 2.0;
    std::string polarizationClosure = "damped_last_scatterings";
    bool measuredLoadBalance = true;
    bool scaleToGreyRosseland = true;
};

void printUsage(int rank)
{
    if (rank != 0) return;
    std::cerr << "Usage: rich [options]\n"
              << "Options:\n"
              << "  --input PATH             Input HDF5 snapshot (default: TDE snap_full_136.h5)\n"
              << "  --output PATH            Output HDF5 file (default: tde_postprocess_output.h5)\n"
              << "  --opacity-dir PATH       STA opacity table directory\n"
              << "  --grey-opacity-dir PATH  Grey STA opacity directory (default: parent of opacity-dir)\n"
              << "  --eos-dir PATH           EOS table directory\n"
              << "  --radius R               Observer sphere radius [cm] (default: 5e14)\n"
              << "  --n-observers N          Observer patches (default: 256)\n"
              << "  --source-dt DT           Source emission dt [s] (default: 1.0)\n"
              << "  --transport-time T       Max transport time [s] (default: 2*R/c)\n"
              << "  --center X Y Z           Observer sphere center [cm]\n"
              << "  --photons-per-cell N     Packets per cell (default: 100)\n"
              << "  --compton                Enable Compton (requires multigroup build)\n"
              << "  --compton-samples N      Compton matrix samples (default: 200000)\n"
              << "  --n-generations N        Split transport into N generations (default: 1)\n"
              << "  --no-ddmc                Disable DDMC thick-cell acceleration\n"
              << "  --no-velocity            Ignore cell velocities (no Doppler shifts)\n"
              << "  --polarization           Enable postprocess linear polarization\n"
              << "  --polarization-manual-scatterings N\n"
              << "  --polarization-depolarization-scatterings N\n"
              << "  --polarization-closure NAME\n"
              << "  --no-measured-lb         Disable first-generation measured load balance\n"
              << "  --no-rosseland-scale     Disable MG absorption scaling to match grey Rosseland\n";
}

bool parseArgs(int argc, char* argv[], Config &cfg, int rank)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) { cfg.inputPath = argv[++i]; }
        else if (arg == "--output" && i + 1 < argc) { cfg.outputPath = argv[++i]; }
        else if (arg == "--opacity-dir" && i + 1 < argc) { cfg.opacityDir = argv[++i]; }
        else if (arg == "--grey-opacity-dir" && i + 1 < argc) { cfg.greyOpacityDir = argv[++i]; }
        else if (arg == "--eos-dir" && i + 1 < argc) { cfg.eosDir = argv[++i]; }
        else if (arg == "--radius" && i + 1 < argc) { cfg.radius = std::atof(argv[++i]); }
        else if (arg == "--n-observers" && i + 1 < argc) { cfg.nObservers = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--source-dt" && i + 1 < argc) { cfg.sourceDt = std::atof(argv[++i]); }
        else if (arg == "--transport-time" && i + 1 < argc) { cfg.transportTime = std::atof(argv[++i]); }
        else if (arg == "--center" && i + 3 < argc) {
            double x = std::atof(argv[++i]);
            double y = std::atof(argv[++i]);
            double z = std::atof(argv[++i]);
            cfg.center = Vector3D(x, y, z);
        }
        else if (arg == "--photons-per-cell" && i + 1 < argc) { cfg.photonsPerCell = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--compton") { cfg.compton = true; }
        else if (arg == "--compton-samples" && i + 1 < argc) { cfg.comptonSamples = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--compton-angle-dependent" && i + 1 < argc) { cfg.comptonAngleDependent = (std::atoi(argv[++i]) != 0); }
        else if (arg == "--vtk-output" && i + 1 < argc) { cfg.vtkOutput = argv[++i]; }
        else if (arg == "--n-generations" && i + 1 < argc) { cfg.nGenerations = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--no-ddmc") { cfg.ddmc = false; }
        else if (arg == "--no-velocity") { cfg.useCellVelocities = false; }
        else if (arg == "--polarization") { cfg.polarization = true; }
        else if (arg == "--polarization-manual-scatterings" && i + 1 < argc) { cfg.polarizationManualScatterings = std::atoi(argv[++i]); }
        else if (arg == "--polarization-depolarization-scatterings" && i + 1 < argc) { cfg.polarizationDepolarizationScatterings = std::atof(argv[++i]); }
        else if (arg == "--polarization-closure" && i + 1 < argc) { cfg.polarizationClosure = argv[++i]; }
        else if (arg == "--no-measured-lb") { cfg.measuredLoadBalance = false; }
        else if (arg == "--no-rosseland-scale") { cfg.scaleToGreyRosseland = false; }
        else { if (rank == 0) std::cerr << "Unknown argument: " << arg << "\n"; return false; }
    }

    if (cfg.radius <= 0.0) { if (rank == 0) std::cerr << "--radius must be positive\n"; return false; }
    if (cfg.nObservers == 0) { if (rank == 0) std::cerr << "--n-observers must be > 0\n"; return false; }
    if (cfg.sourceDt <= 0.0) { if (rank == 0) std::cerr << "--source-dt must be positive\n"; return false; }
    if (cfg.photonsPerCell == 0) { if (rank == 0) std::cerr << "--photons-per-cell must be > 0\n"; return false; }
    if (cfg.nGenerations == 0) { if (rank == 0) std::cerr << "--n-generations must be >= 1\n"; return false; }

    if (cfg.greyOpacityDir.empty()) {
        std::string d = cfg.opacityDir;
        if (d.size() > 3 && d.substr(d.size() - 3) == "MG/")
            d = d.substr(0, d.size() - 3);
        else if (d.size() > 2 && d.substr(d.size() - 2) == "MG")
            d = d.substr(0, d.size() - 2);
        cfg.greyOpacityDir = d;
    }

    if (cfg.transportTime <= 0.0)
        cfg.transportTime = 2.0 * cfg.radius / units::clight;

#if ENERGY_GROUPS_NUM <= 1
    if (cfg.compton) { if (rank == 0) std::cerr << "--compton requires ENERGY_GROUPS_NUM > 1\n"; return false; }
#endif

    return true;
}

// Rosseland weight fraction for a single group with dimensionless boundaries [a, b].
// Uses: integral_a^b x^4 e^x/(e^x-1)^2 dx = a^4/(e^a-1) - b^4/(e^b-1) + 4*(pi^4/15)*planck_integral(a,b)
// Normalized by the full-spectrum integral 4*pi^4/15.
double RosselandWeightFraction(double a, double b)
{
    double const fullIntegral = 4.0 * std::pow(M_PI, 4) / 15.0;
    double boundaryTerm = 0.0;
    if (a > 0.0 && a < 500.0)
        boundaryTerm += std::pow(a, 4) / std::expm1(a);
    if (b > 0.0 && b < 500.0)
        boundaryTerm -= std::pow(b, 4) / std::expm1(b);
    double planckTerm = 4.0 * (std::pow(M_PI, 4) / 15.0) * planck_integral::planck_integral(a, b);
    return (boundaryTerm + planckTerm) / fullIntegral;
}

// Solve for alpha such that:
//   sum_g [ f_g / (alpha * sigmaA[g] + sigmaS[g]) ] = 1/kappaRGrey
// F(alpha) is monotonically decreasing, so bisection converges.
double SolveRosselandAlpha(
    std::vector<double> const& sigmaA,
    std::vector<double> const& sigmaS,
    std::vector<double> const& fRoss,
    double kappaRGrey)
{
    double const target = 1.0 / kappaRGrey;

    auto evalF = [&](double alpha) {
        double sum = 0.0;
        for (size_t g = 0; g < sigmaA.size(); ++g) {
            double denom = alpha * sigmaA[g] + sigmaS[g];
            if (denom > 0.0)
                sum += fRoss[g] / denom;
        }
        return sum - target;
    };

    // F is decreasing: F(0) = sum(f_g/sigmaS_g) - target (large positive if scattering << grey)
    //                   F(inf) → -target (negative)
    // Find bracket
    double lo = 0.0, hi = 1.0;
    while (evalF(hi) > 0.0 && hi < 1e10)
        hi *= 2.0;

    if (evalF(lo) <= 0.0)
        return lo;
    if (evalF(hi) >= 0.0)
        return hi;

    for (int iter = 0; iter < 100; ++iter) {
        double mid = 0.5 * (lo + hi);
        if (evalF(mid) > 0.0)
            lo = mid;
        else
            hi = mid;
        if ((hi - lo) < 1e-12 * (lo + hi + 1e-300))
            break;
    }
    return 0.5 * (lo + hi);
}

} // anonymous namespace

int main(int argc, char* argv[])
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    int rank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#else
    int rank = 0, mpiSize = 1;
#endif

    try {
        Config cfg;
        if (!parseArgs(argc, argv, cfg, rank)) {
            printUsage(rank);
#ifdef RICH_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        if (rank == 0) {
            std::cout << "=== TDE IMC Post-Processing ===\n"
                      << "Input:           " << cfg.inputPath << "\n"
                      << "Output:          " << cfg.outputPath << "\n"
                      << "Opacity dir:     " << cfg.opacityDir << "\n"
                      << "Grey opacity:    " << cfg.greyOpacityDir << "\n"
                      << "EOS dir:         " << cfg.eosDir << "\n"
                      << "Radius:          " << cfg.radius << " cm\n"
                      << "Observers:       " << cfg.nObservers << "\n"
                      << "Source dt:       " << cfg.sourceDt << " s\n"
                      << "Transport time:  " << cfg.transportTime << " s\n"
                      << "Photons/cell:    " << cfg.photonsPerCell << "\n"
                      << "Center:          (" << cfg.center.x << ", " << cfg.center.y << ", " << cfg.center.z << ")\n"
                      << "Compton:         " << (cfg.compton ? "yes" : "no") << "\n"
                      << "DDMC:            " << (cfg.ddmc ? "yes" : "no") << "\n"
                      << "Cell velocities: " << (cfg.useCellVelocities ? "yes" : "no") << "\n"
                      << "Polarization:    " << (cfg.polarization ? "yes" : "no") << "\n"
                      << "Measured LB:     " << (cfg.measuredLoadBalance ? "requested" : "disabled") << "\n"
                      << "Rosseland scale: " << (cfg.scaleToGreyRosseland ? "enabled" : "disabled") << "\n"
                      << "Generations:     " << cfg.nGenerations << "\n"
                      << "MPI ranks:       " << mpiSize << "\n"
                      << std::endl;
        }

        // ============================================================
        // Code-unit scale factors (for snapshot → CGS conversion)
        // ============================================================
        double const lscale = 7e10;   // cm
        double const mscale = 2e33;   // g
        double const tscale = 1603;   // s

        double const rho_factor = mscale / (lscale * lscale * lscale);
        double const vel_factor = lscale / tscale;
        double const energy_factor = lscale * lscale / (tscale * tscale);

        // ============================================================
        // Load EOS (identity scales: inputs will already be CGS)
        // ============================================================
        auto eos = std::make_shared<OndrejEOS>(
            cfg.eosDir + "density.txt",
            cfg.eosDir + "Pfile.txt",
            cfg.eosDir + "csfile.txt",
            cfg.eosDir + "Sfile.txt",
            cfg.eosDir + "Ufile.txt",
            cfg.eosDir + "Tfile.txt",
            cfg.eosDir + "CVfile.txt",
            1.0, 1.0, 1.0);

        if (rank == 0)
            std::cout << "EOS loaded (CGS mode, lscale=" << lscale << " mscale=" << mscale << " tscale=" << tscale << ")." << std::endl;

        // ============================================================
        // Load STA multigroup opacity
        // ============================================================
        auto opacity = std::make_shared<STAMGopacityMC>(cfg.opacityDir);

#if ENERGY_GROUPS_NUM > 1
        if (opacity->energy_groups_boundary.size() != ENERGY_GROUPS_NUM + 1)
        {
            UniversalError eo("STA opacity table group count does not match ENERGY_GROUPS_NUM");
            eo.addEntry("Table boundaries", opacity->energy_groups_boundary.size());
            eo.addEntry("Expected boundaries", ENERGY_GROUPS_NUM + 1);
            throw eo;
        }
        for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
            ComputationalCell3D::energyBoundaries[g] = opacity->energy_groups_boundary[g];
#endif

        if (rank == 0)
        {
            std::cout << "Opacity loaded: " << opacity->energy_groups_center.size() << " groups." << std::endl;
            std::cout << "ENERGY_GROUPS_NUM=" << ENERGY_GROUPS_NUM << "  energyBoundaries:";
            for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
                std::cout << " " << ComputationalCell3D::energyBoundaries[g];
            std::cout << std::endl;
        }

        // ============================================================
        // Read snapshot (MPI-written)
        // ============================================================
        if (rank == 0)
            std::cout << "Reading snapshot..." << std::endl;

        Snapshot3D snapshot;
#ifdef RICH_MPI
        snapshot = ReadSnapshot3DParallel(cfg.inputPath);
        int const fileRanks = GetNumberOfRanksInHDF(cfg.inputPath);
        if (mpiSize < fileRanks && rank == 0) {
            for (int fileRank = mpiSize; fileRank < fileRanks; ++fileRank) {
                Snapshot3D extra = ReadSnapshot3DParallel(cfg.inputPath, fileRank);
                snapshot.cells.insert(snapshot.cells.end(),
                                      extra.cells.begin(), extra.cells.end());
                snapshot.mesh_points.insert(snapshot.mesh_points.end(),
                                            extra.mesh_points.begin(),
                                            extra.mesh_points.end());
            }
        }
#else
        snapshot = ReadSnapshot3D(cfg.inputPath);
#endif

        if (snapshot.mesh_points.empty()) {
            if (rank == 0) std::cerr << "Empty snapshot\n";
#ifdef RICH_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        if (rank == 0)
            std::cout << "Snapshot read: " << snapshot.mesh_points.size() << " points, time=" << snapshot.time << std::endl;

        // ============================================================
        // Convert snapshot from code units to CGS
        // ============================================================
        snapshot.ll = snapshot.ll * lscale;
        snapshot.ur = snapshot.ur * lscale;
        snapshot.time *= tscale;

        for (auto& pt : snapshot.mesh_points)
            pt = pt * lscale;

        for (auto& c : snapshot.cells) {
            c.density *= rho_factor;
            c.pressure *= mscale / (tscale * tscale * lscale);
            c.internal_energy *= energy_factor;
            c.velocity = c.velocity * vel_factor;
            c.Erad *= energy_factor;
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                c.Eg[g] *= energy_factor;
        }

        if (rank == 0)
            std::cout << "Converted snapshot to CGS." << std::endl;

        // ============================================================
        // Rebuild tessellation (two-pass for weighted load balancing)
        // ============================================================
#ifdef RICH_MPI
        ComputationalCell3D dummyCell;
        std::vector<double> lbWeights;
        std::vector<Vector3D> localPoints;
        {
            Voronoi3D tempTess(snapshot.ll, snapshot.ur);
            tempTess.BuildParallel(snapshot.mesh_points);
            MPI_exchange_data(tempTess, snapshot.cells, false, 1, &dummyCell);

            size_t N = tempTess.GetPointNo();

            std::vector<double> tauScatVec(N, 0.0);
            std::vector<double> tauPlanckLocal(N, 0.0);
            for (size_t i = 0; i < N; ++i)
            {
                double charLen = std::cbrt(tempTess.GetVolume(i));
                tauPlanckLocal[i] = opacity->CalcPlanckOpacity(snapshot.cells[i]) * charLen;
                tauScatVec[i] = opacity->CalcScatteringOpacity(snapshot.cells[i]) * charLen;
            }

            MPI_exchange_data(tempTess, tauScatVec, true);
            size_t Ntot = tauScatVec.size();

            lbWeights.resize(N);
            localPoints.resize(N);
            for (size_t i = 0; i < N; ++i)
            {
                localPoints[i] = tempTess.GetMeshPoint(i);

                std::vector<size_t> neighbors = tempTess.GetNeighbors(i);
                double avgTauScat = 0.0;
                size_t count = 0;
                avgTauScat += tauScatVec[i];
                ++count;

                for (size_t nb : neighbors)
                {
                    if (nb < N || !tempTess.IsPointOutsideBox(nb)) 
                    {
                        avgTauScat += 0.5 * (tauScatVec[nb] + tauScatVec[i]);
                        ++count;
                    }
                }
                avgTauScat /= static_cast<double>(count);

                double const tauPlanck = tauPlanckLocal[i];
                double maxweight = 3;
                if (avgTauScat > 5.0)
                {
                    maxweight = std::max(maxweight, 0.05 * avgTauScat);
                    lbWeights[i] = 1.0 + (tauPlanck < tauScatVec[i] * 0.1 ? std::min(maxweight, avgTauScat) : 0.0);
                }
            }
        }
        snapshot.mesh_points.clear();
        snapshot.mesh_points.shrink_to_fit();

        Voronoi3D tess(snapshot.ll, snapshot.ur);
        tess.BuildParallel(localPoints, lbWeights);

        localPoints.clear();
        localPoints.shrink_to_fit();
        lbWeights.clear();
        lbWeights.shrink_to_fit();

        MPI_exchange_data(tess, snapshot.cells, false, 1, &dummyCell);
#else
        Voronoi3D tess(snapshot.ll, snapshot.ur);
        tess.Build(snapshot.mesh_points);
#endif

        size_t Ncells = tess.GetPointNo();
        std::vector<ComputationalCell3D> &cells = snapshot.cells;
        if (cells.size() < Ncells)
            throw UniversalError("Snapshot cell count is smaller than tessellation cell count");

        if (rank == 0)
            std::cout << "Tessellation built: " << Ncells << " local cells." << std::endl;

        // Validate cells
        size_t badCells = 0;
        for (size_t i = 0; i < Ncells; ++i) {
            if (cells[i].density <= 0.0 || !std::isfinite(cells[i].density) ||
                cells[i].temperature <= 0.0 || !std::isfinite(cells[i].temperature)) {
                ++badCells;
            }
        }
        if (badCells > 0) {
            UniversalError eo("Invalid density or temperature in post-process snapshot");
            eo.addEntry("Invalid local cells", badCells);
            throw eo;
        }

        // ============================================================
        // Compute grey FLD luminosity per observer patch
        // ============================================================
        auto greyOpacity = std::make_shared<STAgreyOpacity>(cfg.greyOpacityDir);
        if (rank == 0)
            std::cout << "Grey opacity loaded for FLD luminosity." << std::endl;

        // ============================================================
        // Scale MG absorption so total MG Rosseland matches grey
        // ============================================================
#if ENERGY_GROUPS_NUM > 1
        if (cfg.scaleToGreyRosseland) {
            size_t const Ng = opacity->energy_groups_boundary.size() - 1;
            std::unordered_map<size_t, double> scaleFactors;
            scaleFactors.reserve(Ncells);

            double alphaMin = std::numeric_limits<double>::max();
            double alphaMax = 0.0;
            double alphaSum = 0.0;
            size_t alphaOutliers = 0;

            for (size_t i = 0; i < Ncells; ++i) {
                double kT = CG::boltzmann_constant * cells[i].temperature;
                if (kT <= 0.0) continue;

                std::vector<double> fRoss(Ng);
                double fTotal = 0.0;
                for (size_t g = 0; g < Ng; ++g) {
                    double a = opacity->energy_groups_boundary[g] / kT;
                    double b = opacity->energy_groups_boundary[g + 1] / kT;
                    if (a >= b || a > 500.0) { fRoss[g] = 0.0; continue; }
                    b = std::min(b, 500.0);
                    fRoss[g] = RosselandWeightFraction(a, b);
                    fTotal += fRoss[g];
                }
                if (fTotal <= 0.0) continue;
                for (double& f : fRoss)
                    f /= fTotal;

                std::vector<double> sigA(Ng), sigS(Ng);
                for (size_t g = 0; g < Ng; ++g) {
                    sigA[g] = opacity->CalcAbsorptionOpacity(cells[i], opacity->energy_groups_center[g]);
                    sigS[g] = opacity->CalcScatteringOpacity(cells[i], opacity->energy_groups_center[g]);
                }

                double D_grey = greyOpacity->CalcDiffusionCoefficient(cells[i]);
                double kappaRGrey = CG::speed_of_light / (3.0 * D_grey);

                double alpha = SolveRosselandAlpha(sigA, sigS, fRoss, kappaRGrey);
                alpha = std::max(0.01, std::min(alpha, 100.0));

                scaleFactors[cells[i].ID] = alpha;

                alphaMin = std::min(alphaMin, alpha);
                alphaMax = std::max(alphaMax, alpha);
                alphaSum += alpha;
                if (alpha < 0.5 || alpha > 2.0)
                    ++alphaOutliers;
            }

            opacity->SetRosselandScaleFactors(std::move(scaleFactors));

            if (rank == 0) {
                double alphaMean = (Ncells > 0) ? alphaSum / static_cast<double>(Ncells) : 1.0;
                std::cout << "Rosseland scale applied: alpha min=" << alphaMin
                          << " max=" << alphaMax << " mean=" << alphaMean
                          << " outliers(>2x)=" << alphaOutliers << "/" << Ncells << std::endl;
            }
        }
#endif

        // Per-cell radiation energy density, diffusion coefficient, and FLD flux vector
        std::vector<double> Er_vol(Ncells);
        std::vector<double> D_cell(Ncells);
        std::vector<Vector3D> fldFlux(Ncells);

        for (size_t i = 0; i < Ncells; ++i) {
            Er_vol[i] = cells[i].density * cells[i].Erad;
            D_cell[i] = greyOpacity->CalcDiffusionCoefficient(cells[i]);
        }

#ifdef RICH_MPI
        MPI_exchange_data(tess, Er_vol, true);
#endif

        // Green-Gauss gradient of E_r
        std::vector<Vector3D> gradEr(Ncells);
        {
            std::vector<size_t> neighbors;
            for (size_t i = 0; i < Ncells; ++i) {
                auto const& faces = tess.GetCellFaces(i);
                tess.GetNeighbors(i, neighbors);
                Vector3D grad(0, 0, 0);
                for (size_t j = 0; j < neighbors.size(); ++j) {
                    size_t nb = neighbors[j];
                    if (tess.IsPointOutsideBox(nb))
                        continue;
                    double Er_face = 0.5 * (Er_vol[i] + Er_vol[nb]);
                    Vector3D r_ij = tess.GetMeshPoint(nb) - tess.GetMeshPoint(i);
                    double dist = fastabs(r_ij);
                    if (dist < 1e-200)
                        continue;
                    Vector3D nhat = r_ij * (1.0 / dist);
                    grad += nhat * (tess.GetArea(faces[j]) * Er_face);
                }
                double vol = tess.GetVolume(i);
                if (vol > 0.0)
                    grad *= 1.0 / vol;
                gradEr[i] = grad;
            }
        }

        // Apply flux limiter and compute FLD flux vector F = -lambda * D * grad(Er)
        for (size_t i = 0; i < Ncells; ++i) {
            double lambda = CG::CalcSingleFluxLimiter(gradEr[i], D_cell[i], Er_vol[i]);
            fldFlux[i] = gradEr[i] * (-lambda * D_cell[i]);
        }

        if (rank == 0)
            std::cout << "FLD flux computed for " << Ncells << " cells." << std::endl;

        // ============================================================
        // Build extensives
        // ============================================================
        std::vector<Conserved3D> extensives(Ncells);
        for (size_t i = 0; i < Ncells; ++i)
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

        // ============================================================
        // Energy group boundaries
        // ============================================================
        std::vector<double> groupBoundaries;
#if ENERGY_GROUPS_NUM > 1
        groupBoundaries.resize(ENERGY_GROUPS_NUM + 1);
        for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
            groupBoundaries[g] = ComputationalCell3D::energyBoundaries[g];
#endif

        // ============================================================
        // Construct observer
        // ============================================================
        auto observer = std::make_shared<SphericalObserver>(
            cfg.center, cfg.radius, cfg.nObservers, groupBoundaries);

        // ============================================================
        // Map FLD flux to observer patches
        // ============================================================
        size_t nObs = observer->getNumObservers();
        std::vector<Vector3D> const& obsDirections = observer->getDirections();
        std::vector<double> const& obsSolidAngles = observer->getObserverSolidAngles();

        std::vector<double> fldLuminosity(nObs, 0.0);
        std::vector<double> patchMinDist(nObs, std::numeric_limits<double>::max());

        for (size_t p = 0; p < nObs; ++p) {
            Vector3D spherePoint = cfg.center + obsDirections[p] * cfg.radius;
            double patchArea_p = obsSolidAngles[p] * cfg.radius * cfg.radius;
            for (size_t i = 0; i < Ncells; ++i) {
                double dist = fastabs(tess.GetMeshPoint(i) - spherePoint);
                if (dist < patchMinDist[p]) {
                    patchMinDist[p] = dist;
                    double radialFlux = ScalarProd(fldFlux[i], obsDirections[p]);
                    fldLuminosity[p] = std::max(0.0, radialFlux) * patchArea_p;
                }
            }
        }

#ifdef RICH_MPI
        {
            struct DistVal { double dist; int rank; };
            std::vector<DistVal> localDV(nObs), globalDV(nObs);
            for (size_t p = 0; p < nObs; ++p) {
                localDV[p].dist = patchMinDist[p];
                localDV[p].rank = rank;
            }
            MPI_Allreduce(localDV.data(), globalDV.data(),
                          static_cast<int>(nObs), MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
            for (size_t p = 0; p < nObs; ++p) {
                if (globalDV[p].rank != rank)
                    fldLuminosity[p] = 0.0;
            }
            MPI_Allreduce(MPI_IN_PLACE, fldLuminosity.data(),
                          static_cast<int>(nObs), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif

        // Free FLD intermediates no longer needed
        Er_vol.clear(); Er_vol.shrink_to_fit();
        D_cell.clear(); D_cell.shrink_to_fit();
        gradEr.clear(); gradEr.shrink_to_fit();
        fldFlux.clear(); fldFlux.shrink_to_fit();
        patchMinDist.clear(); patchMinDist.shrink_to_fit();

        double totalFldLum = 0.0;
        for (size_t p = 0; p < nObs; ++p)
            totalFldLum += fldLuminosity[p];

        if (rank == 0)
            std::cout << "FLD luminosity mapped to " << nObs << " patches, total = "
                      << totalFldLum << " erg/s" << std::endl;

        // ============================================================
        // Construct boundary condition
        // ============================================================
        auto boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        // ============================================================
        // Construct RadiationIMC
        // ============================================================
        RadiationIMCParameters params;
        size_t genPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / cfg.nGenerations);
        params.newPhotonsPerCell = genPhotonsPerCell;
        params.withHydro = false;
        params.noHydroFeedback = true;
        params.withRandomWalk = true;
        params.rwMinCellOpticalDepth = 15;
        params.withDDMC = cfg.ddmc;
        params.ddmcMinCellOpticalDepth = 15;
        params.ddmcMinParticleOpticalDepth = 5;
        params.ddmcUseMultigroupPGRW = true;
        params.MMC = false;
        params.diffusionPressureGradient = false;
#if ENERGY_GROUPS_NUM > 1
        params.withMultigroupOpacity = true;
#else
        params.withMultigroupOpacity = false;
#endif
        params.withCompton = cfg.compton;
        params.comptonMatrixSamples = cfg.comptonSamples;
        params.comptonAngleDependent = cfg.comptonAngleDependent;

        params.postProcess.enabled = true;
        params.postProcess.sourceDt = cfg.sourceDt;
        params.postProcess.transportTime = cfg.transportTime;
        params.postProcess.useCellVelocities = cfg.useCellVelocities;
        params.postProcess.polarization.enabled = cfg.polarization;
        params.postProcess.polarization.manualScatteringsAfterAcceleration = cfg.polarizationManualScatterings;
        params.postProcess.polarization.depolarizationScatterings = cfg.polarizationDepolarizationScatterings;
        params.postProcess.polarization.acceleratedClosure = cfg.polarizationClosure;

        auto physics = std::make_shared<RadiationIMC>(
            tess, boundary, cells, extensives, eos, opacity, params);
        physics->setObserver(observer);

        auto popControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

        // ============================================================
        // Construct manager
        // ============================================================
        std::shared_ptr<MonteCarloManager3D> manager;
#ifdef RICH_MPI
        manager = std::make_shared<RDMAMonteCarloManager3D>(
            tess, physics, popControl, boundary);
#else
        manager = std::make_shared<MonteCarloManagerSerial3D>(
            tess, physics, popControl, boundary);
#endif

        if (rank == 0)
        {
            std::cout << "Starting transport..." << std::endl;
            size_t diagCells = std::min<size_t>(5, cells.size());
            for (size_t ci = 0; ci < diagCells; ++ci)
            {
                auto const& cc = cells[ci];
                double vol = tess.GetVolume(ci);
                double charLen = std::cbrt(vol);
                double sigAbs = opacity->CalcAbsorptionOpacity(cc, opacity->energy_groups_center[0]);
                double sigScat = opacity->CalcScatteringOpacity(cc, opacity->energy_groups_center[0]);
                double sigScatGrey = opacity->CalcScatteringOpacity(cc);
                double tauAbs = sigAbs * charLen;
                double tauScat = sigScat * charLen;
                std::cerr << "[DIAG] cell " << ci
                          << ": rho=" << cc.density
                          << " T=" << cc.temperature
                          << " vol=" << vol
                          << " L=" << charLen
                          << " sig_abs(g0)=" << sigAbs
                          << " sig_scat(g0)=" << sigScat
                          << " sig_scat_grey=" << sigScatGrey
                          << " tau_abs=" << tauAbs
                          << " tau_scat=" << tauScat
                          << "\n";
            }
            std::cerr << std::flush;
        }

        // ============================================================
        // Run transport (generation loop)
        // ============================================================
        using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

        bool const measuredLBActive = cfg.measuredLoadBalance && cfg.nGenerations > 1;
#if ENERGY_GROUPS_NUM > 1
        bool const isMultigroup = true;
#else
        bool const isMultigroup = false;
#endif

        if (rank == 0)
            std::cout << "Measured LB active: " << (measuredLBActive ? "yes" : "no") << std::endl;

        imc_measured_lb::Parameters measuredLBParams;
        measuredLBParams.floorCost = 1.0;
        measuredLBParams.stepWeight = 1.0;
        measuredLBParams.particleWeight = 0.0;
        measuredLBParams.medianClampFactor = isMultigroup ? 50.0 : 30.0;
        measuredLBParams.missingCellCost = isMultigroup ? 5.0 : 2.0;
        measuredLBParams.grayZeroStepInflation = 2.0;
        measuredLBParams.multigroupZeroStepInflation = 5.0;
        measuredLBParams.useMedianClamp = true;

        for (size_t gen = 0; gen < cfg.nGenerations; ++gen)
        {
            if (rank == 0)
                std::cout << "Generation " << (gen + 1) << "/" << cfg.nGenerations << std::endl;

            physics->reseedRNG(static_cast<uint64_t>(rank+12345678) * cfg.nGenerations + gen);

            std::vector<Particle3D> empty;
            auto remaining = manager->step(std::move(empty), cells, cfg.transportTime);
            (void)remaining;

#ifdef RICH_MPI
            // Generation 0 is retained as part of the final postprocess result.
            // Its measured step counters are used only to repartition generations 1..N-1.
            // Repartition assumptions:
            //   - Each generation is independent (no census particles carried between them).
            //   - noHydroFeedback is true: gas state is not modified by MC generations.
            //   - Extensives can be rebuilt from cell primitives after repartition.
            //   - The observer accumulates escaped packets globally and is preserved.
            if (gen == 0 && measuredLBActive) {
                if (!params.noHydroFeedback) {
                    throw UniversalError("Measured load balance repartition requires noHydroFeedback=true");
                }

                auto const& localSteps = manager->GetCellsStepsCounters();

                std::vector<size_t> cellIDs(Ncells);
                for (size_t i = 0; i < Ncells; ++i)
                    cellIDs[i] = cells[i].ID;

                std::vector<size_t> noParticles;
                auto localMeasurements = imc_measured_lb::BuildLocalMeasurements(cellIDs, localSteps, noParticles);
                auto globalMeasurements = imc_measured_lb::GatherMeasurementsAllRanks(localMeasurements, MPI_COMM_WORLD);

                size_t globalTotalSteps = 0;
                for (auto const& m : globalMeasurements)
                    globalTotalSteps += m.stepCount;

                if (globalTotalSteps == 0) {
                    if (rank == 0)
                        std::cerr << "MEASURED_LB: generation 0 had zero total steps, skipping repartition\n";
                } else {
                    auto costByCellID = imc_measured_lb::BuildMeasuredCosts(globalMeasurements, measuredLBParams, isMultigroup);
                    imc_measured_lb::PrintMeasuredLBDiagnostics(localMeasurements, costByCellID, isMultigroup, MPI_COMM_WORLD);

                    IMCStepCounterCostCalculator::Parameters costCalcParams;
                    costCalcParams.floorCost = measuredLBParams.floorCost;
                    costCalcParams.missingCellCost = measuredLBParams.missingCellCost;
                    IMCStepCounterCostCalculator measuredCostCalc(costByCellID, costCalcParams);

                    std::vector<Vector3D> currentPoints(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        currentPoints[i] = tess.GetMeshPoint(i);

                    auto lbWeightsNew = measuredCostCalc.CalculateCost(tess, cells);
                    tess.BuildParallel(currentPoints, lbWeightsNew);
                    MPI_exchange_data(tess, cells, false, 1, &dummyCell);

                    Ncells = tess.GetPointNo();
                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

                    boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    physics = std::make_shared<RadiationIMC>(
                        tess, boundary, cells, extensives, eos, opacity, params);
                    // Keep the same observer so generation-0 tallies remain part of the result.
                    physics->setObserver(observer);

                    popControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                    manager = std::make_shared<RDMAMonteCarloManager3D>(
                        tess, physics, popControl, boundary);

                    std::vector<size_t> newCellIDs(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        newCellIDs[i] = cells[i].ID;
                    imc_measured_lb::PrintPostRepartitionDiagnostics(
                        costByCellID, newCellIDs, measuredLBParams.missingCellCost, isMultigroup, MPI_COMM_WORLD);

                    if (rank == 0)
                        std::cout << "MEASURED_LB: repartitioned after generation 0, new local cells=" << Ncells << std::endl;
                }
            }
#endif // RICH_MPI
        }

        // ============================================================
        // Finish diagnostics
        // ============================================================
        observer->addBoxEscapeEnergy(boundary->getEscapedEnergy());
        observer->mpiReduceToRank0();

        if (cfg.nGenerations > 1)
            observer->scale(1.0 / static_cast<double>(cfg.nGenerations));

        if (rank == 0) {
            SphericalObserver::Diagnostics diag;
            diag.sourceDt = cfg.sourceDt;
            diag.transportTime = cfg.transportTime;
            diag.mpiRanks = mpiSize;
            diag.comptonEnabled = cfg.compton ? 1 : 0;
            diag.emittedEnergy = observer->getEmittedEnergy();
            diag.absorbedEnergy = observer->getAbsorbedEnergy();
            diag.boxEscapeEnergy = observer->getBoxEscapeEnergy();
            diag.timedOutEnergy = observer->getTimedOutEnergy();
            diag.cutoffEnergy = observer->getCutoffEnergy();
            diag.snapshotTime = snapshot.time;
            diag.snapshotCycle = snapshot.cycle;
            diag.nGenerations = static_cast<int>(cfg.nGenerations);

            observer->writeHDF5(cfg.outputPath, diag);

            if (!cfg.vtkOutput.empty()) {
                observer->writeVTK(cfg.vtkOutput, cfg.sourceDt);

                // Append FLD luminosity scalars to the VTK file
                std::ofstream vtkAppend(cfg.vtkOutput, std::ios::app);
                if (vtkAppend.is_open()) {
                    vtkAppend << std::scientific << std::setprecision(12);

                    double fourPi = 4.0 * M_PI;

                    vtkAppend << "SCALARS fld_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p)
                        vtkAppend << fldLuminosity[p] << "\n";

                    vtkAppend << "SCALARS fld_isotropic_equivalent_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double isoEquiv = (obsSolidAngles[p] > 0.0)
                            ? fldLuminosity[p] * fourPi / obsSolidAngles[p] : 0.0;
                        vtkAppend << isoEquiv << "\n";
                    }

                    vtkAppend << "SCALARS log10_fld_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double val = (fldLuminosity[p] > 0.0) ? std::log10(fldLuminosity[p]) : -99.0;
                        vtkAppend << val << "\n";
                    }

                    vtkAppend << "SCALARS fld_flux double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double patchArea_p = obsSolidAngles[p] * cfg.radius * cfg.radius;
                        double flux = (patchArea_p > 0.0) ? fldLuminosity[p] / patchArea_p : 0.0;
                        vtkAppend << flux << "\n";
                    }
                }
            }

            double totalLum = observer->getTotalCrossingEnergy() / cfg.sourceDt;
            double residual = diag.emittedEnergy - diag.absorbedEnergy
                            - diag.boxEscapeEnergy - diag.timedOutEnergy - diag.cutoffEnergy;
            double timedOutFrac = (diag.emittedEnergy > 0.0)
                ? diag.timedOutEnergy / diag.emittedEnergy : 0.0;

            std::cout << "\n=== TDE Post-Processing Results ===\n"
                      << "Generations:              " << cfg.nGenerations << "\n"
                      << "Photons/cell/gen:         " << genPhotonsPerCell << "\n"
                      << "Total crossing luminosity: " << totalLum << " erg/s\n"
                      << "Total FLD luminosity:     " << totalFldLum << " erg/s\n"
                      << "Emitted energy:           " << diag.emittedEnergy << " erg\n"
                      << "Absorbed energy:          " << diag.absorbedEnergy << " erg\n"
                      << "Box escape energy:        " << diag.boxEscapeEnergy << " erg\n"
                      << "Timed-out energy:         " << diag.timedOutEnergy << " erg\n"
                      << "Cutoff energy:            " << diag.cutoffEnergy << " erg\n"
                      << "Sink residual:            " << residual << " erg\n"
                      << "Timed-out fraction:       " << timedOutFrac << "\n"
                      << "Output written to:        " << cfg.outputPath << "\n"
                      << std::endl;
        }

        // ============================================================
        // Release MG objects before grey run to avoid OOM
        // ============================================================
        manager.reset();
        physics.reset();
        popControl.reset();
        observer.reset();
        boundary.reset();
        opacity.reset();

        if (rank == 0)
            std::cout << "MG resources released." << std::endl;

        // ============================================================
        // Grey IMC run (half generations)
        // ============================================================
        size_t nGreyGens = std::max<size_t>(1, cfg.nGenerations / 2);
        size_t greyPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / (2 * nGreyGens));

        if (rank == 0)
            std::cout << "\n=== Starting Grey IMC run (" << nGreyGens << " generations) ===" << std::endl;

        auto greyObserver = std::make_shared<SphericalObserver>(
            cfg.center, cfg.radius, cfg.nObservers, std::vector<double>());

        auto greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        RadiationIMCParameters greyParams;
        greyParams.newPhotonsPerCell = greyPhotonsPerCell;
        greyParams.withHydro = false;
        greyParams.noHydroFeedback = true;
        greyParams.withRandomWalk = true;
        greyParams.rwMinCellOpticalDepth = 15;
        greyParams.withDDMC = cfg.ddmc;
        greyParams.ddmcMinCellOpticalDepth = 15;
        greyParams.ddmcMinParticleOpticalDepth = 5;
        greyParams.ddmcUseMultigroupPGRW = false;
        greyParams.MMC = false;
        greyParams.diffusionPressureGradient = false;
        greyParams.withMultigroupOpacity = false;
        greyParams.withCompton = false;
        greyParams.postProcess.enabled = true;
        greyParams.postProcess.sourceDt = cfg.sourceDt;
        greyParams.postProcess.transportTime = cfg.transportTime;
        greyParams.postProcess.useCellVelocities = cfg.useCellVelocities;

        auto greyPhysics = std::make_shared<RadiationIMC>(
            tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
        greyPhysics->setObserver(greyObserver);

        auto greyPopControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

        std::shared_ptr<MonteCarloManager3D> greyManager;
#ifdef RICH_MPI
        greyManager = std::make_shared<RDMAMonteCarloManager3D>(
            tess, greyPhysics, greyPopControl, greyBoundary);
#else
        greyManager = std::make_shared<MonteCarloManagerSerial3D>(
            tess, greyPhysics, greyPopControl, greyBoundary);
#endif

        bool const greyMeasuredLBActive = cfg.measuredLoadBalance && nGreyGens > 1;

        if (rank == 0)
            std::cout << "Grey measured LB active: " << (greyMeasuredLBActive ? "yes" : "no") << std::endl;

        imc_measured_lb::Parameters greyLBParams;
        greyLBParams.floorCost = 1.0;
        greyLBParams.stepWeight = 1.0;
        greyLBParams.particleWeight = 0.0;
        greyLBParams.medianClampFactor = 30.0;
        greyLBParams.missingCellCost = 2.0;
        greyLBParams.grayZeroStepInflation = 2.0;
        greyLBParams.multigroupZeroStepInflation = 5.0;
        greyLBParams.useMedianClamp = true;

        for (size_t gen = 0; gen < nGreyGens; ++gen) {
            if (rank == 0)
                std::cout << "Grey generation " << (gen + 1) << "/" << nGreyGens << std::endl;

            greyPhysics->reseedRNG(static_cast<uint64_t>(rank + 87654321) * nGreyGens + gen);

            std::vector<Particle3D> empty;
            auto remaining = greyManager->step(std::move(empty), cells, cfg.transportTime);
            (void)remaining;

#ifdef RICH_MPI
            // Same repartition logic as multigroup: use generation-0 step counters
            // to rebalance subsequent grey generations.
            if (gen == 0 && greyMeasuredLBActive) {
                if (!greyParams.noHydroFeedback) {
                    throw UniversalError("Grey measured load balance repartition requires noHydroFeedback=true");
                }

                auto const& greyLocalSteps = greyManager->GetCellsStepsCounters();

                std::vector<size_t> greyCellIDs(Ncells);
                for (size_t i = 0; i < Ncells; ++i)
                    greyCellIDs[i] = cells[i].ID;

                std::vector<size_t> noParticles;
                auto greyLocalMeas = imc_measured_lb::BuildLocalMeasurements(greyCellIDs, greyLocalSteps, noParticles);
                auto greyGlobalMeas = imc_measured_lb::GatherMeasurementsAllRanks(greyLocalMeas, MPI_COMM_WORLD);

                size_t greyGlobalTotalSteps = 0;
                for (auto const& m : greyGlobalMeas)
                    greyGlobalTotalSteps += m.stepCount;

                if (greyGlobalTotalSteps == 0) {
                    if (rank == 0)
                        std::cerr << "MEASURED_LB_GREY: generation 0 had zero total steps, skipping repartition\n";
                } else {
                    auto greyCostByCellID = imc_measured_lb::BuildMeasuredCosts(greyGlobalMeas, greyLBParams, false);
                    imc_measured_lb::PrintMeasuredLBDiagnostics(greyLocalMeas, greyCostByCellID, false, MPI_COMM_WORLD);

                    IMCStepCounterCostCalculator::Parameters greyCostCalcParams;
                    greyCostCalcParams.floorCost = greyLBParams.floorCost;
                    greyCostCalcParams.missingCellCost = greyLBParams.missingCellCost;
                    IMCStepCounterCostCalculator greyCostCalc(greyCostByCellID, greyCostCalcParams);

                    std::vector<Vector3D> greyCurrentPoints(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        greyCurrentPoints[i] = tess.GetMeshPoint(i);

                    auto greyLBWeights = greyCostCalc.CalculateCost(tess, cells);
                    tess.BuildParallel(greyCurrentPoints, greyLBWeights);
                    MPI_exchange_data(tess, cells, false, 1, &dummyCell);

                    Ncells = tess.GetPointNo();
                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after grey measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

                    greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    greyPhysics = std::make_shared<RadiationIMC>(
                        tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
                    greyPhysics->setObserver(greyObserver);

                    greyPopControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                    greyManager = std::make_shared<RDMAMonteCarloManager3D>(
                        tess, greyPhysics, greyPopControl, greyBoundary);

                    std::vector<size_t> newGreyCellIDs(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        newGreyCellIDs[i] = cells[i].ID;
                    imc_measured_lb::PrintPostRepartitionDiagnostics(
                        greyCostByCellID, newGreyCellIDs, greyLBParams.missingCellCost, false, MPI_COMM_WORLD);

                    if (rank == 0)
                        std::cout << "MEASURED_LB_GREY: repartitioned after generation 0, new local cells=" << Ncells << std::endl;
                }
            }
#endif // RICH_MPI
        }

        greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
        greyObserver->mpiReduceToRank0();

        if (nGreyGens > 1)
            greyObserver->scale(1.0 / static_cast<double>(nGreyGens));

        if (rank == 0) {
            std::string greyVtk;
            if (!cfg.vtkOutput.empty()) {
                size_t dotPos = cfg.vtkOutput.rfind('.');
                if (dotPos != std::string::npos)
                    greyVtk = cfg.vtkOutput.substr(0, dotPos) + "_grey" + cfg.vtkOutput.substr(dotPos);
                else
                    greyVtk = cfg.vtkOutput + "_grey";
            }

            if (!greyVtk.empty()) {
                greyObserver->writeVTK(greyVtk, cfg.sourceDt);

                std::vector<double> const& greySolidAngles = greyObserver->getObserverSolidAngles();
                std::ofstream vtkAppend(greyVtk, std::ios::app);
                if (vtkAppend.is_open()) {
                    vtkAppend << std::scientific << std::setprecision(12);

                    double fourPi = 4.0 * M_PI;

                    vtkAppend << "SCALARS fld_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p)
                        vtkAppend << fldLuminosity[p] << "\n";

                    vtkAppend << "SCALARS fld_isotropic_equivalent_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double isoEquiv = (greySolidAngles[p] > 0.0)
                            ? fldLuminosity[p] * fourPi / greySolidAngles[p] : 0.0;
                        vtkAppend << isoEquiv << "\n";
                    }

                    vtkAppend << "SCALARS log10_fld_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double val = (fldLuminosity[p] > 0.0) ? std::log10(fldLuminosity[p]) : -99.0;
                        vtkAppend << val << "\n";
                    }

                    vtkAppend << "SCALARS fld_flux double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double patchArea_p = greySolidAngles[p] * cfg.radius * cfg.radius;
                        double flux = (patchArea_p > 0.0) ? fldLuminosity[p] / patchArea_p : 0.0;
                        vtkAppend << flux << "\n";
                    }
                }
            }

            double greyTotalLum = greyObserver->getTotalCrossingEnergy() / cfg.sourceDt;
            double greyEmitted = greyObserver->getEmittedEnergy();
            double greyAbsorbed = greyObserver->getAbsorbedEnergy();
            double greyBoxEscape = greyObserver->getBoxEscapeEnergy();
            double greyTimedOut = greyObserver->getTimedOutEnergy();
            double greyCutoff = greyObserver->getCutoffEnergy();
            double greyResidual = greyEmitted - greyAbsorbed - greyBoxEscape - greyTimedOut - greyCutoff;
            double greyTimedOutFrac = (greyEmitted > 0.0) ? greyTimedOut / greyEmitted : 0.0;

            std::cout << "\n=== Grey IMC Results ===\n"
                      << "Generations:              " << nGreyGens << "\n"
                      << "Photons/cell/gen:         " << greyPhotonsPerCell << "\n"
                      << "Total crossing luminosity: " << greyTotalLum << " erg/s\n"
                      << "Total FLD luminosity:     " << totalFldLum << " erg/s\n"
                      << "Emitted energy:           " << greyEmitted << " erg\n"
                      << "Absorbed energy:          " << greyAbsorbed << " erg\n"
                      << "Box escape energy:        " << greyBoxEscape << " erg\n"
                      << "Timed-out energy:         " << greyTimedOut << " erg\n"
                      << "Cutoff energy:            " << greyCutoff << " erg\n"
                      << "Sink residual:            " << greyResidual << " erg\n"
                      << "Timed-out fraction:       " << greyTimedOutFrac << "\n"
                      << "Grey VTK written to:      " << greyVtk << "\n"
                      << std::endl;
        }

    } catch (UniversalError const& eo) {
        std::cerr << "UniversalError on rank " << rank << ":\n";
        reportError(eo, std::cerr);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Exception on rank " << rank << ": " << e.what() << "\n";
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
