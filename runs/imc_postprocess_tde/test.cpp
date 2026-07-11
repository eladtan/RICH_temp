#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/3D/output/read3D.hpp"
#include "source/3D/output/Snapshot3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/monte/MonteCarloManager3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/SphericalObserver.hpp"
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

#include "postprocess_config.hpp"
#include "postprocess_runtime.hpp"
#include "adaptive_statistics.hpp"
#include "photosphere_calculation.hpp"
#include "forward_calculation.hpp"
#include "grey_calculation.hpp"

using namespace imc_postprocess_tde;

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
#ifndef RICH_IMC_DDMC_ENABLED
        if (cfg.ddmc) {
            if (rank == 0)
                std::cerr << "Error: DDMC is enabled, but this executable was built without RadiationIMC_DDMC.cpp. "
                          << "Rebuild with forward DDMC support or pass --no-ddmc.\n";
#ifdef RICH_MPI
            MPI_Finalize();
#endif
            return 1;
        }
#endif

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
                      << "Photosphere:     " << (cfg.photosphere ? "yes" : "no") << "\n"
                      << "Measured LB:     " << (cfg.measuredLoadBalance ? "requested" : "disabled") << "\n"
                      << "  weight compression: " << EffectiveMeasuredLBWeightCompression(cfg) << "\n"
                      << "  max cell imbalance: " << MEASURED_LB_MAX_CELL_IMBALANCE << "\n"
                      << "  adaptive cadence: learned-only probe LB, then every 10 learned-final steps before the last\n"
                      << "Opacity scale:   " << (cfg.opacityScaleMode == OpacityScaleMode::Planck ? "planck" :
                                                  cfg.opacityScaleMode == OpacityScaleMode::Rosseland ? "rosseland" : "disabled") << "\n"
                      << "Adaptive source: " << (cfg.adaptiveSourceCells ? "enabled" : "disabled") << "\n"
                      << "  MG schedule:   1 exact-1 burn-in, 14 exact-3 burn-in, learned-only exact-75 probe, LB, "
                      << cfg.nGenerations << " learned-only final steps (min=500 max=2000)\n"
                      << "  final LB cadence: every 10 learned-final steps before the last\n"
                      << "  min esc frac:  " << cfg.adaptiveSourceMinEscapedFrac << "\n"
                      << "  strength:      " << cfg.adaptiveSourceStrength << "\n"
                      << "  EMA:           " << cfg.adaptiveSourceEma << "\n"
                      << "  max factor:    " << cfg.adaptiveSourceMaxFactor << "\n"
                      << "  learned reserve frac:      " << cfg.adaptiveSourceLearnedReserveFrac << "\n"
                      << "  learned min factor:        " << cfg.adaptiveSourceLearnedMinFactor << "\n"
                      << "  observer equity:           " << ((cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) ? "enabled" : "disabled") << "\n"
                      << "  observer target neff:      " << cfg.adaptiveObserverTargetNeff << "\n"
                      << "  observer target pol SNR:   " << cfg.adaptiveObserverTargetPolSnr << "\n"
                      << "  observer deficit max/EMA:  " << cfg.adaptiveObserverDeficitMax << "/" << cfg.adaptiveObserverDeficitEma << "\n"
                      << "  observer extra budget max: " << cfg.adaptiveObserverExtraBudgetFrac << "\n"
                      << "  burnin/adapt LB: " << ((cfg.adaptiveSourceCells && cfg.measuredLoadBalance) ? "requested" : "disabled") << "\n"
                      << "Requested generations: " << cfg.nGenerations << "\n"
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
        std::shared_ptr<EquationOfState> eos = std::make_shared<OndrejEOS>(
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
        // Scale MG absorption to match grey mean
        // ============================================================
#if ENERGY_GROUPS_NUM > 1
        if (cfg.opacityScaleMode != OpacityScaleMode::None) {
            RecomputeOpacityScaleFactors(
                *opacity, *greyOpacity, cells, Ncells, rank, cfg.opacityScaleMode, "initial");
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
        params.withRandomWalk = cfg.randomWalk;
        params.rwMinCellOpticalDepth = 15;
        params.withDDMC = cfg.ddmc;
        params.ddmcMinCellOpticalDepth = 15;
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

        PostprocessRuntime runtime{
            rank, mpiSize, tess, cells, extensives, eos, opacity, greyOpacity,
            observer, boundary, popControl, physics, manager, params, Ncells,
            snapshot.time, snapshot.cycle, dummyCell, fldLuminosity, totalFldLum};

        if (cfg.photosphere) {
            observer->setPhotosphereData(ComputeObserverPhotospheres(cfg, runtime));
        }

        ForwardPostprocessResult forwardResult = RunForwardPostprocess(cfg, runtime);
        RunGreyPostprocess(cfg, runtime, forwardResult);
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
