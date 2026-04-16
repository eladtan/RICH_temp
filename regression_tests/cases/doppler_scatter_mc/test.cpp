#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>
#include <functional>
#ifdef RICH_MPI
#include <mpi.h>
#endif

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/MultigroupDiffusionBoundaryCalculator.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/monte/population/NoControl.hpp"
#include "source/monte/boundary/BoundaryCondition.hpp"
#include "source/3D/tessellation/utils/RandomOnFace.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/3D/radiation/IMCCostCalculator.hpp"

namespace fs = std::filesystem;

namespace
{

// ---------------------------------------------------------------------------
// MC opacity: pure scattering, no absorption/emission
// ---------------------------------------------------------------------------
class ScatterOnlyOpacity : public OpacityCalculator
{
public:
    ScatterOnlyOpacity(double kappa_s,
                       const std::vector<double> &centers,
                       const std::vector<double> &boundaries)
        : kappa_s_(kappa_s)
    {
        energy_groups_center = centers;
        energy_groups_boundary = boundaries;
    }

    double CalcPlanckOpacity(const ComputationalCell3D & /*cell*/) const override
    { return 0.0; }

    double CalcAbsorptionOpacity(const ComputationalCell3D & /*cell*/, double /*energy*/) const override
    { return 1e-100; }

    double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/, double /*energy*/) const override
    { return kappa_s_; }

    double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/) const override
    { return kappa_s_; }

private:
    double kappa_s_;
};

// ---------------------------------------------------------------------------
// Rejection-sampling of a truncated Planck spectrum f(E) ∝ E^3/(exp(E/kT)-1)
// ---------------------------------------------------------------------------
double sampleTruncatedPlanck(double E_lo, double E_hi, double T_kelvin,
                             std::mt19937_64 &rng)
{
    double kT = units::k_boltz * T_kelvin;
    double peak = 2.82 * kT;
    double E_mode = std::clamp(peak, E_lo, E_hi);
    double f_max = E_mode * E_mode * E_mode / std::expm1(E_mode / kT);

    std::uniform_real_distribution<double> uE(E_lo, E_hi);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (;;)
    {
        double E = uE(rng);
        double f = E * E * E / std::expm1(E / kT);
        if (u01(rng) * f_max <= f)
            return E;
    }
}

// ---------------------------------------------------------------------------
// Escaped-photon record stored by the boundary condition
// ---------------------------------------------------------------------------
struct EscapedPhoton
{
    double lab_energy;
    double comoving_energy;
    double weight;
    size_t scatterings;
    bool right_side; // true = escaped at x=L, false = escaped at x=0
};

// ---------------------------------------------------------------------------
// MC boundary condition: transparent on all sides, records escaping photon
// data.  Photon injection is handled externally as seed particles.
// ---------------------------------------------------------------------------
class TransparentBC : public BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    TransparentBC(const Tessellation3D &grid, double v_right)
        : BoundaryCondition<Vector3D, Tessellation3D>(grid),
          v_right_(v_right)
    {
    }

    MonteCarloParticleStatus apply(MonteCarloParticle<Vector3D, Tessellation3D> &particle) override
    {
        const auto &[ll, ur] = this->grid.GetBoxCoordinates();
        bool is_right = std::abs(particle.location.x - ur.x) < std::abs(particle.location.x - ll.x);

        double v_local = is_right ? v_right_ : 0.0;
        double beta = v_local / units::clight;
        double mu_lab = particle.velocity.x / abs(particle.velocity);
        double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
        double comoving_energy = particle.frequency * gamma * (1.0 - beta * mu_lab);

        EscapedPhoton ep;
        ep.lab_energy = particle.frequency;
        ep.comoving_energy = comoving_energy;
        ep.weight = particle.weight;
        ep.scatterings = particle.steps;
        ep.right_side = is_right;
        escaped_.push_back(ep);

        return MonteCarloParticleStatus::REMOVE;
    }

    std::vector<MonteCarloParticle<Vector3D, Tessellation3D>>
    generateNewBoundaryParticles(double /*fullDt*/) override
    {
        return {};
    }

    const std::vector<EscapedPhoton> &getEscaped() const { return escaped_; }

private:
    double v_right_;
    std::vector<EscapedPhoton> escaped_;
};

// ---------------------------------------------------------------------------
// Diffusion boundary: Planck source on left x-face, open on right x-face,
// closed on y/z faces
// ---------------------------------------------------------------------------
class LeftPlanckRightOpenBoundary : public MultigroupDiffusionBoundaryCalculator
{
public:
    LeftPlanckRightOpenBoundary(double T_kelvin,
                                const std::vector<double> &centers,
                                const std::vector<double> &boundaries)
        : side_(T_kelvin, centers, boundaries)
    {
    }

    void setBoundaryValuesGroup(std::size_t group, Tessellation3D const &tess,
                                std::size_t index, std::size_t outside_point,
                                double dt,
                                std::vector<ComputationalCell3D> const &cells,
                                double Area, double &A, double &b,
                                std::size_t face_index) const override
    {
        if (IsLeftXBoundary(tess, index, outside_point))
            side_.setBoundaryValuesGroup(group, tess, index, outside_point, dt,
                                         cells, Area, A, b, face_index);
        else if (IsRightXBoundary(tess, index, outside_point))
            open_.setBoundaryValuesGroup(group, tess, index, outside_point, dt,
                                         cells, Area, A, b, face_index);
        else
            closed_.setBoundaryValuesGroup(group, tess, index, outside_point, dt,
                                           cells, Area, A, b, face_index);
    }

    void getOutsideValuesGroup(std::size_t group, Tessellation3D const &tess,
                               std::size_t index, std::size_t outside_point,
                               std::vector<ComputationalCell3D> const &cells,
                               double Eg_i, double &Eg_outside,
                               Vector3D &v_outside) const override
    {
        if (IsLeftXBoundary(tess, index, outside_point))
            side_.getOutsideValuesGroup(group, tess, index, outside_point, cells,
                                        Eg_i, Eg_outside, v_outside);
        else if (IsRightXBoundary(tess, index, outside_point))
            open_.getOutsideValuesGroup(group, tess, index, outside_point, cells,
                                        Eg_i, Eg_outside, v_outside);
        else
            closed_.getOutsideValuesGroup(group, tess, index, outside_point, cells,
                                          Eg_i, Eg_outside, v_outside);
    }

private:
    static bool IsXBoundary(Tessellation3D const &tess, std::size_t index,
                            std::size_t outside_point)
    {
        double R = tess.GetWidth(index);
        double dx = std::abs(tess.GetMeshPoint(index).x -
                             tess.GetMeshPoint(outside_point).x);
        return dx > R * 1e-4;
    }

    static bool IsLeftXBoundary(Tessellation3D const &tess, std::size_t index,
                                std::size_t outside_point)
    {
        if (!IsXBoundary(tess, index, outside_point))
            return false;
        return tess.GetMeshPoint(outside_point).x < tess.GetMeshPoint(index).x;
    }

    static bool IsRightXBoundary(Tessellation3D const &tess, std::size_t index,
                                 std::size_t outside_point)
    {
        if (!IsXBoundary(tess, index, outside_point))
            return false;
        return tess.GetMeshPoint(outside_point).x > tess.GetMeshPoint(index).x;
    }

    MultigroupDiffusionSideBoundary side_;
    MultigroupDiffusionOpenBoundary open_;
    MultigroupDiffusionClosedBoundary closed_;
};

// ---------------------------------------------------------------------------
// MPI helpers
// ---------------------------------------------------------------------------
#ifdef RICH_MPI
std::vector<EscapedPhoton> gatherEscapedPhotons(
    const std::vector<EscapedPhoton> &local)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localCount = static_cast<int>(local.size());
    std::vector<int> counts(size);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    // Pack each EscapedPhoton into 4 doubles + 1 size_t + 1 int
    // Use a flat double buffer: lab_energy, comoving_energy, weight, (double)scatterings, (double)right_side
    constexpr int fieldsPerPhoton = 5;
    int localN = localCount * fieldsPerPhoton;
    std::vector<double> localBuf(static_cast<size_t>(localN));
    for (int i = 0; i < localCount; ++i)
    {
        localBuf[static_cast<size_t>(i * fieldsPerPhoton + 0)] = local[static_cast<size_t>(i)].lab_energy;
        localBuf[static_cast<size_t>(i * fieldsPerPhoton + 1)] = local[static_cast<size_t>(i)].comoving_energy;
        localBuf[static_cast<size_t>(i * fieldsPerPhoton + 2)] = local[static_cast<size_t>(i)].weight;
        localBuf[static_cast<size_t>(i * fieldsPerPhoton + 3)] = static_cast<double>(local[static_cast<size_t>(i)].scatterings);
        localBuf[static_cast<size_t>(i * fieldsPerPhoton + 4)] = local[static_cast<size_t>(i)].right_side ? 1.0 : 0.0;
    }

    std::vector<int> recvCounts(size), displs(size);
    int totalN = 0;
    if (rank == 0)
    {
        for (int r = 0; r < size; ++r)
        {
            recvCounts[r] = counts[r] * fieldsPerPhoton;
            displs[r] = totalN;
            totalN += recvCounts[r];
        }
    }

    std::vector<double> allBuf;
    if (rank == 0)
        allBuf.resize(static_cast<size_t>(totalN));

    MPI_Gatherv(localBuf.data(), localN, MPI_DOUBLE,
                allBuf.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    std::vector<EscapedPhoton> result;
    if (rank == 0)
    {
        int totalPhotons = totalN / fieldsPerPhoton;
        result.resize(static_cast<size_t>(totalPhotons));
        for (int i = 0; i < totalPhotons; ++i)
        {
            result[static_cast<size_t>(i)].lab_energy = allBuf[static_cast<size_t>(i * fieldsPerPhoton + 0)];
            result[static_cast<size_t>(i)].comoving_energy = allBuf[static_cast<size_t>(i * fieldsPerPhoton + 1)];
            result[static_cast<size_t>(i)].weight = allBuf[static_cast<size_t>(i * fieldsPerPhoton + 2)];
            result[static_cast<size_t>(i)].scatterings = static_cast<size_t>(allBuf[static_cast<size_t>(i * fieldsPerPhoton + 3)]);
            result[static_cast<size_t>(i)].right_side = allBuf[static_cast<size_t>(i * fieldsPerPhoton + 4)] > 0.5;
        }
    }
    return result;
}
#endif

} // anonymous namespace

int main(int argc, char *argv[])
{
    int rank = 0;
    int worldSize = 1;
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
#else
    (void)argc;
    (void)argv;
#endif

    try
    {
        // =================================================================
        //  Common parameters from chat.tex
        // =================================================================
        constexpr size_t G = ENERGY_GROUPS_NUM;
        static_assert(G >= 10, "Need at least 10 energy groups");

        double const L = 1.0e10;                           // cm
        double const H = 3.0e-2;                           // s^-1
        double const v_right = H * L;                      // 3e8 cm/s = 0.01c
        double const kappa_s = 3.0e-8;                     // cm^-1
        double const T_src_keV = 1.0;                      // keV
        double const T_src = T_src_keV * units::kev_kelvin; // Kelvin
        double const E_trunc_lo = 0.5 * units::kev;        // erg
        double const E_trunc_hi = 3.0 * units::kev;        // erg
        size_t const N_pkt = 1000000;
        size_t const Nx = 64;
        double const rho = 1.0;                             // g/cm^3

        // Energy group grid (log-spaced, 100 eV to 100 keV)
        std::vector<double> energy_groups_center(G);
        std::vector<double> energy_groups_boundary(G + 1);
        double const Emin = units::kev * 1e-1;  // 100 eV
        double const Emax = units::kev * 1e2;   // 100 keV
        energy_groups_boundary[0] = Emin;
        for (size_t g = 0; g < G; ++g)
        {
            energy_groups_boundary[g + 1] =
                std::pow(Emax / Emin, 1.0 / G) * energy_groups_boundary[g];
            energy_groups_center[g] =
                0.5 * (energy_groups_boundary[g + 1] + energy_groups_boundary[g]);
        }
        for (size_t g = 0; g <= G; ++g)
            ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];

        // =================================================================
        //  Mesh: 1D slab [0,L] x [-dy,dy] x [-dz,dz]
        // =================================================================
        double const cellHalf = 0.5 * L / Nx;
        Vector3D ll(0, -cellHalf, -cellHalf);
        Vector3D ur(L, cellHalf, cellHalf);

        std::vector<Vector3D> points;
#ifdef RICH_MPI
        if (worldSize == 1)
        {
            points = CartesianMesh(Nx, 1, 1, ll, ur);
        }
        else
        {
            if (rank == 0)
                points = CartesianMesh(Nx, 1, 1, ll, ur);
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
#else
        points = CartesianMesh(Nx, 1, 1, ll, ur);
#endif

        Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
        if (worldSize == 1)
            tess.Build(points);
        else
            tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        // EOS (irrelevant — no hydro feedback)
        double const cv = 1e15 / units::kev_kelvin;
        IdealGas eos(1.4, cv, 1, 0);

        // Initial cells: homologous velocity v(x) = H*x, no radiation
        size_t const N = tess.GetPointNo();
        std::vector<ComputationalCell3D> initialCells(N);
        for (size_t i = 0; i < N; ++i)
        {
            auto &c = initialCells[i];
            c.density = rho;
            c.temperature = T_src;
            c.internal_energy = eos.dT2e(c.density, c.temperature, c.tracers,
                                         ComputationalCell3D::tracerNames);
            c.pressure = eos.de2p(c.density, c.internal_energy, c.tracers,
                                  ComputationalCell3D::tracerNames);
            double x = tess.GetMeshPoint(i).x;
            c.velocity = Vector3D(H * x, 0, 0);

            for (size_t g = 0; g < G; ++g)
                c.Eg[g] = 0.0;
            c.Erad = 0.0;
        }

        std::string const caseDir = fs::path(__FILE__).parent_path().string();

        // =================================================================
        //  PHASE 1: Monte Carlo transport
        // =================================================================
        if (rank == 0)
            std::cout << "=== Phase 1: Monte Carlo transport ===" << std::endl;

        // Diffusion time: t_diff ~ tau^2 / (kappa_s * c) = 300^2 / (3e-8 * 3e10) ≈ 1e5 s
        double const tau = kappa_s * L;
        double const t_diff = tau * tau / (kappa_s * units::clight);

        {
            Simulation sim(tess, initialCells, eos);
            auto tsc = std::make_shared<ManualTimeStep>();
            sim.SetTimeStepFunction(tsc);

            auto &cells = sim.getCells();
            auto &extensives = sim.getExtensives();
            extensives.resize(cells.size());
            for (size_t i = 0; i < cells.size(); ++i)
                PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

            auto eosPtr = std::make_shared<IdealGas>(eos);
            auto opacityPtr = std::make_shared<ScatterOnlyOpacity>(
                kappa_s, energy_groups_center, energy_groups_boundary);

            double const dt_mc = t_diff * 0.5;
            size_t const maxSteps = 200;

            auto boundaryCond = std::make_shared<TransparentBC>(tess, v_right);

            // --- Build seed photons on the left boundary (x=0) ---
            std::vector<Particle3D> seedParticles;
            {
                std::vector<size_t> leftCells;
                std::vector<size_t> leftFaces;
                for (size_t i = 0; i < N; ++i)
                {
                    for (size_t fIdx : tess.GetCellFaces(i))
                    {
                        auto [n1, n2] = tess.GetFaceNeighbors(fIdx);
                        size_t neighbor = (n1 == i) ? n2 : n1;
                        if (neighbor >= N && tess.IsPointOutsideBox(neighbor))
                        {
                            Vector3D normal = normalize(
                                tess.GetMeshPoint(neighbor) - tess.GetMeshPoint(i));
                            if (normal.x < -0.99)
                            {
                                leftCells.push_back(i);
                                leftFaces.push_back(fIdx);
                                break;
                            }
                        }
                    }
                }

                size_t localLeftCells = leftCells.size();
                size_t totalLeftCells = localLeftCells;
#ifdef RICH_MPI
                MPI_Allreduce(&localLeftCells, &totalLeftCells, 1,
                              MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

                size_t packetsPerCell = N_pkt / totalLeftCells;
                size_t remainder = N_pkt % totalLeftCells;

                size_t globalCellOffset = 0;
#ifdef RICH_MPI
                MPI_Exscan(&localLeftCells, &globalCellOffset, 1,
                           MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
                if (rank == 0)
                    globalCellOffset = 0;
#endif
                size_t localTotal = 0;
                for (size_t c = 0; c < localLeftCells; ++c)
                    localTotal += packetsPerCell + ((globalCellOffset + c) < remainder ? 1 : 0);

                size_t idOffset = 0;
#ifdef RICH_MPI
                MPI_Exscan(&localTotal, &idOffset, 1, MPI_UNSIGNED_LONG,
                           MPI_SUM, MPI_COMM_WORLD);
                if (rank == 0)
                    idOffset = 0;
#endif

                double T4 = T_src * T_src * T_src * T_src;
                std::mt19937_64 rng(42 + static_cast<unsigned>(rank));
                std::uniform_real_distribution<double> u01(0.0, 1.0);

                size_t idCounter = idOffset;
                for (size_t c = 0; c < localLeftCells; ++c)
                {
                    size_t cellIdx = leftCells[c];
                    size_t faceIdx = leftFaces[c];
                    double area = tess.GetArea(faceIdx);

                    size_t nPkt = packetsPerCell +
                                  ((globalCellOffset + c) < remainder ? 1 : 0);

                    double totalEnergy = units::sigma_sb * T4 * area * dt_mc;
                    double weightPerPacket = totalEnergy / static_cast<double>(nPkt);

                    for (size_t k = 0; k < nPkt; ++k)
                    {
                        Particle3D p;
                        p.location = RandomPointOnFace(tess, faceIdx);

                        double mu = std::sqrt(u01(rng));
                        double phi = 2.0 * M_PI * u01(rng);
                        double sinTheta = std::sqrt(1.0 - mu * mu);
                        p.velocity = Vector3D(mu, sinTheta * std::cos(phi),
                                              sinTheta * std::sin(phi));
                        p.velocity *= units::clight;

                        p.frequency = sampleTruncatedPlanck(E_trunc_lo, E_trunc_hi,
                                                             T_src, rng);
                        p.weight = weightPerPacket;
                        p.initialWeight = weightPerPacket;
                        p.timeLeft = 0;
                        p.cellIndex = cellIdx;
                        p.cellID = cells[cellIdx].ID;
                        p.id = idCounter++;
                        p.steps = 0;
                        p.on_track = false;
                        p.sent = false;
                        seedParticles.push_back(p);
                    }
                }

                if (rank == 0)
                    std::cout << "Built " << localTotal << " seed photons on rank 0 left boundary "
                              << "(total across ranks: " << N_pkt << ")" << std::endl;
            }

            RadiationIMCParameters imcParams = {
                .newPhotonsPerCell = 0,
                .withHydro = true,
                .diffusionPressureGradient = false,
                .MMC = false,
                .withMultigroupOpacity = true,
                .withRandomWalk = false,
                .noHydroFeedback = true};

            auto physics = std::make_shared<RadiationIMC>(
                tess, boundaryCond, cells, extensives, eosPtr, opacityPtr,
                imcParams);

            auto popControl =
                std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

            auto mcStep = std::make_shared<RadiationMCStep>(
                tess, cells, extensives, physics, popControl, boundaryCond,
                seedParticles, 0, true
#ifdef RICH_MPI
                ,
                RadiationMCStep::ManagerType::AUTO_RDMA
#endif
            );

            sim.addPhysics(mcStep);
#ifdef RICH_MPI
            mcStep->setCost(std::make_shared<IMCCostCalculator>(mcStep->getManager()));
            sim.setForceRebalanceSteps(4);
            sim.addMigrationBuffer(mcStep->getManager()->GetCellsStepsCounters());
#endif

            if (rank == 0)
            {
                std::cout << "MC parameters: Nx=" << Nx << " cells (global), "
                          << G << " groups, N_pkt=" << N_pkt
                          << ", kappa_s=" << kappa_s
                          << ", tau=" << tau
                          << ", dt=" << dt_mc
                          << ", max_steps=" << maxSteps << std::endl;
            }

            auto wallStart = std::chrono::high_resolution_clock::now();

            for (size_t step = 0; step < maxSteps; ++step)
            {
                sim.SetTimeStep(dt_mc);
                sim.step();

                size_t localRemaining = mcStep->getParticles().size();
                size_t globalRemaining = localRemaining;
#ifdef RICH_MPI
                MPI_Allreduce(&localRemaining, &globalRemaining, 1,
                              MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

                if (rank == 0 && (step % 10 == 0 || globalRemaining == 0))
                {
                    double elapsed = std::chrono::duration<double>(
                                         std::chrono::high_resolution_clock::now() - wallStart)
                                         .count();
                    std::cout << "MC step " << step + 1 << "/" << maxSteps
                              << "  remaining=" << globalRemaining
                              << "  elapsed=" << elapsed << "s" << std::endl;
                }

                if (globalRemaining == 0)
                {
                    if (rank == 0)
                        std::cout << "All photons escaped after "
                                  << step + 1 << " steps." << std::endl;
                    break;
                }
            }

            // Gather escaped photon data
            const auto &localEscaped = boundaryCond->getEscaped();
            std::vector<EscapedPhoton> allEscaped;
#ifdef RICH_MPI
            allEscaped = gatherEscapedPhotons(localEscaped);
#else
            allEscaped = localEscaped;
#endif

            // Bin right-escaped photons into comoving-frame spectrum
            std::vector<double> Eg_mc_comoving(G, 0.0);
            size_t rightCount = 0;
            if (rank == 0)
            {
                for (const auto &ep : allEscaped)
                {
                    if (!ep.right_side)
                        continue;
                    rightCount++;
                    size_t g = opacityPtr->findGroup(ep.comoving_energy);
                    if (g < G)
                        Eg_mc_comoving[g] += ep.weight;
                }
                double totalWeight = 0;
                for (size_t g = 0; g < G; ++g)
                    totalWeight += Eg_mc_comoving[g];

                // Normalize: convert to energy density per group
                // (weight is in energy units; divide by group width for spectral density)
                if (totalWeight > 0)
                {
                    for (size_t g = 0; g < G; ++g)
                        Eg_mc_comoving[g] /= totalWeight;
                }

                std::cout << "MC done: " << allEscaped.size() << " total escaped, "
                          << rightCount << " at right boundary." << std::endl;

                double wallTotal = std::chrono::duration<double>(
                                       std::chrono::high_resolution_clock::now() - wallStart)
                                       .count();
                std::cout << "MC wall time: " << wallTotal << "s" << std::endl;
            }

#ifdef RICH_MPI
            MPI_Bcast(Eg_mc_comoving.data(), static_cast<int>(G), MPI_DOUBLE,
                      0, MPI_COMM_WORLD);
#endif

            // =============================================================
            //  PHASE 2: Multigroup diffusion
            // =============================================================
            if (rank == 0)
                std::cout << "\n=== Phase 2: Multigroup diffusion ===" << std::endl;

            {
                // Re-create simulation for diffusion
                Simulation simDiff(tess, initialCells, eos);
                auto tscDiff = std::make_shared<ManualTimeStep>();
                simDiff.SetTimeStepFunction(tscDiff);

                auto &cellsDiff = simDiff.getCells();
                auto &extensivesDiff = simDiff.getExtensives();
                extensivesDiff.resize(cellsDiff.size());
                for (size_t i = 0; i < cellsDiff.size(); ++i)
                    PrimitiveToConserved(cellsDiff[i], tess.GetVolume(i),
                                         extensivesDiff[i]);

                AnalyticOpacity diffOpacity(
                    [&](ComputationalCell3D const & /*cell*/, double /*energy*/) -> double
                    {
                        return units::clight / (3.0 * kappa_s);
                    },
                    [](ComputationalCell3D const & /*cell*/, double /*energy*/) -> double
                    {
                        return 1e-100;
                    },
                    [&](ComputationalCell3D const & /*cell*/, double /*energy*/) -> double
                    {
                        return kappa_s;
                    },
                    energy_groups_center,
                    energy_groups_boundary);

                LeftPlanckRightOpenBoundary diffBoundary(T_src,
                                                         energy_groups_center,
                                                         energy_groups_boundary);

                // MultigroupDiffusion(centers, boundaries, opacity, boundary, eos,
                //                     tracerNames, flux_limiter, hydro_on, compton_on, doppler_on, ...)
                MultigroupDiffusion diffusion(
                    energy_groups_center, energy_groups_boundary,
                    diffOpacity, diffBoundary, eos,
                    std::vector<std::string>(),
                    true,   // flux_limiter
                    false,  // hydro_on
                    false,  // compton_on
                    true,   // doppler_on
                    -1,     // minimum_temperature (disabled)
                    false,  // protections_on
                    false); // cooling_time_limiter_on

                auto radStep = std::make_shared<RadiationStep>(
                    tess, simDiff.getCells(), simDiff.getExtensives(),
                    simDiff.getTracker(),
#ifdef RICH_MPI
                    nullptr,
#endif
                    diffusion, true); // no_hydro = true

                simDiff.addPhysics(radStep);

                // Time-step until steady state
                // Diffusion time ~ t_diff ≈ 1e5 s; run for ~5 * t_diff
                double const dt_diff_init = 1.0;
                double const t_end_diff = 5.0 * t_diff;
                size_t const maxDiffSteps = 100000;
                double dt_diff = dt_diff_init;

                auto wallStartDiff = std::chrono::high_resolution_clock::now();

                if (rank == 0)
                    std::cout << "Diffusion: t_end=" << t_end_diff
                              << ", t_diff=" << t_diff << std::endl;

                for (size_t step = 0; step < maxDiffSteps; ++step)
                {
                    simDiff.SetTimeStep(dt_diff);
                    simDiff.step();

                    // Ramp up dt
                    double suggested = simDiff.GetTimeStep();
                    dt_diff = std::min(suggested * 1.05, t_diff * 0.1);
                    dt_diff = std::max(dt_diff, dt_diff_init);

                    if (rank == 0 && step % 500 == 0)
                    {
                        double elapsed = std::chrono::duration<double>(
                                             std::chrono::high_resolution_clock::now() - wallStartDiff)
                                             .count();
                        std::cout << "Diffusion step " << step
                                  << "  t=" << simDiff.GetTime()
                                  << "  dt=" << dt_diff
                                  << "  elapsed=" << elapsed << "s" << std::endl;
                    }

                    if (simDiff.GetTime() >= t_end_diff)
                    {
                        if (rank == 0)
                            std::cout << "Diffusion reached t_end at step "
                                      << step + 1 << std::endl;
                        break;
                    }
                }

                // Extract right-boundary cell Eg (comoving frame)
                // Find the cell closest to x=L
                size_t rightCellIdx = 0;
                double maxX = -1;
                for (size_t i = 0; i < tess.GetPointNo(); ++i)
                {
                    if (tess.GetMeshPoint(i).x > maxX)
                    {
                        maxX = tess.GetMeshPoint(i).x;
                        rightCellIdx = i;
                    }
                }

                // In MPI, find the rank with the rightmost cell
                int rightRank = 0;
#ifdef RICH_MPI
                struct
                {
                    double x;
                    int rank;
                } localMax{maxX, rank}, globalMax;
                MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE_INT,
                              MPI_MAXLOC, MPI_COMM_WORLD);
                rightRank = globalMax.rank;
#endif

                std::vector<double> Eg_diff_comoving(G, 0.0);
                if (rank == rightRank)
                {
                    const auto &rightCell = simDiff.getCells()[rightCellIdx];
                    double totalEg = 0;
                    for (size_t g = 0; g < G; ++g)
                    {
                        // Eg is stored as specific (per density); multiply by density
                        Eg_diff_comoving[g] = rightCell.Eg[g] * rightCell.density;
                        totalEg += Eg_diff_comoving[g];
                    }
                    if (totalEg > 0)
                    {
                        for (size_t g = 0; g < G; ++g)
                            Eg_diff_comoving[g] /= totalEg;
                    }
                }

#ifdef RICH_MPI
                MPI_Bcast(Eg_diff_comoving.data(), static_cast<int>(G),
                          MPI_DOUBLE, rightRank, MPI_COMM_WORLD);
#endif

                if (rank == 0)
                {
                    double wallDiff = std::chrono::duration<double>(
                                          std::chrono::high_resolution_clock::now() - wallStartDiff)
                                          .count();
                    std::cout << "Diffusion wall time: " << wallDiff << "s" << std::endl;
                }

                // ==========================================================
                //  Write final output with both spectra
                // ==========================================================
                if (rank == 0)
                {
                    std::string const specPath =
                        caseDir + "/doppler_scatter_spectrum.txt";
                    std::ofstream out(specPath);
                    out << std::scientific << std::setprecision(12);

                    out << "# Doppler scatter benchmark: MC vs diffusion\n";
                    out << "# L " << L << "\n";
                    out << "# H " << H << "\n";
                    out << "# v_right " << v_right << "\n";
                    out << "# kappa_s " << kappa_s << "\n";
                    out << "# tau " << kappa_s * L << "\n";
                    out << "# T_src_kelvin " << T_src << "\n";
                    out << "# E_trunc_lo " << E_trunc_lo << "\n";
                    out << "# E_trunc_hi " << E_trunc_hi << "\n";
                    out << "# N_pkt " << N_pkt << "\n";
                    out << "# Nx " << Nx << "\n";
                    out << "# right_escaped " << rightCount << "\n";
                    out << "# t_final_diff " << simDiff.GetTime() << "\n";
                    out << "# columns: group E_lo E_hi Eg_mc_comoving Eg_diff_comoving\n";

                    for (size_t g = 0; g < G; ++g)
                    {
                        out << g
                            << " " << energy_groups_boundary[g]
                            << " " << energy_groups_boundary[g + 1]
                            << " " << Eg_mc_comoving[g]
                            << " " << Eg_diff_comoving[g]
                            << "\n";
                    }
                    out.close();
                    std::cout << "Wrote " << specPath << std::endl;
                }
            }
        }
    }
    catch (const UniversalError &e)
    {
        std::cerr << "=== UniversalError ===" << std::endl;
        reportError(e);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "=== std::exception: " << e.what() << " ===" << std::endl;
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
