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
#ifdef RICH_MPI
#include <mpi.h>
#endif

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/monte/population/Comb.hpp"
#include "source/monte/boundary/Rigid.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"

namespace fs = std::filesystem;

namespace
{
    // Transparent to absorption/emission, high scattering so photons undergo
    // frequent Lorentz boosts that produce the Doppler frequency shift.
    class DopplerMCOpacity : public OpacityCalculator
    {
    public:
        DopplerMCOpacity(double kappa_s,
                         const std::vector<double> &centers,
                         const std::vector<double> &boundaries)
            : kappa_s_(kappa_s)
        {
            energy_groups_center = centers;
            energy_groups_boundary = boundaries;
        }

        double CalcPlanckOpacity(const ComputationalCell3D & /*cell*/) const override
        {
            return 0.0;
        }

        double CalcAbsorptionOpacity(const ComputationalCell3D & /*cell*/, double /*energy*/) const override
        {
            return 1e-100;
        }

        double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/, double /*energy*/) const override
        {
            return kappa_s_;
        }

        double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/) const override
        {
            return kappa_s_;
        }

    private:
        double kappa_s_;
    };

    // Sample a photon energy from a truncated Planck distribution via rejection
    double sampleTruncatedPlanck(double E_lo, double E_hi, double T,
                                 std::mt19937_64 &rng)
    {
        double kT = units::k_boltz * T;
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
}

int main(int argc, char *argv[])
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
#endif

    try
    {
        constexpr size_t G = ENERGY_GROUPS_NUM;
        static_assert(G >= 10, "Need at least 10 energy groups for the Doppler MC test");

        // --- energy group grid (log-spaced) ---
        std::vector<double> energy_groups_center(G);
        std::vector<double> energy_groups_boundary(G + 1);
        double const Emin = units::kev * 1e-4;
        double const Emax = units::kev * 1e2;
        energy_groups_boundary[0] = Emin;
        for (size_t g = 0; g < G; ++g)
        {
            energy_groups_boundary[g + 1] = std::pow(Emax / Emin, 1.0 / G) * energy_groups_boundary[g];
            energy_groups_center[g] = 0.5 * (energy_groups_boundary[g + 1] + energy_groups_boundary[g]);
        }
        for (size_t g = 0; g <= G; ++g)
            ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];

        // --- physical parameters ---
        double const v0 = 1e9;                       // cm/s
        double const T  = units::kev_kelvin;         // 1 keV in Kelvin
        double const rho = 1.0;                      // g/cm^3
        double const kappa_s = 10.0;                 // scattering opacity [cm^-1]

        double const E_trunc_lo = 1.12 * units::kev; // truncated Planck window
        double const E_trunc_hi = 8.12 * units::kev;

        // --- timing ---
        double const tf = 4e-8;                      // 40 ns
        double const dt = 1e-10;                     // 0.1 ns
        size_t const iterations = static_cast<size_t>(tf / dt);

        constexpr size_t photonsPerCell = 10000;

        // --- geometry: two 5x5x5 cells ---
        double const width = 10.0;
        size_t const Nx = 2;
        Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx);
        Vector3D ur(width, 0.5 * width / Nx, 0.5 * width / Nx);

        std::vector<Vector3D> points = CartesianMesh(Nx, 1, 1, ll, ur);
        Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        // --- EOS (irrelevant, no hydro evolution) ---
        double const cv = 1e15 / units::kev_kelvin;
        IdealGas eos(1.4, cv, 1, 0);

        // --- initial cells ---
        size_t const N = tess.GetPointNo();
        std::vector<ComputationalCell3D> initialCells(N);
        for (size_t i = 0; i < N; ++i)
        {
            auto &c = initialCells[i];
            c.density = rho;
            c.temperature = T;
            c.internal_energy = eos.dT2e(c.density, c.temperature, c.tracers, ComputationalCell3D::tracerNames);
            c.pressure = eos.de2p(c.density, c.internal_energy, c.tracers, ComputationalCell3D::tracerNames);

            // Fill truncated Planck radiation
            for (size_t g = 0; g < G; ++g)
            {
                double Elo = energy_groups_boundary[g];
                double Ehi = energy_groups_boundary[g + 1];
                if (Ehi <= E_trunc_lo || Elo >= E_trunc_hi)
                {
                    c.Eg[g] = 0.0;
                    continue;
                }
                double a = std::max(Elo, E_trunc_lo);
                double b = std::min(Ehi, E_trunc_hi);
                c.Eg[g] = planck_integral::planck_energy_density_group_integral(a, b, T) / c.density;
            }
            c.Erad = std::accumulate(c.Eg.begin(), c.Eg.end(), 0.0);

            bool leftCell = tess.GetMeshPoint(i).x < (0.5 * width);
            c.velocity = leftCell ? Vector3D(0, 0, 0) : Vector3D(v0, 0, 0);
        }

        // --- record initial spectrum (per-volume, before simulation modifies cells) ---
        std::vector<std::vector<double>> Eg_init(N, std::vector<double>(G));
        for (size_t i = 0; i < N; ++i)
            for (size_t g = 0; g < G; ++g)
                Eg_init[i][g] = initialCells[i].Eg[g] * initialCells[i].density;

        // --- simulation ---
        Simulation sim(tess, initialCells, eos);
        auto tsc = std::make_shared<ManualTimeStep>();
        sim.SetTimeStepFunction(tsc);

        auto &cells = sim.getCells();
        auto &extensives = sim.getExtensives();
        extensives.resize(cells.size());
        for (size_t i = 0; i < cells.size(); ++i)
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

        auto eosPtr = std::make_shared<IdealGas>(eos);
        auto opacityPtr = std::make_shared<DopplerMCOpacity>(kappa_s, energy_groups_center, energy_groups_boundary);

        RadiationIMCParameters radiationIMCParameters = {
            .newPhotonsPerCell = 0,
            .withHydro = true,
            .diffusionPressureGradient = false,
            .MMC = false,
            .withMultigroupOpacity = true,
            .withRandomWalk = false
        };

        auto boundaryCond =
            std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        auto physics = std::make_shared<RadiationIMC>(
            tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

        auto popControl =
            std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, photonsPerCell, 1.0);

        // --- build initial photons manually (truncated Planck, isotropic) ---
        std::vector<Particle3D> initialParticles;
        initialParticles.reserve(photonsPerCell * N);
        std::mt19937_64 rng(42);

        for (size_t i = 0; i < N; ++i)
        {
            double totalErad = 0;
            for (size_t g = 0; g < G; ++g)
                totalErad += Eg_init[i][g];
            totalErad *= tess.GetVolume(i);

            double weightPerPhoton = totalErad / photonsPerCell;
            Vector3D center = tess.GetMeshPoint(i);
            double halfW = 0.5 * width / Nx;

            std::uniform_real_distribution<double> distX(center.x - halfW, center.x + halfW);
            std::uniform_real_distribution<double> distYZ(-halfW, halfW);
            std::uniform_real_distribution<double> dist01(-1.0 + 1e-14, 1.0 - 1e-14);

            for (size_t k = 0; k < photonsPerCell; ++k)
            {
                Particle3D p;
                p.location = Vector3D(distX(rng), distYZ(rng), distYZ(rng));
                double vx = dist01(rng), vy = dist01(rng), vz = dist01(rng);
                p.velocity = normalize(Vector3D(vx, vy, vz)) * units::clight;
                p.frequency = sampleTruncatedPlanck(E_trunc_lo, E_trunc_hi, T, rng);
                p.weight = weightPerPhoton;
                p.initialWeight = weightPerPhoton;
                p.cellIndex = i;
                p.cellID = cells[i].ID;
                p.id = initialParticles.size();
                p.timeLeft = 0;
                p.steps = 0;
                p.on_track = false;
                p.sent = false;
                initialParticles.push_back(p);
            }
        }

        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, popControl, boundaryCond,
            initialParticles, 0, true);

        sim.addPhysics(mcStep);
        sim.SetTimeStep(dt);

        std::cout << "Doppler MC test: " << N << " cells, " << G << " groups, "
                  << photonsPerCell << " photons/cell, v0=" << v0
                  << ", kappa_s=" << kappa_s << ", tf=" << tf << std::endl;

        auto wallStart = std::chrono::high_resolution_clock::now();

        for (size_t step = 0; step < iterations; ++step)
        {
            sim.step();
            sim.SetTimeStep(dt);

            if (step % 50 == 0 || step + 1 == iterations)
            {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - wallStart).count();
                std::cout << "Step " << step + 1 << "/" << iterations
                          << "  t=" << sim.GetTime()
                          << "  elapsed=" << elapsed << "s" << std::endl;
            }
        }

        double wallTotal = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wallStart).count();
        std::cout << "Done. Wall time: " << wallTotal << "s" << std::endl;

        // --- histogram final photon spectrum per cell ---
        const auto &particles = mcStep->getParticles();
        std::vector<std::vector<double>> Eg_final(N, std::vector<double>(G, 0.0));
        for (const auto &p : particles)
        {
            if (p.cellIndex >= N)
                continue;
            size_t g = opacityPtr->findGroup(p.frequency);
            if (g < G)
                Eg_final[p.cellIndex][g] += p.weight;
        }
        for (size_t i = 0; i < N; ++i)
        {
            double vol = tess.GetVolume(i);
            for (size_t g = 0; g < G; ++g)
                Eg_final[i][g] /= vol;
        }

        // --- write output ---
        std::string const caseDir = fs::path(__FILE__).parent_path().string();
        std::string const specPath = caseDir + "/doppler_mc_spectrum.txt";
        std::ofstream out(specPath);
        out << std::scientific << std::setprecision(12);

        double const L_x = width / Nx;
        double const divv_left = v0 / (2.0 * L_x);
        double const K_left  = -divv_left / 3.0;
        double const K_right =  divv_left / 3.0;

        out << "# Doppler MC spectrum\n";
        out << "# v0 " << v0 << "\n";
        out << "# L_x " << L_x << "\n";
        out << "# t_final " << sim.GetTime() << "\n";
        out << "# K_left " << K_left << "\n";
        out << "# K_right " << K_right << "\n";
        out << "# T_kelvin " << T << "\n";
        out << "# E_trunc_lo " << E_trunc_lo << "\n";
        out << "# E_trunc_hi " << E_trunc_hi << "\n";
        out << "# columns: group E_lo E_hi Eg_init_left Eg_init_right Eg_final_left Eg_final_right\n";

        size_t idx_left = 0, idx_right = 0;
        for (size_t i = 0; i < N; ++i)
        {
            if (tess.GetMeshPoint(i).x < 0.5 * width)
                idx_left = i;
            else
                idx_right = i;
        }

        for (size_t g = 0; g < G; ++g)
        {
            out << g
                << " " << energy_groups_boundary[g]
                << " " << energy_groups_boundary[g + 1]
                << " " << Eg_init[idx_left][g]
                << " " << Eg_init[idx_right][g]
                << " " << Eg_final[idx_left][g]
                << " " << Eg_final[idx_right][g]
                << "\n";
        }
        out.close();
        std::cout << "Wrote " << specPath << std::endl;
    }
    catch (const UniversalError &e)
    {
        std::cerr << "=== UniversalError ===" << std::endl;
        reportError(e);
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "=== std::exception: " << e.what() << " ===" << std::endl;
        return 1;
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
