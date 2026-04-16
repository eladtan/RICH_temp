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
#include "source/3D/radiation/LorentzTransformation.hpp"
#include "source/monte/population/NoControl.hpp"
#include "source/monte/boundary/Rigid.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"

namespace fs = std::filesystem;

namespace
{
    // Opacity class that mimics a linear velocity gradient v(x) = v0 * x / width.
    // At each scatter event it computes the virtual local velocity from the
    // photon's x-position, Lorentz-transforms to the comoving frame, samples an
    // isotropic scatter direction, and transforms back to the lab frame.
    class VelocityGradientOpacity : public OpacityCalculator
    {
    public:
        VelocityGradientOpacity(double kappa_s, double v0, double width,
                                const std::vector<double> &centers,
                                const std::vector<double> &boundaries)
            : kappa_s_(kappa_s), v0_(v0), width_(width)
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

        Vector3D getNewScatterVelocity(ComputationalCell3D const &cell, MCParticle &particle) const override
        {
            Vector3D v_local(v0_ * particle.location.x / width_, 0, 0);
            LorentzTransformation(particle, v_local);
            particle.velocity = getRandomVelocity(cell);
            LorentzTransformation(particle, -1.0 * v_local);
            return particle.velocity;
        }

    private:
        double kappa_s_;
        double v0_;
        double width_;
    };

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

int main(int /*argc*/, char * /*argv*/[])
{
    try
    {
        constexpr size_t G = ENERGY_GROUPS_NUM;
        static_assert(G >= 10, "Need at least 10 energy groups for the Doppler MC test");

        // --- energy group grid (log-spaced) ---
        std::vector<double> energy_groups_center(G);
        std::vector<double> energy_groups_boundary(G + 1);
        double const Emin = units::kev * 1e-1;
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
        // Preserves div(v) = v0/width = 5e6 s^-1, same K*t as original test.
        double const v0      = 2.5e8;       // cm/s   (virtual max velocity at x = width)
        double const T       = units::kev_kelvin;
        double const rho     = 1.0;         // g/cm^3
        double const kappa_s = 40.0;        // scattering opacity [cm^-1]

        double const E_trunc_lo = 1.12 * units::kev;
        double const E_trunc_hi = 8.12 * units::kev;

        // --- timing ---
        double const dt = 4e-10;
        size_t const iterations = 50;

        size_t const totalPhotons = 100000;

        // --- geometry: single cell, optically thick slab ---
        // Diffusion distance ~ 3 cm in t_total, width = 50 gives ~17x margin.
        double const width     = 50.0;
        size_t const Nx        = 1;
        double const cellHalfY = 0.025;
        Vector3D ll(0, -cellHalfY, -cellHalfY);
        Vector3D ur(width, cellHalfY, cellHalfY);

        std::vector<Vector3D> points = CartesianMesh(Nx, 1, 1, ll, ur);
        Voronoi3D tess(ll, ur);
        tess.Build(points);

        // --- EOS (irrelevant, no hydro evolution) ---
        double const cv = 1e15 / units::kev_kelvin;
        IdealGas eos(1.4, cv, 1, 0);

        // --- single cell: zero velocity ---
        size_t const N = tess.GetPointNo();
        std::vector<ComputationalCell3D> initialCells(N);
        for (size_t i = 0; i < N; ++i)
        {
            auto &c = initialCells[i];
            c.density = rho;
            c.temperature = T;
            c.internal_energy = eos.dT2e(c.density, c.temperature, c.tracers, ComputationalCell3D::tracerNames);
            c.pressure = eos.de2p(c.density, c.internal_energy, c.tracers, ComputationalCell3D::tracerNames);

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
            c.velocity = Vector3D(0, 0, 0);
        }

        // --- record initial cell spectrum (truncated Planck, for reference) ---
        std::vector<double> Eg_init_cell(G, 0.0);
        for (size_t g = 0; g < G; ++g)
            Eg_init_cell[g] = initialCells[0].Eg[g] * initialCells[0].density;

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
        auto opacityPtr = std::make_shared<VelocityGradientOpacity>(
            kappa_s, v0, width, energy_groups_center, energy_groups_boundary);

        // --- seed photons at the left boundary (x ~ 0) ---
        std::vector<Particle3D> initialParticles;
        initialParticles.reserve(totalPhotons);

        double totalErad = 0;
        for (size_t g = 0; g < G; ++g)
            totalErad += initialCells[0].Eg[g] * initialCells[0].density;
        totalErad *= tess.GetVolume(0);
        double const weightPerPhoton = totalErad / totalPhotons;

        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> distYZ(-cellHalfY, cellHalfY);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double const nudge = 1e-6 * width;

        for (size_t k = 0; k < totalPhotons; ++k)
        {
            Particle3D p;
            p.location = Vector3D(nudge, distYZ(rng), distYZ(rng));

            // Lambert cosine law directed into the domain (+x)
            double mu = std::sqrt(u01(rng));
            double sinTheta = std::sqrt(1.0 - mu * mu);
            double phi = 2.0 * M_PI * u01(rng);
            p.velocity = Vector3D(mu, sinTheta * std::cos(phi), sinTheta * std::sin(phi)) * units::clight;

            p.frequency = sampleTruncatedPlanck(E_trunc_lo, E_trunc_hi, T, rng);
            p.weight = weightPerPhoton;
            p.initialWeight = weightPerPhoton;
            p.cellIndex = 0;
            p.cellID = cells[0].ID;
            p.id = k;
            p.timeLeft = 0;
            p.steps = 0;
            p.on_track = false;
            p.sent = false;
            initialParticles.push_back(p);
        }

        // --- record initial photon spectrum ---
        std::vector<double> Eg_init_photons(G, 0.0);
        for (const auto &p : initialParticles)
        {
            size_t g = opacityPtr->findGroup(p.frequency);
            if (g < G)
                Eg_init_photons[g] += p.weight;
        }
        double const vol = tess.GetVolume(0);
        for (size_t g = 0; g < G; ++g)
            Eg_init_photons[g] /= vol;

        // --- configure MC physics ---
        RadiationIMCParameters radiationIMCParameters = {
            .newPhotonsPerCell = 0,
            .withHydro = true,
            .diffusionPressureGradient = false,
            .MMC = false,
            .withMultigroupOpacity = true,
            .withRandomWalk = false,
            .noHydroFeedback = true
        };

        auto boundaryCond =
            std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        auto physics = std::make_shared<RadiationIMC>(
            tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

        auto popControl =
            std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, popControl, boundaryCond,
            initialParticles, 0, true);

        sim.addPhysics(mcStep);
        sim.SetTimeStep(dt);

        std::cout << "Doppler MC (single cell, velocity-gradient opacity): "
                  << G << " groups, " << totalPhotons << " photons, v0=" << v0
                  << ", width=" << width << ", kappa_s=" << kappa_s
                  << ", dt=" << dt << ", steps=" << iterations << std::endl;

        auto wallStart = std::chrono::high_resolution_clock::now();

        for (size_t step = 0; step < iterations; ++step)
        {
            sim.step();
            sim.SetTimeStep(dt);

            if (step % 10 == 0 || step + 1 == iterations)
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

        // --- collect final spectrum ---
        const auto &particles = mcStep->getParticles();
        std::vector<double> Eg_final(G, 0.0);
        for (const auto &p : particles)
        {
            size_t g = opacityPtr->findGroup(p.frequency);
            if (g < G)
                Eg_final[g] += p.weight;
        }
        for (size_t g = 0; g < G; ++g)
            Eg_final[g] /= vol;

        // --- write spectrum file ---
        std::string const caseDir = fs::path(__FILE__).parent_path().string();
        std::string const specPath = caseDir + "/doppler_mc_spectrum.txt";
        std::ofstream out(specPath);
        out << std::scientific << std::setprecision(12);

        double const divv = v0 / width;
        double const K = -divv / 3.0;

        out << "# Doppler MC spectrum (single cell, velocity-gradient opacity)\n";
        out << "# v0 " << v0 << "\n";
        out << "# width " << width << "\n";
        out << "# Nx " << Nx << "\n";
        out << "# t_final " << sim.GetTime() << "\n";
        out << "# K " << K << "\n";
        out << "# T_kelvin " << T << "\n";
        out << "# E_trunc_lo " << E_trunc_lo << "\n";
        out << "# E_trunc_hi " << E_trunc_hi << "\n";
        out << "# columns: group E_lo E_hi Eg_init_cell Eg_init_photons Eg_final\n";

        for (size_t g = 0; g < G; ++g)
        {
            out << g
                << " " << energy_groups_boundary[g]
                << " " << energy_groups_boundary[g + 1]
                << " " << Eg_init_cell[g]
                << " " << Eg_init_photons[g]
                << " " << Eg_final[g]
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

    return 0;
}
