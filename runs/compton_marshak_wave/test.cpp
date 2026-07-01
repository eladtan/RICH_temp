#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cfenv>
#include <memory>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusionBoundaryCalculator.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/monte/boundary/SideTemperature.hpp"
#include "source/monte/population/Comb.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/serialize/mpi_commands.hpp"
#endif

namespace fs = std::filesystem;

namespace {

constexpr double ev      = 1.602176634e-12;
constexpr double kev     = 1e3 * ev;
constexpr double ev_K    = ev / CG::boltzmann_constant;
constexpr double kev_K   = 1e3 * ev_K;

double radiation_temperature(ComputationalCell3D const& cell)
{
    return std::pow(std::max(0.0, cell.Erad * cell.density) / CG::radiation_constant, 0.25);
}

class FreeFreeOpacityMC : public FreeFreeAbsorptionOpacityMultigroup
{
public:
    using FreeFreeAbsorptionOpacityMultigroup::FreeFreeAbsorptionOpacityMultigroup;

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
    {
        double const kT = CG::boltzmann_constant * cell.temperature;
        double weightedSum = 0.0;
        double totalWeight = 0.0;
        for (std::size_t g = 0; g < energy_groups_center.size(); ++g) {
            double a = energy_groups_boundary[g] / kT;
            double b = energy_groups_boundary[g + 1] / kT;
            double Bg = ::planck_integral::planck_integral(a, b);
            double sigma_g = CalcAbsorptionOpacity(cell, energy_groups_center[g]);
            weightedSum += sigma_g * Bg;
            totalWeight += Bg;
        }
        return (totalWeight > 0.0) ? weightedSum / totalWeight : 0.0;
    }

};

enum class Mode { MC, MCIsotropic, Diffusion };

Mode parse_mode(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--diffusion")    return Mode::Diffusion;
        if (arg == "--mc-isotropic") return Mode::MCIsotropic;
        if (arg == "--mc")           return Mode::MC;
    }
    return Mode::MC;
}

std::string mode_string(Mode m)
{
    switch (m) {
        case Mode::MC:          return "mc";
        case Mode::MCIsotropic: return "mc_iso";
        case Mode::Diffusion:   return "diffusion";
    }
    return "mc";
}

}  // namespace

int main(int argc, char* argv[])
{
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
#endif

    try {
        std::size_t const G = ENERGY_GROUPS_NUM;
        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

        Mode const mode = parse_mode(argc, argv);
        std::string const mode_str = mode_string(mode);
        if (rank == 0)
            std::cout << "Marshak wave (" << mode_str << ")" << std::endl;

        // Energy groups (log-spaced)
        std::vector<double> energy_groups_center(G);
        std::vector<double> energy_groups_boundary(G + 1);

        double const Emin = kev * 1e-4;
        double const Emax = kev * 1e3;

        energy_groups_boundary[0] = Emin;
        for (std::size_t g = 0; g < G; ++g) {
            energy_groups_boundary[g + 1] = std::pow(Emax / Emin, 1.0 / G) * energy_groups_boundary[g];
            energy_groups_center[g] = 0.5 * (energy_groups_boundary[g + 1] + energy_groups_boundary[g]);
        }
        for (std::size_t g = 0; g <= G; ++g)
            ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];

        // EOS with Cv = 1e8 erg/g/K
        double constexpr cv = 1e8;
        IdealGas eos(5.0 / 3.0, cv, 1.0, 0.0);

        // Mesh: 256 cells in x over 0 cm, 1 cell in y and z
        constexpr std::size_t Nx = 256;
        double const Lx = 200.0;
        double const dy = Lx / Nx;
        Vector3D ll(0.0, -0.5 * dy, -0.5 * dy);
        Vector3D ur(Lx,   0.5 * dy,  0.5 * dy);
        Voronoi3D tess(ll, ur);
        std::vector<Vector3D> points;
        if (rank == 0)
            points = CartesianMesh(Nx, 1, 1, ll, ur);
#ifdef RICH_MPI
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        // Initial conditions: T_gas = T_rad = 0.1 keV
        double const T_init = 0.1 * kev_K;
        double const T_bath = 10.0 * kev_K;

        ComputationalCell3D init_cell;
        init_cell.density = 1.0;
        init_cell.temperature = T_init;
        init_cell.internal_energy = eos.dT2e(init_cell.density, init_cell.temperature,
                                              init_cell.tracers, ComputationalCell3D::tracerNames);
        init_cell.pressure = eos.de2p(init_cell.density, init_cell.internal_energy,
                                       init_cell.tracers, ComputationalCell3D::tracerNames);
        init_cell.Erad = CG::radiation_constant * std::pow(T_init, 4) / init_cell.density;
        for (std::size_t g = 0; g < G; ++g) {
            init_cell.Eg[g] = planck_integral::planck_energy_density_group_integral(
                energy_groups_boundary[g], energy_groups_boundary[g + 1], T_init);
            init_cell.Eg[g] /= init_cell.density;
            init_cell.Eg[g] = std::max(init_cell.Eg[g], init_cell.Erad * 1e-8);
        }

        Simulation simulation(tess, std::vector<ComputationalCell3D>(tess.GetPointNo(), init_cell), eos);
        simulation.SetTimeStepFunction(std::make_shared<ManualTimeStep>());

        auto& cells = simulation.getCells();
        auto& extensives = simulation.getExtensives();
        extensives.resize(cells.size());
        for (std::size_t i = 0; i < cells.size(); ++i)
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

        auto eos_ptr = std::make_shared<IdealGas>(eos);

        std::shared_ptr<::RadiationIMC> mc_physics;
        std::vector<double> min_fleck_history;

        // Diffusion objects must outlive the simulation loop
        std::unique_ptr<FreeFreeOpacityMC> diff_opacity;
        std::unique_ptr<MultigroupDiffusionSideBoundary> D_boundary;
        std::unique_ptr<MultigroupDiffusion> diffusion;

        double dt = 1e-14;
        double dt_max = 1e-11;
        if (mode == Mode::MC || mode == Mode::MCIsotropic) {
            constexpr std::size_t new_photons   = 50;
            constexpr std::size_t init_photons  = 20;
            constexpr std::size_t bdy_photons   = 3000;

            auto opacity_ptr = std::make_shared<FreeFreeOpacityMC>(
                1.0, energy_groups_center, energy_groups_boundary);
            auto boundary_cond = std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
                tess, cells, T_bath, bdy_photons, true);

            STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> imc_params = {
                .newPhotonsPerCell = new_photons,
                .withHydro = false,
                .diffusionPressureGradient = false,
                .MMC = false,
                .withMultigroupOpacity = true,
                .withRandomWalk = false,
                .noHydroFeedback = false,
                .withEgTimeAvg = true,
                .withCompton = true,
                .comptonUseInduced = true,
                .comptonAllowNZeroFallback = true,
                .comptonAngleDependent = (mode == Mode::MC),
                .comptonMatrixSamples = 500000,
            };

            mc_physics = std::make_shared<::RadiationIMC>(
                tess, boundary_cond, cells, extensives, eos_ptr, opacity_ptr, imc_params);

            auto group_classifier = [opacity_ptr](const Particle3D& particle) -> std::size_t {
                return opacity_ptr->findGroup(particle.frequency);
            };
            auto pop_control = std::make_shared<StratifiedCombPopulationControl<Vector3D, Tessellation3D>>(
                tess, ENERGY_GROUPS_NUM, group_classifier, 1000, 2.0, 2);
            std::vector<Particle3D> initial_particles;
            auto mc_step = std::make_shared<RadiationMCStep>(
                tess, cells, extensives, mc_physics, pop_control, boundary_cond,
                initial_particles, init_photons, false);
            simulation.addPhysics(mc_step);

        } else {
            // dt_max = 1e-11;
            diff_opacity = std::make_unique<FreeFreeOpacityMC>(
                1.0, energy_groups_center, energy_groups_boundary);
            D_boundary = std::make_unique<MultigroupDiffusionSideBoundary>(
                T_bath, energy_groups_center, energy_groups_boundary);
            diffusion = std::make_unique<MultigroupDiffusion>(
                energy_groups_center, energy_groups_boundary,
                *diff_opacity, *D_boundary, eos, std::vector<std::string>(),
                true, false, true, false, -1, false);

            auto radStep = std::make_shared<RadiationStep>(
                tess, simulation.getCells(), simulation.getExtensives(),
                simulation.getTracker(),
            #ifdef RICH_MPI
                nullptr,
            #endif
                *diffusion, false);
            simulation.addPhysics(radStep);
        }

        double const tf = 4e-9;

        while (simulation.GetTime() < tf) {
            double const dt_step = std::min(dt, tf - simulation.GetTime());
            if (dt_step <= 0.0) break;

            simulation.SetTimeStep(dt_step);
            simulation.step();

            if ((mode == Mode::MC || mode == Mode::MCIsotropic) && mc_physics) {
                auto const& fleck = mc_physics->getFactorFleck();
                if (!fleck.empty()) {
                    double min_f_local = *std::min_element(fleck.begin(), fleck.end());
                    double min_f = min_f_local;
#ifdef RICH_MPI
                    MPI_Allreduce(&min_f_local, &min_f, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
                    min_fleck_history.push_back(min_f);
                }
            }

            if (rank == 0 && (simulation.GetCycle() % 50 == 0 || simulation.GetCycle() <= 5)) {
                std::cout << "Cycle " << simulation.GetCycle()
                          << " dt " << dt_step
                          << " t " << simulation.GetTime()
                          << std::endl;
            }

            dt = std::min(dt * 1.1, dt_max);
        }

        // Collect final profiles (in MPI, each rank has only its owned cells)
        std::size_t const Nlocal = tess.GetPointNo();
        std::vector<double> x_local(Nlocal), Tgas_local(Nlocal), Trad_local(Nlocal);
        for (std::size_t i = 0; i < Nlocal; ++i) {
            x_local[i] = tess.GetMeshPoint(i).x;
            Tgas_local[i] = cells[i].temperature;
            Trad_local[i] = radiation_temperature(cells[i]);
        }

#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        int nlocal = static_cast<int>(Nlocal);
        int nprocs = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        std::vector<int> counts(nprocs), displs(nprocs);
        MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            displs[0] = 0;
            for (int p = 1; p < nprocs; ++p)
                displs[p] = displs[p-1] + counts[p-1];
        }
        int Ntotal = 0;
        if (rank == 0) for (int p = 0; p < nprocs; ++p) Ntotal += counts[p];

        std::vector<double> x_all(Ntotal), Tgas_all(Ntotal), Trad_all(Ntotal);
        MPI_Gatherv(x_local.data(), nlocal, MPI_DOUBLE, x_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(Tgas_local.data(), nlocal, MPI_DOUBLE, Tgas_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(Trad_local.data(), nlocal, MPI_DOUBLE, Trad_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Sort by x position
        if (rank == 0) {
            std::vector<std::size_t> idx(Ntotal);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b){ return x_all[a] < x_all[b]; });
            std::vector<double> x_sorted(Ntotal), Tg_sorted(Ntotal), Tr_sorted(Ntotal);
            for (std::size_t i = 0; i < static_cast<std::size_t>(Ntotal); ++i) {
                x_sorted[i] = x_all[idx[i]];
                Tg_sorted[i] = Tgas_all[idx[i]];
                Tr_sorted[i] = Trad_all[idx[i]];
            }
            x_all = std::move(x_sorted);
            Tgas_all = std::move(Tg_sorted);
            Trad_all = std::move(Tr_sorted);
        }
#else
        std::vector<double>& x_all = x_local;
        std::vector<double>& Tgas_all = Tgas_local;
        std::vector<double>& Trad_all = Trad_local;
#endif

        if (rank == 0) {
            std::string const case_dir = fs::path(__FILE__).parent_path().string();
            std::string const suffix = "_" + mode_str;
            write_vector(x_all,      case_dir + "/x_pos" + suffix + ".txt");
            write_vector(Tgas_all,   case_dir + "/Tgas" + suffix + ".txt");
            write_vector(Trad_all,   case_dir + "/Trad" + suffix + ".txt");
            if (!min_fleck_history.empty())
                write_vector(min_fleck_history, case_dir + "/min_fleck.txt");
            std::cout << "Done (" << mode_str << ")" << std::endl;
        }

#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    } catch (UniversalError const& eo) {
        reportError(eo);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "=== std::exception: " << e.what() << " ===" << std::endl;
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef RICH_MPI
    MPI_Finalize();
    _exit(0); // skip atexit/static destructors after local MPI resources are released
#endif
    return 0;
}
