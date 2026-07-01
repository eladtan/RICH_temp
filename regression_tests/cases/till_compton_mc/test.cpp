#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fenv.h>
#include <filesystem>
#include <iostream>
#include <optional>
#include <system_error>
#include <string_view>
#include <vector>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/monte/boundary/RigidBoundary.hpp"
#include "source/monte/population/NoPopulationControl.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"

namespace fs = std::filesystem;

namespace {

static constexpr double ev = 1.602176634e-12;
static constexpr double kev = 1e3 * ev;

static constexpr double ev_kelvin = ev / CG::boltzmann_constant;
static constexpr double kev_kelvin = 1e3 * ev_kelvin;

double radiation_temperature(const ComputationalCell3D& cell)
{
  return std::pow(std::max(0.0, cell.Erad * cell.density) / CG::radiation_constant, 0.25);
}

std::string shell_quote(std::string const& value)
{
  std::string quoted = "'";
  for (char const c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

struct RuntimeOptions {
  std::optional<double> force_time_step;
  bool include_plasma_cutoff = true;
};

std::optional<double> parse_positive_double(std::string_view value)
{
  double parsed = -1.0;
  auto const result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed <= 0.0) {
    return {};
  }

  return parsed;
}

std::optional<bool> parse_plasma_cutoff_value(std::string_view value)
{
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  return {};
}

RuntimeOptions parse_runtime_options(int argc, char* argv[])
{
  RuntimeOptions options;

  for (int i = 1; i < argc; ++i) {
    std::string_view const arg = argv[i];
    constexpr std::string_view plasma_prefix = "--plasma-cutoff=";
    constexpr std::string_view dt_prefix = "--dt=";

    if (arg == "--plasma-cutoff") {
      options.include_plasma_cutoff = true;
      continue;
    }
    if (arg == "--no-plasma-cutoff") {
      options.include_plasma_cutoff = false;
      continue;
    }
    if (arg.size() > plasma_prefix.size() && arg.substr(0, plasma_prefix.size()) == plasma_prefix) {
      std::optional<bool> const parsed = parse_plasma_cutoff_value(arg.substr(plasma_prefix.size()));
      if (parsed.has_value()) {
        options.include_plasma_cutoff = *parsed;
      } else {
        std::cout << "Ignoring invalid plasma cutoff value: " << arg << std::endl;
      }
      continue;
    }
    if (arg.size() > dt_prefix.size() && arg.substr(0, dt_prefix.size()) == dt_prefix) {
      std::optional<double> const parsed = parse_positive_double(arg.substr(dt_prefix.size()));
      if (parsed.has_value()) {
        options.force_time_step = *parsed;
      } else {
        std::cout << "Ignoring non-positive forced timestep: " << arg << std::endl;
      }
      continue;
    }
    if (arg == "--dt" && i + 1 < argc) {
      std::string_view const dt_value = argv[++i];
      std::optional<double> const parsed = parse_positive_double(dt_value);
      if (parsed.has_value()) {
        options.force_time_step = *parsed;
      } else {
        std::cout << "Ignoring non-positive forced timestep: " << dt_value << std::endl;
      }
      continue;
    }

    std::optional<double> const parsed_dt = parse_positive_double(arg);
    if (parsed_dt.has_value()) {
      options.force_time_step = *parsed_dt;
      continue;
    }

    std::cout << "Ignoring unrecognized argument: " << arg << std::endl;
  }

  if (options.force_time_step.has_value()) {
    std::cout << "Force Time Step ON" << std::endl;
    std::cout << "Time step = " << *options.force_time_step << std::endl;
  }
  std::cout << "Plasma cutoff = " << (options.include_plasma_cutoff ? "ON" : "OFF") << std::endl;

  return options;
}

}  // namespace

int main(int argc, char* argv[])
{
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  try {
    std::size_t const G = ENERGY_GROUPS_NUM;
    constexpr std::size_t Nx = 1;
    constexpr std::size_t new_photons_per_cell = 10000;
    constexpr std::size_t initial_photons_per_cell = 4000;

    std::cout << "Running case: Till MC" << std::endl;
    std::cout << "T_mat = 1 KeV, T_rad = 10 KeV, compton = ON, absorption = ON" << std::endl;

    RuntimeOptions const runtime_options = parse_runtime_options(argc, argv);
    std::optional<double> const force_time_step = runtime_options.force_time_step;

    std::vector<double> energy_groups_center(G);
    std::vector<double> energy_groups_boundary(G + 1);

    double const Emin = kev * 1e-4;
    double const Emax = kev * 1e3;

    energy_groups_boundary[0] = Emin;
    for (std::size_t g = 0; g < G; ++g) {
      energy_groups_boundary[g + 1] = std::pow(Emax / Emin, 1.0 / G) * energy_groups_boundary[g];
      energy_groups_center[g] = 0.5 * (energy_groups_boundary[g + 1] + energy_groups_boundary[g]);
    }

    if (energy_groups_center.size() != ENERGY_GROUPS_NUM) {
      std::cout << "Error: energy_groups_center size does not match ENERGY_GROUPS_NUM" << std::endl;
      return 1;
    }

    if (energy_groups_boundary.size() != ENERGY_GROUPS_NUM + 1) {
      std::cout << "Error: energy_groups_boundary size does not match ENERGY_GROUPS_NUM+1" << std::endl;
      return 1;
    }

    for (std::size_t g = 0; g <= G; ++g) {
      ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];
    }

    double constexpr m_p = 1.6726231e-24;
    double constexpr cv = 1.3 * 3.0 * CG::boltzmann_constant / m_p;
    IdealGas eos(5.0 / 3.0, /*f=*/cv, /*beta=*/1.0, /*mu=*/0.0);

    double const width = 1.0;
    Vector3D ll(0.0, -0.5 * width / Nx, -0.5 * width / Nx);
    Vector3D ur(width, 0.5 * width / Nx, 0.5 * width / Nx);
    Voronoi3D tess(ll, ur);
    std::vector<Vector3D> points = CartesianMesh(Nx, 1, 1, ll, ur);
    tess.Build(points);

    ComputationalCell3D init_cell;
    double const T_mat = 1.0 * kev_kelvin;
    double const T_rad = 10.0 * kev_kelvin;

    init_cell.density = 1.0;
    init_cell.temperature = T_mat;
    init_cell.internal_energy =
        eos.dT2e(init_cell.density, init_cell.temperature, init_cell.tracers, ComputationalCell3D::tracerNames);
    init_cell.pressure =
        eos.de2p(init_cell.density, init_cell.internal_energy, init_cell.tracers, ComputationalCell3D::tracerNames);
    init_cell.Erad = CG::radiation_constant * std::pow(T_rad, 4) / init_cell.density;
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
      init_cell.Eg[g] =
          planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g],
                                                                 energy_groups_boundary[g + 1],
                                                                 T_rad);
      init_cell.Eg[g] /= init_cell.density;
      init_cell.Eg[g] = std::max(init_cell.Eg[g], init_cell.Erad * 1e-8);
    }

    Simulation simulation(tess, std::vector<ComputationalCell3D>(tess.GetPointNo(), init_cell), eos);
    simulation.SetTimeStepFunction(std::make_shared<ManualTimeStep>());

    auto& cells = simulation.getCells();
    auto& extensives = simulation.getExtensives();
    extensives.resize(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
      PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
    }

    auto eos_ptr = std::make_shared<IdealGas>(eos);
    auto opacity_ptr =
        std::make_shared<FreeFreeAbsorptionOpacityMultigroup>(1.0,
                                                              energy_groups_center,
                                                              energy_groups_boundary,
                                                              runtime_options.include_plasma_cutoff,
                                                              true);
    auto boundary_cond = std::make_shared<RigidBoundary<Vector3D, Tessellation3D>>(tess);

    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> imc_params = {
        .newPhotonsPerCell = new_photons_per_cell,
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
        .comptonMatrixSamples = 2000000,
    };

    auto physics = std::make_shared<::RadiationIMC>(
        tess, boundary_cond, cells, extensives, eos_ptr, opacity_ptr, imc_params);

    auto pop_control =
        std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, 10000);

    std::vector<Particle3D> initial_particles;
    auto mc_step = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, pop_control, boundary_cond,
        initial_particles, initial_photons_per_cell, false);
    simulation.addPhysics(mc_step);

    double init_dt = 1e-13;
    double const tf = 3e-8;
    double old_dt = force_time_step.value_or(init_dt);
    simulation.SetTimeStep(old_dt);

    std::vector<double> Tgas;
    std::vector<double> Trad;
    std::vector<double> Etotal;
    std::vector<double> time;
    Tgas.push_back(init_cell.temperature);
    Trad.push_back(radiation_temperature(init_cell));
    Etotal.push_back(init_cell.internal_energy + init_cell.Erad);
    time.push_back(0.0);

    while (simulation.GetTime() < tf) {
      double const dt_step = std::min(old_dt, tf - simulation.GetTime());
      if (dt_step <= 0.0) {
        break;
      }

      simulation.SetTimeStep(dt_step);
      simulation.step();

      double const elapsed_time = simulation.GetTime();
      std::cout << "Cycle " << simulation.GetCycle()
                << " dt " << dt_step
                << " elapsed " << elapsed_time
                << " particles " << mc_step->getParticles().size()
                << std::endl;

      auto const& cell = simulation.getCells()[0];
      Tgas.push_back(cell.temperature);
      Trad.push_back(radiation_temperature(cell));
      Etotal.push_back(cell.internal_energy + cell.Erad);
      time.push_back(elapsed_time);

      old_dt = force_time_step.value_or(std::min(old_dt * 1.2, 1e-10));
    }

    fs::path const case_dir = fs::path(__FILE__).parent_path();
    write_vector(time, (case_dir / "time.txt").string());
    write_vector(Tgas, (case_dir / "Tgas.txt").string());
    write_vector(Trad, (case_dir / "Trad.txt").string());
    write_vector(Etotal, (case_dir / "Etotal.txt").string());

    fs::path const root_dir = case_dir.parent_path().parent_path().parent_path();
    fs::path const plot_script = root_dir / "regression_tests" / "plot_results.py";
    fs::path const plot_dir = root_dir / "regression_tests" / "plots";
    std::string const plot_command =
        "python3 " + shell_quote(plot_script.string()) +
        " --test till_compton_mc --output-dir " + shell_quote(plot_dir.string());
    std::cout << "Generating Till MC plot: " << plot_command << std::endl;
    int const plot_status = std::system(plot_command.c_str());
    if (plot_status != 0) {
      std::cerr << "Failed to generate Till MC plot" << std::endl;
      return 1;
    }

    std::cout << "Done" << std::endl;
    return 0;
  } catch (UniversalError const& eo) {
    reportError(eo);
    return 1;
  } catch (std::exception const& e) {
    std::cerr << "=== std::exception: " << e.what() << " ===" << std::endl;
    return 1;
  }
}
