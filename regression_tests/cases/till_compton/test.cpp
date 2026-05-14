#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/MultigroupDiffusionBoundaryCalculator.hpp"
#include "source/misc/int2str.hpp"
#include <boost/numeric/odeint.hpp>
#include "source/newtonian/three_dimensional/LagrangianExtensiveUpdater3D.hpp"
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <exception>
#include <fenv.h>
#include <filesystem>
#include "source/3D/GeometryCommon/UpdateBox.hpp"
namespace fs = std::filesystem;
#include <sys/stat.h>
#include <boost/math/tools/roots.hpp>
#include <sstream>
#include <source/Radiation/CMMC/src/planck_integral/planck_integral.hpp>
#include <algorithm>
#include "boost/math/special_functions/pow.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include <string_view>
#include <charconv>
#include <optional>

namespace {

static constexpr double ev = 1.602176634e-12;
static constexpr double kev = 1e3 * ev;

static constexpr double ev_kelvin = ev / CG::boltzmann_constant;
static constexpr double kev_kelvin = 1e3 * ev_kelvin;

struct Case {
  std::string const description;
  double const T_mat;
  double const T_rad;
  bool const compton_on;
  bool const absorption_on;
};

Case get_case(std::string_view const case_num_sv) {
  int case_num = -1;
  std::from_chars(case_num_sv.data(), case_num_sv.data() + case_num_sv.size(), case_num);

  switch (case_num) {
    case 0:
      return {"Winslow", 20.0 * kev_kelvin, 1.0 * kev_kelvin, true, true};
    case 1:
      return {"Winslow, no absorption", 20.0 * kev_kelvin, 1.0 * kev_kelvin, true, false};
    case 2:
      return {"Winslow, no compton", 20.0 * kev_kelvin, 1.0 * kev_kelvin, false, true};
    case 3:
      return {"Till", 1.0 * kev_kelvin, 10.0 * kev_kelvin, true, true};
    case 4:
      return {"Till, no compton", 1.0 * kev_kelvin, 10.0 * kev_kelvin, false, true};
    default:
      std::cout << "Error! No Such case as: " << case_num_sv << std::endl;
      std::cout << "Available cases: 0, 1, 2, 3, 4" << std::endl;
      exit(1);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  int rank = 0;
#ifdef RICH_MPI
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  std::size_t const G = ENERGY_GROUPS_NUM;

  // Always run Till case (option 3) in regression.
  auto const current_case = get_case("3");
  std::cout << "Running case: " << current_case.description << std::endl;
  std::cout << "T_mat = " << current_case.T_mat / kev_kelvin << " KeV, T_rad = " << current_case.T_rad / kev_kelvin
            << " KeV, compton = " << std::string(current_case.compton_on ? "ON " : "OFF")
            << ", absorption = " << std::string(current_case.absorption_on ? "ON " : "OFF") << std::endl;

  // Optional: allow forcing timestep as the first runtime argument.
  std::optional<double> force_time_step{};
  if (argc >= 2) {
    std::cout << "Force Time Step ON" << std::endl;
    double time_step = -1.0;
    std::string_view time_step_sv = argv[1];
    std::from_chars(time_step_sv.data(), time_step_sv.data() + time_step_sv.size(), time_step);
    force_time_step = time_step;
    std::cout << "Time step = " << *force_time_step << std::endl;
  }

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
    std::cout << "Error: energy_groups_boundries size does not match ENERGY_GROUPS_NUM+1" << std::endl;
    return 1;
  }

  double const lscale = 1.;
  double const mscale = 1.;
  double const tscale = 1.;

  double constexpr m_p = 1.6726231e-24;
  double constexpr cv = 3.0 * CG::boltzmann_constant / m_p;
  IdealGas eos(5.0 / 3.0, /*f=*/cv, /*beta=*/1.0, /*mu=*/0.0);

  const double width = 1 / lscale;
  size_t const Nx = 1;
  Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx), ur(width, 0.5 * width / Nx, 0.5 * width / Nx);
  Voronoi3D tess(ll, ur);

  using boost::math::pow;

  FreeFreeAbsorptionOpacityMultigroup opacity(current_case.absorption_on ? 1.0 : 1e-80, energy_groups_center,
                                               energy_groups_boundary);

  ComputationalCell3D init_cell;

  double const T_mat = current_case.T_mat;
  double const T_rad = current_case.T_rad;

  try {
    init_cell.density = 1. * lscale * lscale * lscale / mscale;
    init_cell.temperature = T_mat;
    init_cell.internal_energy =
        eos.dT2e(init_cell.density, init_cell.temperature, init_cell.tracers, ComputationalCell3D::tracerNames);
    init_cell.pressure =
        eos.de2p(init_cell.density, init_cell.internal_energy, init_cell.tracers, ComputationalCell3D::tracerNames);
    init_cell.Erad = CG::radiation_constant * pow<4>(T_rad) * tscale * tscale / (init_cell.density * mscale / lscale);
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
      init_cell.Eg[g] =
          planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g + 1], T_rad);
      init_cell.Eg[g] *= tscale * tscale / (init_cell.density * mscale / lscale);
      init_cell.Eg[g] = std::max(init_cell.Eg[g], init_cell.Erad * 1e-8);
    }
  } catch (UniversalError const& eo) {
    reportError(eo);
    throw;
  }

  vector<Vector3D> points;
  if (rank == 0) {
    points = CartesianMesh(Nx, 1, 1, ll, ur);
  }
  try {
#ifdef RICH_MPI
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif
  } catch (UniversalError const& eo) {
    reportError(eo);
    throw;
  }
  vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);
  for (size_t i = 0; i < cells.size(); ++i) {
    if (tess.GetCellCM(i).x < 2.0) {
      cells[i].tracers[0] = 1.0;
    } else {
      cells[i].tracers[1] = 1.0;
    }
  }

  Hllc3D rs;
  RigidWallGenerator3D ghost;
  LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);

  Lagrangian3D bpm;
  RoundCells3D pm(bpm, eos, 3.75, 0.01, false, 1.25);

  MultigroupDiffusionClosedBoundary D_boundary{};

  constexpr bool flux_limiter = true;
  constexpr bool hydro_on = false;
  const bool compton_on = current_case.compton_on;
  constexpr bool doppler_on = false;
  constexpr bool protections_on = false;

  MultigroupDiffusion matrix_builder{energy_groups_center,
                                     energy_groups_boundary,
                                     opacity,
                                     D_boundary,
                                     eos,
                                     std::vector<std::string>(),
                                     flux_limiter,
                                     hydro_on,
                                     compton_on,
                                     doppler_on,
                                     -1,
                                     protections_on};

  matrix_builder.length_scale_ = lscale;
  matrix_builder.time_scale_ = tscale;
  matrix_builder.mass_scale_ = mscale;
  ZeroForce3D force = ZeroForce3D();

  DefaultCellUpdater cu(false, 0.0, true, 0.0, &matrix_builder);

  RigidWallFlux3D rigidflux(rs);
  RegularFlux3D* regular_flux = new RegularFlux3D(rs);
  IsBoundaryFace3D* boundary_face = new IsBoundaryFace3D();
  IsBulkFace3D* bulk_face = new IsBulkFace3D();
  vector<pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*>> flux_vector;
  flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*>(boundary_face, &rigidflux));
  flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*>(bulk_face, regular_flux));
  ConditionActionFlux1 fc(flux_vector, interp);

  vector<pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
  ConditionExtensiveUpdater3D eu(eu_sequence);

  auto tsf = std::make_shared<CourantFriedrichsLewy>(0.25, 1, force);

  Simulation simulation(tess, cells, eos);
  simulation.SetTimeStepFunction(tsf);
  HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
              std::pair<std::vector<std::string>, std::vector<std::string>>(ComputationalCell3D::tracerNames,
                                                                              ComputationalCell3D::stickerNames));

  double init_dt = 1e-12 / tscale;
  double const tf = 3e-8 / tscale;
  simulation.SetTimeStep(init_dt);
  double old_dt = init_dt;
  double old_time = simulation.GetTime();

  double new_dt = force_time_step ? *force_time_step : init_dt;
  std::vector<double> Tgas, Trad, Etotal, time;
  Tgas.push_back(init_cell.temperature);
  Trad.push_back(std::pow(init_cell.Erad / CG::radiation_constant, 0.25));
  Etotal.push_back(init_cell.internal_energy + init_cell.Erad);
  time.push_back(0.0);

  auto radStep = std::make_shared<RadiationStep>(tess, simulation.getCells(), simulation.getExtensives(),
      simulation.getTracker(),
#ifdef RICH_MPI
      nullptr,
#endif
      matrix_builder, false);
  simulation.addPhysics(radStep);

  while (simulation.GetTime() < tf) {
    try {
      simulation.SetTimeStep(old_dt);
      simulation.step();
      new_dt = simulation.GetTimeStep();
      double const elapsed_time = simulation.GetTime();
      double const dt_step = elapsed_time - old_time;
      if (rank == 0) {
        std::cout << "Cycle " << simulation.GetCycle()
                  << " dt " << dt_step
                  << " elapsed " << elapsed_time << std::endl;
      }
      old_time = elapsed_time;

      auto const& cell = simulation.getCells()[0];
      Tgas.push_back(cell.temperature);
      Trad.push_back(std::pow(cell.Erad * cell.density / CG::radiation_constant, 0.25));
      Etotal.push_back(cell.internal_energy + cell.Erad);
      time.push_back(simulation.GetTime());

      if (force_time_step) {
        new_dt = force_time_step.value();
      } else {
        new_dt = std::min(old_dt * 1.2, 1e-10);
      }

      old_dt = new_dt;
    } catch (UniversalError const& eo) {
      reportError(eo);
      throw;
    }
  }
  std::string const case_dir = fs::path(__FILE__).parent_path().string();
  write_vector(time, case_dir + "/time.txt");
  write_vector(Tgas, case_dir + "/Tgas.txt");
  write_vector(Trad, case_dir + "/Trad.txt");
  write_vector(Etotal, case_dir + "/Etotal.txt");
  std::cout << "Done" << std::endl;
#ifdef RICH_MPI
  MPI_Finalize();
#endif
  return 0;
}
