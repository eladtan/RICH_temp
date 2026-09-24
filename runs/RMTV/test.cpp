// Spherical strong-conduction RMTV, evolved with RICH hydro + STORM radiation.
// See README.md for the asymptotic model, dimensional scaling, and limitations.
#ifdef RICH_MPI
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "source/3D/radiation/IMCCostCalculator.hpp"
#endif
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include "misc/mesh_generator3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "3D/tessellation/voronoi/exception/MadVoroException.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include "newtonian/three_dimensional/eulerian_3d.hpp"
#include "newtonian/three_dimensional/Lagrangian3D.hpp"
#include "newtonian/three_dimensional/RoundCells3D.hpp"
#include "newtonian/three_dimensional/Hllc3D.hpp"
#include "newtonian/three_dimensional/LinearGauss3D.hpp"
#include "newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "newtonian/three_dimensional/default_cell_updater.hpp"
#include "newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "monte/boundary/RigidBoundary.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "RMTVReference.hpp"
#include "RMTVOpacity.hpp"
#ifdef RICH_MPI
#endif

static_assert(ENERGY_GROUPS_NUM == 1, "Build RMTV with --energy_groups_num=1");

namespace
{
struct Options
{
    int n = 32, q = 8, photons = 32, initial = 16, maxPhotons = 64, maxSteps = 100000, dump = 20;
    double box = 1.2, rf0 = 0.45, rf1 = 0.9, cfl = 0.2, dtFraction = 0.005, floor = 1.0, cap = 1.e8,
           ddmcMinTau = 3.0,
           // Measured: the per-cell step count IS the cost, so it must dominate.
           // The shipped 0.005/10 made the weight ~97% particle count, which
           // misses the ~9x spread in steps per packet and left the real work
           // 35x imbalanced however often the balancer ran.
           costStepScale = 1.0, costParticleScale = 5.0;
    size_t lbInterval = 1;
    std::string mcManager = "rdma";
    double lbTolerance = 3.0;
    std::string motion = "eulerian";
    double roundChi = 1.25, roundEta = 0.02;
    unsigned seed = 12345;
    double jitter = 0; // initial lattice displacement, fraction of dx per axis
    bool ddmc = true, initOnly = false, hydroOnly = false, radiationOnly = false, adaptiveRadiationDt = false,
         octant = false, help = false;
    std::string reference = "reference/shape.dat", output = "output";
    rmtv::Scales scales;
};
Options parse(int argc, char **argv, int rank)
{
    Options o;
    ArgumentParser arguments("RMTV radiative shock benchmark (Reinicke & Meyer-ter-Vehn 1991)");
    arguments.addOption<int>("n", o.n, "cells per Cartesian axis");
    arguments.addOption<int>("quadrature", o.q, "quadrature order of the cell-averaged reference");
    arguments.addOption<int>("photons", o.photons, "packets emitted per cell per step");
    arguments.addOption<int>("initial-photons", o.initial, "packets per cell at t_start");
    arguments.addOption<int>("max-photons", o.maxPhotons, "population control target per cell");
    arguments.addOption<int>("max-steps", o.maxSteps, "cycle limit");
    arguments.addOption<int>("dump", o.dump, "snapshot interval, in cycles");
    arguments.addOption<unsigned>("seed", o.seed, "Monte Carlo seed");
    arguments.addOption<double>("box", o.box, "box half width, in units of the length scale");
    arguments.addOption<double>("rf-start", o.rf0, "heat front radius at t_start");
    arguments.addOption<double>("rf-end", o.rf1, "heat front radius at t_end");
    arguments.addOption<double>("cfl", o.cfl, "acoustic Courant factor");
    arguments.addOption<double>("dt-fraction", o.dtFraction, "step bound as a fraction of the age");
    arguments.addOption<double>("floor", o.floor, "temperature floor [K]");
    arguments.addOption<double>("opacity-cap", o.cap, "macroscopic opacity bound [1/cm]");
    arguments.addOption<double>("time-unit", o.scales.time, "time scale [s]");
    arguments.addOption<double>("temperature-unit", o.scales.temperature, "temperature scale [K]");
    arguments.addOption<double>("ddmc-min-tau", o.ddmcMinTau,
                                "cells at or above this sigma*chord are transported by DDMC");
    arguments.addOption<std::string>("motion", o.motion, "eulerian or lagrangian");
    arguments.addOption<double>("jitter", o.jitter,
                                "lattice degeneracy break, as a fraction of a cell; "
                                "lagrangian defaults to 1e-6");
    arguments.addOption<double>("round-chi", o.roundChi, "RoundCells3D correction strength");
    arguments.addOption<double>("round-eta", o.roundEta, "RoundCells3D activation threshold");
    arguments.addOption<std::string>("mc-manager", o.mcManager,
                                     "MC transport backend: rdma, rdma-ibv, mpi-rma or p2p");
    arguments.addOption<size_t>("lb-interval", o.lbInterval,
                                "MC steps between load balance checks, 0 to never check");
    arguments.addOption<double>("lb-tolerance", o.lbTolerance,
                                "rebalance once the busiest rank exceeds this times the mean");
    arguments.addOption<double>("cost-step-scale", o.costStepScale,
                                "load balance weight per transport step");
    arguments.addOption<double>("cost-particle-scale", o.costParticleScale,
                                "load balance weight per packet");
    arguments.addOption<std::string>("output", o.output, "output directory");
    arguments.addOption<std::string>("reference", o.reference, "reference shape table");
    arguments.addFlag("octant", "evolve one octant behind reflecting symmetry planes");
    arguments.addFlag("imc", "transport every packet with IMC instead of DDMC");
    arguments.addFlag("init-only", "write the initial state and stop");
    arguments.addFlag("hydro-only", "no radiation transport");
    arguments.addFlag("radiation-only", "no hydrodynamics");
    arguments.addFlag("adaptive-radiation-dt", "also honour STORM's suggested step");

    if(!arguments.parse(argc, argv))
    {
        if(rank == 0)
        {
            std::cout << arguments.help() << std::endl;
        }
        o.help = true;
        return o;
    }

    o.n = arguments.get<int>("n");
    o.q = arguments.get<int>("quadrature");
    o.photons = arguments.get<int>("photons");
    o.initial = arguments.get<int>("initial-photons");
    o.maxPhotons = arguments.get<int>("max-photons");
    o.maxSteps = arguments.get<int>("max-steps");
    o.dump = arguments.get<int>("dump");
    o.seed = arguments.get<unsigned>("seed");
    o.box = arguments.get<double>("box");
    o.rf0 = arguments.get<double>("rf-start");
    o.rf1 = arguments.get<double>("rf-end");
    o.cfl = arguments.get<double>("cfl");
    o.dtFraction = arguments.get<double>("dt-fraction");
    o.floor = arguments.get<double>("floor");
    o.cap = arguments.get<double>("opacity-cap");
    o.scales.time = arguments.get<double>("time-unit");
    o.scales.temperature = arguments.get<double>("temperature-unit");
    o.ddmcMinTau = arguments.get<double>("ddmc-min-tau");
    o.motion = arguments.get<std::string>("motion");
    o.jitter = arguments.get<double>("jitter");
    o.roundChi = arguments.get<double>("round-chi");
    o.roundEta = arguments.get<double>("round-eta");
    o.mcManager = arguments.get<std::string>("mc-manager");
    o.lbInterval = arguments.get<size_t>("lb-interval");
    o.lbTolerance = arguments.get<double>("lb-tolerance");
    o.costStepScale = arguments.get<double>("cost-step-scale");
    o.costParticleScale = arguments.get<double>("cost-particle-scale");
    o.output = arguments.get<std::string>("output");
    o.reference = arguments.get<std::string>("reference");
    o.octant = arguments.get<bool>("octant");
    o.ddmc = !arguments.get<bool>("imc");
    o.initOnly = arguments.get<bool>("init-only");
    o.hydroOnly = arguments.get<bool>("hydro-only");
    o.radiationOnly = arguments.get<bool>("radiation-only");
    o.adaptiveRadiationDt = arguments.get<bool>("adaptive-radiation-dt");

    if(o.motion != "eulerian" && o.motion != "lagrangian")
    {
        throw std::runtime_error("--motion must be eulerian or lagrangian");
    }
    if(o.mcManager != "rdma" && o.mcManager != "rdma-ibv" && o.mcManager != "mpi-rma" &&
       o.mcManager != "p2p")
    {
        throw std::runtime_error("--mc-manager must be rdma, rdma-ibv, mpi-rma or p2p");
    }
    if(!(o.jitter >= 0 && o.jitter < 0.5))
    {
        throw std::runtime_error("--jitter must be in [0, 0.5) of a cell");
    }
    if(!(o.lbTolerance >= 1))
    {
        throw std::runtime_error("--lb-tolerance must be at least 1");
    }
    if((o.hydroOnly && o.radiationOnly) || o.n < 4 || o.n % 2 || o.q < 2 || o.q % 2 || o.q > 32 ||
        o.photons < 1 || o.initial < 1 || o.maxPhotons < 1 || o.maxSteps < 1 || o.dump < 1 ||
        !(o.rf0 > 0 && o.rf1 > o.rf0 && o.box > o.rf1) ||
        !(o.cfl > 0 && o.cfl <= 0.3 && o.dtFraction > 0 && o.dtFraction <= 0.1) ||
        !(o.floor > 0 && o.cap > 0 && o.scales.time > 0 && o.scales.temperature > 0))
    {
        throw std::runtime_error(
            "Invalid configuration; require even n/q and box > rf-end > rf-start > 0");
    }
    for(double v : {o.scales.cv(), o.scales.chi0(), o.scales.age(o.rf0), o.scales.age(o.rf1)})
    {
        if(!(v > 0) || !std::isfinite(v))
        {
            throw std::runtime_error("Unrepresentable physical scales");
        }
    }
    // A moving mesh is re-tessellated every cycle, and the exact lattice is a
    // degenerate Delaunay input whose ties two ranks can break differently.
    // 1e-6 of a cell is eight orders above predicate round-off and shifts the
    // initial cell averages by ~3e-6 relative, far below anything measured.
    if(o.motion == "lagrangian" && o.jitter == 0)
    {
        o.jitter = 1e-6;
        if(rank == 0)
        {
            std::cout << "RMTV: --motion lagrangian without --jitter; using --jitter 1e-6\n";
        }
    }
    return o;
}
double sum(double v)
{
#ifdef RICH_MPI
    double out;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
#else
    return v;
#endif
}
double minimum(double v)
{
#ifdef RICH_MPI
    double out;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return out;
#else
    return v;
#endif
}
// Positive returned maximum is also used for domain-wide CFL diagnostics.
double maximum(double v)
{
    return -minimum(-v);
}

void manifest(const Options &o, int rank, int size)
{
    if(rank)
    {
        return;
    }
    std::ofstream f(o.output + "/config.json");
    f << std::setprecision(17) << "{\n\"n\":" << o.n << ",\"quadrature\":" << o.q
      << ",\"box\":" << o.box << ",\"rf_start\":" << o.rf0 << ",\"rf_end\":" << o.rf1
      << ",\"time_unit\":" << o.scales.time << ",\"temperature_unit\":" << o.scales.temperature
      << ",\"cv\":" << o.scales.cv() << ",\"chi0\":" << o.scales.chi0()
      << ",\"floor_K\":" << o.floor << ",\"opacity_cap\":" << o.cap
      << ",\"t_start\":" << o.scales.age(o.rf0) << ",\"t_end\":" << o.scales.age(o.rf1)
      << ",\"cfl\":" << o.cfl << ",\"dt_fraction\":" << o.dtFraction << ",\"seed\":" << o.seed
      << ",\"photons\":" << o.photons << ",\"initial_photons\":" << o.initial
      << ",\"max_photons\":" << o.maxPhotons << ",\"ddmc\":" << (o.ddmc ? "true" : "false")
      << ",\"ddmc_min_cell_tau\":" << o.ddmcMinTau
      << ",\"mc_manager\":\"" << o.mcManager << '"'
      << ",\"lb_tolerance\":" << o.lbTolerance << ",\"jitter\":" << o.jitter
      << ",\"motion\":\"" << o.motion << '"'
      << ",\"round_chi\":" << o.roundChi << ",\"round_eta\":" << o.roundEta
      << ",\"lb_interval\":" << o.lbInterval
      << ",\"cost_step_scale\":" << o.costStepScale
      << ",\"cost_particle_scale\":" << o.costParticleScale
      << ",\"hydro_only\":" << (o.hydroOnly ? "true" : "false")
      << ",\"radiation_only\":" << (o.radiationOnly ? "true" : "false")
      << ",\"octant\":" << (o.octant ? "true" : "false") << ",\"radiation_momentum\":false"
      << ",\"adaptive_radiation_dt\":" << (o.adaptiveRadiationDt ? "true" : "false")
      << ",\"init_only\":" << (o.initOnly ? "true" : "false") << ",\"ranks\":" << size
      << ",\"max_steps\":" << o.maxSteps << ",\"dump\":" << o.dump << "\n}\n";
}

void snapshot(const Options &o, const std::string &label, int rank, const Tessellation3D &tess,
              const std::vector<ComputationalCell3D> &cells, double age, const rmtv::Reference &ref)
{
    std::ofstream f(o.output + "/" + label + "_rank" + std::to_string(rank) + ".csv");
    if(!f)
    {
        throw std::runtime_error("Cannot write snapshot");
    }
    f << std::setprecision(17) << "# physical_age_s=" << age << "\n"
      << "x,y,z,volume,rho,vx,vy,vz,T,p,e,Er,rho_ref,vx_ref,vy_ref,vz_ref,T_ref,p_ref,e_ref\n";
    for(size_t i = 0; i < tess.GetPointNo(); ++i)
    {
        Vector3D x = tess.GetCellCM(i);
        const ComputationalCell3D &c = cells[i];
        // Average the exact solution over a cube of the cell's own volume. On a
        // uniform Cartesian mesh this is the cell exactly; on a moving mesh the
        // cells compress, and a cube of the fixed initial size would compare
        // against the wrong amount of material.
        const double cellSize = std::cbrt(tess.GetVolume(i));
        std::array<double, 6> a = ref.average(x.x, x.y, x.z, cellSize, age, o.scales, o.q, o.floor);
        const double vx = a[1] / a[0], vy = a[2] / a[0], vz = a[3] / a[0];
        const double e = a[4] / a[0] - 0.5 * (vx * vx + vy * vy + vz * vz);
        f << x.x << ',' << x.y << ',' << x.z << ',' << tess.GetVolume(i) << ',' << c.density << ','
          << c.velocity.x << ',' << c.velocity.y << ',' << c.velocity.z << ',' << c.temperature
          << ',' << c.pressure << ',' << c.internal_energy << ',' << c.Erad * c.density << ','
          << a[0] << ',' << vx << ',' << vy << ',' << vz << ',' << e / o.scales.cv() << ','
          << (rmtv::gamma - 1) * a[0] * e << ',' << e << '\n';
    }
}

// Keeps the generating points inside the box. For a point within half a cell of
// a wall the wall-normal velocity component is dropped. On the octant's three
// symmetry planes the exact velocity is already tangential, so this only
// suppresses numerical drift across them; on the outer walls the gas is
// quiescent at these ages. Without it a Lagrangian point can be pushed out of
// the domain, which the rigid-wall ghost generator cannot represent.
class BoxConfinedMotion3D : public PointMotion3D
{
    const PointMotion3D &base_;
    double lower_, upper_;

    void confine(const Tessellation3D &tess, std::vector<Vector3D> &velocities) const
    {
        const size_t n = std::min(velocities.size(), tess.GetPointNo());
        for(size_t i = 0; i < n; ++i)
        {
            const Vector3D point = tess.GetMeshPoint(i);
            const double guard = 0.5 * std::cbrt(tess.GetVolume(i));
            double *component[3] = {&velocities[i].x, &velocities[i].y, &velocities[i].z};
            const double position[3] = {point.x, point.y, point.z};
            for(int d = 0; d < 3; ++d)
            {
                if(position[d] - lower_ < guard && *component[d] < 0)
                {
                    *component[d] = 0;
                }
                if(upper_ - position[d] < guard && *component[d] > 0)
                {
                    *component[d] = 0;
                }
            }
        }
    }

  public:
    BoxConfinedMotion3D(const PointMotion3D &base, double lower, double upper)
        : base_(base), lower_(lower), upper_(upper)
    {
    }

    void operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
                    double time, vector<Vector3D> &res) const override
    {
        base_(tess, cells, time, res);
        confine(tess, res);
    }

    // RoundCells3D does its centring correction here, so it has to be forwarded
    // and the confinement reapplied afterwards.
    void ApplyFix(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
                  double time, double dt, vector<Vector3D> &velocities) const override
    {
        base_.ApplyFix(tess, cells, time, dt, velocities);
        confine(tess, velocities);
    }
};

#ifdef RICH_MPI
// RDMA by default: the transport moves few, small messages, and the previous
// P2P choice was paired with an ob1/tcp BTL restriction that ruled out the
// one-sided paths altogether.
RadiationMCStep::ManagerType managerType(const std::string &name)
{
    if(name == "p2p")
    {
        return RadiationMCStep::ManagerType::P2P;
    }
    if(name == "rdma-ibv")
    {
        return RadiationMCStep::ManagerType::RDMA_IBV;
    }
    if(name == "mpi-rma")
    {
        return RadiationMCStep::ManagerType::LEGACY_MPI_RMA;
    }
    return RadiationMCStep::ManagerType::RDMA;
}
#endif

void run(const Options &o, int rank, int size)
{
    namespace fs = std::filesystem;
    // Refuse to mix a new run with old snapshots or a different decomposition.
    if(rank == 0)
    {
        if(fs::exists(o.output) && !fs::is_empty(o.output))
        {
            throw std::runtime_error("Output directory must be empty: " + o.output);
        }
        fs::create_directories(o.output);
    }
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    manifest(o, rank, size);
    if(rank == 0)
    {
        fs::copy_file(o.reference, o.output + "/reference_shape.dat");
    }
    rmtv::Reference ref(o.reference);
    const double L = o.box * o.scales.length, lower = o.octant ? 0 : -L, dx = (L - lower) / o.n;
    const double t0 = o.scales.age(o.rf0), tend = o.scales.age(o.rf1);
    Voronoi3D tess(Vector3D(lower, lower, lower), Vector3D(L, L, L));
    std::vector<Vector3D> points;
    if(rank == 0)
    {
        // The generator breaks the lattice's Delaunay ties itself (see
        // CartesianMesh); --jitter is the displacement as a fraction of a cell.
        points = CartesianMesh(o.n, o.n, o.n, Vector3D(lower, lower, lower), Vector3D(L, L, L),
                               o.jitter);
    }
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif
    if(tess.GetPointNo() == 0)
    {
        throw std::runtime_error(
            "RMTV requires an owned cell on each rank; increase n or reduce ranks");
    }
    IdealGas eos(rmtv::gamma, o.scales.cv(), 1, 0);
    std::vector<ComputationalCell3D> initial(tess.GetPointNo());
    double floorEnergy = 0;
    for(size_t i = 0; i < initial.size(); ++i)
    {
        Vector3D x = tess.GetCellCM(i);
        // The reference is averaged over a cube of each cell's own volume, so
        // a jittered lattice is fine; the uniformity check only guards the
        // unjittered case against a mis-built mesh.
        if(o.jitter == 0 && std::abs(tess.GetVolume(i) / (dx * dx * dx) - 1) > 1e-8)
        {
            throw std::runtime_error(
                "Cell-average initialization requires uniform Cartesian Voronoi cells");
        }
        std::array<double, 6> a =
            ref.average(x.x, x.y, x.z, std::cbrt(tess.GetVolume(i)), t0, o.scales, o.q, o.floor);
        ComputationalCell3D &c = initial[i];
        c.density = a[0];
        c.velocity = Vector3D(a[1], a[2], a[3]) / a[0];
        c.internal_energy = a[4] / a[0] - 0.5 * ScalarProd(c.velocity, c.velocity);
        c.temperature = c.internal_energy / o.scales.cv();
        c.pressure = (rmtv::gamma - 1) * c.density * c.internal_energy;
        const double eosCv =
            eos.dT2cv(c.density, c.temperature, c.tracers, ComputationalCell3D::tracerNames);
        if(std::abs(eosCv / (c.density * o.scales.cv()) - 1) > 1e-12)
        {
            throw std::runtime_error("EOS heat capacity is not the required volumetric rho*Cv");
        }
        // LTE of the represented finite-volume material state, not <T^4>.
        c.Erad = units::arad * std::pow(c.temperature, 4) / c.density;
        floorEnergy += a[5] * tess.GetVolume(i);
    }
    floorEnergy = sum(floorEnergy);
    Simulation sim(tess, initial, eos);
    sim.SetTime(t0);
    std::shared_ptr<ManualTimeStep> tsc = std::make_shared<ManualTimeStep>();
    sim.SetTimeStepFunction(tsc);
    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &ext = sim.getExtensives();
    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);
    IsBoundaryFace3D boundaryFace;
    IsBulkFace3D bulkFace;
    RigidWallFlux3D wallFlux(rs);
    RegularFlux3D bulkFlux(rs);
    ConditionActionFlux1 flux({{&boundaryFace, &wallFlux}, {&bulkFace, &bulkFlux}}, interp);
    // No ongoing temperature floor: do not conceal energy injection by hydro.
    DefaultCellUpdater cu(false, 0, true, 0);
    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
                          const ConditionExtensiveUpdater3D::Action3D *>>
        actions;
    ConditionExtensiveUpdater3D eu(actions);
    ZeroForce3D force;
    Eulerian3D eulerianMotion;
    Lagrangian3D lagrangianMotion;
    RoundCells3D roundCells(lagrangianMotion, eos, o.roundChi, o.roundEta);
    BoxConfinedMotion3D confinedMotion(roundCells, lower, L);
    PointMotion3D &motion = (o.motion == "lagrangian")
                                ? static_cast<PointMotion3D &>(confinedMotion)
                                : static_cast<PointMotion3D &>(eulerianMotion);
    HDSim3D hydro(tess, cells, ext, eos, sim.getTracker(), motion, *tsc, flux, cu, eu, force,
        std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
    std::shared_ptr<HydroStep> hydroStep = std::make_shared<HydroStep>(hydro, HydroStep::TIMEADVANCE_2);
    std::shared_ptr<RMTVOpacity> opacity = std::make_shared<RMTVOpacity>(o.scales, o.floor, o.cap);
    std::shared_ptr<STORM::RigidBoundary<Vector3D, Tessellation3D>> wall = std::make_shared<STORM::RigidBoundary<Vector3D, Tessellation3D>>(tess);
    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> params;
    params.newPhotonsPerCell = o.photons;
    params.withHydro = false;
    params.withDDMC = o.ddmc; // Thermal exchange only; RMTV has no radiation force.
    // The hot, rarefied centre has sigma*chord ~ 7.5 now and ~4.5 by t_end (rho_c ~ t^-1.6),
    // below the default 15 (and below 5 late in the run). IMC
    // there costs ~tau^2 effective scatters per crossing on a handful of cells
    // that no load balancer can split. 3 keeps the centre in DDMC throughout.
    params.ddmcMinCellOpticalDepth = o.ddmcMinTau;
    params.withRandomWalk = false;
    params.withMultigroupOpacity = false;
    params.energyBoundaries.front() = 0;
    params.energyBoundaries.back() = 1e30;
    params.energyBoundariesProvided = true;
    std::shared_ptr<::RadiationIMC> physics = std::make_shared<::RadiationIMC>(
        tess, wall, cells, ext, std::make_shared<IdealGas>(eos), opacity, params);
    physics->reseedRNG(o.seed);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Tessellation3D>> population =
        std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(
        tess, o.maxPhotons, 1);
    // The step's withHydro flag remaps photons after mesh redistribution;
    // params.withHydro stays false to keep radiation forces disabled.
    std::shared_ptr<RadiationMCStep> mc = std::make_shared<RadiationMCStep>(
        tess, cells, ext, physics, population, wall, std::vector<Particle3D>{}, o.initial, true
#ifdef RICH_MPI
                                                ,
                                                managerType(o.mcManager)
#endif
    );
#ifdef RICH_MPI
    // Rebalancing drags every packet in the moved cells with it: the hot cells
    // hold ~40k packets each, so one move relocates ~1e6 particles and costs
    // seconds, while the decomposition itself takes ~0.2s. The stock 1.15
    // trigger then spends ~4.5s to recover ~0.4s of loop time. Break-even with
    // a ~2s loop is a ratio near 3.
#ifdef RICH_MPI
    tess.SetImbalanceTolerance(o.lbTolerance);
#endif
    // The balancer only reconsiders the MC partition every `lbInterval` steps.
    // The hotspot here drifts back to a ~35x imbalance within a couple of steps,
    // so the core default of 10 leaves it stale most of the time.
    mc->setRebalanceInterval(o.lbInterval);
    mc->setCost(std::make_shared<IMCCostCalculator>(mc->getManager(), o.costStepScale,
                                                    o.costParticleScale));
    sim.addMigrationBuffer(mc->getManager()->GetCellsStepsCounters());
    sim.addMigrationBuffer(mc->getManager()->GetBeginningParticleCount());
#endif
    if(!o.radiationOnly)
    {
        sim.addPhysics(hydroStep);
    }
    if(!o.hydroOnly)
    {
        sim.addPhysics(mc);
    }
    std::ofstream diag;
    if(rank == 0)
    {
        diag.open(o.output + "/diagnostics.csv");
        diag << std::setprecision(17);
        diag << "cycle,time,dt,mass,gas_energy,packet_energy,total_energy,energy_drift,"
             << "mass_drift,max_radiation_heat_capacity_ratio,max_v_over_c,min_cell_tau,"
             << "opacity_capped_cells,min_T,max_boundary_T,initial_floor_energy,hydro_energy_"
                "change,radiation_energy_change,ddmc_steps,ddmc_leaks\n";
    }
    // With hydro disabled, a region's gas-plus-census energy loss is its exact
    // net transport through its mesh-face boundary. Emission and absorption
    // cancel locally; comb population control preserves each cell's energy.
    const std::array<double, 4> ledgerRadii = {0.32, 0.37, 0.42, 4 * o.box};
    std::ofstream transportLedger;
    if(o.radiationOnly && rank == 0)
    {
        transportLedger.open(o.output + "/transport_ledger.csv");
        if(!transportLedger)
        {
            throw std::runtime_error("Cannot open transport ledger");
        }
        transportLedger << std::setprecision(17) << "cycle,start_time,dt,radius,energy_before,energy_after,net_outward_energy\n";
    }
    auto regionEnergy = [&](double radius)
    {
        long double energy = 0;
        for(size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
        {
            if(abs(tess.GetCellCM(cellIndex)) < radius * o.scales.length)
            {
                energy += ext[cellIndex].energy;
            }
        }
        for(const Particle3D &particle : mc->getParticles())
        {
            if(abs(tess.GetCellCM(particle.cellIndex)) < radius * o.scales.length)
            {
                energy += particle.weight;
            }
        }
        return sum(static_cast<double>(energy));
    };
    double E0 = 0, M0 = 0, hydroEnergyChange = 0, radiationEnergyChange = 0;
    auto totalEnergy = [&]()
    {
        double value = 0;
        for(size_t i = 0; i < tess.GetPointNo(); ++i)
        {
            value += ext[i].energy;
        }
        for(const auto &p : mc->getParticles())
        {
            value += p.weight;
        }
        return sum(value);
    };
    auto diagnostics = [&](double dt)
    {
        double M = 0, E = 0, Er = 0, ratio = 0, beta = 0, tau = 1e300, capped = 0, Tmin = 1e300, Tboundary = 0;
        for(size_t i = 0; i < tess.GetPointNo(); ++i)
        {
            const auto &c = cells[i];
            if(!(c.density > 0 && c.temperature > 0 && c.pressure > 0 && c.internal_energy > 0) ||
                !std::isfinite(c.temperature) || !std::isfinite(ext[i].energy) ||
                !std::isfinite(c.Erad) || c.Erad < 0 || !std::isfinite(c.velocity.x) ||
                !std::isfinite(c.velocity.y) || !std::isfinite(c.velocity.z))
            {
                throw std::runtime_error("Invalid evolved material state");
            }
            const double representedEnergy =
                ext[i].internal_energy +
                0.5 * ScalarProd(ext[i].momentum, ext[i].momentum) / ext[i].mass;
            if(not std::isfinite(representedEnergy) or std::abs(representedEnergy / ext[i].energy - 1) > 1e-10)
            {
                throw std::runtime_error("Gas total/internal/kinetic energies are inconsistent");
            }
            M += ext[i].mass;
            E += ext[i].energy;
            ratio = std::max(ratio, 4 * units::arad * std::pow(c.temperature, 3) /
                                        (c.density * o.scales.cv()));
            beta = std::max(beta, abs(c.velocity) / units::clight);
            tau = std::min(tau, opacity->CalcPlanckOpacity(c) * dx);
            capped += opacity->raw(c.density, c.temperature) > o.cap;
            Tmin = std::min(Tmin, c.temperature);
            auto x = tess.GetCellCM(i);
            if(std::max({std::abs(x.x), std::abs(x.y), std::abs(x.z)}) > L - 1.1 * dx)
            {
                Tboundary = std::max(Tboundary, c.temperature);
            }
        }
        for(const auto &p : mc->getParticles())
        {
            if(!std::isfinite(p.weight) || p.weight < 0)
            {
                throw std::runtime_error("Invalid packet energy");
            }
            Er += p.weight;
        }
        M = sum(M);
        E = sum(E);
        Er = sum(Er);
        ratio = maximum(ratio);
        beta = maximum(beta);
        tau = minimum(tau);
        capped = sum(capped);
        Tmin = minimum(Tmin);
        Tboundary = maximum(Tboundary);
        double ddmcSteps = sum(physics->getDDMCStepCount()),
               ddmcLeaks = sum(physics->getDDMCLeakCount());
        if(sim.GetCycle() == 0)
        {
            E0 = E + Er;
            M0 = M;
        }
        if(rank == 0)
        {
            diag << sim.GetCycle() << ',' << sim.GetTime() << ',' << dt << ',' << M << ',' << E
                 << ',' << Er << ',' << E + Er << ',' << (E + Er - E0) / E0 << ',' << (M - M0) / M0
                 << ',' << ratio << ',' << beta << ',' << tau << ',' << capped << ',' << Tmin << ','
                 << Tboundary << ',' << floorEnergy << ',' << hydroEnergyChange << ','
                 << radiationEnergyChange << ',' << ddmcSteps << ',' << ddmcLeaks << '\n';
        }
        if(beta >= 0.01)
        {
            throw std::runtime_error("RMTV scale is not sufficiently nonrelativistic");
        }
    };
    snapshot(o, "initial", rank, tess, cells, t0, ref);
    diagnostics(0);
    if(o.initOnly)
    {
        return;
    }
    double radiationDt = std::numeric_limits<double>::max();
    while(sim.GetTime() < tend && sim.GetCycle() < size_t(o.maxSteps))
    {
        // Sum of directional acoustic rates is conservative for a 3D grid.
        double dt = o.dtFraction * sim.GetTime();
        for(size_t i = 0; i < tess.GetPointNo(); ++i)
        {
            const auto &c = cells[i];
            double cs = std::sqrt(rmtv::gamma * c.pressure / c.density);
            dt = std::min(dt, o.cfl * std::cbrt(tess.GetVolume(i)) /
                                  (std::abs(c.velocity.x) + std::abs(c.velocity.y) +
                                   std::abs(c.velocity.z) + 3 * cs));
        }
        dt = minimum(std::min({dt, radiationDt, tend - sim.GetTime()}));
        if(!(dt > 0) || sim.GetTime() + dt == sim.GetTime())
        {
            throw std::runtime_error("Non-advancing timestep");
        }
        std::array<double, 4> regionBefore{};
        if(o.radiationOnly)
        {
            for(size_t regionIndex = 0; regionIndex < ledgerRadii.size(); ++regionIndex)
            {
                regionBefore[regionIndex] = regionEnergy(ledgerRadii[regionIndex]);
            }
        }
        const double before = totalEnergy();
        sim.SetTimeStep(dt);
        sim.step();
        if(!o.hydroOnly)
        {
            {
                // STORM's withHydro=false still exchanges thermal energy, but
                // does not synchronize gas total energy/velocity. Hydro owns
                // momentum here: close E = U + |p|^2/(2m) after heat exchange.
                for(size_t i = 0; i < tess.GetPointNo(); ++i)
                {
                    ext[i].energy =
                        ext[i].internal_energy +
                        0.5 * ScalarProd(ext[i].momentum, ext[i].momentum) / ext[i].mass;
                }
            }
            if(o.radiationOnly)
            {
                for(size_t regionIndex = 0; regionIndex < ledgerRadii.size(); ++regionIndex)
                {
                    const double after = regionEnergy(ledgerRadii[regionIndex]);
                    if(rank == 0)
                    {
                        transportLedger << sim.GetCycle() << ',' << sim.GetTime() - dt << ',' << dt << ','
                                        << ledgerRadii[regionIndex] << ',' << regionBefore[regionIndex] << ','
                                        << after << ',' << regionBefore[regionIndex] - after << '\n';
                    }
                }
                if(rank == 0)
                {
                    transportLedger.flush();
                }
            }
            if(o.adaptiveRadiationDt)
            {
                radiationDt = mc->suggestTimeStep();
            }
        }
        else
        {
            // The debugging control freezes radiation in space. Erad is
            // specific energy, so its denominator must follow the gas mass.
            std::vector<double> census(tess.GetPointNo(), 0.0);
            for(const auto &p : mc->getParticles())
            {
                census.at(p.cellIndex) += p.weight;
            }
            for(size_t i = 0; i < tess.GetPointNo(); ++i)
            {
                cells[i].Erad = census[i] / ext[i].mass;
            }
        }
#ifdef RICH_MPI
        // Radiation changes primitives on each owner. Refresh remote material
        // states before the next hydro reconstruction.
        MPI_exchange_data(tess, cells, true);
#endif
        radiationEnergyChange += totalEnergy() - before;
        diagnostics(dt);
        if(sim.GetCycle() % o.dump == 0)
        {
            snapshot(o, "step" + std::to_string(sim.GetCycle()), rank, tess, cells, sim.GetTime(),
                     ref);
        }
        if(rank == 0 && (sim.GetCycle() <= 3 || sim.GetCycle() % o.dump == 0))
        {
            std::cout << "RMTV cycle=" << sim.GetCycle() << " age=" << sim.GetTime() << " dt=" << dt
                      << std::endl;
        }
    }
    snapshot(o, "final", rank, tess, cells, sim.GetTime(), ref);
    if(rank == 0)
    {
        std::cout << "RMTV "
                  << (sim.GetTime() >= tend ? "reached final time"
                                            : "stopped at max-steps before final time")
                  << std::endl;
        std::ofstream status(o.output + "/status.json");
        status << std::setprecision(17)
               << "{\"reached_end\":" << (sim.GetTime() >= tend ? "true" : "false")
               << ",\"age\":" << sim.GetTime() << ",\"cycles\":" << sim.GetCycle() << "}\n";
    }
}
} // namespace
int main(int argc, char **argv)
{
    int rank = 0, size = 1;
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    int code = 0;
    try
    {
        const Options options = parse(argc, argv, rank);
        if(!options.help)
        {
            run(options, rank, size);
        }
    }
    catch(const UniversalError &e)
    {
        reportError(e);
        code = 1;
    }
    // MadVoroException does not derive from std::exception, so without this the
    // tessellation's own diagnostics are lost to terminate().
    catch(const MadVoro::Exception::MadVoroException &e)
    {
        std::cerr << "RMTV rank " << rank << " MadVoro: " << e.getErrorMessage() << '\n';
        MadVoro::Exception::reportError(e, std::cerr);
        code = 1;
    }
    catch(const std::exception &e)
    {
        std::cerr << "RMTV rank " << rank << ": " << e.what() << '\n';
        code = 1;
    }
#ifdef RICH_MPI
    if(code)
    {
        MPI_Abort(MPI_COMM_WORLD, code);
    }
    MPI_Finalize();
#endif
    return code;
}
