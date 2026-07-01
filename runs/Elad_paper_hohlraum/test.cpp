#include <mpi.h>
#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "mpi/mpi_commands.hpp"
#include "misc/mesh_generator3D.hpp"
#include "3D/GeometryCommon/RoundGrid3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/common/MixedEos.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "3D/output/write3D.hpp"
#include "3D/output/read3D.hpp"
#include "3D/output/MC/read_write_particles.hpp"
#include <MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp>
#include <MeshDecomposer3D/load_balancing/CurveLoadBalancer.hpp>
#include  "utils/debug/cleanNode.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"

#include "3D/monte/Voronoi3DMovement.hpp"
#include "HohlraumOpacity.hpp"
#include "HohlraumBoundary.hpp"
#include "utils/debug/vtune.h"

namespace fs = std::filesystem;

/*
 * 2D Cylindrical Hohlraum benchmark from:
 *   McClarren & Urbatsch (2009), as presented in
 *   Steinberg & Heizler (2021), arXiv:2108.13453, Section 4.2.
 *
 * Original problem is in RZ geometry.  Here we generate a full 3D
 * version by revolving the 2D cross-section around the symmetry axis.
 * In 3D the x-axis is the symmetry axis and r = sqrt(y^2 + z^2).
 *
 * Domain (RZ):  z in [0, 1.4],  r in [0, 0.65]  (cm)
 * 3D box:       x in [0, 1.4],  y in [-0.65, 0.65],  z in [-0.65, 0.65]
 *
 * Material regions (absorbing, blue in paper Fig. 3):
 *   Left wall:      [0.10, 0.15] x [0, 0.45]
 *   Capsule:        [0.55, 0.95] x [0, 0.45]
 *   Right end cap:  [1.35, 1.40] x [0, 0.65]
 *   Outer wall:     [0.10, 1.40] x [0.60, 0.65]
 *
 * Material:   sigma_a = 300 (T/keV)^{-3} cm^{-1},  Cv = 3e15 erg/keV/cm^3
 * Vacuum:     sigma_a ~ 0,                          Cv ~ negligible
 *
 * BC:         x=Lx blackbody at T = 1 keV;  all other boundaries vacuum
 * Init:       T = 300 K
 * Timestep:   dt = 1e-11 s,  run until t = 10 ns  (1000 steps)
 */

#ifdef RICH_MPI
class HohlraumCostCalculator : public CostCalculator3D
{
public:
    HohlraumCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics)
        : manager(manager), physics(physics)
    {}

    std::vector<double> CalculateCost(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const override
    {
        size_t N = tess.GetPointNo();
        std::vector<double> weights(N, 50);
        const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
        const std::vector<size_t> &particleCounts = manager->GetBeginningParticleCount();
        const bool useCounters = counters.size() == N;
        const bool useParticleCounts = particleCounts.size() == N;
        if(!useCounters || !useParticleCounts)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(rank == 0)
                std::cout << "WARNING: CalculateCost size mismatch: N=" << N
                          << " counters=" << counters.size()
                          << " particleCounts=" << particleCounts.size() << std::endl;
        }

        constexpr double particleWeight = 10;
        constexpr double countersWeight = 1.0;

        for(size_t i = 0; i < N; i++)
        {
            if(useParticleCounts)
            {
                weights[i] += particleWeight * static_cast<double>(particleCounts[i]);
            }
            if(useCounters)
            {
                weights[i] += countersWeight * counters[i];
            }
        }

        return weights;
    }

    void Dump(size_t cycle) const override
    {
        const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
        const std::vector<size_t> &particleCounts = manager->GetBeginningParticleCount();

        size_t totalSteps = 0;
        for(size_t c : counters)
            totalSteps += c;

        size_t totalParticles = 0;
        for(size_t p : particleCounts)
            totalParticles += p;

        constexpr double particleWeight = 10.0;
        double totalWeight = 50.0 * static_cast<double>(counters.size())
            + particleWeight * static_cast<double>(totalParticles)
            + static_cast<double>(totalSteps);

        size_t endParticles = manager->GetEndParticleCount();
        size_t initialParticles = manager->GetInitialParticleCount();
        size_t numCells = counters.size();

        int rank, ws;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);

        std::vector<double> allWeights(ws);
        std::vector<size_t> allSteps(ws);
        std::vector<size_t> allParticles(ws);
        std::vector<size_t> allEndParticles(ws);
        std::vector<size_t> allInitialParticles(ws);
        std::vector<size_t> allCells(ws);

        MPI_Gather(&totalWeight, 1, MPI_DOUBLE, allWeights.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&totalSteps, 1, MPI_UNSIGNED_LONG_LONG, allSteps.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&totalParticles, 1, MPI_UNSIGNED_LONG_LONG, allParticles.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&endParticles, 1, MPI_UNSIGNED_LONG_LONG, allEndParticles.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&initialParticles, 1, MPI_UNSIGNED_LONG_LONG, allInitialParticles.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&numCells, 1, MPI_UNSIGNED_LONG_LONG, allCells.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        #ifdef TIMING
        double computeTime = manager->GetPureComputeTime();
        std::vector<double> allComputeTime(ws);
        MPI_Gather(&computeTime, 1, MPI_DOUBLE, allComputeTime.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        #endif

        if(rank == 0)
        {
            std::string filename = "after_rebalance_info_" + std::to_string(cycle) + ".csv";
            std::ofstream out(filename);
            #ifdef TIMING
            out << "rank,cells,weight,steps,particles,end_particles,initial_particles,compute_time\n";
            for(int r = 0; r < ws; r++)
                out << r << "," << allCells[r] << "," << allWeights[r] << "," << allSteps[r] << "," << allParticles[r]
                    << "," << allEndParticles[r] << "," << allInitialParticles[r]
                    << "," << allComputeTime[r] << "\n";
            #else
            out << "rank,cells,weight,steps,particles,end_particles,initial_particles\n";
            for(int r = 0; r < ws; r++)
                out << r << "," << allCells[r] << "," << allWeights[r] << "," << allSteps[r] << "," << allParticles[r]
                    << "," << allEndParticles[r] << "," << allInitialParticles[r] << "\n";
            #endif
            out.close();
            std::cout << "Wrote " << filename << std::endl;
        }
    }

private:
    const std::shared_ptr<MonteCarloManager3D> manager;
    const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> physics;
};
#endif // RICH_MPI

static vector<Vector3D> RandCylindar(std::size_t PointNum, double Rin, double Rout, double Xmin, double Xmax)
{
    typedef boost::mt19937_64 base_generator_type;
    double ran[3];
    std::vector<Vector3D> res;
    Vector3D point;
    base_generator_type generator;
    boost::random::uniform_real_distribution<> dist;
    for(size_t i = 0; i < PointNum; ++i)
    {
        ran[0] = dist(generator);
        ran[1] = dist(generator);
        ran[2] = dist(generator);
        double r = std::sqrt(ran[0] * (Rout * Rout - Rin * Rin) + Rin * Rin);
        point.z = r * std::sin(2 * M_PI * ran[1]);
        point.y = r * std::cos(2 * M_PI * ran[1]);
        point.x = ran[2] * (Xmax - Xmin) + Xmin;
        res.push_back(point);
    }
    return res;
}

bool isMaterial(double x, double r)
{
    // Left wall
    if(x >= 0.10 and x <= 0.15 and r <= 0.45)
        return true;
    // Capsule
    if(x >= 0.55 and x <= 0.95 and r <= 0.45)
        return true;
    // Right end cap
    if(x >= 1.35 and x <= 1.40 and r <= 0.65)
        return true;
    // Outer cylindrical wall
    if(x >= 0.10 and x <= 1.40 and r >= 0.60 and r <= 0.65)
        return true;
    return false;
}

std::vector<ComputationalCell3D> Initialize(Voronoi3D &tess, const EquationOfState &eos, size_t N_base, double T_init, bool mode2d)
{
    // --- Fresh start: generate ring mesh ---
    auto [ll, ur] = tess.GetBoxCoordinates();

    int rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    double domainVol = (ur.x - ll.x) * (ur.y - ll.y) * (ur.z - ll.z);

    std::vector<Vector3D> points;
    if(rank == 0)
    {
        points = RandRectangular(N_base, ll, ur);
        
        if(!mode2d)
        {
            double dx = 0.02;
            size_t gridFactor = 4;
            size_t Np = N_base * gridFactor;

            auto pts = RandCylindar(Np, 0.45, 0.45 + dx, 0.10, 0.15);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(Np, 0.45, 0.45 + dx, 0.55, 0.95);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(2 * Np, 0, 0.45, 0.55, 0.95);
            points.insert(points.end(), pts.begin(), pts.end());

            pts = RandCylindar(Np, 0.60, 0.60 + dx, 0.10, 1.40);
            points.insert(points.end(), pts.begin(), pts.end());

            pts = RandCylindar(Np, 0.65, 0.65 + dx, 0.10, 1.40);
            points.insert(points.end(), pts.begin(), pts.end());

            pts = RandCylindar(Np, 0, 0.65, 0.10-dx/2, 0.10 + dx);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(Np, 0, 0.45, 0.15-dx, 0.15 + dx/2);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(Np, 0, 0.45, 0.55-dx, 0.55 + dx/2);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(Np, 0, 0.45, 0.95-dx/2, 0.95 + dx);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(Np, 0, 0.65, 1.35-dx/2, 1.35 + dx);
            points.insert(points.end(), pts.begin(), pts.end());
            pts = RandCylindar(Np, 0, 0.65, 1.40-dx/2, 1.40 + dx);
            points.insert(points.end(), pts.begin(), pts.end());
        }

        std::cout << "Generated " << points.size() << " points"
            << " (N_base=" << N_base << ")"
            << std::endl;
        if(static_cast<rank_t>(points.size()) < ws)
        {
            std::cerr << "ERROR: only " << points.size() << " cells for " << ws
                << " MPI ranks. Reduce N_base." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    try
    {
        points = RoundGrid3D(points, ll, ur, 5);
        tess.BuildParallel(points);
    }
    catch(const MadVoro::Exception::MadVoroException &eo)
    {
        MadVoro::Exception::reportError(eo);
        throw;
    }
    points = tess.getMeshPoints();
    points.resize(tess.GetPointNo());

    // --- Initial conditions ---
    ComputationalCell3D templateCell;
    templateCell.density = 1.0;
    templateCell.temperature = T_init;
    templateCell.velocity = Vector3D(0, 0, 0);

    size_t N = tess.GetPointNo();
    std::vector<ComputationalCell3D> initialCells(N, templateCell);

    for(size_t i = 0; i < N; i++)
    {
        Vector3D p = tess.GetCellCM(i);
        double r = mode2d ? std::abs(p.y) : std::sqrt(p.y * p.y + p.z * p.z);
        if(isMaterial(p.x, r))
            initialCells[i].tracers[0] = 1.0;
        else
            initialCells[i].tracers[1] = 1.0;

        initialCells[i].internal_energy = eos.dT2e(
            initialCells[i].density, initialCells[i].temperature,
            initialCells[i].tracers, ComputationalCell3D::tracerNames);
        initialCells[i].Erad = units::arad * std::pow(T_init, 4) / initialCells[i].density;
    }
    return initialCells;
}

struct CellData : public Serializable
{
    double x;
    double r;
    double temperature;

    CellData() : x(0), r(0), temperature(0) {}
    CellData(double x_, double r_, double T_) : x(x_), r(r_), temperature(T_) {}

    size_t dump(Serializer *serializer) const override
    {
        size_t off = 0;
        off += serializer->insert(this->x);
        off += serializer->insert(this->r);
        off += serializer->insert(this->temperature);
        return off;
    }

    size_t load(const Serializer *serializer, std::size_t offset) override
    {
        size_t rd = 0;
        rd += serializer->extract(this->x, offset);
        rd += serializer->extract(this->r, offset + rd);
        rd += serializer->extract(this->temperature, offset + rd);
        return rd;
    }

    bool operator<(const CellData &o) const { return x < o.x; }
};

void WriteProfile(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells, const std::string &filename, double time_ns, size_t N_base, bool mode2d)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double r_line = 0.05;
    const double r_tol  = 0.03;
    size_t nPts = tess.GetPointNo();
    std::vector<CellData> cellData;
    for(size_t i = 0; i < nPts; i++)
    {
        Vector3D p = tess.GetMeshPoint(i);
        double ri = mode2d ? std::abs(p.y) : std::sqrt(p.y * p.y + p.z * p.z);
        if(std::abs(ri - r_line) < r_tol)
        {
            cellData.emplace_back(p.x, ri, cells[i].temperature);
        }
    }

    cellData = MPI_Gatherv_serializable(cellData, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::sort(cellData.begin(), cellData.end());
        std::ofstream out(filename);
        out << "# t_ns=" << time_ns << " r_line=" << r_line << "\n";
        out << "# x(cm), T(K), T(keV)\n";
        for(const auto &cd : cellData)
            out << cd.x << ", " << cd.temperature << ", "
                << cd.temperature / units::kev_kelvin << "\n";
        out.close();
        std::cout << "Wrote " << filename << " (" << cellData.size() << " cells)" << std::endl;
    }
}

void WriteVTK(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells, const std::shared_ptr<MonteCarloRadiationPhysics3D> &physics, const std::string &filename)
{
    size_t nPts = tess.GetPointNo();
    std::vector<double> temps(nPts), tKeV(nPts), tr0(nPts), tr1(nPts), erads(nPts), volumes(nPts);
    for(size_t i = 0; i < nPts; i++)
    {
        temps[i] = cells[i].temperature;
        tKeV[i]  = cells[i].temperature / units::kev_kelvin;
        tr0[i]   = cells[i].tracers[0];
        tr1[i]   = cells[i].tracers[1];
        erads[i] = cells[i].Erad;
        volumes[i] = tess.GetVolume(i);
    }
    WriteVoronoiVTKOnly(tess, filename,
        {temps, tKeV, tr0, tr1, erads, physics->getEradTimeAvg(), volumes},
        {"Temperature_K", "Temperature_keV", "Material", "Vacuum", "Erad", "Erad_time_avg", "Volume"});
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
    {
        std::cout << "Wrote " << filename << std::endl;
    }
}

int main(int argc, char *argv[]) {
    rank_t rank = -1, ws = 0;
    try {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);

        vtune_stop();
        DISABLE_TIMERS();

        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

        // char hostname[MPI_MAX_PROCESSOR_NAME];
        // int name_len;
        // MPI_Get_processor_name(hostname, &name_len);
        // std::cerr << "Rank " << rank << "/" << ws
        //         << " on " << hostname
        //         << " pid=" << getpid() << std::endl;

        ensureCleanNode();

        if(rank == 0 and argc < 2)
        {
            std::cerr << "Usage: " << argv[0]
                << " <N_base> [new_per_cell] [max_per_cell] [--2d] [--p2p] "
                "[--ibv] [--new_ibv] [--new-rdma] [--mpi-rma]"
                << " [--hold-small-idle-flushes]"
                << " [--resume <dump_number>]" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t N_base = std::stoul(argv[1]);
        size_t newPhotonsPerCell = 5;
        size_t minPhotonsPerCell = 15;
        bool mode2d = false;
        int resumeDump = -1;

        bool resumeAuto = false;
        bool useP2P = false;
        bool useIBV = false;
        bool useMpiRma = false;
        bool useNewRdma = false;
        bool useNewIbv = false;
        bool holdSmallIdleFlushes = true;
        std::vector<std::string> positionalArgs;
        for(int a = 2; a < argc; a++)
        {
            std::string arg(argv[a]);
            if(arg == "--2d")
                mode2d = true;
            else if(arg == "--p2p")
                useP2P = true;
            else if(arg == "--ibv")
                useIBV = true;
            else if(arg == "--mpi-rma")
                useMpiRma = true;
            else if(arg == "--new-rdma")
                useNewRdma = true;
            else if(arg == "--new_ibv" or arg == "--new-ibv")
                useNewIbv = true;
            else if(arg == "--hold-small-idle-flushes")
                holdSmallIdleFlushes = true;
            else if(arg == "--idle-flush-holdoff")
            {
                if(rank == 0)
                    std::cerr << "--idle-flush-holdoff was removed; the holdoff value is "
                        "now internal/adaptive."
                        << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else if(arg == "--resume")
            {
                if(a + 1 < argc and argv[a + 1][0] != '-')
                {
                    resumeDump = std::stoi(argv[++a]);
                } else
                    resumeAuto = true;
            } else
                positionalArgs.push_back(arg);
        }
        if(positionalArgs.size() >= 1)
        {
            newPhotonsPerCell = std::stoul(positionalArgs[0]);
        }
        if(positionalArgs.size() >= 2)
        {
            minPhotonsPerCell = std::stoul(positionalArgs[1]);
        }

        const std::string hohlraumDir = useP2P ? "Hohlraum_P2P" : "Hohlraum";
        const std::string outputDir =
            "/data/shared/maorm/Hohlraum/" + hohlraumDir + "/N_base_" +
            std::to_string(N_base) + "/size_" + std::to_string(ws);
        char prefixBuf[256];
        std::snprintf(prefixBuf, sizeof(prefixBuf), "Hohlraum_%s_%d_", argv[1], ws);
        const std::string prefix = outputDir + "/" + prefixBuf;

        if(rank == 0)
        {
            fs::create_directories(outputDir);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // --- Physical parameters (Section 4.2) ---
        const double T_boundary_keV = 1.0;
        const double T_boundary = T_boundary_keV * units::kev_kelvin; // Kelvin
        const double T_init = 300.0;                                  // Kelvin
        const double init_dt = 1e-11; // 1e-11 / (std::pow(ws / (static_cast<double>(5 * 112)), 0.333333333)); // seconds
        const double t_final = 3e-9; // dt * 150;                    // 1e-9;
        const double max_dt = 5e-11; 
        constexpr size_t boundaryPhotonsPerCell = 1000;
        constexpr size_t dumpInterval = 25;

        // Domain: 3D revolves around x-axis; 2D is a thin slab at z=1
        // Pad the domain beyond the material boundaries so that no material
        // interface coincides with the domain boundary (avoids particles
        // landing exactly on a boundary face with zero distance to all faces).
        const double Lx = 1.4, Ly = 0.65;
        const double Lz = mode2d ? 1.0 : 0.65;
        const double pad = 2 * 0.03;
        Vector3D ll(0, -(Ly + pad), mode2d ? 0.0 : -(Lz + pad));
        Vector3D ur(Lx + pad, Ly + pad, mode2d ? 2.0 : (Lz + pad));

        // --- Equations of State ---
        IdealGas eosMaterial(1.5, 3e15 / units::kev_kelvin, 1, 0);
        IdealGas eosVacuum(1.5, 1e15 / units::kev_kelvin, 1, 0);
        std::vector<EquationOfState *> eosList = {&eosMaterial, &eosVacuum};
        MixedEOS eos(eosList);

        if(resumeAuto)
        {
            int found = 0;
            if(rank == 0)
            {
                std::string simFile = prefix + "latest_sim.h5";
                if(fs::exists(simFile))
                {
                    found = 1;
                    std::cout << "Auto-detected simulation file: " << simFile
                        << std::endl;
                } else
                    std::cout << "No simulation file found, starting fresh" << std::endl;
            }
            MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(found)
                resumeDump = 0;
        }

        Voronoi3D tess(ll, ur);
        std::vector<ComputationalCell3D> initialCells;
        size_t startCycle = 0;

        double simTime = 0;
        size_t dumpCount = 0;

        ComputationalCell3D::tracerNames = {"Material", "Vacuum"};

        if(resumeDump < 0)
        {
            initialCells = Initialize(tess, eos, N_base, T_init, mode2d);
        }

        // --- Simulation ---
        Simulation sim(tess, initialCells, eos);
        std::shared_ptr<TimeStepFunction3D> tsc =
            std::make_shared<ManualTimeStep>();
        sim.SetTimeStepFunction(tsc);

        std::vector<ComputationalCell3D> &cells = sim.getCells();
        std::vector<Conserved3D> &extensives = sim.getExtensives();

        // --- MC radiation transport ---
        constexpr bool withHydro = false;

        auto eosPtr = std::make_shared<MixedEOS>(eos);
        auto opacityPtr = std::make_shared<HohlraumOpacity>();

        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
            std::make_shared<HohlraumBoundary<Vector3D, Tessellation3D>>(
                tess, cells, T_boundary, boundaryPhotonsPerCell);

        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> params = {.newPhotonsPerCell = newPhotonsPerCell,
                .withHydro = withHydro,
                .withRandomWalk = true};
        std::shared_ptr<MonteCarloRadiationPhysics3D> physics =
            std::make_shared<::RadiationIMC>(tess, boundaryCond, cells, extensives,
                eosPtr, opacityPtr, params);

        size_t comb_factor = 6;
        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
            std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(
                tess, minPhotonsPerCell, comb_factor);

        MonteCarloConfig monteCarloConfig;
        monteCarloConfig.holdSmallIdleFlushes = holdSmallIdleFlushes;
        if(rank == 0)
        {
            std::cout << "Hohlraum MonteCarloConfig: holdSmallIdleFlushes="
                << monteCarloConfig.holdSmallIdleFlushes
                << ", smallIdleFlushHoldoffCycles="
                << monteCarloConfig.GetSmallIdleFlushHoldoffCycles()
                << std::endl;
        }

        std::vector<Particle3D> initialParticles;
        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, popControl, boundaryCond,
            initialParticles, 0, withHydro
#ifdef RICH_MPI
            ,
            useMpiRma
            ? RadiationMCStep::ManagerType::MPI_RMA
            : (useP2P ? RadiationMCStep::ManagerType::P2P
                : (useNewIbv ? RadiationMCStep::ManagerType::NEW_IBV_RDMA
                    : (useIBV ? RadiationMCStep::ManagerType::IBV_RDMA
                        : (useNewRdma ? RadiationMCStep::ManagerType::NEW_RDMA
                            : RadiationMCStep::ManagerType::AUTO_RDMA)))),
            nullptr, monteCarloConfig
#endif
        );
        sim.addPhysics(mcStep);
#ifdef RICH_MPI
        mcStep->setCost(std::make_shared<HohlraumCostCalculator>(
                mcStep->getManager(), physics));
        sim.addMigrationBuffer(mcStep->getManager()->GetCellsStepsCounters());
        sim.addMigrationBuffer(mcStep->getManager()->GetBeginningParticleCount());
#endif // RICH_MPI

        double dt = init_dt;
        sim.SetTimeStep(dt);

        if(resumeDump >= 0)
        {
            ReadSimulation(prefix + "latest_sim.h5", sim);
            startCycle = sim.GetCycle();
            simTime = sim.GetTime();
            dumpCount = startCycle / dumpInterval;

            if(rank == 0)
                std::cout << "Resumed from " << prefix
                    << "latest_sim.h5: cycle=" << startCycle
                    << ", t=" << simTime * 1e9 << " ns"
                    << ", dumpCount=" << dumpCount << std::endl;
        }

        extensives.resize(cells.size());
        for(size_t i = 0; i < cells.size(); i++)
        {
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
        }

        size_t N = tess.GetPointNo();
        if(rank == 0)
        {
            size_t nMaterial = 0, nVacuum = 0;
            for(size_t i = 0; i < N; i++)
            {
                if(cells[i].tracers[0] > 0.5)
                    nMaterial++;
                else
                    nVacuum++;
            }
            std::cout << "Hohlraum (" << (mode2d ? "2D" : "3D")
                << "): N_base=" << N_base << " cm, cells=" << N
                << ", material=" << nMaterial << ", vacuum=" << nVacuum
                << ", dt=" << dt << " s"
                << ", t_final=" << t_final << " s"
                << ", new/cell=" << newPhotonsPerCell
                << ", max/cell=" << minPhotonsPerCell << ", output=" << prefix
                << (resumeDump >= 0
                    ? ", resumed from dump " + std::to_string(resumeDump)
                    : "")
                << std::endl;
        }

        WriteSimulation(sim, prefix + "latest_sim.h5");

        if(resumeDump < 0)
        {
            WriteVTK(tess, cells, physics, prefix + "init.vtu");
        }

        // --- Main time-stepping loop ---
        auto startWall = std::chrono::high_resolution_clock::now();
        double computeTotal = 0;
        size_t stepsSinceLastDump = 0;

        while(simTime < t_final)
        {
            auto stepStart = std::chrono::high_resolution_clock::now();
            sim.step();
            auto stepEnd = std::chrono::high_resolution_clock::now();
            double computeSec = std::chrono::duration<double>(stepEnd - stepStart).count();
            computeTotal += computeSec;

            stepsSinceLastDump++;
            simTime = sim.GetTime();

            if(simTime >= 1e-9)
            {
                dt = std::min(1.05 * dt, max_dt);
            }
            sim.SetTimeStep(dt);

            double fraction = simTime / t_final;
            double eta =
                (fraction > 0) ? computeTotal * (1.0 - fraction) / fraction : 0;

            if(rank == 0)
            {
                int pct = static_cast<int>(fraction * 100);
                int etaMin = static_cast<int>(eta) / 60;
                int etaSec = static_cast<int>(eta) % 60;
                int totalMin = static_cast<int>(computeTotal) / 60;
                int totalSec = static_cast<int>(computeTotal) % 60;
                std::cout << "Cycle " << sim.GetCycle() << "  t=" << simTime * 1e9
                    << " ns"
                    << " (" << pct << "%)"
                    << "  step=" << computeSec << "s"
                    << "  compute=" << totalMin << "m" << totalSec << "s"
                    << "  ETA=" << etaMin << "m" << etaSec << "s" << std::endl;
            }

            if(stepsSinceLastDump >= dumpInterval)
            {
                stepsSinceLastDump = 0;
                char buf[512];

                std::snprintf(buf, sizeof(buf), "%s%05zu.pvtu", prefix.c_str(),
                    dumpCount);
                WriteVTK(tess, cells, physics, buf);

                WriteSimulation(sim, prefix + "latest_sim.h5");
                if(rank == 0)
                {
                    std::cout << "Wrote simulation: " << prefix << "latest_sim.h5"
                        << std::endl;
                }

                double t_ns = simTime * 1e9;
                std::snprintf(buf, sizeof(buf), "%s%05zu.txt", prefix.c_str(),
                    dumpCount);
                WriteProfile(tess, cells, buf, t_ns, N_base, mode2d);

                dumpCount++;
            }
        }

        auto endWall = std::chrono::high_resolution_clock::now();
        double wallSec = std::chrono::duration<double>(endWall - startWall).count();
        if(rank == 0)
            std::cout << "Total wall time: " << wallSec << "s"
                << "  (compute: " << computeTotal << "s"
                << ", I/O: " << wallSec - computeTotal << "s)" << std::endl;

        // --- Final output ---
        WriteProfile(tess, cells, prefix + "final.txt", simTime * 1e9, N_base,mode2d);
        WriteVTK(tess, cells, physics, prefix + "final.vtu");
        WriteSimulation(sim, prefix + "latest_sim.h5");
    } catch (const UniversalError &e) {
        std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
        reportError(e, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::exception &e) {
        std::cerr << "=== std::exception on rank " << rank << ": " << e.what()
            << " ===" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
}
