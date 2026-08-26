#include "utils/debug/SmartTimer.hpp"
#include <mpi.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "mpi/mpi_commands.hpp"
#include "misc/mesh_generator3D.hpp"
#include "3D/GeometryCommon/RoundGrid3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "CMMC/src/units/units.hpp"
#include "newtonian/common/MixedEos.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "3D/output/write3D.hpp"
#include "3D/output/MC/read_write_particles.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "3D/monte/STORMVoronoi3DMovement.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "CrookedPipeBoundary.hpp"
#include "CrookedPipeOpacity.hpp"
#ifdef STORM_WITH_GPU
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

std::shared_ptr<RadiationMCStep> mcStep = nullptr;
bool do_output = false;

void Output(const std::string &fname)
{
    if(not do_output)
    {
        return;
    }
    if(mcStep == nullptr)
    {
        return;
    }

    #ifdef RICH_MPI
    rank_t rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
    {
        std::cout << "Writing output to " << fname << std::endl;
    }
    #endif // RICH_MPI

    const std::vector<ComputationalCell3D> &cells = mcStep->getCells();
    const Voronoi3D &tess = reinterpret_cast<const Voronoi3D&>(mcStep->getTessellation());

    size_t N = tess.GetPointNo();
    assert(cells.size() >= N);

    std::vector<double> temperatures(N);
    std::vector<double> tracers0(N), tracers1(N);
    for(size_t i = 0; i < N; i++)
    {
        temperatures[i] = cells[i].temperature;
        tracers0[i] = cells[i].tracers[0];
        tracers1[i] = cells[i].tracers[1];
    }

    // TODO: serial write
    WriteSnapshot3DParallel_AOS(tess, cells, fname + "_aos.h5");
    WriteParticlesParallel(fname + "_particles.h5", mcStep->getParticles());
    WriteVoronoiVTKOnly(tess, fname + ".vtu", {temperatures, tracers0, tracers1}, {"Temperature", "Tracer0", "Tracer1"});
}

vector<Vector3D> RandCylindar(std::size_t PointNum, double Rin, double Rout, double Zmin, double Zmax)
{
	typedef boost::mt19937_64 base_generator_type;
	double ran[3];

	std::vector<Vector3D> res;
	Vector3D point;
	base_generator_type generator;
	boost::random::uniform_real_distribution<> dist;

	for (size_t i = 0; i < PointNum; ++i)
	{
		ran[0] = dist(generator);
		ran[1] = dist(generator);
		ran[2] = dist(generator);
        double r = ran[0] * (Rout - Rin) + Rin;
		point.z = r * std::sin(2 * M_PI * ran[1]);
		point.y = r * std::cos(2 * M_PI * ran[1]);
		point.x = ran[2] * (Zmax - Zmin) + Zmin;
		res.push_back(point);
	}
	return res;
}

std::vector<Vector3D> GeneratePoints(size_t N, const Vector3D &ll, const Vector3D &ur)
{
    std::vector<Vector3D> points;
    double dx = 0.01;
    double gridFactor = 4;
    size_t Np = N /*1e5*/ * gridFactor;
    points = RandRectangular(N, ll, ur);
    auto points2 = RandCylindar(Np, 0.5, 0.5+dx, 0, 2.5);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 0.5, 0.5+dx, 4.5, 7);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 1.5, 1.5+dx, 2.5, 4.5);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 1 - dx, 1, 2.5, 4.5);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 0.5, 1.5, 2.5-dx, 2.5);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 0.5, 1.5, 4.5, 4.5+dx);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 0, 1, 3, 3+dx);
    points.insert(points.end(), points2.begin(), points2.end());
    points2 = RandCylindar(Np, 0, 1, 4-dx, 4);
    points.insert(points.end(), points2.begin(), points2.end());
    return points;
}

struct ProbeHit : public Serializable
{
    int probeIndex;
    double dist2;
    double temperature;

    ProbeHit() : probeIndex(-1), dist2(1e300), temperature(0) {}
    ProbeHit(int probeIndex_, double dist2_, double temperature_)
        : probeIndex(probeIndex_), dist2(dist2_), temperature(temperature_) {}

    size_t dump(Serializer *serializer) const override
    {
        size_t off = 0;
        off += serializer->insert(this->probeIndex);
        off += serializer->insert(this->dist2);
        off += serializer->insert(this->temperature);
        return off;
    }

    size_t load(const Serializer *serializer, std::size_t offset) override
    {
        size_t rd = 0;
        rd += serializer->extract(this->probeIndex, offset);
        rd += serializer->extract(this->dist2, offset + rd);
        rd += serializer->extract(this->temperature, offset + rd);
        return rd;
    }
};

struct AxisSample : public Serializable
{
    double x;
    double r;
    double temperature;

    AxisSample() : x(0), r(0), temperature(0) {}
    AxisSample(double x_, double r_, double temperature_)
        : x(x_), r(r_), temperature(temperature_) {}

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

    bool operator<(const AxisSample &o) const { return x < o.x; }
};

constexpr size_t nCrookedPipeProbes = 5;
const double crookedPipeProbeX[nCrookedPipeProbes] = {0.25, 2.75, 3.5, 4.25, 6.75};
const double crookedPipeProbeR[nCrookedPipeProbes] = {0.0, 0.0, 1.25, 0.0, 0.0};

void WriteProbeHeader(const std::string &path)
{
    std::ofstream out(path, std::ios::trunc);
    out << "# Crooked-pipe probe temperatures, Steinberg & Heizler 2022 Fig. 8\n";
    out << "# Probes (r, z_axis=x): P1 (0, 0.25), P2 (0, 2.75), P3 (1.25, 3.5), P4 (0, 4.25), P5 (0, 6.75) cm\n";
    out << "# t_ns, cycle, T1_keV, T2_keV, T3_keV, T4_keV, T5_keV\n";
}

void AppendProbes(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                  double time_s, int cycle, const std::vector<std::string> &paths)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<ProbeHit> localHits(nCrookedPipeProbes);
    for(size_t ip = 0; ip < nCrookedPipeProbes; ip++)
    {
        localHits[ip].probeIndex = static_cast<int>(ip);
    }
    size_t nPts = tess.GetPointNo();
    for(size_t i = 0; i < nPts; i++)
    {
        Vector3D p = tess.GetCellCM(i);
        double r = std::sqrt(p.y * p.y + p.z * p.z);
        for(size_t ip = 0; ip < nCrookedPipeProbes; ip++)
        {
            double dx = p.x - crookedPipeProbeX[ip];
            double dr = r - crookedPipeProbeR[ip];
            double dist2 = dx * dx + dr * dr;
            if(dist2 < localHits[ip].dist2)
            {
                localHits[ip] = ProbeHit(static_cast<int>(ip), dist2, cells[i].temperature);
            }
        }
    }

    std::vector<ProbeHit> allHits = MPI_Gatherv_serializable(localHits, 0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        return;
    }

    double TkeV[nCrookedPipeProbes];
    double bestDist[nCrookedPipeProbes];
    for(size_t ip = 0; ip < nCrookedPipeProbes; ip++)
    {
        TkeV[ip] = 0;
        bestDist[ip] = 1e300;
    }
    for(size_t i = 0; i < allHits.size(); i++)
    {
        int ip = allHits[i].probeIndex;
        if(ip < 0 || ip >= static_cast<int>(nCrookedPipeProbes))
        {
            continue;
        }
        if(allHits[i].dist2 < bestDist[ip])
        {
            bestDist[ip] = allHits[i].dist2;
            TkeV[ip] = allHits[i].temperature / units::kev_kelvin;
        }
    }

    for(size_t f = 0; f < paths.size(); f++)
    {
        std::ofstream out(paths[f], std::ios::app);
        out << time_s * 1e9 << ", " << cycle;
        for(size_t ip = 0; ip < nCrookedPipeProbes; ip++)
        {
            out << ", " << TkeV[ip];
        }
        out << "\n";
    }
}

void WriteAxisProfile(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                      double time_s, const std::string &path)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double rMax = 0.15;
    size_t nPts = tess.GetPointNo();
    std::vector<AxisSample> samples;
    for(size_t i = 0; i < nPts; i++)
    {
        Vector3D p = tess.GetMeshPoint(i);
        double r = std::sqrt(p.y * p.y + p.z * p.z);
        if(r < rMax)
        {
            samples.emplace_back(p.x, r, cells[i].temperature);
        }
    }

    samples = MPI_Gatherv_serializable(samples, 0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        return;
    }
    std::sort(samples.begin(), samples.end());
    std::ofstream out(path);
    out << "# Crooked-pipe axis line-out  t_ns=" << time_s * 1e9 << "  r_max=" << rMax << "\n";
    out << "# x(cm), r(cm), T(K), T(keV)\n";
    for(size_t i = 0; i < samples.size(); i++)
    {
        out << samples[i].x << ", " << samples[i].r << ", " << samples[i].temperature << ", "
            << samples[i].temperature / units::kev_kelvin << "\n";
    }
    std::cout << "Wrote " << path << " (" << samples.size() << " cells)" << std::endl;
}

#ifdef RICH_MPI
class CrookedPipeCostCalculator : public CostCalculator3D
{
public:
    CrookedPipeCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics)
        : manager(manager), physics(physics)
    {}

    std::vector<double> CalculateCost(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const override
    {
        size_t N = tess.GetPointNo();
        std::vector<double> weights(N, 50.0);
        const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
        const std::vector<size_t> &particleCounts = manager->GetBeginningParticleCount();
        bool useCounters = counters.size() == N;
        bool useParticleCounts = particleCounts.size() == N;

        if(!useCounters || !useParticleCounts)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(rank == 0)
            {
                std::cout << "WARNING: Rebalance size mismatch: N=" << N
                          << " counters=" << counters.size()
                          << " particleCounts=" << particleCounts.size() << std::endl;
            }
        }

        constexpr double particleWeight = 10.0;
        constexpr double countersWeight = 1.0;

        for(size_t j = 0; j < N; j++)
        {
            if(useParticleCounts)
            {
                weights[j] += particleWeight * static_cast<double>(particleCounts[j]);
            }
            if(useCounters)
            {
                weights[j] += countersWeight * static_cast<double>(counters[j]);
            }
        }
        return weights;
    }

private:
    const std::shared_ptr<MonteCarloManager3D> manager;
    const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> physics;
};
#endif // RICH_MPI

int main(int argc, char *argv[])
{
    DISABLE_TIMERS();

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    rank_t rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try
    {
        ArgumentParser arguments("Crooked pipe Monte Carlo benchmark");
        arguments.addPositional<size_t>("points", "number of background mesh points").required();
        arguments.addPositional<size_t>("particles_per_cell", "population-control photon cap per cell").required();
        arguments.addOption<std::string>("output", "", "output directory");
        arguments.addOption<size_t>("cycles", 200, "maximum number of cycles")
            .optionAlias("iterations");
        arguments.addFlag("random-walk", "enable random walk acceleration")
            .defaultValue(true)
            .flagAlias("rw", true)
            .flagAlias("no-rw", false);
        arguments.addOption<std::string>("manager", "new-rdma-auto", "Monte Carlo communication manager")
            .choices({"new-rdma-auto", "new-rdma-ibv", "p2p"})
            .flagAlias("new-rdma", "new-rdma-auto")
            .flagAlias("rdma", "new-rdma-auto")
            .flagAlias("new-ibv", "new-rdma-ibv")
            .flagAlias("new_ibv", "new-rdma-ibv")
            .flagAlias("ibv", "new-rdma-ibv")
            .flagAlias("p2p", "p2p");
        arguments.addOption<size_t>("profile-cycle", 0,
            "If >0, resume ROCTx collection only for this 1-based cycle (use with rocprofv3)");

        try
        {
            if(!arguments.parse(argc, argv))
            {
                if(rank == 0)
                    std::cout << arguments.help() << std::endl;
                MPI_Finalize();
                return 0;
            }
        }
        catch(const std::exception &e)
        {
            if(rank == 0)
            {
                std::cerr << e.what() << std::endl;
                std::cerr << arguments.help() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t N = arguments.get<size_t>("points");
        size_t particlesPerCell = arguments.get<size_t>("particles_per_cell");
        std::string outputDir = arguments.get<std::string>("output");
        do_output = !outputDir.empty();
        bool withRandomWalk = arguments.get<bool>("random-walk");
        size_t cycles = arguments.get<size_t>("cycles");
        size_t profileCycle = arguments.get<size_t>("profile-cycle");
        std::string managerName = arguments.get<std::string>("manager");
#ifdef STORM_WITH_GPU
        if(profileCycle > 0)
        {
            roctxProfilerPause(0);
            if(rank == 0)
            {
                std::cout << "ROCTx profiler paused; will resume for cycle "
                          << profileCycle << std::endl;
            }
        }
#else
        if(profileCycle > 0 and rank == 0)
        {
            std::cerr << "--profile-cycle requires a --with-gpu build" << std::endl;
        }
#endif

        #ifdef RICH_MPI
        RadiationMCStep::ManagerType managerType =
            managerName == "p2p" ? RadiationMCStep::ManagerType::P2P :
            managerName == "new-rdma-ibv" ? RadiationMCStep::ManagerType::NEW_IBV_RDMA :
            RadiationMCStep::ManagerType::NEW_RDMA;
        #endif // RICH_MPI

        Vector3D ll(0, -2, -2), ur(7, 2, 2);

        std::vector<Vector3D> points;
        if(rank == 0)
        {
            std::cout << "Generating " << N << " points..." << std::endl;
            points = GeneratePoints(N, ll, ur);
            std::cout << "Generated " << points.size() << " points." << std::endl;
        }
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        points = RoundGrid3D(points, ll, ur, 5);

        Voronoi3D tess(ll, ur);

        tess.BuildParallel(points);
        points = tess.getMeshPoints();
        points.resize(tess.GetPointNo());
        tess.BuildParallel(points);
        points = tess.getMeshPoints();
        points.resize(tess.GetPointNo());

        IdealGas eos1(1.5, 1e16 / units::kev_kelvin, 1, 0);
        IdealGas eos2(1.5, 1e13 / units::kev_kelvin, 1, 0);
        std::vector<EquationOfState*> eosList = {&eos1, &eos2};
        MixedEOS eos(eosList);

        ComputationalCell3D initialCell;
        initialCell.density = 1;
        initialCell.temperature = 0.05 * units::kev_kelvin;
        initialCell.velocity = Vector3D(0, 0, 0);
        std::vector<ComputationalCell3D> initialCells(points.size(), initialCell);

        N = tess.GetPointNo();
        ComputationalCell3D::tracerNames = {"Thick", "Thin"};
        for(size_t i = 0; i < N; i++)
        {
            Vector3D p = tess.GetCellCM(i);
            double const r = std::sqrt(p.z * p.z + p.y * p.y);
            if(r > 1.5)
            {
                initialCells[i].tracers[0] = 1.0;
            }
            else
            {
                if(r < 0.5)
                {
                    if(p.x > 3 && p.x < 4)
                        initialCells[i].tracers[0] = 1.0;
                    else
                        initialCells[i].tracers[1] = 1.0;
                }
                else
                {
                    if(p.x < 2.5 || p.x > 4.5)
                        initialCells[i].tracers[0] = 1.0;
                    else
                    {
                        if(p.x > 3 && p.x < 4 && r < 1)
                            initialCells[i].tracers[0] = 1.0;
                        else
                            initialCells[i].tracers[1] = 1.0;
                    }
                }
            }
            initialCells[i].internal_energy = eos.dT2e(initialCells[i].density, initialCells[i].temperature, initialCells[i].tracers, ComputationalCell3D::tracerNames);
        }

        std::string prefix;
        std::vector<std::string> probeFiles;
        std::vector<std::string> axisFiles;
        if(do_output)
        {
            prefix = outputDir + "/";
            probeFiles = {prefix + "crookedpipe_probes.txt"};
            axisFiles = {prefix + "crookedpipe_profile.txt"};
            if(rank == 0)
            {
                std::filesystem::create_directories(outputDir);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(do_output && rank == 0)
        {
            for(size_t i = 0; i < probeFiles.size(); i++)
            {
                WriteProbeHeader(probeFiles[i]);
            }
            std::cout << "Crooked pipe benchmark:"
                      << " points=" << N
                      << ", particles/cell=" << particlesPerCell
                      << ", random_walk=" << withRandomWalk
                      << ", manager=" << managerName
                      << ", cycles=" << cycles
                      << ", injected photons/cell=" << particlesPerCell / 4
                      << ", output=" << (do_output ? outputDir : "<disabled>")
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Simulation sim(tess, initialCells, eos);
        std::shared_ptr<TimeStepFunction3D> tsc = std::make_shared<ManualTimeStep>();
        sim.SetTimeStepFunction(tsc);

        std::vector<ComputationalCell3D> &cells = sim.getCells();
        std::vector<Conserved3D> &extensives = sim.getExtensives();

        constexpr bool withHydro = false;
        auto eosPtr = std::make_shared<MixedEOS>(eos);
        auto opacityPtr = std::make_shared<CrookedPipeOpacity>();

        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
            std::make_shared<CrookedPipeBoundaryCondition<Vector3D, Tessellation3D>>(tess, cells);

        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> params = {
            .newPhotonsPerCell = particlesPerCell / 4,
            .withHydro = withHydro,
            .withMultigroupOpacity = false,
            .withRandomWalk = withRandomWalk,
            .rwMinCellOpticalDepth = 25,
            .energyBoundaries = {0.0, 1.0e30},
            .energyBoundariesProvided = true,
            .postProcess = {}
        };
        std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<::RadiationIMC>(
            tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, params);

        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(tess, particlesPerCell, 4);

        std::vector<Particle3D> initialParticles;

        mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, popControl, boundaryCond, initialParticles, 0, withHydro
            #ifdef RICH_MPI
                , managerType
            #endif
        );
        sim.addPhysics(mcStep);

        #ifdef RICH_MPI
            mcStep->setCost(std::make_shared<CrookedPipeCostCalculator>(mcStep->getManager(), physics));
        #endif

        if(do_output)
        {
            Output(prefix + "start");
        }

        double dt = 1e-11;
        double time = 0;
        const double totalTime = 1e-6;
        double max_step = 1e-9;

        if(do_output)
        {
            AppendProbes(tess, cells, time, sim.GetCycle(), probeFiles);
            for(size_t i = 0; i < axisFiles.size(); i++)
            {
                WriteAxisProfile(tess, cells, time, axisFiles[i]);
            }
        }

        sim.SetTimeStep(dt);

        auto start_total = std::chrono::high_resolution_clock::now();

        while(time < totalTime and sim.GetCycle() < cycles)
        {
            const size_t upcomingCycle = sim.GetCycle() + 1;
#ifdef STORM_WITH_GPU
            const bool profileThisCycle =
                profileCycle > 0 and upcomingCycle == profileCycle;
            if(profileThisCycle)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if(rank == 0)
                {
                    std::cout << "ROCTx profiler resume for cycle "
                              << profileCycle << std::endl;
                }
                roctxRangePush("crookedpipe_profile_cycle");
                roctxProfilerResume(0);
            }
#endif
            auto stepStart = std::chrono::high_resolution_clock::now();
            sim.step();
            auto stepEnd = std::chrono::high_resolution_clock::now();
#ifdef STORM_WITH_GPU
            if(profileThisCycle)
            {
                if(Kokkos::is_initialized())
                {
                    Kokkos::fence();
                }
                roctxProfilerPause(0);
                roctxRangePop();
                MPI_Barrier(MPI_COMM_WORLD);
                if(rank == 0)
                {
                    std::cout << "ROCTx profiler pause after cycle "
                              << profileCycle << std::endl;
                }
            }
#endif

            time += dt;
            dt = std::min(1.1 * dt, std::min(max_step, totalTime - time));
            sim.SetTimeStep(dt);

            if(rank == 0)
            {
                double timeTaken = std::chrono::duration<double>(stepEnd - stepStart).count();
                std::cout << "Ended Cycle " << sim.GetCycle() << ", dt: " << dt << ", simulation time: " << time << " (" << time / totalTime * 100 << "%), time taken: " << timeTaken << " seconds" << std::endl;
            }
            if(do_output)
            {
                AppendProbes(tess, cells, time, sim.GetCycle(), probeFiles);
                if(sim.GetCycle() % 5 == 0)
                {
                    Output(prefix + std::to_string(sim.GetCycle()));
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        double totalTimeTaken = std::chrono::duration<double>(end_total - start_total).count();
        if(rank == 0)
        {
            std::cout << "Total time taken: " << totalTimeTaken << " seconds" << std::endl;
        }

        if(do_output)
        {
            Output(prefix + "final");
            for(size_t i = 0; i < axisFiles.size(); i++)
            {
                WriteAxisProfile(tess, cells, time, axisFiles[i]);
            }
        }
        mcStep.reset();
    }
    catch(const UniversalError &e)
    {
        std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
        reportError(e);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch(const std::exception &e)
    {
        std::cerr << "=== std::exception on rank " << rank << ": " << e.what() << " ===" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    RMAFactory::Finalize(RDMA_Type::AUTO_RDMA);
#ifdef STORM_WITH_GPU
    STORM::gpu::KokkosRuntime::Finalize();
#endif
    MPI_Finalize();
}
