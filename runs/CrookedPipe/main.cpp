#include <mpi.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
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
#include "3D/monte/Voronoi3DMovement.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "CrookedPipeBoundary.hpp"
#include "CrookedPipeOpacity.hpp"

namespace fs = std::filesystem;

std::shared_ptr<RadiationMCStep> mcStep = nullptr;
bool do_output;

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

#ifdef RICH_MPI
class CrookedPipeCostCalculator : public CostCalculator3D
{
public:
    CrookedPipeCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics)
        : manager(manager), physics(physics)
    {}

    std::vector<double> CalculateCost(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const override
    {
        // const std::vector<double> &factorFlecks = physics->getFactorFleck();
        // const std::vector<double> &planckOpacities = physics->getPlanckOpacities();
        size_t N = tess.GetPointNo();
        // weights.resize(N, 1.0);
        // double weight = static_cast<double>(manager.GetStepCounter()) / N;
        // for(size_t j = 0; j < N; j++)
        // {
        //     // weights[j] = std::max(1., planckOpacities[j] * tess.GetWidth(j) * (1 - factorFlecks[j]));
        //     weights[j] = weight;
        // }
        std::vector<double> weights(N, 0.01);
        const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
        assert(counters.size() == N);

        for(size_t j = 0; j < N; j++)
        {
            weights[j] = static_cast<double>(counters[j]);
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
        arguments.addOption<size_t>("iterations", std::numeric_limits<size_t>::max(), "maximum number of cycles");
        arguments.addOption<bool>("random-walk", false, "enable random walk acceleration")
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
        do_output = arguments.wasSet("output");
        bool withRandomWalk = arguments.get<bool>("random-walk");
        size_t iterations = arguments.get<size_t>("iterations");
        std::string managerName = arguments.get<std::string>("manager");

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

        std::string prefix = outputDir.empty()
            ? "/data/shared/maorm/OSC_CrookedPIPE_" + std::to_string(size) + "_"
            : outputDir + "/";
        if(rank == 0 and do_output)
        {
            fs::create_directories(prefix);
        }
        if(rank == 0)
        {
            std::cout << "Crooked pipe benchmark:"
                      << " points=" << N
                      << ", particles/cell=" << particlesPerCell
                      << ", random_walk=" << withRandomWalk
                      << ", manager=" << managerName
                      << ", iterations=" << iterations
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
            .withRandomWalk = withRandomWalk,
            .rwMinCellOpticalDepth = 25
        };
        std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<::RadiationIMC>(
            tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, params);

        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
            std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, particlesPerCell, 4);

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

        Output(prefix + "start");

        double dt = 1e-11;
        double time = 0;
        const double totalTime = 1e-6;
        double max_step = 1e-9;

        sim.SetTimeStep(dt);

        auto start_total = std::chrono::high_resolution_clock::now();

        while(time < totalTime and sim.GetCycle() < iterations)
        {
            auto stepStart = std::chrono::high_resolution_clock::now();
            sim.step();
            auto stepEnd = std::chrono::high_resolution_clock::now();

            time += dt;
            dt = std::min(1.1 * dt, std::min(max_step, totalTime - time));
            sim.SetTimeStep(dt);

            if(rank == 0)
            {
                double timeTaken = std::chrono::duration<double>(stepEnd - stepStart).count();
                std::cout << "Ended Cycle " << sim.GetCycle() << ", dt: " << dt << ", simulation time: " << time << " (" << time / totalTime * 100 << "%), time taken: " << timeTaken << " seconds" << std::endl;
            }
            if(sim.GetCycle() % 5 == 0)
            {
                Output(prefix + std::to_string(sim.GetCycle()));
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        double totalTimeTaken = std::chrono::duration<double>(end_total - start_total).count();
        if(rank == 0)
        {
            std::cout << "Total time taken: " << totalTimeTaken << " seconds" << std::endl;
        }

        Output(prefix + "final");
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

    MPI_Finalize();
}
