#include <mpi.h>
#include <random>
#include <filesystem>
#include "mpi/mpi_commands.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp"
#include "misc/mesh_generator3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "newtonian/common/MixedEos.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "3D/output/write3D.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/Comb.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "3D/GeometryCommon/RoundGrid3D.hpp"
#include "CrookedPipeBoundary.hpp"
#include "CrookedPipeOpacity.hpp"
#include "3D/tessellation/voronoi/pointsManager/ParMETISPointManager.hpp"
#include "3D/output/MC/read_write_particles.hpp"

// #define RDMA

namespace fs = std::filesystem;

std::vector<ComputationalCell3D> *cellsPtr;
std::vector<Particle3D> *particlesPtr;
Voronoi3D *tessPtr;
bool do_output;

void Output(const std::string &fname)
{
    if(not do_output)
    {
        return;
    }
    rank_t rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
    {
        std::cout << "Writing output to " << fname << std::endl;
    }

    std::vector<ComputationalCell3D> &cells = *cellsPtr;
    size_t N = cells.size();
    assert(N == tessPtr->GetPointNo());
    std::vector<double> temperatures(N);
    std::vector<double> tracers0(N), tracers1(N);
    for(size_t i = 0; i < N; i++)
    {
        temperatures[i] = cells[i].temperature;
        tracers0[i] = cells[i].tracers[0];
        tracers1[i] = cells[i].tracers[1];
    }

    if(particlesPtr != nullptr)
    {
        WriteParticlesParallel(fname + "_particles.h5", *particlesPtr);
    }
    if(tessPtr != nullptr)
    {
        WriteVoronoiVTKOnly(*tessPtr, fname + ".vtu", {temperatures, tracers0, tracers1}, {"Temperature", "Tracer0", "Tracer1"});
    }
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

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();
    
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0)
    {
        if(argc != 3 and argc != 4 and argc != 5)
        {
            std::cerr << "Usage: " << argv[0] << " <number of points> <particles per cell> [output? = 1] [RDMA/P2P]" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    size_t N = std::stoul(argv[1]);
    size_t particlesPerCell = std::stoul(argv[2]);
    do_output = (argc >= 4) ? std::stoul(argv[3]) : false;
    bool useRDMA = (argc >= 5) ? (std::string(argv[4]) == "RDMA") : false;

    Vector3D ll(0, -2, -2), ur(7, 2, 2);

    std::vector<Vector3D> points;
    if(rank == 0)
    {
        std::cout << "Rank " << rank << " generating points..." << std::endl;
        points = GeneratePoints(N, ll, ur);
        std::cout << "Rank " << rank << " generated " << points.size() << " points." << std::endl;
    }
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "After spreading" << std::endl;
    }

    SILENCE_TIMERS();
    points = RoundGrid3D(points, ll, ur, 5);
    UNSILENCE_TIMERS();

    Voronoi3D tess(ll, ur);

    try
    {
        tessPtr = &tess;
        {
            START_TIMER("First Build");
            tess.BuildParallel(points);
            points = tess.getMeshPoints();
            points.resize(tess.GetPointNo());
        }
        // std::shared_ptr<PointsManager> pointsManager = std::make_shared<ParMETISPointManager>(tess);
        // tess.SetPointsManager(pointsManager);
        tess.BuildParallel(points);
        points = tess.getMeshPoints();
        points.resize(tess.GetPointNo());
    }
    catch(const UniversalError &e)
    {
        reportError(e);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    
    IdealGas eos1(1.5, 1e16 / units::kev_kelvin, 1, 0);
    IdealGas eos2(1.5, 1e13 / units::kev_kelvin, 1, 0);
    std::vector<EquationOfState*> eosList = {&eos1, &eos2};
    MixedEOS eos(eosList);
    ComputationalCell3D cell;
    cell.density = 1;
    cell.temperature = 0.05 * units::kev_kelvin;
    cell.velocity = Vector3D(0, 0, 0);
    std::vector<ComputationalCell3D> cells(points.size(), cell);
    cellsPtr = &cells;
    
    N = tess.GetPointNo();
    ComputationalCell3D::tracerNames = {"Thick", "Thin"};
    for(size_t i = 0; i < N; i++)
    {
        Vector3D p = tess.GetCellCM(i);
        double const r = std::sqrt(p.z * p.z + p.y * p.y);
        if(r > 1.5)
        {
            cells[i].tracers[0] = 1.0;
            
        }
        else
        {
            if(r < 0.5)
            {
                if(p.x > 3 && p.x < 4)
                {
                    cells[i].tracers[0] = 1.0;
                }
                else
                    cells[i].tracers[1] = 1.0;
            }
            else
            {
                if(p.x < 2.5 || p.x > 4.5)
                {
                    cells[i].tracers[0] = 1.0;
                }
                else
                {
                    if(p.x > 3 && p.x < 4 && r < 1)
                        cells[i].tracers[0] = 1.0;
                    else
                        cells[i].tracers[1] = 1.0;
                }
            }
        }
        cells[i].internal_energy = eos.dT2e(cells[i].density, cells[i].temperature, cells[i].tracers, ComputationalCell3D::tracerNames);
    }

    Conserved3D cons;
    std::vector<Conserved3D> conserved(points.size(), cons);
    for(size_t i = 0; i < N; i++)
    {
        PrimitiveToConserved(cells[i], tess.GetVolume(i), conserved[i]);
        if(conserved[i].internal_energy < 1)
        {
            try
            {
                UniversalError eo("Zero internal energy in main");
                eo.addEntry("Cell index", i);
                eo.addEntry("Cell volume", tess.GetVolume(i));
                eo.addEntry("Cell mass", conserved[i].mass);
                eo.addEntry("Internal energy", conserved[i].internal_energy);
                eo.addEntry("Cell Internal energy", cells[i].internal_energy);
                eo.addEntry("Density", cells[i].density);
                eo.addEntry("Temperature", cells[i].temperature);
                eo.addEntry("Tracers", cells[i].tracers);
                eo.addEntry("Tracer names", ComputationalCell3D::tracerNames);
                throw eo;
            }
            catch(const UniversalError &e)
            {
                reportError(e);
                exit(1);
            }
        }
    }
    // MCPowerLawOpacity opacity(10, 0, 0, -3, 0, 0);
    CrookedPipeOpacity opacity;

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond = std::make_shared<CrookedPipeBoundaryCondition<Vector3D, Tessellation3D>>(tess, cells);
    std::shared_ptr<RadiationIMC> physics = std::make_shared<RadiationIMC>(tess, boundaryCond, cells, conserved, eos, opacity, particlesPerCell / 10 /* 100 */);
    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, particlesPerCell /* 100 */);

    std::string prefix = "/data/shared/maorm/OSC_CrookedPIPE_" + std::to_string(size) + "_";
    fs::path path = fs::path(prefix);

    // create prefix (directory)
    if(not fs::exists(path))
    {
        fs::create_directories(path);
    }

    particlesPtr = nullptr;
    Output(prefix + "start");

    {
        std::shared_ptr<MonteCarloManager3D> manager;
        if(useRDMA)
        {
            manager = std::make_shared<RDMAMonteCarloManager3D>(tess, physics, popControl, boundaryCond);
        }
        else
        {
            manager = std::make_shared<TwoSidedMonteCarloManager3D>(tess, physics, popControl, boundaryCond);
        }

        std::vector<MonteCarloParticle<Vector3D, Tessellation3D>> particles;
        particlesPtr = &particles;

        size_t iterations = 10000;
        std::chrono::high_resolution_clock::time_point start, end1, end2;
        std::chrono::high_resolution_clock::time_point start_total, end_total;

        // vtune_start();
        double dt = 1e-11; 
        double time = 0;
        size_t i = 0;
        const double totalTime = 1e-6;

        std::vector<double> weights(points.size(), 1.0);

        double max_step = 1e-9;// 1e-8

        start_total = std::chrono::high_resolution_clock::now();

        while(time < totalTime)
        // for(int k = 0; k < 21; k++)
        {
            start = std::chrono::high_resolution_clock::now();
            // if(k == 50)
            // {
            //     vtune_start();
            // }
            try
            {
                START_TIMER("MC Step");
                particles = manager->step(particles, cells, dt);
            }
            catch(const UniversalError &e)
            {
                reportError(e);
                exit(1);
            }
            time += dt;
            dt = std::min(1.1 * dt, std::min(max_step, totalTime - time));
            end1 = std::chrono::high_resolution_clock::now();
            if(rank == 0 /* and i % 100 == 0 */)
            {
                double timeTaken = std::chrono::duration<double>(end1 - start).count();
                std::cout << "Ended Iteration " << i << ", dt: " << dt << ", simulation time: " << time << " (" << time / totalTime * 100 << "%), time taken: " << timeTaken << " seconds" << std::endl;
            }
            if(i % 5 == 0)
            {
                particlesPtr = &particles;
                Output(prefix + std::to_string(i));
            }
            i++;
            
            MPI_Barrier(MPI_COMM_WORLD);
            // vtune_stop();

            start = std::chrono::high_resolution_clock::now();
            if(i % 10 == 0)
            {                
                vtune_start();
                try
                {
                    START_TIMER("Rebalance & Exchange");
                    {
                        START_TIMER("Rebalance");
                        SILENCE_TIMERS();
                        const std::vector<double> &factorFlecks = physics->getFactorFleck();
                        const std::vector<double> &planckOpacities = physics->getPlanckOpacities();
                        size_t N = tess.GetPointNo();
                        // weights.resize(N, 1.0);
                        // double weight = static_cast<double>(manager.GetStepCounter()) / N;
                        // for(size_t j = 0; j < N; j++)
                        // {
                        //     // weights[j] = std::max(1., planckOpacities[j] * tess.GetWidth(j) * (1 - factorFlecks[j]));
                        //     weights[j] = weight;
                        // }
                        weights.resize(N);
                        const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
                        for(size_t j = 0; j < N; j++)
                        {
                            weights[j] = static_cast<double>(counters[j]);
                        }
                        tess.Rebalance(weights);
                        points = tess.getMeshPoints();
                        points.resize(tess.GetPointNo());
                        
                        MPI_exchange_data(tess, cells, false);
                        MPI_exchange_data(tess, conserved, false);
                        
                        end1 = std::chrono::high_resolution_clock::now();
        
                        UNSILENCE_TIMERS();
                    }
                    {
                        START_TIMER("Exchange");
                        if(rank == 0)
                        {
                            std::cout << "Calling UpdateNewCellsAfterExchange" << std::endl;
                        }
                        UpdateNewCells(tess, particles, cells);
                        // UpdateNewCellsAfterExchange(tess, particles);
                        // std::cout << "Rank " << rank << " has now " << particles.size() << " particles after exchange." << std::endl;
                    }
    
                    end2 = std::chrono::high_resolution_clock::now();
                    MPI_Barrier(MPI_COMM_WORLD);
        
                    vtune_start();
                }
                catch(const UniversalError &e)
                {
                    reportError(e);
                    exit(1);
                }
                vtune_stop();

                if(rank == 0 /* and i % 100 == 0 */)
                {
                    PRINT_TIMES();
                    double timeTaken = std::chrono::duration<double>(end2 - start).count();
                    double timeTakenOnlyRebalance = std::chrono::duration<double>(end1 - start).count();
                    std::cout << "Did rebalance, time taken: " << timeTaken << " seconds, only rebalance (no exchange): " << timeTakenOnlyRebalance << std::endl;
                }
                CLEAR_TIMES();
            }
        }

        end_total = std::chrono::high_resolution_clock::now();
        double totalTimeTaken = std::chrono::duration<double>(end_total - start_total).count();
        if(rank == 0)
        {
            std::cout << "Total time taken: " << totalTimeTaken << " seconds" << std::endl;
        }
    }

    Output(prefix + "final");

    MPI_Finalize();
}