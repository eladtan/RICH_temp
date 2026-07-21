#ifdef RICH_MPI

#include <signal.h>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include "monte/manager/MonteCarloManager.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/Comb.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "monte/Voronoi3DMovement.hpp"
#include "3D/output/write3D.hpp"

std::vector<ComputationalCell3D> *cellsPtr;
Voronoi3D *tessPtr;

void Output(const std::string &fname)
{
    std::vector<ComputationalCell3D> &cells = *cellsPtr;
    size_t N = cells.size();
    std::vector<double> temperatures(N);
    for(size_t i = 0; i < N; i++)
    {
        temperatures[i] = cells[i].temperature;
    }

    WriteVoronoiParallel(*tessPtr, fname, {temperatures}, {"Temperature"});
}

std::vector<Vector3D> RandomPointsInRectangle(const Vector3D &ll, const Vector3D &ur, size_t N)
{
    std::vector<Vector3D> points;
    points.reserve(N);
    std::uniform_real_distribution<double> unifX(ll.x, ur.x);
    std::uniform_real_distribution<double> unifY(ll.y, ur.y);
    std::uniform_real_distribution<double> unifZ(ll.z, ur.z);
    std::mt19937_64 re(0);

    for(size_t i = 0; i < N; i++)
    {
        points.emplace_back(unifX(re), unifY(re), unifZ(re));
    }
    return points;
}

std::vector<Vector3D> GenerateCartesian(const Vector3D &ll, const Vector3D &ur, size_t Nx, size_t Ny, size_t Nz)
{
    std::vector<Vector3D> points;
    double x_len = (ur.x - ll.x) / Nx;
    double y_len = (ur.y - ll.y) / Ny;
    double z_len = (ur.z - ll.z) / Nz;
    double _x = ll.x + x_len / 2;

    for(size_t i = 0; i < Nx; i++)
    {
        double _y = ll.y + y_len / 2;
        for(size_t j = 0; j < Ny; j++)
        {
            double _z = ll.z + z_len / 2;
            for(size_t k = 0; k < Nz; k++)
            {
                points.emplace_back(_x, _y, _z);
                _z += z_len;
            }
            _y += y_len;
        }
        _x += x_len;
    }
    return points;
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <number of points>" << std::endl;
        return 1;
    }
    vtune_stop();
    MPI_Init(&argc, &argv);

    double length = 3;
    Vector3D ll(0, 0, 0), ur(length, 1, 1);
    size_t Nx = 256, Ny = 4, Nz = 4;

    Voronoi3D tess(ll, ur);
    tessPtr = &tess;

    std::vector<Vector3D> points;
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t N = std::stoull(argv[1]);

    if(rank == 0)
    {
        points = GenerateCartesian(ll, ur, Nx, Ny, Nz);
        // points = RandomPointsInRectangle(ll, ur, N);
    }

    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    tess.BuildParallel(points);
    points = tess.getMeshPoints();
    points.resize(tess.GetPointNo());
    std::cout << "Rank " << rank << " has " << points.size() << " points" << std::endl;
    
    IdealGas eos(1.5, 7.14 * units::arad, 1, 0);
    ComputationalCell3D cell;
    cell.density = 1;
    cell.temperature = 0.01;
    cell.velocity = Vector3D(1e9, 0, 0);
    cell.internal_energy = eos.dT2e(cell.density, cell.temperature, cell.tracers, ComputationalCell3D::tracerNames);
    std::vector<ComputationalCell3D> cells(points.size(), cell);
    cellsPtr = &cells;

    Conserved3D cons;
    PrimitiveToConserved(cell, tess.GetVolume(0), cons);
    std::vector<Conserved3D> conserved(points.size(), cons);
    MCPowerLawOpacity opacity(10, 0, 0, -3, 0, 0);

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond = std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(tess, cells, 1, 100);
    std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> physics = std::make_shared<RadiationIMC>(tess, boundaryCond, cells, conserved, eos, opacity, 100);
    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, 100);

    auto randomDirection = [](void)
    {
        static std::uniform_real_distribution<double> unif(-1, 1);
        static std::mt19937_64 re(0);
        return normalize(Vector3D(unif(re), unif(re), unif(re)));
    };

    Output("MCRT_MM_start.vtu");

    // override SIGINT signal handler
    signal(SIGINT, [](int signum)
    {
        std::cout << "Finishing " << std::endl;
        Output("MCRT_MM_signal.vtu");
        MPI_Finalize();
        exit(signum);
    });

    {
        MonteCarloManager<Vector3D, Tessellation3D> manager(tess, physics, popControl, boundaryCond);
        
        std::vector<MonteCarloParticle<Vector3D, Tessellation3D>> particles;
        size_t iterations = 10000;
        std::chrono::high_resolution_clock::time_point start, end;

        vtune_start();
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < iterations; i++)
        {
            if(rank == 0 and i % 100 == 0)
            {
                std::cout << "Iteration " << i << std::endl;
            }
            if(i % 1000 == 0)
            {
                Output("MCRT_MM_" + std::to_string(i) + ".vtu");
            }

            particles = manager.step(particles, 0.03 / units::clight);
            
            // // move points a little bit
            // for(size_t j = 0; j < points.size(); j++)
            // {
            //     Vector3D &point = points[j];
            //     Vector3D newPoint = point + 0.01 * randomDirection();
            //     if(not tess.IsPointOutsideBox(newPoint))
            //     {
            //         point = newPoint;
            //     }
            // }
            
            // tess.BuildParallel(points);
            // points = tess.getMeshPoints();
            // points.resize(tess.GetPointNo());
            // MPI_exchange_data(tess, cells, false);
            // MPI_exchange_data(tess, conserved, false);

            // UpdateNewCells(tess, particles);
        }
        end = std::chrono::high_resolution_clock::now();
        vtune_stop();

        if(rank == 0)
        {
            double totalTime = std::chrono::duration<double>(end - start).count();
            std::cout << "Total time taken in seconds: " << totalTime << std::endl;
            std::cout << "Average for one step: " << totalTime / iterations << " seconds" << std::endl;
        }

        struct CellData : public Serializable 
        {
        public:
            double x;
            double temperature;
    
            inline CellData() : x(0), temperature(0) {}
    
            inline CellData(double x, double temperature) : x(x), temperature(temperature) {}
    
            // dump
            inline size_t dump(Serializer *serializer) const override
            {
                size_t byteOffset = 0;
                byteOffset += serializer->insert(this->x);
                byteOffset += serializer->insert(this->temperature);
                return byteOffset;
            }
    
            // load
            inline size_t load(const Serializer *serializer, std::size_t offset)
            {
                size_t bytesRead = 0;
                bytesRead += serializer->extract(this->x, offset);
                bytesRead += serializer->extract(this->temperature, offset + bytesRead);
                return bytesRead;
            }
    
            // operator <
            inline bool operator<(const CellData &other) const{return this->x < other.x;}
    
            // operator ==
            inline bool operator==(const CellData &other) const{return this->x == other.x;}
        };
    
        std::vector<CellData> cellData(points.size());
        for(size_t i = 0; i < points.size(); i++)
        {
            cellData[i].x = points[i].x;
            cellData[i].temperature = cells[i].temperature;
        }
        
        cellData = MPI_Gatherv_serializable(cellData, 0, MPI_COMM_WORLD);
        if(rank == 0)
        {
            std::cout << "Total cell data is " << cellData.size() << " points" << std::endl;
            std::sort(cellData.begin(), cellData.end());
            std::ofstream out("MCRT_results_parallel_" + std::to_string(size) + ".txt", std::ios::out);
            for(size_t i = 0; i < cellData.size(); i++)
            {
                out << cellData[i].x << ", " << cellData[i].temperature << std::endl;
                // std::cout << cellData[i].x << ", " << cellData[i].temperature << std::endl;
            }
            out.close();
        }
    }

    Output("MCRT_MM_final.vtu");

    MPI_Finalize();
}

#endif // RICH_MPI