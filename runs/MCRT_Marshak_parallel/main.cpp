#ifdef RICH_MPI

#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "monte/manager/parallel/MonteCarloManagerLegacy.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "3D/output/write3D.hpp"

using namespace STORM;

int main(int argc, char *argv[])
{
    vtune_stop();
    MPI_Init(&argc, &argv);

    double length = 3;
    Vector3D ll(0, 0, 0), ur(length, 1, 1);
    size_t Nx = 256, Ny = 4, Nz = 4;

    Voronoi3D tess(ll, ur);

    std::vector<Vector3D> points;
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0)
    {
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
                    points.push_back(Vector3D(_x, _y, _z));
                    _z += z_len;
                }
                _y += y_len;
            }
            _x += x_len;
        }
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
    cell.internal_energy = eos.dT2e(cell.density, cell.temperature, cell.tracers, ComputationalCell3D::tracerNames);
    std::vector<ComputationalCell3D> cells(points.size(), cell);
    Conserved3D cons;
    PrimitiveToConserved(cell, tess.GetVolume(0), cons);
    std::vector<Conserved3D> conserved(points.size(), cons);
    MCPowerLawOpacity opacity(10, 0, 0, -3, 0, 0);

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond = std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(tess, 1, 100);
    std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> physics = std::make_shared<::RadiationIMC>(tess, boundaryCond, cells, conserved, eos, opacity, 100);
    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, 100);

    {
        MonteCarloManagerLegacy<Vector3D, Tessellation3D> manager(tess, physics, popControl, boundaryCond);
        
        std::vector<MonteCarloParticle<Vector3D>> particles;
        size_t iterations = 500 / 0.03 / 4;
        std::chrono::high_resolution_clock::time_point start, end;

        vtune_start();
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < iterations; i++)
        {
            if(i % 100 == 0 and rank == 0)
            {
                std::cout << "Iteration " << i << " (out of " << iterations << ")" << std::endl;
            }
            particles = manager.step(particles, 0.03 / units::clight);
            if(i % 100 == 0 and rank == 0)
            {
                std::cout << "Particles: " << particles.size() << std::endl;
            }
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

    std::vector<double> temperatures(points.size());
    for(size_t i = 0; i < points.size(); i++)
    {
        temperatures[i] = cells[i].temperature;
    }

    WriteVoronoiParallel(tess, "MCRT_results.vtu", {temperatures}, {"Temperature"});
    MPI_Finalize();
}

#endif // RICH_MPI
