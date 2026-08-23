#include "3D/tessellation/Voronoi3D.hpp"
#include "monte/manager/serial/MonteCarloManagerSerial.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/boundary/SideTemperature.hpp"

int main(int argc, char *argv[])
{
    double length = 3;
    Vector3D ll(0, 0, 0), ur(length, 1, 1);
    size_t N = 128;

    Voronoi3D tess(ll, ur);

    std::vector<Vector3D> points;

    double cellLength = (ur.x - ll.x) / N;
    double _x = ll.x + cellLength / 2;
    for(size_t i = 0; i < N; i++)
    {
        points.push_back(Vector3D(_x, 0.5, 0.5));
        _x += cellLength;
    }

    tess.Build(points);

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

    MonteCarloManagerSerial<Vector3D, Tessellation3D> manager(tess, physics, popControl, boundaryCond);
    
    std::vector<MonteCarloParticle<Vector3D>> particles;
    size_t iterations = 500 / 0.03;
    for(int i = 0; i < iterations; i++)
    {
        if(i % 100 == 0)
        {
            std::cout << "Iteration " << i << "(out of " << iterations << ")" << std::endl;
        }
        particles = manager.step(particles, 0.03 / units::clight, 1e4);
        if(i % 100 == 0)
        {
            std::cout << "Particles: " << particles.size() << std::endl;
        }
    }

    N = tess.GetPointNo();
    std::vector<double> x_s(N);
    std::vector<double> temperatures(N);
    std::ofstream out("results.txt", std::ios::out);
    for(size_t i = 0; i < N; i++)
    {
        x_s[i] = tess.GetMeshPoint(i).x;
        temperatures[i] = cells[i].temperature;
        out << x_s[i] << ", " << temperatures[i] << std::endl;
        std::cout << x_s[i] << ", " << temperatures[i] << std::endl;
    }
    out.close();
}
