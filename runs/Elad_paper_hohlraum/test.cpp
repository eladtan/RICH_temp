#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "mpi/mpi_commands.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/common/MixedEos.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "3D/output/write3D.hpp"

#include "3D/radiation/RadiationIMC.hpp"
#include "monte/population/Comb.hpp"
#include "newtonian/three_dimensional/RadiationMCStep.hpp"

#include "HohlraumOpacity.hpp"
#include "HohlraumBoundary.hpp"

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

static bool isMaterial(double x, double r)
{
    // Left wall
    if(x >= 0.10 && x <= 0.15 && r <= 0.45)
        return true;
    // Capsule
    if(x >= 0.55 && x <= 0.95 && r <= 0.45)
        return true;
    // Right end cap
    if(x >= 1.35 && x <= 1.40 && r <= 0.65)
        return true;
    // Outer cylindrical wall
    if(x >= 0.10 && x <= 1.40 && r >= 0.60 && r <= 0.65)
        return true;
    return false;
}

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    rank_t rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    if(rank == 0 && argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <delta_cm> [output_prefix] [new_per_cell] [max_per_cell] [--2d]"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

  try
  {
    double delta = std::stod(argv[1]);
    std::string prefix = (argc >= 3) ? argv[2] : "hohlraum";
    size_t newPhotonsPerCell = (argc >= 4) ? std::stoul(argv[3]) : 50;
    size_t maxPhotonsPerCell = (argc >= 5) ? std::stoul(argv[4]) : 200;

    bool mode2d = false;
    for(int a = 1; a < argc; a++)
        if(std::string(argv[a]) == "--2d") mode2d = true;

    // --- Physical parameters (Section 4.2) ---
    const double T_boundary_keV = 1.0;
    const double T_boundary     = T_boundary_keV * units::kev_kelvin;   // Kelvin
    const double T_init         = 300.0;                                 // Kelvin
    const double dt             = 1e-11;                                 // seconds
    const double t_final        = 10e-9;                                 // 10 ns
    const size_t iterations     = static_cast<size_t>(t_final / dt + 0.5);
    constexpr size_t boundaryPhotonsPerCell = 100;

    // Domain: 3D revolves around x-axis; 2D is a thin slab at z=1
    const double Lx = 1.4, Ly = 0.65;
    const double Lz = mode2d ? 1.0 : 0.65;
    Vector3D ll(0, -Ly, mode2d ? 0.0 : -Lz);
    Vector3D ur(Lx, Ly, mode2d ? 2.0 : Lz);

    // --- Generate grid points on rank 0, spread to all ranks ---
    size_t Nx = static_cast<size_t>(std::ceil(Lx / delta));
    size_t Ny = static_cast<size_t>(std::ceil(2 * Ly / delta));
    size_t Nz = mode2d ? 1 : static_cast<size_t>(std::ceil(2 * Lz / delta));

    std::vector<Vector3D> points;
    if(rank == 0)
    {
        double dx = Lx / Nx;
        double dy = 2 * Ly / Ny;
        double dz = 2 * Lz / Nz;
        for(size_t i = 0; i < Nx; i++)
            for(size_t j = 0; j < Ny; j++)
                for(size_t k = 0; k < Nz; k++)
                    points.emplace_back(ll.x + (i + 0.5) * dx,
                                        ll.y + (j + 0.5) * dy,
                                        ll.z + (k + 0.5) * dz);
        std::cout << "Generated " << points.size() << " points ("
                  << Nx << " x " << Ny << " x " << Nz
                  << "), delta=" << delta << " cm" << std::endl;
        if(static_cast<rank_t>(points.size()) < ws)
        {
            std::cerr << "ERROR: only " << points.size() << " cells for " << ws
                      << " MPI ranks. Reduce delta." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    Voronoi3D tess(ll, ur);
    tess.BuildParallel(points);

    // --- Equations of State ---
    // Material: Cv = 3e15 erg/keV/cm^3  =>  f = 3e15 / kev_kelvin (erg/K/cm^3)
    // Vacuum:   negligible Cv
    IdealGas eosMaterial(1.5, 3e15 / units::kev_kelvin, 1, 0);
    IdealGas eosVacuum(1.5, 1e5 / units::kev_kelvin, 1, 0);
    std::vector<EquationOfState *> eosList = {&eosMaterial, &eosVacuum};
    MixedEOS eos(eosList);

    // --- Initial conditions ---
    ComputationalCell3D::tracerNames = {"Material", "Vacuum"};

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

    // --- Simulation ---
    Simulation sim(tess, initialCells, eos);
    std::shared_ptr<TimeStepFunction3D> tsc = std::make_shared<ManualTimeStep>();
    sim.SetTimeStepFunction(tsc);

    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &extensives = sim.getExtensives();
    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    // --- MC radiation transport ---
    constexpr bool withHydro = false;

    auto eosPtr = std::make_shared<MixedEOS>(eos);
    auto opacityPtr = std::make_shared<HohlraumOpacity>();

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
        std::make_shared<HohlraumBoundary<Vector3D, Tessellation3D>>(
            tess, cells, T_boundary, boundaryPhotonsPerCell);

    std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<RadiationIMC>(
        tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, newPhotonsPerCell, withHydro);

    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
        std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, maxPhotonsPerCell);

    std::vector<Particle3D> initialParticles;
    bool withRDMA = true;
    auto mcStep = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, popControl, boundaryCond, initialParticles, withHydro
        #ifdef RICH_MPI
            , withRDMA
        #endif
    );
    sim.addPhysics(mcStep);

    sim.SetTimeStep(dt);

    if(rank == 0)
    {
        size_t nMaterial = 0, nVacuum = 0;
        for(size_t i = 0; i < N; i++)
        {
            if(initialCells[i].tracers[0] > 0.5) nMaterial++;
            else nVacuum++;
        }
        std::cout << "Hohlraum (" << (mode2d ? "2D" : "3D") << "): delta=" << delta
                  << " cm, cells=" << Nx << "x" << Ny << "x" << Nz
                  << " (" << Nx * Ny * Nz << ")"
                  << ", material=" << nMaterial
                  << ", vacuum=" << nVacuum
                  << ", dt=" << dt << " s"
                  << ", t_final=" << t_final << " s"
                  << ", iterations=" << iterations
                  << ", new/cell=" << newPhotonsPerCell
                  << ", max/cell=" << maxPhotonsPerCell
                  << std::endl;
    }

    // --- Helper: write temperature profile along r=0.05 cm ---
    // Collect cells near the axis (r < 0.1) and output T(x)
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

        size_t load(const Serializer *serializer, std::size_t offset)
        {
            size_t rd = 0;
            rd += serializer->extract(this->x, offset);
            rd += serializer->extract(this->r, offset + rd);
            rd += serializer->extract(this->temperature, offset + rd);
            return rd;
        }

        bool operator<(const CellData &o) const { return x < o.x; }
    };

    auto writeProfile = [&](const std::string &filename, double time_ns)
    {
        const double r_line = 0.05;
        const double r_tol  = delta;
        size_t nPts = tess.GetPointNo();
        std::vector<CellData> cellData;
        for(size_t i = 0; i < nPts; i++)
        {
            Vector3D p = tess.GetMeshPoint(i);
            double ri = mode2d ? std::abs(p.y) : std::sqrt(p.y * p.y + p.z * p.z);
            if(std::abs(ri - r_line) < r_tol)
                cellData.emplace_back(p.x, ri, cells[i].temperature);
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
    };

    // --- Helper: write VTK snapshot ---
    auto writeVTK = [&](const std::string &filename)
    {
        size_t nPts = tess.GetPointNo();
        std::vector<double> temps(nPts), tKeV(nPts), tr0(nPts), tr1(nPts);
        for(size_t i = 0; i < nPts; i++)
        {
            temps[i] = cells[i].temperature;
            tKeV[i]  = cells[i].temperature / units::kev_kelvin;
            tr0[i]   = cells[i].tracers[0];
            tr1[i]   = cells[i].tracers[1];
        }
        WriteVoronoiVTKOnly(tess, filename,
                            {temps, tKeV, tr0, tr1},
                            {"Temperature_K", "Temperature_keV", "Material", "Vacuum"});
    };

    writeVTK(prefix + "_t0.vtu");

    constexpr size_t dumpInterval = 20;
    size_t dumpCount = 0;

    // --- Main time-stepping loop ---
    double simTime = 0;
    auto startWall = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < iterations; i++)
    {
        auto stepStart = std::chrono::high_resolution_clock::now();

        sim.step();

        simTime += dt;
        sim.SetTimeStep(dt);

        auto stepEnd = std::chrono::high_resolution_clock::now();
        double stepSec = std::chrono::duration<double>(stepEnd - stepStart).count();
        double elapsedWall = std::chrono::duration<double>(stepEnd - startWall).count();

        double fraction = static_cast<double>(i + 1) / iterations;
        double eta = (fraction > 0) ? elapsedWall * (1.0 - fraction) / fraction : 0;

        if(rank == 0 && (i % 10 == 0 || i + 1 == iterations))
        {
            int pct = static_cast<int>(fraction * 100);
            int etaMin = static_cast<int>(eta) / 60;
            int etaSec = static_cast<int>(eta) % 60;
            std::cout << "Cycle " << i + 1 << "/" << iterations
                      << " (" << pct << "%)"
                      << "  t=" << simTime * 1e9 << " ns"
                      << "  step=" << stepSec << "s"
                      << "  ETA=" << etaMin << "m" << etaSec << "s"
                      << std::endl;
        }

        if((i + 1) % dumpInterval == 0)
        {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s_%05zu.pvtu", prefix.c_str(), dumpCount);
            writeVTK(buf);

            double t_ns = simTime * 1e9;
            std::snprintf(buf, sizeof(buf), "%s_%05zu.txt", prefix.c_str(), dumpCount);
            writeProfile(buf, t_ns);

            dumpCount++;
        }
    }

    auto endWall = std::chrono::high_resolution_clock::now();
    double wallSec = std::chrono::duration<double>(endWall - startWall).count();
    if(rank == 0)
        std::cout << "Total wall time: " << wallSec << "s" << std::endl;

    // --- Final output ---
    writeProfile(prefix + "_final.txt", simTime * 1e9);
    writeVTK(prefix + "_final.vtu");

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
