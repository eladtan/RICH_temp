#ifdef RICH_MPI
#include "source/mpi/MeshPointsMPI.hpp"
#endif
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "source/tessellation/geometry.hpp"
#include "source/newtonian/two_dimensional/hdsim2d.hpp"
#include "source/tessellation/tessellation.hpp"
#include "source/tessellation/RoundGrid.hpp"
#include "source/newtonian/common/hllc.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/tessellation/VoronoiMesh.hpp"
#include "source/newtonian/two_dimensional/spatial_distributions/uniform2d.hpp"
#include "source/newtonian/two_dimensional/point_motions/eulerian.hpp"
#include "source/newtonian/two_dimensional/point_motions/lagrangian.hpp"
#include "source/newtonian/two_dimensional/point_motions/round_cells.hpp"
#include "source/newtonian/two_dimensional/source_terms/zero_force.hpp"
#include "source/newtonian/two_dimensional/geometric_outer_boundaries/SquareBox.hpp"
#include "source/newtonian/two_dimensional/ghost_point_generators/RigidWallGenerator.hpp"
#include "source/newtonian/two_dimensional/diagnostics.hpp"
#include "source/newtonian/two_dimensional/source_terms/cylindrical_complementary.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/mesh_generator.hpp"
#include "source/newtonian/two_dimensional/condition_action_sequence_2.hpp"
#include "source/newtonian/two_dimensional/hdf5_diagnostics.hpp"
#include "source/newtonian/two_dimensional/interpolations/LinearGaussImproved.hpp"
#include "source/newtonian/two_dimensional/simple_cell_updater.hpp"
#include "source/newtonian/two_dimensional/simple_extensive_updater.hpp"
#include "source/newtonian/two_dimensional/stationary_box.hpp"
#ifdef RICH_MPI
#include "source/mpi/MeshPointsMPI.hpp"
#include "source/mpi/ConstNumberPerProc.hpp"
#include <cmath>
#endif

// comment this #define in order to run with slab symmetry
// #define cylindar_run


// simulation settings
struct SimulationParameters{

    // The Astrophysical Journal Supplement Series, 226:2 (26pp), 2016 section 3.1
    std::string const name = "shocktube";

    double const gamma = 5./3.;
    double const rho_left = 1.;
    double const P_left = 1.;
    double const rho_right = 0.125;
    double const P_right = 0.1;

    // End time of run
    double const tend = 0.25;

    // box size
    double const L = 1.;

    // Number of linear mesh points
    size_t const Np = 128;


    // voronoi random or xy grid
    bool const xy_grid = false;
    // bool const xy_grid = true;
};

int main(void)
{
    int rank = 0;
#ifdef RICH_MPI
	MPI_Init(NULL,NULL);
    int ws = 0;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ws);
#endif

    SimulationParameters const params;
    
    // Create box and outer boundary conditions
    Vector2D lower_left(0, 0), upper_right(params.L, params.L);
    SquareBox outer_boundary(lower_left, upper_right);

    size_t const Np = params.Np;
    std::vector<Vector2D> mesh_points;
#ifdef RICH_MPI
    //Create domain decomposition
    std::vector<Vector2D> cpu_points;
    VoronoiMesh cpu_tess;

    if(params.xy_grid){
        double const nx_cpu_d = std::sqrt(ws);
        std::size_t const nx_cpu = static_cast<std::size_t>(nx_cpu_d);
        printf("number of cpu on each direction %g\n", nx_cpu_d);
        if(nx_cpu != nx_cpu_d){
            printf("number of cpu is not a square %d %g\n", ws, nx_cpu_d);
            exit(1);
        }
        cpu_points = cartesian_mesh(nx_cpu, nx_cpu, {lower_left.x, lower_left.y}, {upper_right.x, upper_right.y});
        cpu_tess.Initialise(cpu_points, outer_boundary);
        mesh_points = SquareMeshM(Np, Np, cpu_tess, {lower_left.x, lower_left.y}, {upper_right.x, upper_right.y});
    }
    else{
        cpu_points = RandSquare(ws, lower_left.x, upper_right.x, lower_left.y, upper_right.y);
        // Create initial random mesh
        cpu_tess.Initialise(cpu_points, outer_boundary);
        mesh_points = RandSquare(Np * Np, cpu_tess, lower_left, upper_right);
    }
#else
    if(params.xy_grid){
        mesh_points = cartesian_mesh(Np, Np, {lower_left.x, lower_left.y}, {upper_right.x, upper_right.y});
    }
    else{
     // Create initial random mesh
        mesh_points = RandSquare(Np * Np, lower_left.x, upper_right.x, lower_left.y, upper_right.y);
    }
#endif
    // Make mesh nice and round
#ifdef RICH_MPI
    mesh_points = RoundGridV(mesh_points, outer_boundary, cpu_tess);
#else
    mesh_points = RoundGridV(mesh_points, outer_boundary);
#endif
    // Create voronoi mesh
    VoronoiMesh tess;
#ifdef RICH_MPI
    tess.Initialise(mesh_points, cpu_tess, outer_boundary);
#else
    tess.Initialise(mesh_points, outer_boundary);
#endif
    // Choose geometry
#ifdef cylindar_run 
    printf("CYL SYM\n");
    CylindricalSymmetry geometry(Vector2D(0., 0.), Vector2D(1., 0.));
#else
    printf("SLAB SYM\n");
    SlabSymmetry geometry;
#endif

    // Set how outer interfaces move
    StationaryBox interface_velocity_calc;
    // Set source/force
#ifdef cylindar_run 
    CylindricalComplementary source(geometry.getAxis());
#else
    ZeroForce source;
#endif
    // Set time step calculator
    SimpleCFL time_step(0.3);
    // Riemann solver
    Hllc rs;
    //Equation of state
    IdealGas eos(params.gamma);
    //Ghost point generator
    RigidWallGenerator ghost;
    // Spatial interpolation scheme
    LinearGaussImproved interpolation(eos, ghost);
    // Set flux calculator
    std::vector<std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence::Action*> > first_order;
    std::vector<std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence2::Action2*> > second_order;
    RegularFlux2 reg_flux(rs);
    RigidWallFlux2 rigid_flux(rs);
    second_order.push_back(std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence2::Action2*>(new IsBoundaryEdge(), &rigid_flux));
    second_order.push_back(std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence2::Action2*>(new IsBulkEdge(), &reg_flux));
    ConditionActionSequence2 flux_calc(first_order, second_order, interpolation);
    // Set mesh point motion
    Lagrangian base_point_motion;
    RoundCells point_motion(base_point_motion, eos);
    //How to update the extensive variables
    SimpleExtensiveUpdater extensive_update;
    //How to update the primitive variables
    SimpleCellUpdater cell_update;


    //------------ Set the initial hydro conditions

    std::vector<ComputationalCell> cells(tess.GetPointNo());

    for(size_t i = 0; i < std::size_t(tess.GetPointNo()); ++i)
    {
        bool const left_side = tess.GetMeshPoint(i).y < params.L / 2.;
        cells[i].density  = left_side ? params.rho_left : params.rho_right;
        cells[i].pressure = left_side ? params.P_left   : params.P_right;
    }
    // Create main sim class
#ifdef RICH_MPI
    // Create load balance scheme
    ConstNumberPerProc cpu_move(outer_boundary);
    hdsim sim(cpu_tess, tess, outer_boundary, geometry, cells, eos, point_motion, interface_velocity_calc, source, time_step, flux_calc, extensive_update, 
        cell_update, std::pair<std::vector<std::string>, std::vector<std::string> > (), &cpu_move);
#else
    hdsim sim(tess, outer_boundary, geometry, cells, eos, point_motion, interface_velocity_calc, source, time_step, flux_calc, extensive_update, cell_update);
#endif
    // Main loop
    while(sim.getTime() < params.tend
        // and sim.getCycle() <1000//50//200
    )
    {
        try
        {
            if(rank == 0){
                printf("%d t=%1.16E\n", sim.getCycle(), sim.getTime());
            }

            if(sim.getCycle() % 20 ==0){
                write_snapshot_to_hdf5(
                    sim, 
                    params.name + std::to_string(sim.getCycle()) +"_rank_" + std::to_string(rank) + ".h5",
                    {},
                    false,
                    true
                );
            }

           sim.TimeAdvance2Heun();
        }
        catch(UniversalError const& eo)
        {
            reportError(eo);
            throw;
        }  
    }
#ifdef RICH_MPI
    write_snapshot_to_hdf5(sim, params.name + "_" + std::to_string(sim.getCycle()) +"_rank_" + std::to_string(rank) + ".h5");
    MPI_Finalize();
#else
    write_snapshot_to_hdf5(sim, params.name  + ".h5");
#endif
    return 0;
}

