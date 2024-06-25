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
#define cylindar_run


// simulation settings
struct SimulationParameters{
    std::string const name = "sedov";

    // energy deposition:
    //  in a "point" (spherical point explosion for cylindar_run=1 or cylyndrical point explosion for cylindar_run=0)
    //  or in a "line" (cylyndrical point explosion for cylindar_run=0 or planar point explosion for cylindar_run=0)
    // bool const deposition_point = false;
    bool const deposition_point = true;

    // in case if deposition_point=true, 
    // choose to deposit energy at corner (0,0) or at (L/2, 0)
    bool const corner_ = false;
    // bool const corner_ = true;
    
    bool const corner = deposition_point and corner_;

    // total energy should be multiplied by 0.5 in case of corner or planar
    // double const Etot = 0.25*1.;// * (corner ? 0.5 : 1.);
    double const Etot = 0.5*1.;// * (corner ? 0.5 : 1.);
    // double const Etot = 1.;// * (corner ? 0.5 : 1.);

    double const rho0 = 1.;
    double const gamma = 5./3.;

    // End time of run
    double const tend = 0.15;

    // Number of linear mesh points
    size_t const Np_ = 64;
    // size_t const Np_ = 500;
    size_t const Np = Np_;// * (corner or not deposition_point ? 1 : 2);

    // box size
    double const L = 1.;//0.6;// * (corner or not deposition_point ? 1. : 2.);

    // voronoi random or xy grid
    // bool const xy_grid = false;
    bool const xy_grid = true;
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
    double const R_dep = 1.5*params.L/ static_cast<double>(Np);
    double const x_dep = params.corner ? 0. : (0.5*params.L);

    std::vector<ComputationalCell> cells(tess.GetPointNo());
    vector<Edge> const& all_edges = tess.getAllEdges();
    
    // get the total mass in the deposition region
    double mass_dep = 0.;
    for(size_t i = 0; i < std::size_t(tess.GetPointNo()); ++i)
    {
        double const x = tess.GetMeshPoint(i).x;
        double const y = tess.GetMeshPoint(i).y;

        if(params.deposition_point){
            bool const in_dep_region = (x-x_dep)*(x-x_dep)+y*y < R_dep*R_dep;
            if(not in_dep_region) continue;
        }
        else{
            bool const in_dep_region = y < R_dep;
            if(not in_dep_region) continue; 
        }

        vector<int> edge_index=tess.GetCellEdges(i);
        vector<Edge> temp_vector;
        for(size_t j=0;j<edge_index.size();++j)
            temp_vector.push_back(all_edges[edge_index[j]]);
        double const volume = geometry.calcVolume(temp_vector);
        mass_dep += params.rho0 * volume;
    }

    printf("%g\n",mass_dep);
    double const sie = params.Etot / mass_dep;
    double const P = (params.gamma-1.) * params.rho0 * sie; 

    for(size_t i = 0; i < std::size_t(tess.GetPointNo()); ++i)
    {
        cells[i].density = params.rho0;

        double const x = tess.GetMeshPoint(i).x;
        double const y = tess.GetMeshPoint(i).y;
        bool in_dep_region = false;
        
        if(params.deposition_point){
            in_dep_region = (x-x_dep)*(x-x_dep)+y*y < R_dep*R_dep;
        }
        else{
            in_dep_region = y < R_dep;
        }

        cells[i].pressure = in_dep_region ? P : 1e-10;
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

