#include "source/3D/tesselation/voronoi/Voronoi3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"

int main(void)
{
    size_t const Np = 1e3;
    // Set up size of the domain
    Vector3D ll(-1, -1, -1), ur(1, 1, 1);
    int rank = 0;
    // Start mpi if needed
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // Create the initial points
    std::vector<Vector3D> points;
    if(rank == 0)
        points = RandRectangular(Np, ll, ur);
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
    // Make the points roundish
    points = RoundGrid3D(points, ll, ur, 10);
    if (rank == 0)
		std::cout << "Done round" << std::endl;
    
    // Create the tesselation
    Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif

    //Set up the EOS
    IdealGas eos(5./3.);

    // Create the hydro data
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);
    ComputationalCell3D inner_cell, outer_cell;
    inner_cell.velocity = Vector3D(0, 0 , 0);
    inner_cell.density = 1;
    inner_cell.internal_energy = 1e5;
    inner_cell.pressure = eos.de2p(inner_cell.density, inner_cell.internal_energy, inner_cell.tracers, ComputationalCell3D::tracerNames);
    outer_cell.velocity = Vector3D(0, 0 , 0);
    outer_cell.density = 1;
    outer_cell.internal_energy = 0.1;
    outer_cell.pressure = eos.de2p(outer_cell.density, outer_cell.internal_energy, outer_cell.tracers, ComputationalCell3D::tracerNames);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        if(abs(tess.GetMeshPoint(i)) < 0.2)
            cells[i] = inner_cell;
        else
            cells[i] = outer_cell;
    }
 
    // Rieamann solver
	Hllc3D rs;

	// Hydro boundary conditions
    RigidWallGenerator3D ghost;

	// Spatial Interpolation scheme
	LinearGauss3D interp(eos, ghost);

	// Flux calculator
	std::vector<pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*> > sequence;
    
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
	ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
	ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*>(isboundary, rigid_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*>(isbulk, normal_flux));
	ConditionActionFlux1 flux(sequence, interp);

	// Extensive updater
	std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
	ConditionExtensiveUpdater3D eu(eu_sequence);

	// Primitive updater
	DefaultCellUpdater cu;

	// External force
	ZeroForce3D force;

	// Time step function
	double const hydro_cfl = 0.3;
	double const force_cfl = 1;
	CourantFriedrichsLewy tsf(hydro_cfl, force_cfl, force);

	// Set point motion
	Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos);


	// Create main simulation object
	HDSim3D sim(tess, cells, eos, pm, tsf, flux, cu, eu, force, std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    double old_time = sim.getTime();
    while(sim.getTime() < 0.0075)
    {
        try
        {
            // Print to screen run time
            if(rank == 0)
            {
                std::cout<<std::endl;
                std::cout<<"Iteration "<<sim.getCycle()<<" dt "<<sim.getTime() - old_time<<" time "<<sim.getTime()<<std::endl;
            }
            old_time = sim.getTime();
            // Write to file every 100 time steps
            if(sim.getCycle() % 100 == 0)
                WriteSnapshot3D(sim, "sedov_"+std::to_string(sim.getCycle())+".h5");   
            // Main time step       
            sim.timeAdvance2();
        }
        catch(UniversalError const& eo)
        {
            reportError(eo);
            throw;
        }
    }
    WriteSnapshot3D(sim, "sedov_final.h5");
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
