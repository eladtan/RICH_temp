#include "source/3D/GeometryCommon/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/hdf_write.hpp"
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
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#ifdef RICH_MPI
#include "source/mpi/ConstNumberPerProc3D.hpp"
#include "source/mpi/SetLoad3D.hpp"
#endif
#include "source/Radiation/conj_grad_solve.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"

namespace
{
    class IsXSide : public ConditionActionFlux1::Condition3D
	{
	public:
        std::pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
			const vector<ComputationalCell3D>& cells) const
        {
            if (!tess.BoundaryFace(face_index))
	        	return pair<bool, bool>(false, false);
            size_t const N1 = tess.GetFaceNeighbors(face_index).first;
            size_t const N2 = tess.GetFaceNeighbors(face_index).second;
            if (N1 > tess.GetPointNo())
                if(tess.GetMeshPoint(N1).x > tess.GetBoxCoordinates().second.x ||
                    tess.GetMeshPoint(N1).x < tess.GetBoxCoordinates().first.x)
                    return pair<bool, bool>(true, false);
                else
                    return pair<bool, bool>(false, false);
            else
                if(tess.GetMeshPoint(N2).x > tess.GetBoxCoordinates().second.x ||
                    tess.GetMeshPoint(N2).x < tess.GetBoxCoordinates().first.x)
                    return pair<bool, bool>(true, true);
                else
                    return pair<bool, bool>(false, false);
        }

	};

    class XSideGhost : public SeveralGhostGenerator3D::GhostCriteria3D
	{
	public:
        size_t GhostChoose(Tessellation3D const& tess, size_t index)const
        {
            if(tess.GetMeshPoint(index).x < tess.GetBoxCoordinates().first.x)
                return 0;
            else
                if(tess.GetMeshPoint(index).x > tess.GetBoxCoordinates().second.x)
                    return 1;
                else
                    return 2;
        }
	};
}

int main(void)
{
    size_t const Np = 3e5;
    Vector3D ll(-0.3, 0, 0), ur(0, 0.025, 0.025);
    int ws = 0, rank = 0;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0)
		std::cout << "ws " << ws << std::endl;
	std::vector<Vector3D> cpu_points = RandRectangular(ws, ll, ur);
	cpu_points = RoundGrid3DSingle(cpu_points, ll, ur, 20);
	if (rank == 0)
		std::cout << "Done round" << std::endl;
	Voronoi3D cpu_tess(ll, ur);
	cpu_tess.Build(cpu_points);
	if (rank == 0)
		std::cout << "Finished cpu tess build" << std::endl;
#endif

    //std::vector<Vector3D> points = CartesianMesh(30, 30, 30, ll, ur, &cpu_tess);
    std::vector<Vector3D> points = RandRectangular(Np, ll, ur
#ifdef RICH_MPI
        , &cpu_tess
#endif
        );
    points = RoundGrid3D(points, ll, ur, 10
#ifdef RICH_MPI
        , &cpu_tess
#endif
        );
    if (rank == 0)
		std::cout << "Done round" << std::endl;
    Voronoi3D tess(ll, ur);
    tess.Build(points
#ifdef RICH_MPI
        , cpu_tess
#endif
    );
    size_t const Nlocal = points.size();
    std::vector<ComputationalCell3D> cells(Nlocal);

    double const f = 1.913e8;
    double const beta = 1;
    double const mu = 0;
    IdealGas eos(5./3., f, beta, mu);

    ComputationalCell3D left_state, right_state;
    left_state.velocity = Vector3D(0, 0 , 0);
    left_state.density = 1;
    left_state.temperature = 1415810;
    left_state.Erad = std::pow(left_state.temperature, 4) * CG::radiation_constant / left_state.density; 
    left_state.internal_energy = eos.dT2e(left_state.density, left_state.temperature, left_state.tracers, ComputationalCell3D::tracerNames);
    left_state.pressure = eos.de2p(left_state.density, left_state.internal_energy, left_state.tracers, ComputationalCell3D::tracerNames);
    right_state.velocity = Vector3D(-19518277.654554, 0 , 0);
    right_state.density = 2.2860785640401;
    right_state.temperature = 2941436.3866907;
    right_state.Erad = std::pow(right_state.temperature, 4) * CG::radiation_constant / right_state.density; 
    right_state.internal_energy = eos.dT2e(right_state.density, right_state.temperature, right_state.tracers, ComputationalCell3D::tracerNames);
    right_state.pressure = eos.de2p(right_state.density, right_state.internal_energy, right_state.tracers, ComputationalCell3D::tracerNames);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        if(tess.GetMeshPoint(i).x > -0.1)
            cells[i] = right_state;
        else
            cells[i] = left_state;
    }
 
    // Rieamann solver
	Hllc3D rs;

	// Hydro boundary conditions
    ConstantPrimitiveGenerator3D left_ghost(left_state);
    ConstantPrimitiveGenerator3D right_ghost(right_state);
    RigidWallGenerator3D rigid_ghost;
    std::vector<Ghost3D*> ghosts;
    ghosts.push_back(&left_ghost);
    ghosts.push_back(&right_ghost);
    ghosts.push_back(&rigid_ghost);
    XSideGhost ghost_choose;
    SeveralGhostGenerator3D ghost(ghosts, ghost_choose);
	

	// Spatial Interpolation scheme
	LinearGauss3D interp(eos, ghost);

	// Flux calculator
	std::vector<pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*> > sequence;
    
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
	ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
	ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    ConditionActionFlux1::Condition3D* is_x_side  = new IsXSide();
        sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*>(is_x_side, normal_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*>(isboundary, rigid_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
		const ConditionActionFlux1::Action3D*>(isbulk, normal_flux));
	ConditionActionFlux1 flux(sequence, interp);

	// Extensive updater
	std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
	ConditionExtensiveUpdater3D eu(eu_sequence);

	// Primitive updater
	DefaultCellUpdater cu(0, 0, true);


    double const T_power = 3.5;
    double const TkeV = 11606000;
    PowerLawOpacity opacity(CG::speed_of_light * std::pow(TkeV, -T_power) / (3 * 0.362), -1.0, T_power);
    //DiffusionSideBoundary diff_side(Tbb);
    DiffusionXInflowBoundary diff_side(left_state, right_state, opacity);
    Diffusion diffusion(opacity, diff_side, eos);

	// External force
	DiffusionForce force(diffusion, eos);

	// Time step function
	double const hydro_cfl = 0.3;
	double const force_cfl = 1;
	CourantFriedrichsLewy tsf(hydro_cfl, force_cfl, force);

	// Set point motion
	Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos, 2.25, 0.05);

#ifdef RICH_MPI
	double const cpu_speed = 0.01;
	double const round_speed_ratio = 0.5;
	ConstNumberPerProc3D cpu_move(0.01, 0.5);
#endif
	// Create main simulation object
	HDSim3D sim(tess
#ifdef RICH_MPI
        , cpu_tess
#endif
    , cells, eos, pm, tsf, flux, cu, eu, force, std::pair<std::vector<std::string>, 
		std::vector<std::string> >(std::vector<std::string>(), std::vector<std::string>()), false
#ifdef RICH_MPI
        , &cpu_move, true, 1.4
#endif
        );


    int total_iters = 0;
    double dt = 1e-17;
    tsf.SetTimeStep(dt);
    double old_time = sim.getTime();
    sim.RadiationTimeStep(dt * 0.01, diffusion);
    while(sim.getTime() < 4e-9)
    {
        try
        {
            if(rank == 0)
                std::cout<<"Iteration "<<sim.getCycle()<<" dt "<<sim.getTime() - old_time<<" time "<<sim.getTime()<<std::endl;
            old_time = sim.getTime();
            if(sim.getCycle() % 100 == 0)
                WriteSnapshot3D(sim, "shayhi_"+std::to_string(sim.getCycle())+".h5");          
            //dt = sim.RadiationTimeStep(dt, diffusion, key);
            sim.timeAdvance2();
            if(rank == 0)
                std::cout<<"Done hydro"<<std::endl;
            sim.RadiationTimeStep(sim.getTimeStep(), diffusion);
        }
        catch(UniversalError const& eo)
        {
            reportError(eo);
            if(rank == 0)
            {
#ifdef RICH_MPI
                cpu_tess.output("vproc.bin");
#endif
            }
            throw;
        }
    }
    WriteSnapshot3D(sim, "shay_final.h5");
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}