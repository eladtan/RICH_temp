#include "3D/tessellation/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/3D/output/write3D.hpp"
#include <filesystem>
#include <fenv.h>
namespace fs = std::filesystem;
namespace
{
    /**
 * @brief Function to initialize computational cells for a 3D simulation.
 *
 * This function reads data from external files, calculates the initial conditions for each computational cell,
 * and populates a vector of ComputationalCell3D objects with the calculated values.
 *
 * @param tess A reference to the 3D tessellation object.
 * @param M The mass of the central object.
 * @param R The radius of the central object.
 * @param eos A reference to the equation of state object.
 * @param G The gravitational constant.
 *
 * @return A vector of ComputationalCell3D objects representing the initial conditions for each cell.
 */
std::vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, IdealGas const &eos, double const G)
{
    // Read data from external files
    vector<double> xsi = read_vector("/home/elads/RICH/data/xsi.txt");
    vector<double> theta = read_vector("/home/elads/RICH/data/theta.txt");
    xsi[0] = 0;

    // Constants for the initial conditions
    double n = 1.5;
    double endfactor = 2.714;

    // Calculate initial parameters
    double alpha = R / xsi.back();
    double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
    double K = G * alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));

    // Initialize the result vector
    size_t N = tess.GetPointNo();
    std::vector<ComputationalCell3D> res(N);

    // Loop over each computational cell
    for (size_t i = 0; i < N; ++i)
    {
        Vector3D const &point = tess.GetMeshPoint(i);
        double r = abs(point);
        double t = 0;

        // Calculate initial conditions based on the cell's position
        if (r < R)
        {
            t = LinearInterpolation(xsi, theta, r / alpha);
            res[i].density = std::max(rho_c * std::pow(t, n), 1e-5);
        }
        else
        {
            t = theta.back() * 10;
            res[i].density = rho_c * std::pow(t, n);
        }

        // Calculate pressure and internal energy
        double const P = K * std::pow(res[i].density, 1 + 1.0 / n);
        res[i].pressure = P;
        res[i].internal_energy = eos.dp2e(res[i].density, P, res[i].tracers, ComputationalCell3D::tracerNames);
    }

    // Return the vector of ComputationalCell3D objects
    return res;
}
}

int main(void)
{
// Initialize rank of the current process in a parallel computing environment
int rank = 0;

// Initialize size of the current process in a parallel computing environment
int ws = 1;

// Include MPI-related code only if the RICH_MPI macro is defined
#ifdef RICH_MPI
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Retrieve the rank of the current process in the MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Retrieve the size of the current process in the MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif

	// Enable floating-point exceptions for division by zero, invalid operations, and overflow
feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

// Constants for the central object's radius, mass, and gravitational constant
double const R = 7e10;
double const M = 2e33;
double const G = 6.674e-8;

// Prefix for output file names
std::string file_name = "snap_";

// Constants for the width of the simulation domain and the number of points
const double width = 2 * R;
size_t const np = 1e6;

// Variables for the lower-left and upper-right corners of the simulation domain
Vector3D ll(-width, -width, -width), ur(width, width, width);

// Object representing the Voronoi tesselation for the simulation domain
Voronoi3D tess(ll, ur);

// Declare a vector to store ComputationalCell3D objects
std::vector<ComputationalCell3D> cells;

// Declare a vector to store Vector3D objects representing points in 3D space
vector<Vector3D> points;

// Check if the current process is rank 0
if(rank == 0)
{
    // Generate random points within a spherical region around the origin
    points = RandSphereR(np, ll, ur, 0, R * 1.1);

    // Generate additional random points within smaller spherical regions 
    vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 * R, R * 1.05);
    vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, R, 1.4 * width);

    // Combine the generated points into a single vector
    points.insert(points.end(), ptemp2.begin(), ptemp2.end());
    points.insert(points.end(), ptemp3.begin(), ptemp3.end());
}

// Spread the points across processes in a parallel computing environment
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    std::cout<<"Finished with MPI_Spread Rank "<<rank<<" has "<<points.size()<<" points"<<std::endl;
#endif


// Round the points to a nicer mesh
points = RoundGrid3D(points, ll, ur, 15);

// Print the number of points assigned to the current process
std::cout<<"Rank "<<rank<<" has point no "<<tess.GetPointNo()<<std::endl;
// Check if the RICH_MPI macro is defined to choose between parallel and sequential tesselation building
#ifdef RICH_MPI
    // Build the Voronoi tesselation in parallel using MPI
    tess.BuildParallel(points);
#else
    // Build the Voronoi tesselation sequentially
    tess.Build(points);
#endif

// Print a message indicating that the tesselation building process is finished
if (rank == 0)
    std::cout << "Finished build" << std::endl;
    
    // Create teh EOS
    IdealGas eos(5.0 / 3.0);
    
    // Assign the hydro initial conditions to the computational cells
    cells = GetCells(tess, M, R, eos, G);
	if (rank == 0)
		std::cout << "Finished cells" << std::endl;

    // Create an HLLC Rieman solver 
	Hllc3D rs;

    // Decide how the ghost cells behave
	RigidWallGenerator3D ghost;

    // Create an interpolation object
	LinearGauss3D interp(eos, ghost);

    // Choose how to move the mesh points, using the fluid velocity and then added a "round" correction term
	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos);

    // Choose how to update the computational cells
	DefaultCellUpdater cu;

    // How to calculate the flux, we choose rigid boundaries at the outer edges
	RigidWallFlux3D rigidflux(rs);
	RegularFlux3D *regular_flux = new RegularFlux3D(rs);
	IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
	IsBulkFace3D *bulk_face = new IsBulkFace3D();
	vector<pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>> flux_vector;
	flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>(boundary_face, &rigidflux));
	flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>(bulk_face, regular_flux));
	ConditionActionFlux1 fc(flux_vector, interp);

	vector<pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
	ConditionExtensiveUpdater3D eu(eu_sequence);
	
    // Set a tree code for calculation the self gravity
    bool const use_quadrapole = true;
    double const opening_angle = 0.7;
	GravityAcceleration3D acc(opening_angle, use_quadrapole, G);
    ConservativeForce3D force(acc);

	CourantFriedrichsLewy tsf(0.25, 1, force, std::vector<std::string> (), false);

	HDSim3D sim(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
    double const tf = 5000;
	WriteSnapshot3D(sim, "init.h5");

    // Initialize variables for time tracking and output frequency
    double old_dt = 0, step_time = 0, old_t = 0, output_dt = 500;

#ifdef RICH_MPI
    // Initialize variables for MPI timing
    double step_tstart = MPI_Wtime();
#endif

    // Initialize variable for the next output time
    double nextT = output_dt;

    // Initialize counter for output files
    int counter = 0;

    // Main simulation loop
    while (sim.getTime() < tf)
    {
        // Print time step and run time information for the current process
        if (rank == 0)
        {
            std::cout << std::endl;
            std::cout << "dt " << old_dt << " run time " << step_time - step_tstart << std::endl;
        }

        // Print simulation cycle and time information for the current process
        if (rank == 0)
            std::cout << "Cycle " << sim.getCycle() << " Time " << sim.getTime() << std::endl;

        // Check if it's time to write an output file
        if (sim.getTime() > nextT)
        {
            // Write an output file with the current simulation state
            WriteSnapshot3D(sim, file_name + std::to_string(counter) + ".h5");

            // Update the next output time
            nextT += output_dt;

            // Increment the output file counter
            ++counter;
        }

        // Advance the simulation by one time step
        try
        {
            sim.timeAdvance2();

            // Update time step information
            old_dt = sim.getTime() - old_t;
            old_t = sim.getTime();

#ifdef RICH_MPI
            // Update MPI timing information
            step_tstart = step_time;
            step_time = MPI_Wtime();
#endif
        }
        catch (UniversalError const &eo)
        {
            // Report any errors that occur during the simulation
            reportError(eo);
            throw;
        }
    }
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}

