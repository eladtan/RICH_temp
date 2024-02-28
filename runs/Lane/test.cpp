#include "source/3D/tesselation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/common/TillotsonOrg.hpp"
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
#include "source/3D/output/hdf_write.hpp"
#include <filesystem>
#include <fenv.h>
namespace fs = std::filesystem;
namespace
{
    class Ax: public DiagnosticAppendix3D
    {
    public:
        std::vector<double> data;
        /*! \brief Calculates additional data
        \param sim Hydrodynamic simulation
        \return Calculated data
        */
        virtual vector<double> operator()(const HDSim3D& /*sim*/) const{return data;}

        virtual string getName(void) const {return "Ax";}
    };

    class Ay: public DiagnosticAppendix3D
    {
    public:
        std::vector<double> data;
        /*! \brief Calculates additional data
        \param sim Hydrodynamic simulation
        \return Calculated data
        */
        virtual vector<double> operator()(const HDSim3D& /*sim*/) const{return data;}

        virtual string getName(void) const {return "Ay";}
    };

    class Az: public DiagnosticAppendix3D
    {
    public:
        std::vector<double> data;
        /*! \brief Calculates additional data
        \param sim Hydrodynamic simulation
        \return Calculated data
        */
        virtual vector<double> operator()(const HDSim3D& /*sim*/) const{return data;}

        virtual string getName(void) const {return "Az";}
    };

    std::vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, IdealGas const &eos, double const G)
    {
        vector<double> xsi = read_vector("/home/esternberg/RICH/data/xsi32.txt");
        vector<double> theta = read_vector("/home/esternberg/RICH/data/theta32.txt");
        xsi[0] = 0;

        double n = 1.5;
        double endfactor = 2.714;

        double alpha = R / xsi.back();
        double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
        double K = G * alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));

        size_t N = tess.GetPointNo();
        std::vector<ComputationalCell3D> res(N);
        for (size_t i = 0; i < N; ++i)
        {
            Vector3D const &point = tess.GetMeshPoint(i);
            double r = abs(point);
            double t = 0;
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
            double const P = K * std::pow(res[i].density, 1 + 1.0 / n);
            res[i].pressure = P;
            res[i].internal_energy = eos.dp2e(res[i].density, P, res[i].tracers, ComputationalCell3D::tracerNames);
        }
        return res;
    }
}

int main(void)
{
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	double const R = 7e10;
	double const M = 2e33;
	double const G = 6.674e-8;

	std::string file_name = "snap2_";
	
	const double width = 2 * 7e10;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
	Voronoi3D tproc(ll, ur);
#endif
	int counter = 0;
	std::vector<ComputationalCell3D> cells;
#ifdef RICH_MPI
    vector<Vector3D> procpoints = RoundGrid3DSingle(RandSphereR2(ws, ll, ur, 0, width), ll, ur);
    tproc.Build(procpoints);
#endif


    size_t const np = 1e6;
    vector<Vector3D> ptemp = RandSphereR(np, ll, ur, 0, R * 1.1, Vector3D()
#ifdef RICH_MPI
    , &tproc
#endif
    );
    vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 * R, R * 1.05, Vector3D()
#ifdef RICH_MPI
    , &tproc
#endif
    );
    vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, R, 1.4 * width, Vector3D()
#ifdef RICH_MPI
    , &tproc
#endif
    );
    ptemp.insert(ptemp.end(), ptemp2.begin(), ptemp2.end());
    ptemp.insert(ptemp.end(), ptemp3.begin(), ptemp3.end());
    vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15
#ifdef RICH_MPI
    , &tess
#endif
    );
	points = tess.getMeshPoints();
	points.resize(tess.GetPointNo());
	std::cout<<"Rank "<<rank<<" point no "<<tess.GetPointNo()<<std::endl;
    tess.BuildHilbert(points);
	if (rank == 0)
		std::cout << "Finished build" << std::endl;
    IdealGas eos(5.0 / 3.0);
    cells = GetCells(tess, M, R, eos, G);
	if (rank == 0)
		std::cout << "Finished cells" << std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.75, 0.03);

	DefaultCellUpdater cu;

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
	GravityAcceleration3D acc(0.7, true, G);
    ConservativeForce3D force(acc, false);

	CourantFriedrichsLewy tsf(0.25, 1, force, std::vector<std::string> (), false);
	std::unique_ptr<HDSim3D> sim;
	sim = std::make_unique<HDSim3D>(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, true);
    double const tf = 5000;
    std::vector<Vector3D> acc_result;
    acc(tess, sim->getCells(), std::vector<Conserved3D>(), 0, acc_result);
    Az az;
    Ax ax;
    Ay ay;
    std::vector<DiagnosticAppendix3D*> diag;
    size_t const N = tess.GetPointNo();
    for(size_t i = 0; i < N; ++i)
    {
        ax.data.push_back(acc_result[i].x);
        ay.data.push_back(acc_result[i].y);
        az.data.push_back(acc_result[i].z);
    }
    diag.push_back(&ax);
    diag.push_back(&ay);
    diag.push_back(&az);
	WriteSnapshot3D(*sim, "init.h5", diag);
//     double old_dt = 0, step_time = 0, old_t = 0;
// #ifdef RICH_MPI
//     double step_tstart = MPI_Wtime();
// #endif
//     double nextT = 100;
// 	while (sim->getTime() < tf)
// 	{
// 			if (rank == 0)
// 			{
// 				std::cout<<std::endl;
// 				std::cout << "dt " << old_dt << " run time " << step_time - step_tstart << std::endl;
// 			}
// 			if (rank == 0)
// 				std::cout << "Cycle " << sim->getCycle() << " Time " << sim->getTime() << std::endl;
// 		if (sim->getTime() > nextT)
// 		{
// 			WriteSnapshot3D(*sim, file_name + std::to_string(counter) + ".h5");
// 			nextT += 100;
// 			++counter;
// 		}
// 		try
// 		{
// 			sim->timeAdvance2();
// 			old_dt = sim->getTime() - old_t;
// 			old_t = sim->getTime();
// #ifdef RICH_MPI
// 			step_tstart = step_time;
// 			step_time = MPI_Wtime();
// #endif
// 		}
// 		catch (UniversalError const &eo)
// 		{
// 			reportError(eo);
// 			throw;
// 		}
// 	}
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}

