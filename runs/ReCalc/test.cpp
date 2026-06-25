#include "3D/tessellation/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/output/read3D.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/misc/int2str.hpp"
#include <boost/numeric/odeint.hpp>
#include "source/newtonian/three_dimensional/LagrangianExtensiveUpdater3D.hpp"
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <exception>
#include <fenv.h>
#include <filesystem>
#include "source/3D/GeometryCommon/UpdateBox.hpp"
namespace fs = std::filesystem;
#include <sys/stat.h>
#include <boost/math/tools/roots.hpp>
#include <sstream>
#include <MeshDecomposer3D/kernels/Rectangle.hpp>
#include "source/newtonian/three_dimensional/Dissipation.hpp"

typedef std::array<double, 4> state_type;

#define smooth_factor 0.6
// #define hi_res 1
namespace
{
	class DissipationDiag: public DiagnosticAppendix3D
	{
		private:
			Dissipation const& dissipation_;
		public:
		DissipationDiag(Dissipation const& dissipation):dissipation_(dissipation){}

		std::vector<double> operator()(const HDSim3D& sim) const
		{
			int rank = 0;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if(rank == 0)
				std::cout<<"Starting dissipation diag"<<std::endl;
			return dissipation_.CalcDissipation(sim);
		}

		std::string getName(void) const
		{
			return std::string("Dissipation");
		}
	};

	class GradDiag: public DiagnosticAppendix3D
	{
		private:
			LinearGauss3D &interp_;
			size_t const direction_, value_;
			bool const limited_;
		public:
		GradDiag(size_t const direction, size_t const value, LinearGauss3D &interp, bool limited): direction_(direction), value_(value), interp_(interp), limited_(limited){}

		std::vector<double> operator()(const HDSim3D& sim) const
		{
		    std::vector<Slope3D> slopes = limited_ ? interp_.GetSlopes() : interp_.GetSlopesUnlimited();
			size_t const N = sim.getTesselation().GetPointNo();
			std::vector<double> res(N, 0);
			switch(value_)
			{
				case 0:
					switch(direction_)
					{
						case 0:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].xderivative.internal_energy;
							break;
						case 1:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].yderivative.internal_energy;
							break;
						case 2:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].zderivative.internal_energy;
							break;
					}
					break;
				case 1:
					switch(direction_)
					{
						case 0:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].xderivative.pressure;
							break;
						case 1:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].yderivative.pressure;
							break;
						case 2:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].zderivative.pressure;
							break;
					}
					break;
				case 2:
					switch(direction_)
					{
						case 0:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].xderivative.density;
							break;
						case 1:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].yderivative.density;
							break;
						case 2:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].zderivative.density;
							break;
					}
					break;
				case 3:
					for(size_t i = 0; i < N; ++i)
						res[i] = slopes[i].xderivative.velocity.x + slopes[i].yderivative.velocity.y + slopes[i].zderivative.velocity.z;
					break;
			}
			return res;
		}

		std::string getName(void) const
		{
			switch(value_)
			{
				case 0:
					switch(direction_)
					{
						case 0:
							return std::string("DsieDx") + (limited_ ? "Limited" : "");
							break;
						case 1:
							return std::string("DsieDy") + (limited_ ? "Limited" : "");
							break;
						case 2:
							return std::string("DsieDz") + (limited_ ? "Limited" : "");
							break;
					}
					break;
				case 1:
					switch(direction_)
					{
						case 0:
							return std::string("DpDx") + (limited_ ? "Limited" : "");
							break;
						case 1:
							return std::string("DpDy") + (limited_ ? "Limited" : "");
							break;
						case 2:
							return std::string("DpDz") + (limited_ ? "Limited" : "");
							break;
					}
					break;
				case 2:
					switch(direction_)
					{
						case 0:
							return std::string("DrhoDx") + (limited_ ? "Limited" : "");
							break;
						case 1:
							return std::string("DrhoDy") + (limited_ ? "Limited" : "");
							break;
						case 2:
							return std::string("DrhoDz") + (limited_ ? "Limited" : "");
							break;
					}
					break;
				case 3:
					return std::string("divV") + (limited_ ? "Limited" : "");
					break;
			}
		}
	};
}

int main(int argc, char* argv[])
{
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	if(argc < 3)
		std::cout<<"Error, too few input arguments"<<std::endl;
	std::string in_name(argv[1]);
	std::string out_name(argv[2]);

	std::string eos_location("/home/esternberg/RICH/data/EOS/"); // CHANGE LOCATION HERE

	std::unique_ptr<EquationOfState> eos_ptr;
	bool const TDE_sim = true; // CHANGE HERE AS NEEDED
	if(TDE_sim)
	{
		double const dmin_eos = -50.656872045869001;
		double const dmax_eos = 6.815166376138933;
		double const dd_eos = 0.095310179804322;
		double const lscale = 7e10;
		double const mscale = 2e33;
		double const tscale = 1603;
		eos_ptr = std::make_unique<OndrejEOS>(dmin_eos, dmax_eos, dd_eos, eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
	}
	else
		eos_ptr = std::make_unique<IdealGas>(5.0 / 3.0, 1, 1, 0);
	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);

	vector<ComputationalCell3D> cells;
	Snapshot3D snap;
	int hdf5_rank = -1;
	int NranksInFile = GetNumberOfRanksInHDF(in_name);
	if(rank == 0)
		std::cout<<"Reading from file "<<in_name<<" file has "<<NranksInFile<<" ranks"<<std::endl;
	snap = ReadSnapshot3D(in_name, true, hdf5_rank);
	if(ws < NranksInFile && rank == 0)
	{
		for(int j = ws; j < NranksInFile; ++j)
		{
			Snapshot3D snap_temp = ReadSnapshot3D(in_name, true, j);
			snap.cells.insert(snap.cells.end(), snap_temp.cells.begin(), snap_temp.cells.end());
			snap.mesh_points.insert(snap.mesh_points.end(), snap_temp.mesh_points.begin(), snap_temp.mesh_points.end());
		}
	}
	ll = snap.ll;
	ur = snap.ur;
	if(rank == 0)
		std::cout<<"Box is ll="<<ll<<" ur="<<ur<<std::endl;
	tess.SetBox(snap.ll, snap.ur);
	tess.BuildParallel(snap.mesh_points);
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess, snap.cells, false, &cdummy);
	cells = snap.cells;
	ComputationalCell3D::tracerNames = snap.tracerstickernames.first;
	if (rank == 0)
		std::cout << "Finished build" << std::endl;


	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(*eos_ptr.get(), ghost, true, 0.2, 0.25, 0.75);
	double Tmin = 1e3;

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, *eos_ptr.get(), 3, 0.005, false, 1.25);
	DefaultCellUpdater cu(false, 0, true);

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
	ZeroForce3D force;
	CourantFriedrichsLewy tsf(0.325, 1, force, std::vector<std::string> (),	false);

	std::unique_ptr<HDSim3D> sim;
	sim = std::make_unique<HDSim3D>(tess, cells, *eos_ptr.get(), pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, false);
	sim->SetTime(snap.time);
	sim->SetCycle(snap.cycle);
	try
	{
		std::vector<pair<ComputationalCell3D, ComputationalCell3D> >  interp_values;
		interp(tess, sim->getCells(), 0, interp_values);;
		vector<DiagnosticAppendix3D *> appendices;
		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 3; ++j)
			{
				appendices.push_back(new GradDiag(i, j, interp, false));
				appendices.push_back(new GradDiag(i, j, interp, true));
			}
		}
		appendices.push_back(new GradDiag(0, 3, interp, false));
		appendices.push_back(new GradDiag(0, 3, interp, true));
		Dissipation dissipation(rs, *eos_ptr.get());
		dissipation.face_values = interp_values;
		DissipationDiag DissDiag(dissipation);
		appendices.push_back(&DissDiag);
		
		if(rank == 0)
			std::cout<<"Outputing new file "<<out_name<<std::endl;
		WriteSnapshot3D(*sim, out_name, appendices, true, false);
	}
	catch(UniversalError const& eo)
	{
		reportError(eo);
		throw;
	}
	MPI_Finalize();
	return 0;
}
