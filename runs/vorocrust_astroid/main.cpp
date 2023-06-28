#include "source/3D/GeometryCommon/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/3D/GeometryCommon/hdf_write.hpp"
#include <filesystem>
namespace fs = std::filesystem;
#include <sstream>

int main(void)
{
    // READ HERE POINTS
    std::vector<double> in_seeds_x = read_vector("/home/itamarg/workspace/RICH/VoroCrust/output/astroid/dump/zone_in_volume_seeds/x.txt");
    std::vector<double> in_seeds_y = read_vector("/home/itamarg/workspace/RICH/VoroCrust/output/astroid/dump/zone_in_volume_seeds/y.txt");
    std::vector<double> in_seeds_z = read_vector("/home/itamarg/workspace/RICH/VoroCrust/output/astroid/dump/zone_in_volume_seeds/z.txt");

    std::vector<double> out_seeds_x = read_vector("/home/itamarg/workspace/RICH/VoroCrust/output/astroid/dump/zone_out_seeds/x.txt");
    std::vector<double> out_seeds_y = read_vector("/home/itamarg/workspace/RICH/VoroCrust/output/astroid/dump/zone_out_seeds/y.txt");
    std::vector<double> out_seeds_z = read_vector("/home/itamarg/workspace/RICH/VoroCrust/output/astroid/dump/zone_out_seeds/z.txt");

    size_t const out_size = out_seeds_x.size();
    size_t const in_size = in_seeds_x.size();
    size_t const tot_size = out_size + in_size;

    std::vector<Vector3D> points(tot_size, Vector3D()); // fill this vector
    

    // set in seeds
    for(size_t i=0; i<in_size; ++i){
        // std::cout << i << std::setprecision(16) << ", " << in_seeds_x[i] << ", " << in_seeds_y[i] << ", " << in_seeds_z[i] << std::endl;
        points[i].Set(in_seeds_x[i], in_seeds_y[i], in_seeds_z[i]);
    }

    // set out seeds
    for(size_t i=in_size; i<tot_size; ++i){
        // std::cout << i << std::setprecision(16) << ", " << out_seeds_x[i-in_size] << ", " << out_seeds_y[i-in_size] << ", " << out_seeds_z[i-in_size] << std::endl;

        points[i].Set(out_seeds_x[i-in_size], out_seeds_y[i-in_size], out_seeds_z[i-in_size]);
    }

    double ll_x, ll_y, ll_z, ur_x, ur_y, ur_z;
    ll_x = ll_y = ll_z = std::numeric_limits<double>::max();
    ur_x = ur_y = ur_z = -std::numeric_limits<double>::max();
    
    for(size_t i=0; i<tot_size; ++i){
        // std::cout << i << std::endl;
        Vector3D const& p = points[i];
        ll_x = std::min(p.x, ll_x);
        ll_y = std::min(p.y, ll_y);
        ll_z = std::min(p.z, ll_z);

        ur_x = std::max(p.x, ur_x);
        ur_y = std::max(p.y, ur_y);
        ur_z = std::max(p.z, ur_z);
    }


    double const off = 0.1;
    Vector3D ll(ll_x - off, ll_y - off, ll_z - off), ur(ur_x + off, ur_y + off, ur_z + off); // GIVE HERE VALUES
	Voronoi3D tess(ll, ur);
    try{
	    tess.Build(points);
    } catch (UniversalError const& eo){
        std::cout << eo.getErrorMessage() << std::endl;
        exit(1);
    }
  
    ComputationalCell3D init_cell(1, 1, 1, 1, Vector3D());
	std::vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);
    ComputationalCell3D::stickerNames.push_back("Inside");
    
    for(size_t i = 0; i < tot_size; ++i)
    {  
        // std::cout << i << std::endl;
        cells[i].stickers[0] = i < in_size;
    }

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	PCM3D interp(ghost);

	Lagrangian3D pm;
	DefaultCellUpdater cu;

	vector<pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>> flux_vector;
	ConditionActionFlux1 fc(flux_vector, interp);

	vector<pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
	ConditionExtensiveUpdater3D eu(eu_sequence);

    ZeroForce3D force;
	CourantFriedrichsLewy tsf(0.25, 1, force);

    IdealGas eos(5.0 / 3.0);
	HDSim3D sim(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false
		, true);
	vector<DiagnosticAppendix3D *> appendices;
	WriteSnapshot3D(sim, "astroid.h5", appendices, true);

	return 0;
}
