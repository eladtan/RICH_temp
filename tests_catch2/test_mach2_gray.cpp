#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/output/write3D.hpp"
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
#include "source/Radiation/GrayDiffusion/Diffusion.hpp"
#include "source/Radiation/GrayDiffusion/DiffusionForce.hpp"


#include <catch2/catch_test_macros.hpp>
#include "utils_for_tests.hpp"
#include "snapshot.hpp"

using utils_for_tests::named_vector;
using utils_for_tests::extract_data_from_cells;
using utils_for_tests::make_named_vector;
namespace mpi = utils_for_tests::mpi;

namespace
{
   /**
     * @brief Class to determine if a point is to the left or right of a box.
     *
     * This class is used to check if a given point in a 3D simulation is to the left or right of a box.
     * It inherits from the ConditionActionFlux1::Condition3D class.
     */
    class IsPointLeftRightBox3D : public ConditionActionFlux1::Condition3D
    {
    public:
          /**
         * @brief Checks if a given point is to the left or right of the box.
         *
         * @param face_index The index of the face in the tessellation.
         * @param tess The tessellation of the simulation domain.
         * @param cells The computational cells of the simulation.
         *
         * @return A pair of booleans. The first boolean indicates whether the ghost point is to the left or to the right of the box. The second boolean indicates whether the ghost point is to the right of the face or left.
         *
         */
        pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
            const vector<ComputationalCell3D>& cells) const override {
            if (!tess.BoundaryFace(face_index)) return std::pair<bool, bool>(false, false);

            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& first_point = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).first);
            Vector3D const& second_point = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).second);

            const bool is_left = first_point.x < box.first.x || first_point.x > box.second.x;
            const bool is_right = second_point.x < box.first.x || second_point.x > box.second.x;

            return std::make_pair(is_left || is_right, is_right);
        }
    };

    /**
     * @brief Class to choose which ghost generator to use
     *
     * This class is used to determine which ghost cell to use for a given point in the simulation.
     * It inherits from the SeveralGhostGenerator3D::GhostCriteria3D class.
     */

    /**
     * @brief Chooses the ghost cell for a given point in the simulation.
     *
     * @param tess The tessellation of the simulation domain.
     * @param index The index of the point in the tessellation.
     * @return The index of the ghost cell to use for the given point.
     *
     * This method determines which ghost cell to use for a given point in the simulation.
     * It checks the x-coordinate of the point and returns the corresponding ghost cell index.
     * If the x-coordinate is less than the minimum x-coordinate of the domain, it returns 0.
     * If the x-coordinate is greater than the maximum x-coordinate of the domain, it returns 1.
     * Otherwise, it returns 2.
     */
    class GhostChooser : public SeveralGhostGenerator3D::GhostCriteria3D {
    public:
        size_t GhostChoose(Tessellation3D const& tess, size_t index)const
        {
            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& p = tess.GetMeshPoint(index);
            if (p.x < box.first.x) {
                return 0;
            } if (p.x > box.second.x) {
                return 1;
            } else {
                return 2;
            }
        }
    };

static constexpr double ev = 1.602176634e-12;
static constexpr double kev = 1e3*ev;

auto mach2gray(
    double const mass_scale=1., 
    double const length_scale=1., 
    double  const time_scale=1.)
{
    size_t const Np = 256;

    // Set up size of the domain
    double const box_size = 1e3;
    double const dy = 3 * box_size / (2 * Np);
    Vector3D ll(-box_size, -dy, -dy), ur(2 * box_size, dy, dy);
    int rank = 0;
    // Start mpi if needed
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // Create the initial points
    std::vector<Vector3D> points;
    if (rank == 0)
        points = CartesianMesh(Np, 1, 1, ll, ur);
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

    // Create the tesselation
    Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif

    //Set up the EOS
    constexpr double N_avogadro = 6.0221408e+23;
    IdealGas eos(5./3., CG::boltzmann_constant*N_avogadro / (5.0 / 3.0 - 1), 1, 0);

    // Create the hydro data
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);
    ComputationalCell3D left_cell, right_cell;
    left_cell.velocity = Vector3D(2.3547e5, 0, 0);
    left_cell.density = 5.45887e-13;
    left_cell.temperature = 100;
    left_cell.internal_energy = eos.dT2e(left_cell.density, left_cell.temperature, left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.pressure = eos.de2p(left_cell.density, left_cell.internal_energy, left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.Erad = CG::radiation_constant * std::pow(left_cell.temperature, 4) / left_cell.density;

    right_cell.velocity = Vector3D(1.03e5, 0, 0);
    right_cell.density = 1.2479e-12;
    right_cell.temperature = 207.757;
    right_cell.Erad = CG::radiation_constant * std::pow(right_cell.temperature, 4) / right_cell.density;
    right_cell.internal_energy = eos.dT2e(right_cell.density, right_cell.temperature, right_cell.tracers, ComputationalCell3D::tracerNames);
    right_cell.pressure = eos.de2p(right_cell.density, right_cell.internal_energy, right_cell.tracers, ComputationalCell3D::tracerNames);

    for (size_t i = 0; i < Nlocal; ++i)
    {
        if (tess.GetMeshPoint(i).x < 0)
            cells[i] = left_cell;
        else
            cells[i] = right_cell;
    }

    // Rieamann solver
    Hllc3D rs;

    // Hydro boundary conditions
    RigidWallGenerator3D rigid_ghost;
    ConstantPrimitiveGenerator3D left_ghost(left_cell), right_ghost(right_cell);
    std::vector<Ghost3D*> ghost_list = { &left_ghost, &right_ghost, &rigid_ghost };
    GhostChooser ghost_chooser;
    SeveralGhostGenerator3D ghost(ghost_list, ghost_chooser);

    // Spatial Interpolation scheme
    LinearGauss3D interp(eos, ghost);

    // Flux calculator
    std::vector<pair<
        const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*> > sequence;

    // The order of inputing values in the sequence is important.
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
    ConditionActionFlux1::Condition3D* is_side = new IsPointLeftRightBox3D();
    ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
    ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);

    sequence.push_back(
        std::pair<
            const ConditionActionFlux1::Condition3D*,
            const ConditionActionFlux1::Action3D*>(is_side, normal_flux)
    );

    sequence.push_back(
        std::pair<
            const ConditionActionFlux1::Condition3D*,
            const ConditionActionFlux1::Action3D*>(isboundary, rigid_flux)
    );

    sequence.push_back(
        std::pair<
            const ConditionActionFlux1::Condition3D*,
            const ConditionActionFlux1::Action3D*>(isbulk, normal_flux)
    );

    ConditionActionFlux1 flux(sequence, interp);

    // Extensive updater
    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    // The diffusion class
    PowerLawOpacity opacity(CG::speed_of_light / (3 * 0.848902), 0, 0, 3.93e-5, 0, 0);
    
    DiffusionXInflowBoundary diffusion_boundary(left_cell, right_cell, opacity);
    
    constexpr bool flux_limiter = true;
	constexpr bool hydro_on = true;
	constexpr bool compton_on = false;
	constexpr bool doppler_on = true;

    Diffusion diffusion(
        opacity, 
        diffusion_boundary, 
        eos, 
        std::vector<std::string>(), 
		flux_limiter, 
		hydro_on, 
		compton_on);
    
    diffusion.length_scale_ = length_scale;
	diffusion.time_scale_ = time_scale;
	diffusion.mass_scale_ = mass_scale;

    // Primitive updater
    DefaultCellUpdater cu(false, 0.0, true, 0.0, &diffusion);

    // Time step function
    double const hydro_cfl = 0.3;
    double const force_cfl = 1;
    DiffusionForce force = DiffusionForce(diffusion, eos);

    double current_dt = 1e-11;
    UserDeterminedTimeStep tsf{current_dt};

    // Set point motion
    Eulerian3D pm;


    // Create main simulation object
    HDSim3D sim(tess, cells, eos, pm, tsf, flux, cu, eu, force, std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    double old_time = sim.getTime();
    while (sim.getTime() < 0.02)
    {
        try
        {
            // Print to screen run time
            if (rank == 0) {
                std::cout<<std::endl;
                std::cout<<"Iteration "<<sim.getCycle()<<" dt "<<current_dt<<" time "<<sim.getTime()<<std::endl;
            }
            // Do radiation transfer step
            double _ = sim.RadiationTimeStep(current_dt, diffusion);
            double const new_dt = std::min(current_dt*1.05, 6e-6);
            tsf.SetTimeStep(new_dt);

            // Main time step       
            old_time = sim.getTime();
            sim.timeAdvance2();
            current_dt = sim.getTime() - old_time;
        } catch (UniversalError const& eo) {
            reportError(eo);
            throw;
        }
    }

    std::vector<ComputationalCell3D> cells_end_sim = sim.getCells();
	cells_end_sim.resize(sim.getTesselation().GetPointNo());

    auto time_output = make_named_vector<double>("time", {}); 
    if(mpi::get_mpi_rank() == mpi::rank_root) time_output.vec.push_back(sim.getTime());

    using utils_for_tests::extract_data_from_cells;
    auto [CMx, CMy, CMz] = utils_for_tests::extract_center_of_mass(sim.getTesselation());
    auto [vx, vy, vz] = extract_data_from_cells("velocity", cells_end_sim, &ComputationalCell3D::velocity);

    return std::make_tuple(
        extract_data_from_cells("ID", cells_end_sim, &ComputationalCell3D::ID), 
		extract_data_from_cells("temperature", cells_end_sim, &ComputationalCell3D::temperature),
		extract_data_from_cells("density", cells_end_sim, &ComputationalCell3D::density),
		extract_data_from_cells("Erad", cells_end_sim, &ComputationalCell3D::Erad),
		CMx, CMy, CMz,
        vx, vy, vz,
        time_output
    );
}

} // namespace

TEST_CASE_METHOD(mpi::RichMpiFixture, "Mach2_Gray", "[mach2][gray_diffusion]"){
    snapshot::SnapShot snap;

    auto const success = std::apply(
		[&snap](auto const& ID, auto const&... run_info){
			return snap.CompareOrSaveGather(
				std::nullopt,
				ID,
				run_info...
			);
		},
		mach2gray()
	);

	REQUIRE(success);
}


TEST_CASE_METHOD(mpi::RichMpiFixture, "Mach2_Gray_different_scales", "[mach2][gray_diffusion][scales]"){
    snapshot::SnapShot snap;

    auto const success = std::apply(
		[&snap](auto const& ID, auto const&... run_info){
			return snap.CompareOrSaveGather(
				std::nullopt,
				ID,
				run_info...
			);
		},
		mach2gray(4., 5., 6.)
	);

	REQUIRE(success);
}
