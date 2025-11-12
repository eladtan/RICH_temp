

#include "source/misc/mesh_generator.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/common/hllc.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/two_dimensional/condition_action_sequence_2.hpp"
#include "source/newtonian/two_dimensional/diagnostics.hpp"
#include "source/newtonian/two_dimensional/geometric_outer_boundaries/SquareBox.hpp"
#include "source/newtonian/two_dimensional/ghost_point_generators/FreeFlowGhostGenerator.hpp"
#include "source/newtonian/two_dimensional/ghost_point_generators/RigidWallGenerator.hpp"
#include "source/newtonian/two_dimensional/hdf5_diagnostics.hpp"
#include "source/newtonian/two_dimensional/hdsim2d.hpp"
#include "source/newtonian/two_dimensional/interpolations/LinearGaussImproved.hpp"
#include "source/newtonian/two_dimensional/point_motions/eulerian.hpp"
#include "source/newtonian/two_dimensional/point_motions/lagrangian.hpp"
#include "source/newtonian/two_dimensional/point_motions/round_cells.hpp"
#include "source/newtonian/two_dimensional/simple_cell_updater.hpp"
#include "source/newtonian/two_dimensional/simple_extensive_updater.hpp"
// #include "source/newtonian/two_dimensional/source_terms/CenterGravity.hpp"
#include "source/newtonian/two_dimensional/source_terms/cylindrical_complementary.hpp"
#include "source/newtonian/two_dimensional/source_terms/zero_force.hpp"
#include "source/newtonian/two_dimensional/spatial_distributions/uniform2d.hpp"
#include "source/newtonian/two_dimensional/stationary_box.hpp"
#include "source/tessellation/RoundGrid.hpp"
#include "source/tessellation/VoronoiMesh.hpp"
#include "source/tessellation/geometry.hpp"
#include "source/tessellation/tessellation.hpp"

class CenterGravWithCentrifugal : public Acceleration {
   public:
    CenterGravWithCentrifugal(double M, double Rmin) : M_(M), Rmin_(Rmin) {};

    Vector2D operator()(const Tessellation& tess, const vector<ComputationalCell>& cells,
                        const vector<Extensive>& fluxes, const double time, const int point) const override {
        const Vector2D pos(tess.GetCellCM(point));
        const double r = abs(pos);
        const double r3 = r * r * r + Rmin_ * Rmin_ * Rmin_;
        return (-M_ / r3) * Vector2D(0, pos.y);
    }

   private:
    const double M_;
    const double Rmin_;
};

int main(void) {
    size_t Ntot = 10000;

    double Ly = 10;
    double Lz = 10;
    double r_min = 1;
    double softening_len = r_min;
    double r_max = Lz;
    double gap_width = Ly / 5;
    double center_y = 0;
    // double gap_start = center_y - gap_width / 2;
    // double gap_end = center_y + gap_width / 2;
    double tend = 20;

    // Set up size of the domain
    Vector2D ll(r_min, 0), ur(Ly, Lz);
    // Vector2D ll(-Ly, -Lz), ur(Ly, Lz);
    int rank = 0;

    // Create the initial points
    std::vector<Vector2D> points;

    points = RandSquare(Ntot, ll, ur);
    // points = CirclePointsRmax_1(Ntot, r_min, r_max, 0, 0, Lz, Lz, 0, 0);

    std::cout << "Done generating points" << std::endl;
    SquareBox outer_boundary(ll, ur);

    // Make the points roundish
    // points = RoundGridV(points, outer_boundary);

    // std::cout << "Done round" << std::endl;

    // Create the tesselation
    VoronoiMesh tess;
    tess.Initialise(points, outer_boundary);

    // tess.Build(points);

    CylindricalSymmetry geometry(Vector2D(0., 0.), Vector2D(0., 1.));
    // SlabSymmetry geometry;

    double gamma = 5. / 3.;
    // Set up the EOS
    IdealGas eos(gamma);

    // Create the hydro data
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell> cells(Nlocal);

    double M_bh = 1;

    double rho_floor = 1e-10;

    for (size_t i = 0; i < Nlocal; ++i) {
        double const r = tess.GetMeshPoint(i).x;
        double const z = tess.GetMeshPoint(i).y;
        double const R = std::sqrt(r * r + z * z);

        double const Omega = sqrt(M_bh / (R * R * R));
        double const h = 1e-1 * std::pow(r, 1. / 8);
        double const H = r * h;
        double const cs = Omega * H;
        double const rho0 = 1 * std::pow(r, -15. / 8);

        cells[i].density = std::max(rho0 * std::exp(-z * z / (2 * H * H)), rho_floor);
        cells[i].pressure = cs * cs * cells[i].density;
        cells[i].velocity = Vector2D(0, 0);

        std::cout << "R: " << R << " Omega: " << Omega << " h: " << h << " H: " << H << " cs: " << cs << ' '
                  << cells[i].density << ' ' << cells[i].pressure << "\n";

        /*cells[i].density = 1;
        cells[i].pressure = 1;
        cells[i].velocity = Vector2D(0, 0);*/
    }

    // Rieamann solver
    Hllc rs;

    // Hydro boundary conditions
    // RigidWallGenerator ghost;
    FreeFlowGenerator ghost;

    // Spatial Interpolation scheme
    LinearGaussImproved interp(eos, ghost);

    // Flux calculator
    /*std::vector<pair<const ConditionActionSequence::Condition*, const ConditionActionSequence::Action*> > sequence;
    ConditionActionSequence::Condition* isbulk = new IsBulkEdge();
    ConditionActionSequence::Condition* isboundary = new IsBoundaryEdge();
    ConditionActionSequence::Action2* normal_flux = new RegularFlux2(rs);
    ConditionActionSequence::Action2* rigid_flux = new RigidWallFlux2(rs);
    sequence.push_back(std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence::Action2*>(
        isboundary, rigid_flux));
    sequence.push_back(std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence::Action2*>(
        isbulk, normal_flux));
    ConditionActionSequence flux(sequence, interp);*/

    std::vector<std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence::Action*> >
        first_order;
    std::vector<std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence2::Action2*> >
        second_order;
    RegularFlux2 reg_flux(rs);
    RigidWallFlux2 rigid_flux(rs);
    second_order.push_back(
        std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence2::Action2*>(
            new IsBoundaryEdge(), &rigid_flux));
    second_order.push_back(
        std::pair<const ConditionActionSequence::Condition*, const ConditionActionSequence2::Action2*>(new IsBulkEdge(),
                                                                                                       &reg_flux));
    ConditionActionSequence2 flux(first_order, second_order, interp);

    // Extensive updater
    /*std::vector<std::pair<const ConditionExtensiveUpdater2D::Condition*, const ConditionExtensiveUpdater2D::Action*> >
        eu_sequence;
    ConditionExtensiveUpdater2D eu(eu_sequence);*/
    SimpleExtensiveUpdater eu;

    // Primitive updater
    // DefaultCellUpdater cu;
    SimpleCellUpdater cu;

    // External force

    CenterGravWithCentrifugal acc(M_bh, softening_len);
    ConservativeForce force(acc);
    // ZeroForce force;

    // Time step function
    // double const hydro_cfl = 0.3;
    // double const force_cfl = 1;
    // CourantFriedrichsLewy tsf(hydro_cfl, force_cfl, force);
    SimpleCFL tsf(0.3);

    // Set point motion
    Lagrangian bpm;
    RoundCells pm(bpm, eos);

    StationaryBox interface_velocity_calc;

    // Create main simulation object
    // hdsim sim(tess, cells, eos, pm, tsf, flux, cu, eu, force,
    //           std::make_pair(ComputationalCell::tracerNames, ComputationalCell::stickerNames));
    hdsim sim(tess, outer_boundary, geometry, cells, eos, pm, interface_velocity_calc, force, tsf, flux, eu, cu);

    double old_time = sim.getTime();
    while (sim.getTime() < tend) {
        try {
            // Print to screen run time
            old_time = sim.getTime();
            // Write to file every 100 time steps
            if (sim.getCycle() % 100 == 0) {
                std::cout << "Iteration " << sim.getCycle() << " dt " << sim.getTime() - old_time << " time "
                          << sim.getTime() << "\n";
                write_snapshot_to_hdf5(sim, "gap" + std::to_string(int(sim.getCycle() / 100)) + ".h5", {}, false, true);
            }
            // Main time step
            sim.TimeAdvance();
        } catch (UniversalError const& eo) {
            reportError(eo);
            throw;
        }
    }
    write_snapshot_to_hdf5(sim, "gap_final.h5");
    return 0;
}
