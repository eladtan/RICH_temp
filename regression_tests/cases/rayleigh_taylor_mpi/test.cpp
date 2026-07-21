#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"

static const double GAMMA = 5.0 / 3.0;
static const double RHO_LIGHT = 1.0;
static const double RHO_HEAVY = 2.0;
static const double GRAVITY = 0.5;
static const double Z_MID = 1.0;
static const double P0 = 10.0;
static const double PERTURBATION_AMP = 0.03;
static const double T_END = 3.0;
static const int LOG_INTERVAL = 10;

int main(void)
{
    Vector3D ll(0, 0, 0), ur(1, 1, 2);
    int rank = 0;
    int world_size = 1;
#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    std::vector<Vector3D> points;
    if (rank == 0) {
        points = CartesianMesh(80, 80, 156, ll, ur);
    }
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

    Voronoi3D tess(ll, ur);
    try {
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif
    } catch (UniversalError const& eo) {
        reportError(eo);
        throw;
    }

    IdealGas eos(GAMMA);
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);

    for (size_t i = 0; i < Nlocal; ++i) {
        Vector3D const& pos = tess.GetCellCM(i);
        double const x = pos.x;
        double const y = pos.y;
        double const z = pos.z;

        double rho, pressure;
        if (z < Z_MID) {
            rho = RHO_LIGHT;
            pressure = P0 - RHO_LIGHT * GRAVITY * z;
        } else {
            rho = RHO_HEAVY;
            double p_at_interface = P0 - RHO_LIGHT * GRAVITY * Z_MID;
            pressure = p_at_interface - RHO_HEAVY * GRAVITY * (z - Z_MID);
        }

        double vz_pert = PERTURBATION_AMP
            * (std::cos(2.0 * M_PI * x) + std::cos(2.0 * M_PI * y))
            * std::exp(-((z - Z_MID) * (z - Z_MID)) / 0.04);

        cells[i].density = rho;
        cells[i].pressure = pressure;
        cells[i].internal_energy = pressure / (rho * (GAMMA - 1.0));
        cells[i].velocity = Vector3D(0, 0, vz_pert);
    }

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);

    std::vector<std::pair<const ConditionActionFlux1::Condition3D*,
                          const ConditionActionFlux1::Action3D*>> sequence;
    ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
    ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
    sequence.push_back(std::make_pair(isboundary, rigid_flux));
    sequence.push_back(std::make_pair(isbulk, normal_flux));
    ConditionActionFlux1 flux(sequence, interp);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
                          const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);
    DefaultCellUpdater cu;

    ConstantAcceleration3D acc(Vector3D(0, 0, -GRAVITY));
    ConservativeForce3D force(acc, false);
    CourantFriedrichsLewy tsf(0.3, 1.0, force);

    Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos);

    HDSim3D sim(tess, cells, eos, pm, tsf, flux, cu, eu, force,
                std::make_pair(ComputationalCell3D::tracerNames,
                               ComputationalCell3D::stickerNames));

    std::ofstream ek_file;
    if (rank == 0) {
        ek_file.open("rt_kinetic_energy.txt");
    }

    double old_time = sim.getTime();
    while (sim.getTime() < T_END) {
        try {
            if (sim.getCycle() % LOG_INTERVAL == 0) {
                double local_ekz = 0.0;
                size_t const Nloc = sim.getTesselation().GetPointNo();
                for (size_t i = 0; i < Nloc; ++i) {
                    double vz = sim.getCells()[i].velocity.z;
                    double rho = sim.getCells()[i].density;
                    double vol = sim.getTesselation().GetVolume(i);
                    local_ekz += 0.5 * rho * vol * vz * vz;
                }
                double global_ekz = local_ekz;
#ifdef RICH_MPI
                MPI_Allreduce(&local_ekz, &global_ekz, 1, MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD);
#endif
                if (rank == 0) {
                    ek_file << sim.getTime() << " " << global_ekz << "\n";
                    ek_file.flush();
                }
            }

            if (rank == 0) {
                std::cout << "Cycle " << sim.getCycle()
                          << " dt " << sim.getTime() - old_time
                          << " time " << sim.getTime()
                          << " T_end " << T_END << "\n" << std::endl;
            }
            old_time = sim.getTime();
            sim.timeAdvance2();
        } catch (UniversalError const& eo) {
            reportError(eo);
            throw;
        }
    }

    if (rank == 0) {
        ek_file.close();
    }

    // Density slice: cells near y = 0.5
    double const slice_half_width = 0.8 * (ur.y - ll.y) / 80.0;
    size_t const Nfinal = sim.getTesselation().GetPointNo();
    std::vector<double> local_x, local_z, local_rho;
    for (size_t i = 0; i < Nfinal; ++i) {
        Vector3D const cm = sim.getTesselation().GetCellCM(i);
        if (std::abs(cm.y - 0.5) < slice_half_width) {
            local_x.push_back(cm.x);
            local_z.push_back(cm.z);
            local_rho.push_back(sim.getCells()[i].density);
        }
    }

#ifdef RICH_MPI
    int local_count = static_cast<int>(local_x.size());
    std::vector<int> all_counts(world_size, 0);
    MPI_Gather(&local_count, 1, MPI_INT,
               all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(world_size, 0);
    int total_count = 0;
    if (rank == 0) {
        for (int r = 0; r < world_size; ++r) {
            displacements[r] = total_count;
            total_count += all_counts[r];
        }
    }

    std::vector<double> all_x, all_z, all_rho;
    if (rank == 0) {
        all_x.resize(total_count);
        all_z.resize(total_count);
        all_rho.resize(total_count);
    }
    MPI_Gatherv(local_x.data(), local_count, MPI_DOUBLE,
                all_x.data(), all_counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_z.data(), local_count, MPI_DOUBLE,
                all_z.data(), all_counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_rho.data(), local_count, MPI_DOUBLE,
                all_rho.data(), all_counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream slice_out("rt_density_slice.txt");
        for (int i = 0; i < total_count; ++i) {
            slice_out << all_x[i] << " " << all_z[i] << " "
                      << all_rho[i] << "\n";
        }
        slice_out.close();
    }
#else
    std::ofstream slice_out("rt_density_slice.txt");
    for (size_t i = 0; i < local_x.size(); ++i) {
        slice_out << local_x[i] << " " << local_z[i] << " "
                  << local_rho[i] << "\n";
    }
    slice_out.close();
#endif

    WriteSnapshot3D(sim, "rt_final.h5");

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
