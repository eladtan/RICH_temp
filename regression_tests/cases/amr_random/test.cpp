#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"

namespace
{
std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b, std::uint64_t c)
{
    std::uint64_t x = a + 0x9e3779b97f4a7c15ULL;
    x ^= b + 0xbf58476d1ce4e5b9ULL + (x << 6U) + (x >> 2U);
    x ^= c + 0x94d049bb133111ebULL + (x << 7U) + (x >> 3U);
    return x;
}

double rel_diff(double value, double reference)
{
    const double den = std::max(std::abs(reference), 1e-30);
    return std::abs(value - reference) / den;
}

class RandomRefine3D : public CellsToRefine3D
{
public:
    RandomRefine3D(double probability, int rank):
        probability_(probability),
        rank_(rank),
        iter_(0),
        total_refined_(0) {}

    std::pair<std::vector<size_t>, std::vector<Vector3D> > ToRefine(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const&,
        double) const override
    {
        std::mt19937_64 gen(mix_seed(0xA1B2C3D4ULL, static_cast<std::uint64_t>(rank_), iter_));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<size_t> refine;
        refine.reserve(tess.GetPointNo() / 50);
        const size_t n = tess.GetPointNo();
        for(size_t i = 0; i < n; ++i) {
            if(abs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (0.2 * tess.GetWidth(i))) {
                continue;
            }
            if(dist(gen) < probability_) {
                refine.push_back(i);
            }
        }
        std::sort(refine.begin(), refine.end());
        refine.erase(std::unique(refine.begin(), refine.end()), refine.end());
        total_refined_ += refine.size();
        ++iter_;
        return std::make_pair(refine, std::vector<Vector3D>());
    }

    size_t getTotalRefined() const { return total_refined_; }

private:
    const double probability_;
    const int rank_;
    mutable std::uint64_t iter_;
    mutable size_t total_refined_;
};

class RandomRemove3D : public CellsToRemove3D
{
public:
    RandomRemove3D(double probability, int rank):
        probability_(probability),
        rank_(rank),
        iter_(0),
        total_removed_(0) {}

    std::pair<std::vector<size_t>, std::vector<double> > ToRemove(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const&,
        double) const override
    {
        std::mt19937_64 gen(mix_seed(0xF1E2D3C4ULL, static_cast<std::uint64_t>(rank_), iter_));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<size_t> remove;
        std::vector<double> merits;
        remove.reserve(tess.GetPointNo() / 50);
        merits.reserve(tess.GetPointNo() / 50);
        const size_t n = tess.GetPointNo();
        for(size_t i = 0; i < n; ++i) {
            if(abs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (0.2 * tess.GetWidth(i))) {
                continue;
            }
            if(dist(gen) < probability_) {
                remove.push_back(i);
                merits.push_back(dist(gen));
            }
        }
        if(!remove.empty()) {
            std::vector<std::pair<size_t, double> > zipped;
            zipped.reserve(remove.size());
            for(size_t i = 0; i < remove.size(); ++i) {
                zipped.push_back(std::make_pair(remove[i], merits[i]));
            }
            std::sort(
                zipped.begin(),
                zipped.end(),
                [](std::pair<size_t, double> const& a, std::pair<size_t, double> const& b) {
                    return a.first < b.first;
                });

            remove.clear();
            merits.clear();
            remove.reserve(zipped.size());
            merits.reserve(zipped.size());

            size_t current_index = zipped[0].first;
            double current_merit = zipped[0].second;
            for(size_t i = 1; i < zipped.size(); ++i) {
                if(zipped[i].first == current_index) {
                    current_merit = std::max(current_merit, zipped[i].second);
                }
                else {
                    remove.push_back(current_index);
                    merits.push_back(current_merit);
                    current_index = zipped[i].first;
                    current_merit = zipped[i].second;
                }
            }
            remove.push_back(current_index);
            merits.push_back(current_merit);
        }
        total_removed_ += remove.size();
        ++iter_;
        return std::make_pair(remove, merits);
    }

    size_t getTotalRemoved() const { return total_removed_; }

private:
    const double probability_;
    const int rank_;
    mutable std::uint64_t iter_;
    mutable size_t total_removed_;
};
}

int main()
{
    int rank = 0;
    int world_size = 1;
#ifdef RICH_MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    try {
        Vector3D ll(-1.0, -1.0, -1.0);
        Vector3D ur(1.0, 1.0, 1.0);
        const size_t target_points = (world_size > 1) ? static_cast<size_t>(2e6) : static_cast<size_t>(1e4);
        const size_t amr_rounds = 1;
        const double amr_probability = 5e-2;

        std::vector<Vector3D> points;
        if(rank == 0) {
            points = RandRectangular(target_points, ll, ur);
        }
#ifdef RICH_MPI
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
        points = RoundGrid3D(points, ll, ur, 10);

        Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        IdealGas eos(5.0 / 3.0);
        const ComputationalCell3D baseline = [&eos]() {
            ComputationalCell3D c;
            c.density = 1.0;
            c.internal_energy = 2.5;
            c.pressure = eos.de2p(c.density, c.internal_energy, c.tracers, ComputationalCell3D::tracerNames);
            c.velocity = Vector3D(0.0, 0.0, 0.0);
            return c;
        }();

        std::vector<ComputationalCell3D> cells(tess.GetPointNo(), baseline);

        Hllc3D rs;
        RigidWallGenerator3D ghost;
        LinearGauss3D interp(eos, ghost);
        std::vector<std::pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*> > sequence;
        IsBoundaryFace3D is_boundary;
        IsBulkFace3D is_bulk;
        RigidWallFlux3D rigid_flux(rs);
        RegularFlux3D regular_flux(rs);
        sequence.push_back(std::make_pair(&is_boundary, &rigid_flux));
        sequence.push_back(std::make_pair(&is_bulk, &regular_flux));
        ConditionActionFlux1 flux(sequence, interp);
        std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
        ConditionExtensiveUpdater3D eu(eu_sequence);
        DefaultCellUpdater cu;
        ZeroForce3D force;
        CourantFriedrichsLewy tsf(0.3, 1.0, force);
        Lagrangian3D bpm;
        RoundCells3D pm(bpm, eos);
        Simulation simulation(tess, cells, eos);
        HDSim3D sim(
            tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, tsf, flux, cu, eu, force,
            std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

        RandomRefine3D refine(amr_probability, rank);
        RandomRemove3D remove(amr_probability, rank);
        AMR3D amr(eos, refine, remove, interp);

        double max_drift_local = 0.0;
        double max_volume_growth = 0.0;
        for(size_t round = 0; round < amr_rounds; ++round) {
            std::unordered_map<size_t, double> old_volumes;
            old_volumes.reserve(tess.GetPointNo());
            for(size_t i = 0; i < tess.GetPointNo(); ++i) {
                old_volumes[simulation.getCells()[i].ID] = tess.GetVolume(i);
            }
            amr(simulation);
            const std::vector<ComputationalCell3D>& current_cells = sim.getCells();
            const Tessellation3D& current_tess = sim.getTessellation();
            const size_t npoints = std::min(current_tess.GetPointNo(), current_cells.size());
            size_t real_local_points = 0;
            double max_density_drift_local = 0.0;
            double max_ie_drift_local = 0.0;
            double max_pressure_drift_local = 0.0;
            double max_vx_drift_local = 0.0;
            double max_vy_drift_local = 0.0;
            double max_vz_drift_local = 0.0;
            for(size_t i = 0; i < npoints; ++i) {
                if(current_tess.IsGhostPoint(i) || current_tess.IsPointOutsideBox(i)) {
                    continue;
                }
                ++real_local_points;
                const ComputationalCell3D& c = current_cells[i];
                auto old_volume = old_volumes.find(c.ID);
                if(old_volume != old_volumes.end()) {
                    max_volume_growth = std::max(max_volume_growth,
                        current_tess.GetVolume(i) / old_volume->second);
                }
                const double density_drift = rel_diff(c.density, baseline.density);
                const double ie_drift = rel_diff(c.internal_energy, baseline.internal_energy);
                const double pressure_drift = rel_diff(c.pressure, baseline.pressure);
                const double vx_drift = std::abs(c.velocity.x);
                const double vy_drift = std::abs(c.velocity.y);
                const double vz_drift = std::abs(c.velocity.z);

                max_density_drift_local = std::max(max_density_drift_local, density_drift);
                max_ie_drift_local = std::max(max_ie_drift_local, ie_drift);
                max_pressure_drift_local = std::max(max_pressure_drift_local, pressure_drift);
                max_vx_drift_local = std::max(max_vx_drift_local, vx_drift);
                max_vy_drift_local = std::max(max_vy_drift_local, vy_drift);
                max_vz_drift_local = std::max(max_vz_drift_local, vz_drift);
                max_drift_local = std::max(max_drift_local, density_drift);
                max_drift_local = std::max(max_drift_local, ie_drift);
                max_drift_local = std::max(max_drift_local, pressure_drift);
                max_drift_local = std::max(max_drift_local, vx_drift);
                max_drift_local = std::max(max_drift_local, vy_drift);
                max_drift_local = std::max(max_drift_local, vz_drift);
            }

            double max_density_drift = max_density_drift_local;
            double max_ie_drift = max_ie_drift_local;
            double max_pressure_drift = max_pressure_drift_local;
            double max_vx_drift = max_vx_drift_local;
            double max_vy_drift = max_vy_drift_local;
            double max_vz_drift = max_vz_drift_local;
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &max_density_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_ie_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_pressure_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_vx_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_vy_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_vz_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
            if(rank == 0) {
                std::cout << "AMR round " << round + 1 << "/" << amr_rounds
                          << " local_points=" << real_local_points
                          << " max_drift_local=" << max_drift_local
                          << " density=" << max_density_drift
                          << " ie=" << max_ie_drift
                          << " pressure=" << max_pressure_drift
                          << " vx=" << max_vx_drift
                          << " vy=" << max_vy_drift
                          << " vz=" << max_vz_drift
                          << std::endl;
            }
        }

        size_t total_refined_local = refine.getTotalRefined();
        size_t total_removed_local = remove.getTotalRemoved();
        size_t total_refined_global = total_refined_local;
        size_t total_removed_global = total_removed_local;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &total_refined_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &total_removed_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        if(rank == 0) {
            std::cout << std::endl
                      << "Total points refined: " << total_refined_global
                      << ", Total points removed: " << total_removed_global
                      << std::endl << std::endl;
        }

        double max_drift = max_drift_local;
#ifdef RICH_MPI
        MPI_Allreduce(&max_drift_local, &max_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_volume_growth, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        const bool mpi_mode = world_size > 1;
        const double threshold = mpi_mode ? 1e-6 : 1e-8;
        const int passed = (max_drift <= threshold &&
            max_volume_growth <= 3.0 * (1.0 + 1e-8)) ? 1 : 0;
        int all_passed = passed;
#ifdef RICH_MPI
        MPI_Allreduce(&passed, &all_passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

        if(rank == 0) {
            std::ofstream out("amr_random_metrics.txt");
            out.setf(std::ios::scientific);
            out.precision(16);
            out << "mode " << (mpi_mode ? "mpi" : "serial") << "\n";
            out << "rounds " << amr_rounds << "\n";
            out << "target_points " << target_points << "\n";
            out << "max_drift " << max_drift << "\n";
            out << "threshold " << threshold << "\n";
            out << "max_volume_growth " << max_volume_growth << "\n";
            out << "volume_growth_limit 3.0\n";
            out << "pass " << all_passed << "\n";
            out.close();
        }

#ifdef RICH_MPI
        MPI_Finalize();
#endif
        return all_passed ? 0 : 1;
    }
    catch(UniversalError const& e) {
        reportError(e);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        return 2;
    }
}
