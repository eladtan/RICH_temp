#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>
#ifdef RICH_MPI
#include <mpi.h>
#endif

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/monte/population/NoControl.hpp"
#include "source/monte/boundary/Rigid.hpp"
#include "source/monte/boundary/BoundaryCondition.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/3D/radiation/IMCMemoryCostCalculator.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/SourceTerm3D.hpp"
#include "source/3D/output/write3D.hpp"

namespace fs = std::filesystem;

namespace
{

class NullFluxCalculator : public FluxCalculator3D
{
public:
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>>
    operator()(vector<Conserved3D> & /*fluxes*/, const Tessellation3D & /*tess*/,
               const vector<Vector3D> & /*edge_velocities*/,
               const vector<ComputationalCell3D> & /*cells*/,
               const vector<Conserved3D> & /*extensives*/,
               const EquationOfState & /*eos*/,
               double /*time*/, double /*dt*/) const override
    {
        return {};
    }
};

struct OpacityRow { double nu_min; double nu_max; double kappa; };

static const OpacityRow FINE_OPACITY_TABLE[124] = {
    {1.000e-03, 1.229e-03, 1.000e+04}, {1.229e-03, 1.510e-03, 1.000e+04},
    {1.510e-03, 1.856e-03, 1.000e+04}, {1.856e-03, 2.281e-03, 1.000e+04},
    {2.281e-03, 2.804e-03, 1.000e+04}, {2.804e-03, 3.446e-03, 1.000e+04},
    {3.446e-03, 4.234e-03, 1.000e+04}, {4.234e-03, 5.204e-03, 1.000e+04},
    {5.204e-03, 6.396e-03, 1.000e+04}, {6.396e-03, 7.860e-03, 1.000e+04},
    {7.860e-03, 9.660e-03, 1.000e+04}, {9.660e-03, 1.187e-02, 1.000e+04},
    {1.187e-02, 1.459e-02, 1.000e+04}, {1.459e-02, 1.793e-02, 1.000e+04},
    {1.793e-02, 2.204e-02, 1.000e+04}, {2.204e-02, 2.708e-02, 8.933e+03},
    {2.708e-02, 3.328e-02, 8.569e+03}, {3.328e-02, 4.090e-02, 7.335e+03},
    {4.090e-02, 5.027e-02, 5.656e+03}, {5.027e-02, 6.178e-02, 4.031e+03},
    {6.178e-02, 7.593e-02, 2.710e+03}, {7.593e-02, 9.331e-02, 1.770e+03},
    {9.331e-02, 1.147e-01, 1.184e+03}, {1.147e-01, 1.409e-01, 7.924e+02},
    {1.409e-01, 1.732e-01, 5.061e+02}, {1.732e-01, 2.129e-01, 3.230e+02},
    {2.129e-01, 2.616e-01, 2.062e+02}, {2.616e-01, 3.215e-01, 2.100e+02},
    {3.215e-01, 3.951e-01, 1.229e+02}, {3.951e-01, 4.856e-01, 7.579e+01},
    {4.856e-01, 5.968e-01, 4.905e+01}, {5.968e-01, 7.334e-01, 3.110e+01},
    {7.334e-01, 9.014e-01, 1.947e+01}, {9.014e-01, 1.000e+00, 1.196e+01},
    {1.000e+00, 1.014e+00, 1.187e+01}, {1.014e+00, 1.028e+00, 1.149e+01},
    {1.028e+00, 1.042e+00, 1.112e+01}, {1.042e+00, 1.057e+00, 1.076e+01},
    {1.057e+00, 1.072e+00, 1.041e+01}, {1.072e+00, 1.087e+00, 1.007e+01},
    {1.087e+00, 1.102e+00, 9.740e+00}, {1.102e+00, 1.117e+00, 9.416e+00},
    {1.117e+00, 1.133e+00, 9.098e+00}, {1.133e+00, 1.149e+00, 8.785e+00},
    {1.149e+00, 1.165e+00, 8.477e+00}, {1.165e+00, 1.181e+00, 8.180e+00},
    {1.181e+00, 1.198e+00, 7.900e+00}, {1.198e+00, 1.214e+00, 7.635e+00},
    {1.214e+00, 1.231e+00, 7.381e+00}, {1.231e+00, 1.248e+00, 7.138e+00},
    {1.248e+00, 1.266e+00, 6.902e+00}, {1.266e+00, 1.283e+00, 6.674e+00},
    {1.283e+00, 1.301e+00, 6.452e+00}, {1.301e+00, 1.319e+00, 6.237e+00},
    {1.319e+00, 1.338e+00, 6.029e+00}, {1.338e+00, 1.357e+00, 5.827e+00},
    {1.357e+00, 1.375e+00, 5.631e+00}, {1.375e+00, 1.395e+00, 5.438e+00},
    {1.395e+00, 1.414e+00, 5.250e+00}, {1.414e+00, 1.434e+00, 5.066e+00},
    {1.434e+00, 1.454e+00, 4.886e+00}, {1.454e+00, 1.474e+00, 4.709e+00},
    {1.474e+00, 1.495e+00, 4.542e+00}, {1.495e+00, 1.516e+00, 4.387e+00},
    {1.516e+00, 1.537e+00, 4.243e+00}, {1.537e+00, 1.558e+00, 4.117e+00},
    {1.558e+00, 1.580e+00, 4.310e+00}, {1.580e+00, 1.602e+00, 1.572e+01},
    {1.602e+00, 1.625e+00, 4.834e+00}, {1.625e+00, 1.647e+00, 3.726e+00},
    {1.647e+00, 1.670e+00, 3.758e+00}, {1.670e+00, 1.694e+00, 4.706e+00},
    {1.694e+00, 1.717e+00, 3.394e+01}, {1.717e+00, 1.741e+00, 9.034e+02},
    {1.741e+00, 1.765e+00, 1.615e+01}, {1.765e+00, 1.790e+00, 4.098e+00},
    {1.790e+00, 1.815e+00, 3.420e+00}, {1.815e+00, 1.840e+00, 3.389e+00},
    {1.840e+00, 1.866e+00, 3.986e+00}, {1.866e+00, 1.892e+00, 4.350e+00},
    {1.892e+00, 1.919e+00, 3.933e+00}, {1.919e+00, 1.945e+00, 4.258e+00},
    {1.945e+00, 1.972e+00, 4.861e+00}, {1.972e+00, 1.995e+00, 6.836e+00},
    {1.995e+00, 2.089e+00, 4.674e+01}, {2.089e+00, 2.188e+00, 2.108e+01},
    {2.188e+00, 2.291e+00, 2.281e+01}, {2.291e+00, 2.399e+00, 1.963e+01},
    {2.399e+00, 2.512e+00, 1.749e+01}, {2.512e+00, 2.630e+00, 1.590e+01},
    {2.630e+00, 2.754e+00, 1.442e+01}, {2.754e+00, 2.884e+00, 1.294e+01},
    {2.884e+00, 3.020e+00, 1.144e+01}, {3.020e+00, 3.162e+00, 1.014e+01},
    {3.162e+00, 3.311e+00, 9.047e+00}, {3.311e+00, 3.467e+00, 8.057e+00},
    {3.467e+00, 3.631e+00, 7.118e+00}, {3.631e+00, 3.802e+00, 6.219e+00},
    {3.802e+00, 3.981e+00, 5.474e+00}, {3.981e+00, 4.169e+00, 4.861e+00},
    {4.169e+00, 4.365e+00, 4.311e+00}, {4.365e+00, 4.571e+00, 3.792e+00},
    {4.571e+00, 4.786e+00, 3.296e+00}, {4.786e+00, 5.012e+00, 2.888e+00},
    {5.012e+00, 5.248e+00, 2.555e+00}, {5.248e+00, 5.495e+00, 2.258e+00},
    {5.495e+00, 5.754e+00, 1.978e+00}, {5.754e+00, 6.026e+00, 1.713e+00},
    {6.026e+00, 6.310e+00, 1.496e+00}, {6.310e+00, 6.607e+00, 1.320e+00},
    {6.607e+00, 6.918e+00, 1.163e+00}, {6.918e+00, 7.244e+00, 1.016e+00},
    {7.244e+00, 7.586e+00, 8.770e-01}, {7.586e+00, 7.943e+00, 7.641e-01},
    {7.943e+00, 8.318e+00, 6.729e-01}, {8.318e+00, 8.710e+00, 5.919e-01},
    {8.710e+00, 9.120e+00, 5.160e-01}, {9.120e+00, 9.550e+00, 4.442e-01},
    {9.550e+00, 1.070e+01, 3.862e-01}, {1.070e+01, 1.315e+01, 2.385e-01},
    {1.315e+01, 1.616e+01, 1.309e-01}, {1.616e+01, 1.986e+01, 7.143e-02},
    {1.986e+01, 2.441e+01, 3.867e-02}, {2.441e+01, 3.000e+01, 2.076e-02},
};
constexpr size_t N_FINE_GROUPS = 124;
constexpr size_t N_COARSE_GROUPS = 32;
constexpr double COARSE_EMIN_KEV = 1.0e-3;
constexpr double COARSE_EMAX_KEV = 3.0e+1;

struct CollapsedOpacity {
    double boundary_keV[N_COARSE_GROUPS + 1];
    double kappa[N_COARSE_GROUPS];
};

CollapsedOpacity collapse_opacity_planck(double T_kelvin)
{
    CollapsedOpacity result;

    double const ratio = std::pow(COARSE_EMAX_KEV / COARSE_EMIN_KEV,
                                  1.0 / static_cast<double>(N_COARSE_GROUPS));
    result.boundary_keV[0] = COARSE_EMIN_KEV;
    for (size_t g = 0; g < N_COARSE_GROUPS; ++g)
        result.boundary_keV[g + 1] = result.boundary_keV[g] * ratio;
    result.boundary_keV[N_COARSE_GROUPS] = COARSE_EMAX_KEV;

    for (size_t gp = 0; gp < N_COARSE_GROUPS; ++gp)
    {
        double const new_lo = result.boundary_keV[gp] * units::kev;
        double const new_hi = result.boundary_keV[gp + 1] * units::kev;
        double numerator = 0.0;
        double denominator = 0.0;

        for (size_t g = 0; g < N_FINE_GROUPS; ++g)
        {
            double const old_lo = FINE_OPACITY_TABLE[g].nu_min * units::kev;
            double const old_hi = FINE_OPACITY_TABLE[g].nu_max * units::kev;
            double const overlap_lo = std::max(new_lo, old_lo);
            double const overlap_hi = std::min(new_hi, old_hi);
            if (overlap_hi <= overlap_lo)
                continue;
            double const B_overlap =
                planck_integral::planck_energy_density_group_integral(
                    overlap_lo, overlap_hi, T_kelvin);
            numerator += FINE_OPACITY_TABLE[g].kappa * B_overlap;
            denominator += B_overlap;
        }

        result.kappa[gp] = (denominator > 0.0) ? (numerator / denominator) : 0.0;
    }
    return result;
}

class MovingSlabOpacity32 : public OpacityCalculator
{
public:
    MovingSlabOpacity32(double rho_slab,
                        const std::vector<double> &centers,
                        const std::vector<double> &boundaries,
                        const CollapsedOpacity &collapsed)
        : rho_slab_(rho_slab)
    {
        energy_groups_center = centers;
        energy_groups_boundary = boundaries;

        for (size_t g = 0; g < N_COARSE_GROUPS; ++g)
            sigma_g_[g] = rho_slab_ * collapsed.kappa[g];
    }

    bool isVacuum(const ComputationalCell3D &cell) const
    {
        return cell.density < 0.5 * rho_slab_;
    }

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override
    {
        if (isVacuum(cell))
            return 1e-12;
        double T_kelvin = cell.temperature;
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t g = 0; g < N_COARSE_GROUPS; ++g)
        {
            double Elo = energy_groups_boundary[g];
            double Ehi = energy_groups_boundary[g + 1];
            double Bg = planck_integral::planck_energy_density_group_integral(Elo, Ehi, T_kelvin);
            numerator += sigma_g_[g] * Bg;
            denominator += Bg;
        }
        if (denominator <= 0.0)
            return 0.0;
        return numerator / denominator;
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energy) const override
    {
        if (isVacuum(cell))
            return 1e-12;
        size_t g = findGroup(energy);
        if (g >= N_COARSE_GROUPS)
            return 1e-100;
        return sigma_g_[g];
    }

    double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/, double /*energy*/) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/) const override
    {
        return 0.0;
    }

private:
    double rho_slab_;
    double sigma_g_[N_COARSE_GROUPS];
};

// Transparent on left and right x-faces, rigid on y,z faces.
class MovingSlabBC : public BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    MovingSlabBC(const Tessellation3D &grid)
        : BoundaryCondition<Vector3D, Tessellation3D>(grid)
    {
    }

    MonteCarloParticleStatus apply(MonteCarloParticle<Vector3D, Tessellation3D> &particle) override
    {
        const auto &[ll, ur] = this->grid.GetBoxCoordinates();

        double dx_right = std::abs(particle.location.x - ur.x);
        double dx_left  = std::abs(particle.location.x - ll.x);
        double dy_lo    = std::abs(particle.location.y - ll.y);
        double dy_hi    = std::abs(particle.location.y - ur.y);
        double dz_lo    = std::abs(particle.location.z - ll.z);
        double dz_hi    = std::abs(particle.location.z - ur.z);
        double d_min_yz = std::min({dy_lo, dy_hi, dz_lo, dz_hi});

        bool is_right_x = (dx_right < dx_left) && (dx_right < d_min_yz);
        bool is_left_x  = (dx_left < dx_right) && (dx_left < d_min_yz);

        if (is_right_x && particle.velocity.x > 0)
            return MonteCarloParticleStatus::REMOVE;

        if (is_left_x && particle.velocity.x < 0)
            return MonteCarloParticleStatus::REMOVE;

        const std::vector<Tessellation3D::Face_T> &faces = this->grid.GetBoxFaces();
        bool reflected = false;
        for (const auto &face : faces)
        {
            const Vector3D &onFace = face.vertices[0];
            Vector3D u = face.vertices[1] - face.vertices[0];
            Vector3D v = face.vertices[2] - face.vertices[0];
            Vector3D normal = CrossProduct(u, v);
            double absU = abs(u);
            if (std::fabs(ScalarProd(normal, particle.location - onFace)) < EPSILON * absU * absU * absU)
            {
                normal /= abs(normal);
                double signedDist = ScalarProd(particle.location - onFace, normal);
                particle.location -= 2 * signedDist * normal;
                particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
                reflected = true;
            }
        }

        if (reflected)
        {
            const Vector3D &center = this->grid.GetMeshPoint(particle.cellIndex);
            constexpr double nudge = 1e-6;
            particle.location = particle.location * (1 - nudge) + nudge * center;
            return MonteCarloParticleStatus::REFLECT;
        }

        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        size_t localN = this->grid.GetPointNo();
        bool validCellIndex = particle.cellIndex < localN;
        bool insideBox =
            particle.location.x >= ll.x && particle.location.x <= ur.x &&
            particle.location.y >= ll.y && particle.location.y <= ur.y &&
            particle.location.z >= ll.z && particle.location.z <= ur.z;
        std::cerr
            << "MovingSlabBC: particle not on any boundary"
            << " rank=" << rank
            << " id=" << particle.id
            << " cellIndex=" << particle.cellIndex
            << " localN=" << localN
            << " validCellIndex=" << validCellIndex
            << " insideBox=" << insideBox
            << " location=(" << particle.location.x << ", " << particle.location.y << ", " << particle.location.z << ")"
            << " velocity=(" << particle.velocity.x << ", " << particle.velocity.y << ", " << particle.velocity.z << ")"
            << " box_ll=(" << ll.x << ", " << ll.y << ", " << ll.z << ")"
            << " box_ur=(" << ur.x << ", " << ur.y << ", " << ur.z << ")"
            << " dx_left=" << dx_left
            << " dx_right=" << dx_right
            << " dy_lo=" << dy_lo
            << " dy_hi=" << dy_hi
            << " dz_lo=" << dz_lo
            << " dz_hi=" << dz_hi;
        if (validCellIndex)
        {
            const Vector3D &center = this->grid.GetMeshPoint(particle.cellIndex);
            bool inCell = this->grid.IsPointInCell(particle.location, particle.cellIndex);
            std::cerr
                << " cellCenter=(" << center.x << ", " << center.y << ", " << center.z << ")"
                << " inCurrentCell=" << inCell;
        }
        std::cerr << std::endl;
        exit(1);
    }

    std::vector<MonteCarloParticle<Vector3D, Tessellation3D>>
    generateNewBoundaryParticles(double /*fullDt*/) override
    {
        return {};
    }
};

} // anonymous namespace

int main(int argc, char *argv[])
{
    int rank = 0;
    int worldSize = 1;
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
#else
    (void)argc;
    (void)argv;
#endif

    try
    {
        constexpr size_t G = ENERGY_GROUPS_NUM;
        static_assert(G == 32, "Moving slab 32-group benchmark requires exactly 32 energy groups");

        // --- Benchmark parameters (McClarren & Gentile 2021, original vacuum) ---
        double const rho_slab = 0.1;
        double const L_slab   = 0.4;
        double const T_slab_keV = 1.0;
        double const T_slab   = T_slab_keV * units::kev_kelvin;
        double const v_slab_cm_ns = 0.5994;
        double const v_slab   = v_slab_cm_ns * 1e9;
        double const z_O      = 12.0;
        double const x_sym    = 12.0;
        double const t_O_ns   = 10.0;
        double const t_O      = t_O_ns * 1e-9;
        double const rho_vacuum = 1e-10;

        // --- Collapse 124-group opacity to 32 log-spaced groups ---
        CollapsedOpacity const collapsed = collapse_opacity_planck(T_slab);

        if (rank == 0)
        {
            std::cout << "Collapsed 124 -> 32 groups (Planck weight at T="
                      << T_slab_keV << " keV):" << std::endl;
            for (size_t g = 0; g < N_COARSE_GROUPS; ++g)
                std::cout << "  g=" << g
                          << "  [" << collapsed.boundary_keV[g]
                          << ", " << collapsed.boundary_keV[g + 1]
                          << "] keV  kappa=" << collapsed.kappa[g] << std::endl;
        }

        // --- 32-group energy grid (in CGS: erg) ---
        std::vector<double> energy_groups_center(G);
        std::vector<double> energy_groups_boundary(G + 1);
        for (size_t g = 0; g <= G; ++g)
            energy_groups_boundary[g] = collapsed.boundary_keV[g] * units::kev;
        for (size_t g = 0; g < G; ++g)
            energy_groups_center[g] = 0.5 * (energy_groups_boundary[g] + energy_groups_boundary[g + 1]);
        for (size_t g = 0; g <= G; ++g)
            ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];

        // --- Geometry: 1D-like domain [0, z_O + margin] in x, 3x3 grid in YZ ---
        double const x_max = z_O + 0.2;
        double const cellHalfYZ = 0.5;
        constexpr size_t NYZ = 3;
        Vector3D ll(0, -cellHalfYZ, -cellHalfYZ);
        Vector3D ur(x_max, cellHalfYZ, cellHalfYZ);

        std::array<double, NYZ> yz_centers;
        for (size_t k = 0; k < NYZ; ++k)
            yz_centers[k] = -cellHalfYZ + cellHalfYZ * (2.0 * k + 1.0) / NYZ;

        // Build initial mesh points: 20 slab + 60 vacuum, each replicated NYZ*NYZ in YZ
        size_t const N_slab_pts = 20;
        size_t const N_vac_pts  = 60;
        size_t const N_x_pts    = N_slab_pts + N_vac_pts;
        size_t const N_total_pts = N_x_pts * NYZ * NYZ;

        std::vector<double> slab_pts_initial_x(N_slab_pts);
        for (size_t i = 0; i < N_slab_pts; ++i)
            slab_pts_initial_x[i] = L_slab * (static_cast<double>(i) + 0.5) / N_slab_pts;

        auto buildAllPoints = [&](double t) -> std::vector<Vector3D>
        {
            std::vector<Vector3D> x_pts(N_x_pts);
            double shift = v_slab * t;
            for (size_t i = 0; i < N_slab_pts; ++i)
                x_pts[i] = Vector3D(slab_pts_initial_x[i] + shift, 0, 0);
            double vac_start = L_slab + shift;
            size_t N_left = N_vac_pts - 2;
            double h = (x_sym - vac_start) / N_left;
            double d = h / 2.0;
            for (size_t i = 0; i < N_left; ++i)
            {
                double x = vac_start + h * (static_cast<double>(i) + 0.5);
                x_pts[N_slab_pts + i] = Vector3D(x, 0, 0);
            }
            x_pts[N_slab_pts + N_left]     = Vector3D(x_sym, 0, 0);
            x_pts[N_slab_pts + N_left + 1] = Vector3D(x_sym + d, 0, 0);

            std::vector<Vector3D> pts;
            pts.reserve(N_total_pts);
            for (size_t i = 0; i < N_x_pts; ++i)
                for (size_t jy = 0; jy < NYZ; ++jy)
                    for (size_t jz = 0; jz < NYZ; ++jz)
                        pts.emplace_back(x_pts[i].x, yz_centers[jy], yz_centers[jz]);
            return pts;
        };

        std::vector<Vector3D> points;
#ifdef RICH_MPI
        if (worldSize == 1)
        {
            points = buildAllPoints(0.0);
        }
        else
        {
            if (rank == 0)
                points = buildAllPoints(0.0);
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
#else
        points = buildAllPoints(0.0);
#endif

        Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
        if (worldSize == 1)
            tess.Build(points);
        else
            tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        // --- EOS (irrelevant: noHydroFeedback = true) ---
        double const cv = 1e23 / units::kev_kelvin;
        IdealGas eos(1.4, cv, 1, 0);

        // --- Initialize cells from slab position at t=0 ---
        size_t N = tess.GetPointNo();
        std::vector<ComputationalCell3D> initialCells(N);
        for (size_t i = 0; i < N; ++i)
        {
            auto &c = initialCells[i];
            double x = tess.GetMeshPoint(i).x;
            bool in_slab = (x >= 0.0 && x <= L_slab);

            c.density = in_slab ? rho_slab : rho_vacuum;
            c.temperature = in_slab ? T_slab : 1e5;
            c.internal_energy = eos.dT2e(c.density, c.temperature, c.tracers, ComputationalCell3D::tracerNames);
            c.pressure = eos.de2p(c.density, c.internal_energy, c.tracers, ComputationalCell3D::tracerNames);
            c.velocity = in_slab ? Vector3D(v_slab, 0, 0) : Vector3D(0, 0, 0);

            for (size_t g = 0; g < G; ++g)
            {
                if (in_slab)
                {
                    double Elo = energy_groups_boundary[g];
                    double Ehi = energy_groups_boundary[g + 1];
                    c.Eg[g] = planck_integral::planck_energy_density_group_integral(Elo, Ehi, T_slab) / c.density;
                }
                else
                {
                    c.Eg[g] = 0.0;
                }
            }
            c.Erad = std::accumulate(c.Eg.begin(), c.Eg.end(), 0.0);
        }

        Simulation sim(tess, initialCells, eos);
        auto tsc = std::make_shared<ManualTimeStep>();
        sim.SetTimeStepFunction(tsc);

        Eulerian3D pm;
        NullFluxCalculator fc;
        DefaultCellUpdater cu;
        DefaultExtensiveUpdater eu;
        ZeroForce3D force;
        sim.getCells().resize(tess.GetPointNo());
        sim.getExtensives().resize(tess.GetPointNo());
        HDSim3D hdsim(tess, sim.getCells(), sim.getExtensives(), eos,
                      sim.getTracker(), pm, *tsc, fc, cu, eu, force,
                      std::make_pair(ComputationalCell3D::tracerNames,
                                     ComputationalCell3D::stickerNames));

        auto &cells = sim.getCells();
        auto &extensives = sim.getExtensives();
        extensives.resize(cells.size());
        for (size_t i = 0; i < cells.size(); ++i)
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

        auto eosPtr = std::make_shared<IdealGas>(eos);
        auto opacityPtr = std::make_shared<MovingSlabOpacity32>(
            rho_slab, energy_groups_center, energy_groups_boundary, collapsed);

        auto boundaryCond = std::make_shared<MovingSlabBC>(tess);

        size_t const newPhotonsPerCell = 10000 / (NYZ * NYZ);
        RadiationIMCParameters imcParams = {
            .newPhotonsPerCell = newPhotonsPerCell,
            .withHydro = true,
            .diffusionPressureGradient = false,
            .MMC = false,
            .withMultigroupOpacity = true,
            .withRandomWalk = false,
            .noHydroFeedback = true,
            .withEgTimeAvg = true
        };

        auto physics = std::make_shared<RadiationIMC>(
            tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, imcParams);

        auto popControl =
            std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

        std::vector<MonteCarloParticle<Vector3D, Tessellation3D>> emptyParticles;
        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, popControl, boundaryCond,
            emptyParticles, 0, true
#ifdef RICH_MPI
            , RadiationMCStep::ManagerType::AUTO_RDMA
#endif
        );

        sim.addPhysics(mcStep);
#ifdef RICH_MPI
        mcStep->setCost(std::make_shared<IMCMemoryCostCalculator>(mcStep->getManager()));
        sim.PresetLoadBalance("radiation-mc");
#endif

        // --- Time stepping: dt starts at 1e-3 ns, ramps by 1.1x, max 0.1 ns ---
        double dt = 1e-3 * 1e-9;
        double const dt_max = 0.1 * 1e-9;
        double const dt_ramp = 1.1;
        double const t_end = t_O + dt_max / 2.0;

        if (rank == 0)
        {
            std::cout << "Moving slab MC benchmark (32-group, original vacuum): "
                      << G << " groups, "
                      << N_total_pts << " mesh points (" << N_x_pts << "x * " << NYZ << "y * " << NYZ << "z), "
                      << "newPhotonsPerCell=" << newPhotonsPerCell
                      << ", v_slab=" << v_slab << " cm/s"
                      << ", L=" << L_slab << " cm"
                      << ", T=" << T_slab_keV << " keV"
                      << ", z_O=" << z_O << " cm"
                      << ", t_O=" << t_O_ns << " ns"
                      << ", MPI ranks=" << worldSize
                      << std::endl;
        }

        auto wallStart = std::chrono::high_resolution_clock::now();
        size_t stepCount = 0;

        while (sim.GetTime() < t_end)
        {
            double t_now = sim.GetTime();
            double this_dt = std::min(dt, t_end - t_now);
            if (this_dt <= 0)
                break;

            sim.SetTimeStep(this_dt);
#ifdef RICH_MPI
            if (stepCount > 0)
            {
                auto dupProcs = tess.GetDuplicatedProcs();
                auto &ghosts = tess.GetGhostIndeces();
                size_t totalGhosts = 0;
                for (auto &g : ghosts) totalGhosts += g.size();
                std::cerr << "[pre-step] rank=" << rank
                          << " step=" << stepCount
                          << " localN=" << tess.GetPointNo()
                          << " totalPts=" << tess.GetTotalPointNumber()
                          << " dupProcs=" << dupProcs.size()
                          << " totalGhosts=" << totalGhosts
                          << std::endl;
            }
#endif
            sim.step();

#ifdef RICH_MPI
            if (stepCount == 1)
                sim.setForceRebalanceSteps(4);
#endif
            ++stepCount;
            dt = std::min(dt * dt_ramp, dt_max);

            if (stepCount % 5 == 0 && rank == 0)
            {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - wallStart).count();
                double slab_back_now  = v_slab * sim.GetTime();
                double slab_front_now = L_slab + v_slab * sim.GetTime();
                std::cout << "Step " << stepCount
                          << "  t=" << sim.GetTime() * 1e9 << " ns"
                          << "  dt=" << this_dt * 1e9 << " ns"
                          << "  slab=[" << slab_back_now << ", " << slab_front_now << "]"
                          << "  elapsed=" << elapsed << "s" << std::endl;
            }

            if (sim.GetTime() >= t_end)
                break;

            double t_new = sim.GetTime();
            double new_ll_x = v_slab * t_new;
            Vector3D newLL(new_ll_x, -cellHalfYZ, -cellHalfYZ);
#ifdef RICH_MPI
            std::shared_ptr<LoadBalancer> savedLB = nullptr;
            if (worldSize > 1)
            {
                savedLB = tess.GetLoadBalancer();
            }
#endif
            tess.SetBox(newLL, ur);
#ifdef RICH_MPI
            if (worldSize == 1)
            {
                points = buildAllPoints(t_new);
                tess.Build(points);
            }
            else
            {
                double slab_front_old = L_slab + v_slab * t_now;
                double slab_front_new = L_slab + v_slab * t_new;
                double dx = v_slab * (t_new - t_now);
                size_t localN = tess.GetPointNo();
                std::vector<Vector3D> localPoints(localN);
                for (size_t i = 0; i < localN; ++i)
                {
                    Vector3D p = tess.GetMeshPoint(i);
                    if (p.x <= slab_front_old)
                        p.x += dx;
                    else
                    {
                        double f = (p.x - slab_front_old) / (x_sym - slab_front_old);
                        p.x = slab_front_new + f * (x_sym - slab_front_new);
                    }
                    localPoints[i] = p;
                }
                if(savedLB != nullptr)
                {
                    tess.PresetLoadBalancer(savedLB);
                }
                tess.BuildParallel(localPoints, true);
                if (savedLB != nullptr)
                {
                    tess.PresetLoadBalancer(savedLB);
                }
                {
                    auto dupProcs = tess.GetDuplicatedProcs();
                    auto &ghosts = tess.GetGhostIndeces();
                    size_t totalGhosts = 0;
                    for (auto &g : ghosts) totalGhosts += g.size();
                    std::cerr << "[test-rebuild] rank=" << rank
                              << " localN=" << tess.GetPointNo()
                              << " totalPts=" << tess.GetTotalPointNumber()
                              << " dupProcs=" << dupProcs.size()
                              << " totalGhosts=" << totalGhosts
                              << " step=" << stepCount
                              << std::endl;
                }
                MPI_exchange_data(tess, cells, false);
                MPI_exchange_data(tess, mcStep->getManager()->GetCellsStepsCounters(), false);
            }
#else
            points = buildAllPoints(t_new);
            tess.Build(points);
#endif

            {
                auto &parts = mcStep->getParticles();
                parts.erase(
                    std::remove_if(parts.begin(), parts.end(),
                        [&newLL, &ur](const auto &p) {
                            return p.location.x < newLL.x || p.location.x > ur.x;
                        }),
                    parts.end());
            }

            N = tess.GetPointNo();
            extensives.resize(N);

            for (size_t i = 0; i < N; ++i)
            {
                PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
            }
        }

        double wallTotal = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wallStart).count();
        if (rank == 0)
        {
            std::cout << "Done. " << stepCount << " steps, wall time: " << wallTotal << "s" << std::endl;
        }

        std::string const caseDir = fs::path(__FILE__).parent_path().string();
        WriteSnapshot3D(hdsim, caseDir + "/moving_slab_mc_32_final.h5");
        if (rank == 0)
            std::cout << "Wrote snapshot to " << caseDir << std::endl;

        // --- Find the observer cells (all cells at x closest to z_O) on each rank ---
        const auto &EgTA = mcStep->getEgTimeAvg();
        size_t localN = tess.GetPointNo();

        double bestX = 0.0;
        double minDist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < localN; ++i)
        {
            double d = std::abs(tess.GetMeshPoint(i).x - z_O);
            if (d < minDist)
            {
                minDist = d;
                bestX = tess.GetMeshPoint(i).x;
            }
        }

        int writerRank = 0;
#ifdef RICH_MPI
        {
            struct { double dist; int rank; } localBest{minDist, rank}, globalBest;
            MPI_Allreduce(&localBest, &globalBest, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
            writerRank = globalBest.rank;
            MPI_Bcast(&bestX, 1, MPI_DOUBLE, writerRank, MPI_COMM_WORLD);
        }
#endif

        constexpr double xTol = 1e-8;
        std::array<double, G> localEgSum{};
        size_t localObsCount = 0;
        for (size_t i = 0; i < localN; ++i)
        {
            if (std::abs(tess.GetMeshPoint(i).x - bestX) < xTol)
            {
                const auto &egta = EgTA[i];
                for (size_t g = 0; g < G; ++g)
                    localEgSum[g] += egta[g];
                ++localObsCount;
            }
        }

        std::array<double, G> globalEgSum{};
        size_t globalObsCount = 0;
#ifdef RICH_MPI
        MPI_Reduce(localEgSum.data(), globalEgSum.data(), G, MPI_DOUBLE, MPI_SUM, writerRank, MPI_COMM_WORLD);
        MPI_Reduce(&localObsCount, &globalObsCount, 1, MPI_UNSIGNED_LONG, MPI_SUM, writerRank, MPI_COMM_WORLD);
#else
        globalEgSum = localEgSum;
        globalObsCount = localObsCount;
#endif

        if (rank == writerRank)
        {
            std::cout << "Observer cells: " << globalObsCount << " cells at x=" << bestX
                      << " (target z_O=" << z_O << ", dist=" << minDist << ")" << std::endl;

            double invCount = (globalObsCount > 0) ? 1.0 / static_cast<double>(globalObsCount) : 0.0;

            std::string const specPath = caseDir + "/moving_slab_mc_32_spectrum.txt";
            std::ofstream out(specPath);
            out << std::scientific << std::setprecision(12);

            out << "# Moving slab MC benchmark 32-group (original_vacuum)\n";
            out << "# v_slab_cm_per_ns " << v_slab_cm_ns << "\n";
            out << "# L_slab_cm " << L_slab << "\n";
            out << "# T_slab_keV " << T_slab_keV << "\n";
            out << "# rho_slab " << rho_slab << "\n";
            out << "# z_O_cm " << z_O << "\n";
            out << "# t_O_ns " << t_O_ns << "\n";
            out << "# observer_x_cm " << bestX << "\n";
            out << "# observer_yz_cells " << globalObsCount << "\n";
            out << "# steps " << stepCount << "\n";
            out << "# wall_time_s " << wallTotal << "\n";
            out << "# mpi_ranks " << worldSize << "\n";
            out << "# columns: group nu_min_keV nu_max_keV kappa_cm2_per_g Eg_time_avg_erg_per_cm3\n";

            for (size_t g = 0; g < G; ++g)
            {
                out << g
                    << " " << collapsed.boundary_keV[g]
                    << " " << collapsed.boundary_keV[g + 1]
                    << " " << collapsed.kappa[g]
                    << " " << globalEgSum[g] * invCount
                    << "\n";
            }
            out.close();
            std::cout << "Wrote " << specPath << std::endl;
        }
    }
    catch (const UniversalError &e)
    {
        std::cerr << "=== UniversalError ===" << std::endl;
        reportError(e);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "=== std::exception: " << e.what() << " ===" << std::endl;
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
