#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <iomanip>
#include <array>
#include "source/mpi/mpi_commands.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/RadiationOpacity.hpp"
#include "source/monte/population/Comb.hpp"
#include "source/monte/boundary/BoundaryCondition.hpp"
#include "source/3D/radiation/IMCCostCalculator.hpp"
#include <H5Cpp.h>
#include "source/3D/output/write3D.hpp"

static constexpr double pi = 3.14159265358979323846;

namespace fs = std::filesystem;

// ============================================================
// Constants
// ============================================================
struct Constants
{
    double R_core = 0.15;
    double R_cavity = 0.25;
    double R_shell_outer = 0.55;
    double R_measure = 0.75;
    double R_domain = 1.00;
    double box_half_width = 1.10;

    double rho_core = 10.0;
    double rho_cavity = 0.05;
    double rho_shell = 1.00;
    double rho_atm = 0.03;
    double rho_buffer = 0.01;
    double cone_density_factor = 0.20;

    double T_shell_gas = 1.0 * units::ev_kelvin;
    double T_rad_core = 4 * units::kev_kelvin;
    double T_rad_floor = 1.0 * units::ev_kelvin;

    double alpha = 2.0;
    double E0 = 1.0 * units::kev;
    double E_ref = 0.30 * units::kev;
    double tau_ref = 30.0;
    double kappa0 = 0.0;
    double kappa_floor_mass = 1e-4;
    double sigma_floor = 1e-10;
    double sigma_max = 1e10;

    size_t newPhotonsPerCell = 10;
    size_t maxPhotonsPerCell = 100;
    size_t max_cycles = 50;
    double target_time = 0;

    double cone_half_angle_deg = 25.0;
    Vector3D cone_axis = Vector3D(1.0, 1.0, 0.5);

    double gamma = 5.0 / 3.0;

    double E_low_diag = 0.10 * units::kev;
    double E_mid_diag = 0.30 * units::kev;
    double E_high_diag = 3.00 * units::kev;
};

static bool insideCone(Vector3D const &pos, Constants const &C)
{
    double r = abs(pos);
    if (r < 1e-30)
        return false;
    Vector3D rhat(pos.x / r, pos.y / r, pos.z / r);
    double axisLen = abs(C.cone_axis);
    Vector3D axisNorm(C.cone_axis.x / axisLen, C.cone_axis.y / axisLen, C.cone_axis.z / axisLen);
    double cosTheta = ScalarProd(rhat, axisNorm);
    double cosCone = std::cos(C.cone_half_angle_deg * pi / 180.0);
    return cosTheta > cosCone;
}

static double initialDensity(Vector3D const &x, Constants const &C)
{
    double r = abs(x);
    double rho;
    if (r < C.R_core)
        rho = C.rho_core;
    else if (r < C.R_cavity)
        rho = C.rho_cavity;
    else if (r < C.R_shell_outer)
        rho = C.rho_shell;
    else if (r < C.R_domain)
        rho = C.rho_atm;
    else
        rho = C.rho_buffer;

    if (insideCone(x, C) && r > C.R_cavity && r < C.R_domain)
        rho *= C.cone_density_factor;

    return rho;
}

static int regionId(Vector3D const &x, Constants const &C)
{
    double r = abs(x);
    bool cone = insideCone(x, C);
    if (r < C.R_core)
        return 0;
    if (r < C.R_cavity)
        return 1;
    if (r < C.R_shell_outer)
        return cone ? 5 : 2;
    if (r < C.R_domain)
        return cone ? 6 : 3;
    return 4;
}

// ============================================================
// Density-dependent opacity calculator
// ============================================================
namespace
{
class DensityPowerLawOpacity : public OpacityCalculator
{
public:
    DensityPowerLawOpacity(Constants const &C,
                           std::vector<double> const &groupCenters,
                           std::vector<double> const &groupBoundaries)
        : C_(C), centers_(groupCenters), boundaries_(groupBoundaries)
    {
        this->energy_groups_center = groupCenters;
        this->energy_groups_boundary = groupBoundaries;
    }

    double CalcAbsorptionOpacity(ComputationalCell3D const &cell, double energy) const override
    {
        double E = std::clamp(energy, boundaries_.front(), boundaries_.back());
        double kappa_mass = C_.kappa0 * std::pow(E / C_.E0, -C_.alpha) + C_.kappa_floor_mass;
        double sigma = cell.density * kappa_mass;
        return std::clamp(sigma, C_.sigma_floor, C_.sigma_max);
    }

    double CalcPlanckOpacity(ComputationalCell3D const &cell) const override
    {
        double T = std::max(cell.temperature, 1.0);
        double kT = units::k_boltz * T;
        size_t G = centers_.size();

        double weighted = 0.0;
        double total = 0.0;
        for (size_t g = 0; g < G; ++g)
        {
            double a = boundaries_[g] / kT;
            double b = boundaries_[g + 1] / kT;
            double Bg = planck_integral::planck_integral(a, b);
            double sigma_g = CalcAbsorptionOpacity(cell, centers_[g]);
            weighted += sigma_g * Bg;
            total += Bg;
        }
        if (total <= 0.0)
            return CalcAbsorptionOpacity(cell, C_.E0);
        return weighted / total;
    }

    double CalcScatteringOpacity(ComputationalCell3D const &) const override
    {
        return 0.0;
    }

private:
    Constants C_;
    std::vector<double> centers_;
    std::vector<double> boundaries_;
};
// Vacuum radiation boundary: removes all escaping photons on every face,
// emits nothing. Suitable for spherical-in-a-box tests.
template <typename T, typename Grid>
class VacuumBoxBoundary : public BoundaryCondition<T, Grid>
{
public:
    explicit VacuumBoxBoundary(Grid const &grid)
        : BoundaryCondition<T, Grid>(grid) {}

    MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &) override
    {
        return MonteCarloParticleStatus::REMOVE;
    }

    std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double) override
    {
        return {};
    }

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t, size_t, size_t) const override
    {
        return DDMCBoundaryFaceBehavior::Unsupported;
    }
};

} // namespace

// ============================================================
// MPI-serializable cell data for gathering results
// ============================================================
struct CellResult
#ifdef RICH_MPI
    : public Serializable
#endif
{
    Vector3D pos;
    Vector3D velocity;
    double volume, width, density, pressure, temperature, internal_energy;
    double vr, speed;
    double Erad;
    int region;
    std::array<double, ENERGY_GROUPS_NUM> Eg;

    CellResult() : pos(), velocity(), volume(0), width(0), density(0), pressure(0),
                   temperature(0), internal_energy(0), vr(0), speed(0),
                   Erad(0), region(0), Eg{} {}

#ifdef RICH_MPI
    size_t dump(Serializer *s) const override
    {
        size_t off = 0;
        off += s->insert(pos.x);
        off += s->insert(pos.y);
        off += s->insert(pos.z);
        off += s->insert(velocity.x);
        off += s->insert(velocity.y);
        off += s->insert(velocity.z);
        off += s->insert(volume);
        off += s->insert(width);
        off += s->insert(density);
        off += s->insert(pressure);
        off += s->insert(temperature);
        off += s->insert(internal_energy);
        off += s->insert(vr);
        off += s->insert(speed);
        off += s->insert(Erad);
        off += s->insert(static_cast<double>(region));
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            off += s->insert(Eg[g]);
        return off;
    }

    size_t load(Serializer const *s, std::size_t offset)
    {
        size_t rd = 0;
        rd += s->extract(pos.x, offset + rd);
        rd += s->extract(pos.y, offset + rd);
        rd += s->extract(pos.z, offset + rd);
        rd += s->extract(velocity.x, offset + rd);
        rd += s->extract(velocity.y, offset + rd);
        rd += s->extract(velocity.z, offset + rd);
        rd += s->extract(volume, offset + rd);
        rd += s->extract(width, offset + rd);
        rd += s->extract(density, offset + rd);
        rd += s->extract(pressure, offset + rd);
        rd += s->extract(temperature, offset + rd);
        rd += s->extract(internal_energy, offset + rd);
        rd += s->extract(vr, offset + rd);
        rd += s->extract(speed, offset + rd);
        rd += s->extract(Erad, offset + rd);
        double reg_d = 0;
        rd += s->extract(reg_d, offset + rd);
        region = static_cast<int>(reg_d);
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            rd += s->extract(Eg[g], offset + rd);
        return rd;
    }
#endif
};

// ============================================================
// Gather cell results from all MPI ranks
// ============================================================
static std::vector<CellResult> gatherCellResults(
    Voronoi3D const &tess,
    std::vector<ComputationalCell3D> const &cells,
    Constants const &C)
{
    size_t N = tess.GetPointNo();
    std::vector<CellResult> local(N);

    for (size_t i = 0; i < N; ++i)
    {
        Vector3D p = tess.GetMeshPoint(i);
        double r = abs(p);
        double vol = tess.GetVolume(i);
        double w = tess.GetWidth(i);
        double spd = abs(cells[i].velocity);
        double vrad = (r > 1e-30) ? ScalarProd(cells[i].velocity, p) / r : 0.0;

        local[i].pos = p;
        local[i].velocity = cells[i].velocity;
        local[i].volume = vol;
        local[i].width = w;
        local[i].density = cells[i].density;
        local[i].pressure = cells[i].pressure;
        local[i].temperature = cells[i].temperature;
        local[i].internal_energy = cells[i].internal_energy;
        local[i].vr = vrad;
        local[i].speed = spd;
        local[i].Erad = cells[i].Erad;
        local[i].region = regionId(p, C);
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            local[i].Eg[g] = (g < cells[i].Eg.size()) ? cells[i].Eg[g] : 0.0;
    }

#ifdef RICH_MPI
    local = MPI_Gatherv_serializable(local, 0, MPI_COMM_WORLD);
#endif
    return local;
}

// ============================================================
// Optical depth helpers
// ============================================================
static double sigmaAtEnergy(double rho, double energy, Constants const &C)
{
    double kappa = C.kappa0 * std::pow(energy / C.E0, -C.alpha) + C.kappa_floor_mass;
    return std::clamp(rho * kappa, C.sigma_floor, C.sigma_max);
}

static double tauAtEnergy(double rho, double width, double energy, Constants const &C)
{
    return sigmaAtEnergy(rho, energy, C) * width;
}

// ============================================================
// Write radial profile
// ============================================================
static void writeRadialProfile(std::string const &path,
                               std::vector<CellResult> const &data,
                               Constants const &C,
                               std::vector<double> const &groupCenters)
{
    size_t G = groupCenters.size();
    size_t Nbins = 60;
    double rmax = C.box_half_width * std::sqrt(3.0);
    double dr = rmax / Nbins;

    struct Bin
    {
        double volume = 0, mass = 0, rho_w = 0, p_w = 0, vr_w = 0, spd_w = 0;
        double Tgas_w = 0, Erad_w = 0, Prad_pressure_w = 0, gas_mom_r = 0;
        double tau_low_w = 0, tau_mid_w = 0, tau_high_w = 0;
        std::vector<double> Eg_w;
        Bin() : Eg_w() {}
    };
    std::vector<Bin> bins(Nbins);
    for (auto &b : bins)
        b.Eg_w.assign(G, 0.0);

    for (auto const &c : data)
    {
        double r = abs(c.pos);
        size_t bi = static_cast<size_t>(r / dr);
        if (bi >= Nbins)
            bi = Nbins - 1;
        Bin &b = bins[bi];
        double m = c.density * c.volume;
        b.volume += c.volume;
        b.mass += m;
        b.rho_w += c.density * c.volume;
        b.p_w += c.pressure * c.volume;
        b.vr_w += c.vr * m;
        b.spd_w += c.speed * m;
        b.Tgas_w += c.temperature * m;
        double Erad_phys = c.Erad * c.density;
        b.Erad_w += Erad_phys * c.volume;
        double Prad_r = (r > 1e-30 && Erad_phys > 0) ? Erad_phys / 3.0 : 0.0;
        b.Prad_pressure_w += Prad_r * c.volume;
        b.gas_mom_r += m * c.vr;
        b.tau_low_w += tauAtEnergy(c.density, c.width, C.E_low_diag, C) * c.volume;
        b.tau_mid_w += tauAtEnergy(c.density, c.width, C.E_mid_diag, C) * c.volume;
        b.tau_high_w += tauAtEnergy(c.density, c.width, C.E_high_diag, C) * c.volume;
        for (size_t g = 0; g < G && g < ENERGY_GROUPS_NUM; ++g)
            b.Eg_w[g] += c.Eg[g] * c.density * c.volume;
    }

    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    out << "# r_mid volume mass rho_mean pressure_mean vr_mass_mean speed_mass_mean"
        << " Tgas_mass_mean Erad_vol_mean Prad_pressure_mean gas_Pradial"
        << " tau_low_mean tau_mid_mean tau_high_mean";
    for (size_t g = 0; g < G; ++g)
        out << " Eg_" << g;
    out << "\n";

    for (size_t i = 0; i < Nbins; ++i)
    {
        Bin const &b = bins[i];
        double rmid = (i + 0.5) * dr;
        if (b.volume <= 0 || b.mass <= 0)
        {
            out << rmid;
            for (size_t c = 0; c < 13 + G; ++c)
                out << " 0";
            out << "\n";
            continue;
        }
        out << rmid << " " << b.volume << " " << b.mass
            << " " << b.rho_w / b.volume
            << " " << b.p_w / b.volume
            << " " << b.vr_w / b.mass
            << " " << b.spd_w / b.mass
            << " " << b.Tgas_w / b.mass
            << " " << b.Erad_w / b.volume
            << " " << b.Prad_pressure_w / b.volume
            << " " << b.gas_mom_r
            << " " << b.tau_low_w / b.volume
            << " " << b.tau_mid_w / b.volume
            << " " << b.tau_high_w / b.volume;
        for (size_t g = 0; g < G; ++g)
            out << " " << b.Eg_w[g] / b.volume;
        out << "\n";
    }
    out.close();
}

// ============================================================
// Write spectrum at outer measurement shell
// ============================================================
static void writeSpectrum(std::string const &path,
                          std::vector<CellResult> const &data,
                          Constants const &C,
                          std::vector<double> const &groupCenters,
                          std::vector<double> const &groupBoundaries)
{
    size_t G = groupCenters.size();
    std::vector<double> S_outer(G, 0.0), S_cone(G, 0.0), S_offcone(G, 0.0);

    for (auto const &c : data)
    {
        double r = abs(c.pos);
        if (r < C.R_measure || r > C.R_domain)
            continue;
        bool cone = insideCone(c.pos, C);
        for (size_t g = 0; g < G && g < ENERGY_GROUPS_NUM; ++g)
        {
            double val = c.Eg[g] * c.density * c.volume;
            S_outer[g] += val;
            if (cone)
                S_cone[g] += val;
            else
                S_offcone[g] += val;
        }
    }

    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    out << "# g E_center E_low E_high S_outer S_cone S_offcone\n";
    for (size_t g = 0; g < G; ++g)
    {
        out << g << " " << groupCenters[g]
            << " " << groupBoundaries[g] << " " << groupBoundaries[g + 1]
            << " " << S_outer[g] << " " << S_cone[g] << " " << S_offcone[g]
            << "\n";
    }
    out.close();
}

// ============================================================
// Write angular diagnostics (cone vs off-cone)
// ============================================================
static void writeAngular(std::string const &path,
                         std::vector<CellResult> const &data,
                         Constants const &C)
{
    double mass_cone = 0, mass_offcone = 0;
    double Erad_cone = 0, Erad_offcone = 0;
    double mom_cone = 0, mom_offcone = 0;
    double Tgas_cone = 0, Tgas_offcone = 0;

    for (auto const &c : data)
    {
        double r = abs(c.pos);
        if (r < C.R_measure || r > C.R_domain)
            continue;
        double m = c.density * c.volume;
        double Erad_phys = c.Erad * c.density * c.volume;
        double mom_r = c.vr * m;
        bool cone = insideCone(c.pos, C);

        if (cone)
        {
            mass_cone += m;
            Erad_cone += Erad_phys;
            mom_cone += mom_r;
            Tgas_cone += c.temperature * m;
        }
        else
        {
            mass_offcone += m;
            Erad_offcone += Erad_phys;
            mom_offcone += mom_r;
            Tgas_offcone += c.temperature * m;
        }
    }

    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    out << "# quantity cone offcone\n";
    out << "mass " << mass_cone << " " << mass_offcone << "\n";
    out << "Erad " << Erad_cone << " " << Erad_offcone << "\n";
    out << "momentum " << mom_cone << " " << mom_offcone << "\n";
    out << "Tgas_mean " << (mass_cone > 0 ? Tgas_cone / mass_cone : 0)
        << " " << (mass_offcone > 0 ? Tgas_offcone / mass_offcone : 0) << "\n";
    out << "Erad_fraction " << (Erad_cone + Erad_offcone > 0 ? Erad_cone / (Erad_cone + Erad_offcone) : 0)
        << " " << (Erad_cone + Erad_offcone > 0 ? Erad_offcone / (Erad_cone + Erad_offcone) : 0) << "\n";
    out.close();
}

// ============================================================
// Write DDMC diagnostics (optical-depth per group)
// ============================================================
static void writeDDMCDiagnostics(std::string const &path,
                                 std::vector<CellResult> const &data,
                                 Constants const &C,
                                 std::vector<double> const &groupCenters)
{
    size_t G = groupCenters.size();
    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    out << "# g E_center tau_shell_mean ddmc_eligible_count shell_cell_count\n";

    for (size_t g = 0; g < G; ++g)
    {
        double tau_sum = 0;
        size_t shell_count = 0;
        for (auto const &c : data)
        {
            double r = abs(c.pos);
            if (r < C.R_cavity || r > C.R_shell_outer)
                continue;
            if (insideCone(c.pos, C))
                continue;
            double tau = tauAtEnergy(c.density, c.width, groupCenters[g], C);
            tau_sum += tau;
            ++shell_count;
        }
        double tau_mean = (shell_count > 0) ? tau_sum / shell_count : 0.0;
        size_t eligible = 0;
        for (auto const &c : data)
        {
            double r = abs(c.pos);
            if (r < C.R_cavity || r > C.R_shell_outer)
                continue;
            if (insideCone(c.pos, C))
                continue;
            if (tauAtEnergy(c.density, c.width, groupCenters[g], C) > 15.0)
                ++eligible;
        }
        out << g << " " << groupCenters[g] << " " << tau_mean
            << " " << eligible << " " << shell_count << "\n";
    }
    out.close();
}

// ============================================================
// Write HDF5 output
// ============================================================
static void writeHDF5(std::string const &path,
                      std::vector<CellResult> const &data,
                      Constants const &C,
                      std::vector<double> const &groupCenters,
                      std::vector<double> const &groupBoundaries,
                      bool useDDMC,
                      double finalTime, size_t finalCycle)
{
    try
    {
        size_t N = data.size();
        size_t G = groupCenters.size();

        H5::H5File file(path, H5F_ACC_TRUNC);

        auto writeAttrD = [&](H5::H5Object &obj, std::string const &name, double val) {
            H5::DataSpace sc(H5S_SCALAR);
            H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_DOUBLE, sc);
            attr.write(H5::PredType::NATIVE_DOUBLE, &val);
        };
        auto writeAttrI = [&](H5::H5Object &obj, std::string const &name, int val) {
            H5::DataSpace sc(H5S_SCALAR);
            H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_INT, sc);
            attr.write(H5::PredType::NATIVE_INT, &val);
        };
        auto writeAttrS = [&](H5::H5Object &obj, std::string const &name, std::string const &val) {
            H5::StrType strType(H5::PredType::C_S1, val.size() + 1);
            H5::DataSpace sc(H5S_SCALAR);
            H5::Attribute attr = obj.createAttribute(name, strType, sc);
            attr.write(strType, val.c_str());
        };

        writeAttrS(file, "case_name", "spherical_density_hardening_ddmc_push");
        writeAttrS(file, "variant", useDDMC ? "ddmc" : "imc");
        writeAttrI(file, "with_ddmc", useDDMC ? 1 : 0);
        writeAttrD(file, "time", finalTime);
        writeAttrI(file, "cycle", static_cast<int>(finalCycle));
        writeAttrI(file, "energy_groups_num", static_cast<int>(G));
        writeAttrD(file, "R_core", C.R_core);
        writeAttrD(file, "R_cavity", C.R_cavity);
        writeAttrD(file, "R_shell_outer", C.R_shell_outer);
        writeAttrD(file, "R_measure", C.R_measure);
        writeAttrD(file, "R_domain", C.R_domain);
        writeAttrD(file, "alpha", C.alpha);
        writeAttrD(file, "E0", C.E0);
        writeAttrD(file, "E_ref", C.E_ref);
        writeAttrD(file, "tau_ref", C.tau_ref);
        writeAttrD(file, "rho_core", C.rho_core);
        writeAttrD(file, "rho_cavity", C.rho_cavity);
        writeAttrD(file, "rho_shell", C.rho_shell);
        writeAttrD(file, "rho_atm", C.rho_atm);
        writeAttrD(file, "rho_buffer", C.rho_buffer);
        writeAttrD(file, "cone_density_factor", C.cone_density_factor);

        auto write1D = [&](H5::Group &grp, std::string const &name, std::vector<double> const &vec) {
            hsize_t dim = vec.size();
            H5::DataSpace sp(1, &dim);
            H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, sp);
            ds.write(vec.data(), H5::PredType::NATIVE_DOUBLE);
        };
        auto write2D = [&](H5::Group &grp, std::string const &name,
                           std::vector<double> const &flat, hsize_t rows, hsize_t cols) {
            hsize_t dims[2] = {rows, cols};
            H5::DataSpace sp(2, dims);
            H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, sp);
            ds.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
        };

        // /mesh
        H5::Group meshGrp = file.createGroup("/mesh");
        std::vector<double> pts(N * 3), vol(N), wid(N), rad(N);
        std::vector<double> regid(N);
        for (size_t i = 0; i < N; ++i)
        {
            pts[3 * i] = data[i].pos.x;
            pts[3 * i + 1] = data[i].pos.y;
            pts[3 * i + 2] = data[i].pos.z;
            vol[i] = data[i].volume;
            wid[i] = data[i].width;
            rad[i] = abs(data[i].pos);
            regid[i] = data[i].region;
        }
        write2D(meshGrp, "points", pts, N, 3);
        write1D(meshGrp, "volume", vol);
        write1D(meshGrp, "width", wid);
        write1D(meshGrp, "radius", rad);
        write1D(meshGrp, "region_id", regid);

        // /hydro
        H5::Group hydroGrp = file.createGroup("/hydro");
        std::vector<double> rho(N), prs(N), tmp(N), ie(N), vel(N * 3), vrv(N), mass(N), mom(N * 3);
        for (size_t i = 0; i < N; ++i)
        {
            rho[i] = data[i].density;
            prs[i] = data[i].pressure;
            tmp[i] = data[i].temperature;
            ie[i] = data[i].internal_energy;
            vel[3 * i] = data[i].velocity.x;
            vel[3 * i + 1] = data[i].velocity.y;
            vel[3 * i + 2] = data[i].velocity.z;
            vrv[i] = data[i].vr;
            double m = data[i].density * data[i].volume;
            mass[i] = m;
            mom[3 * i] = m * data[i].velocity.x;
            mom[3 * i + 1] = m * data[i].velocity.y;
            mom[3 * i + 2] = m * data[i].velocity.z;
        }
        write1D(hydroGrp, "density", rho);
        write1D(hydroGrp, "pressure", prs);
        write1D(hydroGrp, "temperature", tmp);
        write1D(hydroGrp, "internal_energy", ie);
        write2D(hydroGrp, "velocity", vel, N, 3);
        write1D(hydroGrp, "radial_velocity", vrv);
        write1D(hydroGrp, "mass", mass);
        write2D(hydroGrp, "momentum", mom, N, 3);

        // /radiation
        H5::Group radGrp = file.createGroup("/radiation");
        std::vector<double> erad(N), hardness(N), e_lo(N), e_mi(N), e_hi(N);
        std::vector<double> Eg_specific(N * G, 0.0), Eg_density(N * G, 0.0);
        size_t quarter = G / 4;
        for (size_t i = 0; i < N; ++i)
        {
            erad[i] = data[i].Erad * data[i].density;
            double Elow = 0, Ehigh = 0;
            for (size_t g = 0; g < quarter; ++g)
                Elow += data[i].Eg[g];
            for (size_t g = G - quarter; g < G; ++g)
                Ehigh += data[i].Eg[g];
            hardness[i] = (Elow > 0) ? Ehigh / Elow : 0.0;
            e_lo[i] = data[i].Eg[0] * data[i].density;
            e_mi[i] = (G > 1) ? data[i].Eg[G / 2] * data[i].density : 0.0;
            e_hi[i] = data[i].Eg[G - 1] * data[i].density;
            for (size_t g = 0; g < G && g < ENERGY_GROUPS_NUM; ++g)
            {
                Eg_specific[i * G + g] = data[i].Eg[g];
                Eg_density[i * G + g] = data[i].Eg[g] * data[i].density;
            }
        }
        write1D(radGrp, "Erad", erad);
        write2D(radGrp, "Eg_specific", Eg_specific, N, G);
        write2D(radGrp, "Eg_density", Eg_density, N, G);
        write1D(radGrp, "hardness", hardness);
        write1D(radGrp, "E_low", e_lo);
        write1D(radGrp, "E_mid", e_mi);
        write1D(radGrp, "E_high", e_hi);

        // /opacity
        H5::Group opGrp = file.createGroup("/opacity");
        std::vector<double> sl(N), sm(N), sh(N), tl(N), tm(N), th(N);
        std::vector<double> dl(N), dm(N), dh(N);
        for (size_t i = 0; i < N; ++i)
        {
            sl[i] = sigmaAtEnergy(data[i].density, C.E_low_diag, C);
            sm[i] = sigmaAtEnergy(data[i].density, C.E_mid_diag, C);
            sh[i] = sigmaAtEnergy(data[i].density, C.E_high_diag, C);
            tl[i] = sl[i] * data[i].width;
            tm[i] = sm[i] * data[i].width;
            th[i] = sh[i] * data[i].width;
            dl[i] = tl[i] > 15.0 ? 1.0 : 0.0;
            dm[i] = tm[i] > 15.0 ? 1.0 : 0.0;
            dh[i] = th[i] > 15.0 ? 1.0 : 0.0;
        }
        write1D(opGrp, "sigma_low", sl);
        write1D(opGrp, "sigma_mid", sm);
        write1D(opGrp, "sigma_high", sh);
        write1D(opGrp, "tau_low", tl);
        write1D(opGrp, "tau_mid", tm);
        write1D(opGrp, "tau_high", th);
        write1D(opGrp, "ddmc_eligible_low", dl);
        write1D(opGrp, "ddmc_eligible_mid", dm);
        write1D(opGrp, "ddmc_eligible_high", dh);

        // /groups
        H5::Group grpGrp = file.createGroup("/groups");
        write1D(grpGrp, "energy_center", groupCenters);
        write1D(grpGrp, "energy_boundary", groupBoundaries);

        file.close();
    }
    catch (H5::Exception const &e)
    {
        std::cerr << "HDF5 write failed for " << path << ": " << e.getDetailMsg() << std::endl;
        throw;
    }
}

// ============================================================
// Write summary
// ============================================================
static void writeSummary(std::string const &path,
                         std::vector<CellResult> const &imc,
                         std::vector<CellResult> const &ddmc,
                         Constants const &C,
                         std::vector<double> const &groupCenters,
                         double imcFinalTime, size_t imcFinalCycle,
                         double ddmcFinalTime, size_t ddmcFinalCycle)
{
    auto shellMomentum = [&](std::vector<CellResult> const &d) -> double {
        double mom = 0;
        for (auto const &c : d)
        {
            double r = abs(c.pos);
            if (r < C.R_cavity || r > C.R_shell_outer)
                continue;
            mom += c.vr * c.density * c.volume;
        }
        return mom;
    };

    double mom_imc = shellMomentum(imc);
    double mom_ddmc = shellMomentum(ddmc);

    size_t thick_count = 0, thin_count = 0;
    for (size_t g = 0; g < groupCenters.size(); ++g)
    {
        double tau_sum = 0;
        size_t cnt = 0;
        for (auto const &c : ddmc)
        {
            double r = abs(c.pos);
            if (r < C.R_cavity || r > C.R_shell_outer)
                continue;
            if (insideCone(c.pos, C))
                continue;
            tau_sum += tauAtEnergy(c.density, c.width, groupCenters[g], C);
            ++cnt;
        }
        double tau_mean = cnt > 0 ? tau_sum / cnt : 0;
        if (tau_mean > 15.0)
            ++thick_count;
        if (tau_mean < 1.0)
            ++thin_count;
    }

    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    out << "# Spherical density hardening DDMC push regression summary\n";
    out << "shell_momentum_imc " << mom_imc << "\n";
    out << "shell_momentum_ddmc " << mom_ddmc << "\n";
    out << "shell_momentum_rel " << (std::abs(mom_imc) > 1e-30
                                         ? std::abs(mom_ddmc - mom_imc) / std::abs(mom_imc)
                                         : 0.0)
        << "\n";
    out << "thick_low_energy_groups " << thick_count << "\n";
    out << "thin_high_energy_groups " << thin_count << "\n";
    out << "mom_imc_positive " << (mom_imc > 0 ? 1 : 0) << "\n";
    out << "mom_ddmc_positive " << (mom_ddmc > 0 ? 1 : 0) << "\n";
    out << "time_imc " << imcFinalTime << "\n";
    out << "time_ddmc " << ddmcFinalTime << "\n";
    out << "cycles_imc " << imcFinalCycle << "\n";
    out << "cycles_ddmc " << ddmcFinalCycle << "\n";
    out.close();
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try
    {
        constexpr size_t Nxyz = 64;
        size_t const G = ENERGY_GROUPS_NUM;

        // Energy groups
        std::vector<double> groupBoundary(G + 1);
        std::vector<double> groupCenter(G);
        double const Emin = 0.05 * units::ev;
        double const Emax = 30.0 * units::kev;
        groupBoundary[0] = Emin;
        double ratio = std::pow(Emax / Emin, 1.0 / G);
        for (size_t g = 0; g < G; ++g)
        {
            groupBoundary[g + 1] = groupBoundary[g] * ratio;
            groupCenter[g] = std::sqrt(groupBoundary[g] * groupBoundary[g + 1]);
        }
        for (size_t g = 0; g <= G; ++g)
            ComputationalCell3D::energyBoundaries[g] = groupBoundary[g];

        // Constants
        Constants C;
        double h_shell = 2.0 * C.box_half_width / Nxyz;
        C.kappa0 = C.tau_ref / (C.rho_shell * h_shell) *
                   std::pow(C.E_ref / C.E0, C.alpha);

        C.target_time = 2000.0 * (C.R_shell_outer - C.R_cavity) / units::clight;
        C.max_cycles = 500 * Nxyz / 64;

        // EOS
        double const cv = 1e15 / units::kev_kelvin;
        IdealGas eos(C.gamma, cv, 1, 0);

        // Mesh points (generated on rank 0, spread to all)
        double hw = C.box_half_width;
        Vector3D ll(-hw, -hw, -hw);
        Vector3D ur(hw, hw, hw);

        std::vector<Vector3D> points;
        if (rank == 0)
            points = CartesianMesh(Nxyz, Nxyz, Nxyz, ll, ur);
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Initial timestep seed for CFL controller
        double dt0 = std::min(3.0 * h_shell / units::clight,
                              0.05 * (C.R_shell_outer - C.R_cavity) / units::clight);

        if (rank == 0)
        {
            std::cout << "=== Spherical density-hardening DDMC push regression ===\n"
                      << "  Nxyz=" << Nxyz << "  G=" << G
                      << "  h_shell=" << h_shell
                      << "  kappa0=" << C.kappa0
                      << "  dt0=" << dt0
                      << "\n  target_time=" << C.target_time
                      << "  max_cycles=" << C.max_cycles
                      << "  new/cell=" << C.newPhotonsPerCell
                      << "  max/cell=" << C.maxPhotonsPerCell
                      << std::endl;
        }

        std::string const caseDir = fs::path(__FILE__).parent_path().string();

        // Storage for gathered results and final state per variant
        std::vector<CellResult> imcResults, ddmcResults;
        double imcFinalTime = 0, ddmcFinalTime = 0;
        size_t imcFinalCycle = 0, ddmcFinalCycle = 0;

        // Lambda: build initial conditions into pre-allocated cell vector
        auto initializeCells = [&](Voronoi3D const &tess,
                                   std::vector<ComputationalCell3D> &cells) {
            size_t N = tess.GetPointNo();
            cells.resize(N);
            for (size_t i = 0; i < N; ++i)
            {
                Vector3D p = tess.GetMeshPoint(i);
                double r = abs(p);
                double rho = initialDensity(p, C);

                cells[i].density = rho;
                cells[i].velocity = Vector3D(0, 0, 0);
                cells[i].temperature = (r < C.R_core) ? C.T_rad_core : C.T_shell_gas;
                cells[i].internal_energy = eos.dT2e(rho, cells[i].temperature,
                                                    cells[i].tracers,
                                                    ComputationalCell3D::tracerNames);
                cells[i].pressure = eos.de2p(rho, cells[i].internal_energy,
                                             cells[i].tracers,
                                             ComputationalCell3D::tracerNames);

                double Trad = (r < C.R_core) ? C.T_rad_core : C.T_rad_floor;
                double totalErad = 0;
                for (size_t g = 0; g < G; ++g)
                {
                    double u_g = planck_integral::planck_energy_density_group_integral(
                        groupBoundary[g], groupBoundary[g + 1], Trad);
                    cells[i].Eg[g] = u_g / rho;
                    totalErad += cells[i].Eg[g];
                }
                cells[i].Erad = totalErad;
            }
        };

        // ====================================================
        // Phase 1: Run IMC variant (independent adaptive timestep)
        // ====================================================
        if (rank == 0)
            std::cout << "\n========== Phase 1: IMC (no DDMC) ==========" << std::endl;
        {
            Voronoi3D tess(ll, ur);
            tess.BuildParallel(points);

            std::vector<ComputationalCell3D> initCells;
            initializeCells(tess, initCells);

            Simulation sim(tess, initCells, eos);

            ZeroForce3D zeroForce;
            auto tsf = std::make_shared<CourantFriedrichsLewy>(0.3, 1.0, zeroForce);
            sim.SetTimeStepFunction(tsf);

            Hllc3D rs;
            RigidWallGenerator3D ghost;
            LinearGauss3D interp(eos, ghost);

            IsBulkFace3D *isBulk = new IsBulkFace3D();
            IsBoundaryFace3D *isBoundary = new IsBoundaryFace3D();
            RegularFlux3D *normalFlux = new RegularFlux3D(rs);
            RigidWallFlux3D *rigidFlux = new RigidWallFlux3D(rs);

            std::vector<std::pair<ConditionActionFlux1::Condition3D const *,
                                  ConditionActionFlux1::Action3D const *>>
                fluxSeq;
            fluxSeq.push_back({isBoundary, rigidFlux});
            fluxSeq.push_back({isBulk, normalFlux});
            ConditionActionFlux1 flux(fluxSeq, interp);

            std::vector<std::pair<ConditionExtensiveUpdater3D::Condition3D const *,
                                  ConditionExtensiveUpdater3D::Action3D const *>>
                euSeq;
            ConditionExtensiveUpdater3D eu(euSeq);
            DefaultCellUpdater cu(false, 0, true, 0, nullptr);
            Eulerian3D pm;

            HDSim3D hdsim(tess, sim.getCells(), sim.getExtensives(), eos,
                          sim.getTracker(), pm, *tsf, flux, cu, eu, zeroForce,
                          std::make_pair(ComputationalCell3D::tracerNames,
                                        ComputationalCell3D::stickerNames));

            WriteSnapshot3D(hdsim, caseDir + "/spherical_push_snapshot_init_imc.h5");

            auto hydroStep = std::make_shared<HydroStep>(hdsim, HydroStep::TIMEADVANCE_2);

            auto eosPtr = std::make_shared<IdealGas>(eos);
            auto opacityPtr = std::make_shared<DensityPowerLawOpacity>(C, groupCenter, groupBoundary);

            std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
                std::make_shared<VacuumBoxBoundary<Vector3D, Tessellation3D>>(tess);

            RadiationIMCParameters imcParams = {
                .newPhotonsPerCell = C.newPhotonsPerCell,
                .withHydro = true,
                .diffusionPressureGradient = false,
                .MMC = false,
                .withMultigroupOpacity = true,
                .withRandomWalk = false,
                .withDDMC = false,
                .ddmcUseMultigroupPGRW = false,
                .noHydroFeedback = false,
                .withEgTimeAvg = true};

            auto physics = std::make_shared<RadiationIMC>(
                tess, boundaryCond, sim.getCells(), sim.getExtensives(),
                eosPtr, opacityPtr, imcParams);

            auto popControl = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(
                tess, C.maxPhotonsPerCell, 3);

            std::vector<Particle3D> initParticles;
            auto mcStep = std::make_shared<RadiationMCStep>(
                tess, sim.getCells(), sim.getExtensives(),
                physics, popControl, boundaryCond,
                initParticles, 0, true
#ifdef RICH_MPI
                ,
                RadiationMCStep::ManagerType::AUTO_RDMA
#endif
            );

            sim.addPhysics(hydroStep);
            sim.addPhysics(mcStep);

#ifdef RICH_MPI
            mcStep->setCost(std::make_shared<IMCCostCalculator>(mcStep->getManager()));
            sim.setForceRebalanceSteps(4);
            sim.addMigrationBuffer(mcStep->getManager()->GetCellsStepsCounters());
#endif

            sim.SetTimeStep(dt0);

            std::ofstream dtLog;
            if (rank == 0)
                dtLog.open(caseDir + "/spherical_push_dt_history_imc.txt");
            if (rank == 0)
                dtLog << "# cycle time dt_used dt_next\n" << std::scientific << std::setprecision(8);

            auto wallStart = std::chrono::high_resolution_clock::now();
            while (sim.GetTime() < C.target_time &&
                   sim.GetCycle() < C.max_cycles)
            {
                double tBefore = sim.GetTime();
                sim.step();
                double tAfter = sim.GetTime();
                double dtUsed = tAfter - tBefore;
                double dtNext = sim.GetTimeStep();

                if (rank == 0)
                {
                    dtLog << sim.GetCycle() << " " << tAfter
                          << " " << dtUsed << " " << dtNext << "\n";
                    double elapsed = std::chrono::duration<double>(
                                         std::chrono::high_resolution_clock::now() - wallStart)
                                         .count();
                    std::cout << "[IMC] Cycle " << sim.GetCycle()
                              << "  t=" << tAfter << "  dt=" << dtUsed
                              << "  next_dt=" << dtNext
                              << "  wall=" << static_cast<int>(elapsed) << "s" << std::endl;
                }
            }
            if (rank == 0)
                dtLog.close();

            imcFinalTime = sim.GetTime();
            imcFinalCycle = sim.GetCycle();

            {
                auto &cells = sim.getCells();
                size_t N = tess.GetPointNo();
                auto const &EgTA = mcStep->getEgTimeAvg();
                auto const &EradTA = mcStep->getEradTimeAvg();
                std::vector<double> savedErad(N);
                std::vector<std::vector<double>> savedEg(N);
                for (size_t i = 0; i < N; ++i)
                {
                    savedErad[i] = cells[i].Erad;
                    savedEg[i].assign(cells[i].Eg.begin(), cells[i].Eg.end());
                    if (i < EradTA.size())
                        cells[i].Erad = EradTA[i];
                    if (i < EgTA.size())
                        for (size_t g = 0; g < G; ++g)
                            cells[i].Eg[g] = EgTA[i][g];
                }
                WriteSnapshot3D(hdsim, caseDir + "/spherical_push_snapshot_final_imc.h5");
                for (size_t i = 0; i < N; ++i)
                {
                    cells[i].Erad = savedErad[i];
                    for (size_t g = 0; g < G; ++g)
                        cells[i].Eg[g] = savedEg[i][g];
                }
            }

            imcResults = gatherCellResults(tess, sim.getCells(), C);

            if (rank == 0)
            {
                writeRadialProfile(caseDir + "/spherical_push_imc_radial_profile.txt",
                                   imcResults, C, groupCenter);
                writeSpectrum(caseDir + "/spherical_push_imc_spectrum.txt",
                              imcResults, C, groupCenter, groupBoundary);
                writeAngular(caseDir + "/spherical_push_imc_angular.txt", imcResults, C);
                writeHDF5(caseDir + "/spherical_push_final_imc.h5",
                          imcResults, C, groupCenter, groupBoundary,
                          false, imcFinalTime, imcFinalCycle);
                std::cout << "IMC outputs written (t=" << imcFinalTime
                          << ", cycles=" << imcFinalCycle << ")." << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ====================================================
        // Phase 2: Run DDMC variant (independent adaptive timestep)
        // ====================================================
        if (rank == 0)
            std::cout << "\n========== Phase 2: DDMC ==========" << std::endl;
        {
            Voronoi3D tess(ll, ur);
            tess.BuildParallel(points);

            std::vector<ComputationalCell3D> initCells;
            initializeCells(tess, initCells);

            Simulation sim(tess, initCells, eos);

            ZeroForce3D zeroForce;
            auto tsf = std::make_shared<CourantFriedrichsLewy>(0.3, 1.0, zeroForce);
            sim.SetTimeStepFunction(tsf);

            Hllc3D rs;
            RigidWallGenerator3D ghost;
            LinearGauss3D interp(eos, ghost);

            IsBulkFace3D *isBulk = new IsBulkFace3D();
            IsBoundaryFace3D *isBoundary = new IsBoundaryFace3D();
            RegularFlux3D *normalFlux = new RegularFlux3D(rs);
            RigidWallFlux3D *rigidFlux = new RigidWallFlux3D(rs);

            std::vector<std::pair<ConditionActionFlux1::Condition3D const *,
                                  ConditionActionFlux1::Action3D const *>>
                fluxSeq;
            fluxSeq.push_back({isBoundary, rigidFlux});
            fluxSeq.push_back({isBulk, normalFlux});
            ConditionActionFlux1 flux(fluxSeq, interp);

            std::vector<std::pair<ConditionExtensiveUpdater3D::Condition3D const *,
                                  ConditionExtensiveUpdater3D::Action3D const *>>
                euSeq;
            ConditionExtensiveUpdater3D eu(euSeq);
            DefaultCellUpdater cu(false, 0, true, 0, nullptr);
            Eulerian3D pm;

            HDSim3D hdsim(tess, sim.getCells(), sim.getExtensives(), eos,
                          sim.getTracker(), pm, *tsf, flux, cu, eu,
                          zeroForce,
                          std::make_pair(ComputationalCell3D::tracerNames,
                                        ComputationalCell3D::stickerNames));

            WriteSnapshot3D(hdsim, caseDir + "/spherical_push_snapshot_init_ddmc.h5");

            auto hydroStep = std::make_shared<HydroStep>(hdsim, HydroStep::TIMEADVANCE_2);

            auto eosPtr = std::make_shared<IdealGas>(eos);
            auto opacityPtr = std::make_shared<DensityPowerLawOpacity>(C, groupCenter, groupBoundary);

            std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
                std::make_shared<VacuumBoxBoundary<Vector3D, Tessellation3D>>(tess);

            RadiationIMCParameters ddmcParams = {
                .newPhotonsPerCell = C.newPhotonsPerCell,
                .withHydro = true,
                .diffusionPressureGradient = false,
                .MMC = false,
                .withMultigroupOpacity = true,
                .withRandomWalk = false,
                .withDDMC = true,
                .ddmcMinCellOpticalDepth = 15.0,
                .ddmcUseMultigroupPGRW = true,
                .noHydroFeedback = false,
                .withEgTimeAvg = true};

            auto physics = std::make_shared<RadiationIMC>(
                tess, boundaryCond, sim.getCells(), sim.getExtensives(),
                eosPtr, opacityPtr, ddmcParams);

            auto popControl = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(
                tess, C.maxPhotonsPerCell, 3);

            std::vector<Particle3D> initParticles;
            auto mcStep = std::make_shared<RadiationMCStep>(
                tess, sim.getCells(), sim.getExtensives(),
                physics, popControl, boundaryCond,
                initParticles, 0, true
#ifdef RICH_MPI
                ,
                RadiationMCStep::ManagerType::AUTO_RDMA
#endif
            );

            sim.addPhysics(hydroStep);
            sim.addPhysics(mcStep);

#ifdef RICH_MPI
            mcStep->setCost(std::make_shared<IMCCostCalculator>(mcStep->getManager()));
            sim.setForceRebalanceSteps(4);
            sim.addMigrationBuffer(mcStep->getManager()->GetCellsStepsCounters());
#endif

            sim.SetTimeStep(dt0);

            std::ofstream dtLog;
            if (rank == 0)
                dtLog.open(caseDir + "/spherical_push_dt_history_ddmc.txt");
            if (rank == 0)
                dtLog << "# cycle time dt_used dt_next\n" << std::scientific << std::setprecision(8);

            auto wallStart = std::chrono::high_resolution_clock::now();
            while (sim.GetTime() < C.target_time &&
                   sim.GetCycle() < C.max_cycles)
            {
                double tBefore = sim.GetTime();
                sim.step();
                double tAfter = sim.GetTime();
                double dtUsed = tAfter - tBefore;
                double dtNext = sim.GetTimeStep();

                if (rank == 0)
                {
                    dtLog << sim.GetCycle() << " " << tAfter
                          << " " << dtUsed << " " << dtNext << "\n";
                    double elapsed = std::chrono::duration<double>(
                                         std::chrono::high_resolution_clock::now() - wallStart)
                                         .count();
                    std::cout << "[DDMC] Cycle " << sim.GetCycle()
                              << "  t=" << tAfter << "  dt=" << dtUsed
                              << "  next_dt=" << dtNext
                              << "  wall=" << static_cast<int>(elapsed) << "s" << std::endl;
                }
            }
            if (rank == 0)
                dtLog.close();

            ddmcFinalTime = sim.GetTime();
            ddmcFinalCycle = sim.GetCycle();

            {
                auto &cells = sim.getCells();
                size_t N = tess.GetPointNo();
                auto const &EgTA = mcStep->getEgTimeAvg();
                auto const &EradTA = mcStep->getEradTimeAvg();
                std::vector<double> savedErad(N);
                std::vector<std::vector<double>> savedEg(N);
                for (size_t i = 0; i < N; ++i)
                {
                    savedErad[i] = cells[i].Erad;
                    savedEg[i].assign(cells[i].Eg.begin(), cells[i].Eg.end());
                    if (i < EradTA.size())
                        cells[i].Erad = EradTA[i];
                    if (i < EgTA.size())
                        for (size_t g = 0; g < G; ++g)
                            cells[i].Eg[g] = EgTA[i][g];
                }
                WriteSnapshot3D(hdsim, caseDir + "/spherical_push_snapshot_final_ddmc.h5");
                for (size_t i = 0; i < N; ++i)
                {
                    cells[i].Erad = savedErad[i];
                    for (size_t g = 0; g < G; ++g)
                        cells[i].Eg[g] = savedEg[i][g];
                }
            }

            ddmcResults = gatherCellResults(tess, sim.getCells(), C);

            if (rank == 0)
            {
                writeRadialProfile(caseDir + "/spherical_push_ddmc_radial_profile.txt",
                                   ddmcResults, C, groupCenter);
                writeSpectrum(caseDir + "/spherical_push_ddmc_spectrum.txt",
                              ddmcResults, C, groupCenter, groupBoundary);
                writeAngular(caseDir + "/spherical_push_ddmc_angular.txt", ddmcResults, C);
                writeDDMCDiagnostics(caseDir + "/spherical_push_ddmc_diagnostics.txt",
                                     ddmcResults, C, groupCenter);
                writeHDF5(caseDir + "/spherical_push_final_ddmc.h5",
                          ddmcResults, C, groupCenter, groupBoundary,
                          true, ddmcFinalTime, ddmcFinalCycle);
                std::cout << "DDMC outputs written (t=" << ddmcFinalTime
                          << ", cycles=" << ddmcFinalCycle << ")." << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ====================================================
        // Phase 3: Write summary
        // ====================================================
        if (rank == 0)
        {
            writeSummary(caseDir + "/spherical_push_summary.txt",
                         imcResults, ddmcResults, C, groupCenter,
                         imcFinalTime, imcFinalCycle,
                         ddmcFinalTime, ddmcFinalCycle);

            std::cout << "\n=== All outputs written to " << caseDir << " ===" << std::endl;
        }
    }
    catch (UniversalError const &e)
    {
        std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
        reportError(e);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch (std::exception const &e)
    {
        std::cerr << "=== std::exception on rank " << rank << ": " << e.what() << " ===" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
