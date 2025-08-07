#include <fenv.h>

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/numeric/odeint.hpp>
#include <exception>
#include <filesystem>

#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/GeometryCommon/UpdateBox.hpp"
#include "source/3D/output/DiagnosticAppendix3D.hpp"
#include "source/3D/output/read3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/tesselation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/misc/int2str.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LagrangianExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/MonopoleSelfGravity3D.hpp"
// #include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"
#include "source/newtonian/three_dimensional/TDE_force.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
namespace fs = std::filesystem;
#ifdef RICH_MPI
#include "source/mpi/ConstNumberPerProc3D.hpp"
#include "source/mpi/SetLoad3D.hpp"
#include "source/mpi/mpi_commands.hpp"
#endif
#include <sys/stat.h>

#include <boost/math/tools/roots.hpp>
#include <sstream>

typedef std::array<double, 4> state_type;

#define smooth_factor 0.6

using EOS_type = IdealGas;
namespace {

    class MassRefine : public CellsToRefine3D {
       private:
        double domain_size_, Mbh_, Mstar_, Rstar_;

       public:
        void SetSize(double s) { domain_size_ = s; }

        MassRefine(double domainsize, double Mbh, double Mstar, double Rstar)
            : domain_size_(domainsize), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar) {}

        std::pair<vector<size_t>, vector<Vector3D>> ToRefine(Tessellation3D const &tess,
                                                             vector<ComputationalCell3D> const &cells,
                                                             double time) const {
            std::vector<std::vector<double>> maxr;
            std::vector<std::vector<double>> phi;
            std::vector<double> theta;
            size_t Norg = tess.GetPointNo();
            vector<size_t> res;
            double MaxMass = 1.5e-7;
            std::vector<size_t> neigh;
            std::vector<double> volumes = tess.GetAllVolumes();
            double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
            double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0);
            double const min_cell_size = Rt * 1e-2;
            double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);

            for (size_t i = 0; i < Norg; ++i) {
                if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15)) continue;
                double r_dist = std::max(fastabs(tess.GetMeshPoint(i)), Rt * smooth_factor);
                if (tess.GetWidth(i) < min_cell_size * (r_dist < 0.65 * Rt ? smooth_factor / 0.6 : 1)) continue;

                if ((r_dist > 0.65 * Rt && r_dist < 2 * Rt) || r_dist > apocenter || r_dist < smooth_factor * Rt)
                    continue;

                double MaxMass2 = (tess.GetMeshPoint(i).x > (-apocenter * 2.5)) ? MaxMass : MaxMass * 30;
                MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));

                double V = tess.GetVolume(i);
                tess.GetNeighbors(i, neigh);
                size_t Nneigh = neigh.size();
                bool good = true, good2 = false;
                for (size_t j = 0; j < Nneigh; ++j) {
                    if (!tess.IsPointOutsideBox(neigh[j])) {
                        if (fastabs(tess.GetCellCM(neigh[j]) - tess.GetMeshPoint(neigh[j])) >
                            (0.09 * std::pow(volumes[neigh[j]], 0.33333333333))) {
                            good = false;
                            break;
                        }
                        if ((5 * volumes[neigh[j]]) < V) good2 = true;
                    }
                }
                if (!good) continue;
                if (good2) {
                    res.push_back(i);
                    continue;
                }
                if ((V * cells[i].density) > (MaxMass2 * std::min(r_dist * r_dist / (50 * Rt * Rt), 1.0)) ||
                    V > domain_size_ * 1e-5) {
                    {
                        res.push_back(i);
                        continue;
                    }
                }
            }
            return std::pair<vector<size_t>, vector<Vector3D>>(res, vector<Vector3D>());
        }
    };

    class RemoveBig : public CellsToRemove3D {
       private:
        double domain_size_, Mbh_, Mstar_, Rstar_;
        EOS_type const &eos_;

       public:
        void SetSize(double s) { domain_size_ = s; }

        RemoveBig(double domain_size, EOS_type const &eos, double Mbh, double Mstar, double Rstar)
            : domain_size_(domain_size), eos_(eos), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar) {}

        std::pair<vector<size_t>, vector<double>> ToRemove(Tessellation3D const &tess,
                                                           vector<ComputationalCell3D> const &cells,
                                                           double time) const {
            std::vector<std::vector<double>> maxr;
            std::vector<std::vector<double>> phi;
            std::vector<double> theta;
            vector<size_t> res;
            vector<double> merits;
            vector<size_t> neigh;
            size_t Norg = tess.GetPointNo();
            std::vector<double> volumes = tess.GetAllVolumes();
            double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
            double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0);
            double const time_Rt = std::sqrt(Rt * Rt * Rt / Mbh_);
            double const min_cell_size = Rt * 1e-2;
            double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);

            double MaxMass = 3e-8;
            for (size_t i = 0; i < Norg; ++i) {
                bool good = true;
                // Do we have little mass amount?
                if (Norg < 500) continue;
                double Vol = tess.GetVolume(i);
                double w = tess.GetWidth(i);
                double MaxMass2 = (tess.GetMeshPoint(i).x > -Rt * apocenter * 2.5) ? MaxMass : MaxMass * 30;
                double const r_org = fastabs(tess.GetMeshPoint(i));
                double r_i = std::max(Rt * smooth_factor, r_org);
                MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));
                MaxMass2 = MaxMass2 * std::min(r_i * r_i / (50 * Rt * Rt), 1.0);
                double const dt = w / eos_.dp2c(cells[i].density, cells[i].pressure, cells[i].tracers);
                double const in_factor = r_i < 0.65 * Rt ? smooth_factor / 0.6 : 1;
                MaxMass2 *= std::max(1.0, std::pow(r_i / r_org, 2.0));
                if (Vol * cells[i].density > MaxMass2 && w > (in_factor * 0.7 * min_cell_size) &&
                    dt > (0.02 * time_Rt * in_factor))
                    continue;
                if (Vol > domain_size_ * 0.5e-5) continue;
                // Make sure we are not that much bigger than smallest neighbor
                tess.GetNeighbors(i, neigh);
                size_t Nneigh = neigh.size();
                for (size_t j = 0; j < Nneigh; ++j) {
                    if (!tess.IsPointOutsideBox(neigh[j]))
                        if (volumes[neigh[j]] < Vol * 0.5) {
                            good = false;
                            break;
                        }
                }
                if (good) {
                    // Make sure we are not too high aspect ratio
                    if (fastabs(tess.GetMeshPoint(i) - tess.GetCellCM(i)) > 0.15 * tess.GetWidth(i)) good = false;
                }
                if (good) {
                    res.push_back(i);
                    merits.push_back(1.0 / Vol);
                }
            }
            return std::pair<vector<size_t>, vector<double>>(res, merits);
        }
    };

    /*vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, EOS_type const &eos,
                                         double const Punits) {
        vector<double> xsi = read_vector("/home/esternberg/RICH/data/xsi.txt");
        vector<double> theta = read_vector("/home/esternberg/RICH/data/theta.txt");
        xsi[0] = 0;

        double n = 1.5;
        double endfactor = 2.714;

        double alpha = R / xsi.back();
        double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
        double K = alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));

        size_t N = tess.GetPointNo();
        vector<ComputationalCell3D> res(N);
        for (size_t i = 0; i < N; ++i) {
            Vector3D const &point = tess.GetMeshPoint(i);
            double r = abs(point);
            double t = 0;
            if (r < R) {
                t = LinearInterpolation(xsi, theta, r / alpha);
                res[i].tracers[1] = (1);
                res[i].density = std::max(rho_c * std::pow(t, n), 1e-5);
            } else {
                t = theta.back() * 10;
                res[i].density = rho_c * std::pow(t, n);
                res[i].tracers[1] = (0);
            }
            res[i].tracers[4] = 0;
            double const P = K * std::pow(res[i].density, 1 + 1.0 / n);
            double const a = CG::radiation_constant;
            double const d = res[i].density;
            auto f = [&eos, d, P, a, Punits](double const x) {
                return P - eos.dT2p(d, x) - Punits * a * x * x * x * x / 3;
            };
            boost::math::tools::eps_tolerance<double> tol(10);
            std::uintmax_t it = 150;
            std::pair<double, double> Tres = boost::math::tools::bracket_and_solve_root(f, 1e4, 2.0, false, tol, it);
            double const T = 0.5 * (Tres.first + Tres.second);
            double const wrongT = eos.dp2T(d, P);
            res[i].internal_energy = eos.dT2e(res[i].density, T, res[i].tracers);
            res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy);
            res[i].Erad = 7.5657e-15 * T * T * T * T * 1603 * 1603 / (7e10 * 7e10 * res[i].density);
            res[i].temperature = T;
            res[i].tracers[0] = (eos.dp2s(res[i].density, res[i].pressure, res[i].tracers));
            res[i].tracers[2] = (0);
            res[i].tracers[3] = (0);
        }
        return res;
    }*/

    vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, EOS_type const &eos,
                                         double const /*Punits*/) {
        const size_t N = tess.GetPointNo();
        vector<ComputationalCell3D> res(N);

        const double rho_c = M / (4.0 / 3.0 * M_PI * std::pow(R, 3));
        const double T_c = 1e6;         //
        const double rho_floor = 1e-5;  //
        const double T_floor = 1e5;     //
        const double a_rad = CG::radiation_constant;
        const double sigma = 0.4 * R;         //
        const double scale_height = 0.5 * R;  //

        for (size_t i = 0; i < N; ++i) {
            Vector3D const &point = tess.GetMeshPoint(i);
            double r = abs(point);

            double rho = rho_c * std::exp(-r / scale_height) * std::exp(-r * r / (2 * sigma * sigma));
            rho = std::max(rho, rho_floor);
            res[i].density = rho;

            double T = (r < R) ? T_c : T_floor;
            res[i].temperature = T;

            res[i].internal_energy = eos.dT2e(rho, T);
            res[i].pressure = eos.de2p(rho, res[i].internal_energy);
            res[i].Erad = std::min(a_rad * std::pow(T, 4) / rho, 1e4);

            res[i].tracers[1] = (r < R) ? 1.0 : 0.0;
            res[i].tracers[0] = eos.dp2s(rho, res[i].pressure);
            res[i].tracers[2] = res[i].tracers[3] = res[i].tracers[4] = 0.0;
        }

        return res;
    }

    ComputationalCell3D GetReferenceCell(EOS_type const &eos, Tessellation3D const &tess, double time) {
        double R = 1;
        double M = 1;
        /*double n = 3;
        double endfactor = 2.01824;
        double alpha = R / 6.89684;*/
        double n = 1.5;
        double endfactor = 2.714;
        double alpha = R / 3.65375;
        double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
        double K = alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));
        ComputationalCell3D reference;
        std::pair<Vector3D, Vector3D> box = tess.GetBoxCoordinates();
        double dfactor = std::min(1.0, std::max(0.01, std::exp(-(time - 900) / 100.0)));
        double mindensity =
            dfactor * 1e-6 * M /
            ((box.second.x - box.first.x) * (box.second.z - box.first.z) * (box.second.y - box.first.y));
        mindensity = std::max(mindensity, 1e-20);
        reference.density = mindensity;
        double const Tref = 1e3;
        reference.Erad = 7.5657e-15 * Tref * Tref * Tref * Tref * 1603 * 1603 / (7e10 * 7e10 * reference.density);
        reference.pressure = eos.dT2p(reference.density, Tref, reference.tracers);
        reference.velocity = Vector3D();
        reference.internal_energy = eos.dp2e(reference.density, reference.pressure, reference.tracers);
        reference.temperature = Tref;
        reference.tracers[0] = (eos.dp2s(reference.density, reference.pressure, reference.tracers));
        reference.tracers[1] = (0);
        reference.tracers[2] = (0);
        reference.tracers[3] = (0);
        return reference;
    }

}  // namespace

class ZeroOpacity : public DiffusionCoefficientCalculator {
   public:
    ZeroOpacity() {};

    double CalcDiffusionCoefficient(ComputationalCell3D const &cell) const override { return 0; };

    double CalcPlanckOpacity(ComputationalCell3D const &cell) const override { return 0; };

    double CalcScatteringOpacity(ComputationalCell3D const &cell) const override { return 0; };
};

int main(void) {
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    std::string run_directory("./");
    double const R = 1;
    double const M = 1;
    double const Mbh = 1e6;
    double const beta = 1;
    std::stringstream ss;
    ss << "R" << R << "M" << M << "BH" << Mbh << "beta" << beta << "S" << static_cast<size_t>(smooth_factor * 100);
    std::string const run_name = ss.str();
    run_directory += run_name + "/";
    fs::create_directory(run_directory.c_str());
    double const Rt = R * std::cbrt(Mbh / M);
    double const Rp = Rt / beta;
    double const apocenter = Rp * std::cbrt(Mbh / M);
    std::string file_name = run_directory + "snap_";
    int counter = 0;

    double const lscale = 7e10;
    double const mscale = 2e33;
    double const tscale = 1603;

    std::cout << "start eos" << std::endl;
    EOS_type eos(5. / 3, 2.06e8, 0., 0.);
    std::cout << "end eos" << std::endl;

    // Radiation

    std::cout << "end sta" << std::endl;

    const double width = 5;
    Vector3D ll(-width, -width, -width), ur(width, width, width);
    Voronoi3D tess(ll, ur);

    vector<ComputationalCell3D> cells;
    double tstart = 0;
    Snapshot3D snap;

    // double startfactor = 3;
    // double fstart = -acos(2 * Rp / (startfactor * Rt) - 1);
    // tstart = 0.3333333 * sqrt(2 * Rp * Rp * Rp / Mbh) * tan(0.5 * fstart) * (3 + tan(0.5 * fstart) * tan(0.5 *
    // fstart));

    size_t const np = std::min(1e3, 1e6 * std::sqrt(Mbh / 1e4));
    vector<Vector3D> ptemp = RandSphereR(np, ll, ur, 0, R * 1.1, Vector3D());
    vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 * R, R * 1.05, Vector3D());
    vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, R, 1.4 * width, Vector3D());
    ptemp.insert(ptemp.end(), ptemp2.begin(), ptemp2.end());
    ptemp.insert(ptemp.end(), ptemp3.begin(), ptemp3.end());

    vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15, &tess);
    std::cout << "Points generated: " << points.size() << std::endl;
    // tess.Build(points);
    cells = GetCells(tess, M, R, eos, tscale * tscale * lscale / mscale);
    std::cout << "Cells generated: " << cells.size() << std::endl;
    ComputationalCell3D::tracerNames.push_back("Entropy");
    ComputationalCell3D::tracerNames.push_back("Star");

    std::cout << "Finished build" << std::endl;

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
    double Tmin = 1e3;

    Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos, 3.25, 0.01, false, 1.25);

    /*
    ZeroOpacity opacity;
    DiffusionOpenBoundary D_boundary;
    Diffusion matrix_builder(opacity, D_boundary, eos);
    matrix_builder.length_scale_ = lscale;
    matrix_builder.time_scale_ = tscale;
    matrix_builder.mass_scale_ = mscale;
    std::shared_ptr<DiffusionForce> rad_force = std::make_shared<DiffusionForce>(matrix_builder, eos);
    DefaultCellUpdater cu(false, 0, true, 0, &matrix_builder);*/
    DefaultCellUpdater cu(false, 0, true, 0, nullptr);

    RigidWallFlux3D rigidflux(rs);
    RegularFlux3D *regular_flux = new RegularFlux3D(rs);
    IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
    IsBulkFace3D *bulk_face = new IsBulkFace3D();
    vector<pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>> flux_vector;
    flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>(
        boundary_face, &rigidflux));
    flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>(
        bulk_face, regular_flux));
    ConditionActionFlux1 fc(flux_vector, interp);

    vector<pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>>
        eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);
    MonopoleSelfGravity3D sg(100, 0.1);
    TDEGravity acc(Mbh, M, R, beta, sg, true);
    std::shared_ptr<ConservativeForce3D> gravity_force = std::make_shared<ConservativeForce3D>(acc, false);
    std::vector<std::shared_ptr<SourceTerm3D>> forces;

    forces.push_back(gravity_force);
    // forces.push_back(rad_force);
    SeveralSources3D force(forces);
    CourantFriedrichsLewy tsf(0.225, 1, force, std::vector<std::string>(), false);
    std::unique_ptr<HDSim3D> sim;

    sim = std::make_unique<HDSim3D>(tess, cells, eos, pm, tsf, fc, cu, eu, force,
                                    std::pair<std::vector<std::string>, std::vector<std::string>>(
                                        ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames),
                                    false, true);
    sim->SetTime(tstart);

    // double init_dt = 1e-4;
    double init_dt = 10;
    tsf.SetTimeStep(init_dt);
    std::cout << "Restart time " << sim->getTime() << std::endl;
    ComputationalCell3D reference_cell = GetReferenceCell(eos, tess, sim->getTime());
    double tf = 4 * std::sqrt(apocenter * apocenter * apocenter / Mbh);
    std::cout << "!!!@@@" << tf << std::endl;
    double mindt = 0.001;
    double nextT = 0;
    nextT = sim->getTime();
    nextT += init_dt;

    RemoveBig remove(8 * width * width * width, eos, Mbh, M, R);
    // CellsToRemove3D remove;
    MassRefine refine(8 * width * width * width, Mbh, M, R);
    PCM3D ainterp(ghost);
    AMR3D amr(eos, refine, remove, interp);
    std::pair<Vector3D, Vector3D> box2 = sim->getTesselation().GetBoxCoordinates();
    double newvol2 = (box2.second.x - box2.first.x) * (box2.second.y - box2.first.y) * (box2.second.z - box2.first.z);
    refine.SetSize(newvol2);
    remove.SetSize(newvol2);
    vector<DiagnosticAppendix3D *> appendices;
    double old_t = sim->getTime();
    double old_dt = init_dt;
    double step_time = 0;
    double const restart_wtime = 20000;
    double const min_dt_output = 0.02 * std::sqrt(std::pow(R, 3.0) * Mbh / M);
    WriteSnapshot3D(*sim, "init.h5", appendices, true);
    while (sim->getTime() < tf) {
        if (sim->getCycle() % 10 == 0) {
            std::cout << "Cycle " << sim->getCycle() << " Time " << sim->getTime() << std::endl;
        }
        // if (sim->getTime() > nextT) {
        WriteSnapshot3D(*sim, file_name + int2str(counter) + ".h5", appendices, true);
        nextT = sim->getTime() + init_dt;
        ++counter;
        //}
        try {
            int restart_dump = 0;
            // sim->RadiationTimeStep(old_dt * 1e-2, matrix_builder);
            // sim->RadiationTimeStep(old_dt * 1e-1, matrix_builder);
            // double new_dt = sim->RadiationTimeStep(old_dt * 0.89, matrix_builder);
            // sim->RadiationTimeStep(old_dt * 1e-3, matrix_builder);
            // sim->RadiationTimeStep(old_dt * 1e-3, matrix_builder);
            // double new_dt = sim->RadiationTimeStep(old_dt * 1, matrix_builder);
            // tsf.SetTimeStep(1e6);
            // std::cout << "Finished rad step" << std::endl;
            sim->timeAdvance2();
            std::cout << "Finished hydro step" << std::endl;

            old_dt = sim->getTime() - old_t;
            old_t = sim->getTime();
            reference_cell = GetReferenceCell(eos, tess, sim->getTime());
            if (sim->getCycle() % 7 == 0) {
                UpdateBox(sim->getTesselation(), *sim, 0.5, 1e-5, reference_cell);
                std::pair<Vector3D, Vector3D> box = sim->getTesselation().GetBoxCoordinates();
                double newvol =
                    (box.second.x - box.first.x) * (box.second.y - box.first.y) * (box.second.z - box.first.z);
                refine.SetSize(newvol);
                remove.SetSize(newvol);
            }
        } catch (UniversalError const &eo) {
            reportError(eo);
            throw;
        }
    }
    return 0;
}
