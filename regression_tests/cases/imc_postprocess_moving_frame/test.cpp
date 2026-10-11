// Energy accounting of post-process transport through moving gas.
//
// A hot, opaque, static core sits inside a pure-scattering envelope (no
// absorption) that moves radially with speed beta*c, outward or inward. The
// core is inside the CER flux-source surface, so all radiation is injected at
// the source faces and the envelope only scatters it. With no absorption the
// radiation energy changes only through work exchanged with the moving gas:
//
//   gain G = (box escape + timed out + absorbed + cutoff) / emitted
//
//   static envelope   G = 1 (weights never change)
//   outflow           G < 1 (radiation pushes the gas and does work on it)
//   inflow            G > 1 (the gas compresses the radiation)
//
// Each flow is run with explicit IMC and with DDMC in the envelope (its cells
// are thick enough for the DDMC threshold set below), in both the multigroup and grey
// passes. DDMC must also match IMC at the same flow: it carries radiation
// with the gas through its advection leaks. These cases run with the DDMC
// compression shift unlimited (transport.ddmc-compression-max-dynamical-times
// = inf), as IMC's Doppler shifts are.
//
// The frozen-flow compression clock: DDMC reports the pdV work of its shift
// (diagnostics/compression_work). It is zero without DDMC and in the static
// envelope, positive in the inflow, negative in the outflow. In the inflow a
// packet's shift stops once it has spent the given number of local dynamical
// times min(1/|div v|, r/|v|) in the flow, so the work is exactly zero with a
// limit of 0 and grows with the limit up to the unlimited case.

#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <H5Cpp.h>

#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/misc/universal_error.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/Radiation/conj_grad_solve.hpp"

static_assert(ENERGY_GROUPS_NUM == 4,
              "imc_postprocess_moving_frame is built with --energy_groups_num=4");

namespace
{

constexpr double kHalfBox = 1.0e14;          // cm
constexpr double kCoreRadius = 0.35 * kHalfBox;
constexpr double kCoreDensity = 1e-9;
// Envelope scattering optical depth ~5 per cell (above the DDMC threshold of
// 3 used here) and ~20 in total. beta*tau ~ 0.4 keeps the envelope in the
// diffusive regime: with beta*tau > 1 an inflow traps the radiation and the
// test becomes slow.
constexpr double kEnvelopeDensity = 1.5e-12;
constexpr double kAbsorptionKappa = 1.0;     // cm^2/g, core only
constexpr double kScatteringKappa = 0.2;     // cm^2/g
constexpr double kBeta = 0.02;
constexpr double kGamma = 5.0 / 3.0;
constexpr double kMuMp = 0.6 * 1.67262192e-24;
double SpecificHeatF() { return CG::boltzmann_constant / ((kGamma - 1.0) * kMuMp); }

bool IsCore(ComputationalCell3D const& cell) { return cell.density > 1e-10; }

// Absorbing core, purely scattering envelope; grey and multigroup alike.
class CoreEnvelopeOpacity final : public OpacityCalculator
{
public:
    explicit CoreEnvelopeOpacity(bool multigroup)
    {
        double const eV = 1.602176634e-12;
        if (multigroup) {
            energy_groups_boundary = {1e-3 * eV, 3.0 * eV, 10.0 * eV, 30.0 * eV, 1e4 * eV};
            for (size_t g = 0; g + 1 < energy_groups_boundary.size(); ++g)
                energy_groups_center.push_back(
                    std::sqrt(energy_groups_boundary[g] * energy_groups_boundary[g + 1]));
        }
    }

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
    {
        return CalcPlanckOpacityAtTemperature(cell, cell.temperature);
    }
    double CalcPlanckOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    {
        return Absorption(cell);
    }
    double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcScatteringOpacity(ComputationalCell3D const& cell, double) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double, double) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double) const override
    {
        return Absorption(cell);
    }
    double CalcAbsorptionOpacityAtTemperature(ComputationalCell3D const& cell, double, double) const override
    {
        return Absorption(cell);
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override
    {
        return CG::speed_of_light / (3.0 * (Absorption(cell) + kScatteringKappa * cell.density));
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double) const override
    {
        return CalcDiffusionCoefficient(cell);
    }

private:
    static double Absorption(ComputationalCell3D const& cell)
    {
        return IsCore(cell) ? kAbsorptionKappa * cell.density : 0.0;
    }
};

// flowSign: +1 outflow, -1 inflow, 0 static envelope.
Snapshot3D MakeSnapshot(int flowSign)
{
    size_t const n = 12;
    double const spacing = 2.0 * kHalfBox / static_cast<double>(n);
    std::mt19937_64 rng(20261009);
    std::uniform_real_distribution<double> jitter(-0.2, 0.2);
    Snapshot3D snapshot;
    snapshot.ll = Vector3D(-kHalfBox, -kHalfBox, -kHalfBox);
    snapshot.ur = Vector3D(kHalfBox, kHalfBox, kHalfBox);
    snapshot.time = 1.0;
    snapshot.cycle = 1;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k) {
                Vector3D const point(
                    -kHalfBox + (static_cast<double>(i) + 0.5 + jitter(rng)) * spacing,
                    -kHalfBox + (static_cast<double>(j) + 0.5 + jitter(rng)) * spacing,
                    -kHalfBox + (static_cast<double>(k) + 0.5 + jitter(rng)) * spacing);
                double const r = abs(point);
                bool const core = r < kCoreRadius;
                ComputationalCell3D cell;
                cell.density = core ? kCoreDensity : kEnvelopeDensity;
                cell.temperature = core ? 2e5 : 2e4;
                cell.internal_energy = SpecificHeatF() * cell.temperature;
                cell.pressure = (kGamma - 1.0) * cell.density * cell.internal_energy;
                cell.velocity = core ? Vector3D(0.0, 0.0, 0.0)
                    : point * (static_cast<double>(flowSign) * kBeta * CG::speed_of_light / r);
                cell.Erad = CG::radiation_constant * std::pow(cell.temperature, 4) / cell.density;
                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                    cell.Eg[g] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
                cell.ID = snapshot.cells.size();
                snapshot.mesh_points.push_back(point);
                snapshot.cells.push_back(cell);
            }
    return snapshot;
}

PostProcessIMC::PostProcessScenario MakeScenario(int flowSign, bool ddmc)
{
    using namespace PostProcessIMC;
    PostProcessScenario scenario = MakeStaSnapshotScenario("imc-postprocess-moving-frame");
    scenario.factories.loadSnapshot = [flowSign](PostProcessConfig::Input const&, ParallelContext const&) {
        return MakeSnapshot(flowSign);
    };
    scenario.factories.transformSnapshot = [](Snapshot3D&, ParallelContext const&) {};
    scenario.factories.createEquationOfState = [](PostProcessConfig::Input const&) {
        return std::static_pointer_cast<EquationOfState>(std::make_shared<IdealGas>(kGamma, SpecificHeatF(), 1.0, 0.0));
    };
    scenario.factories.createMultigroupOpacity = [](PostProcessConfig::Input const&) {
        return std::static_pointer_cast<OpacityCalculator>(std::make_shared<CoreEnvelopeOpacity>(true));
    };
    scenario.factories.createGreyOpacity = [](PostProcessConfig::Input const&) {
        return std::static_pointer_cast<OpacityCalculator>(std::make_shared<CoreEnvelopeOpacity>(false));
    };
    scenario.factories.applyOpacityScaleFactors =
        [](OpacityCalculator&, std::unordered_map<size_t, double>) {};

    PostProcessConfig& c = scenario.defaults;
    c.input.snapshot = "synthetic-moving-envelope";
    c.input.multigroupOpacityDirectory = "synthetic";
    c.input.greyOpacityDirectory = "synthetic";
    c.input.eosDirectory = "synthetic";
    c.output.writeVtk = false;
    c.observer.radius = 0.9 * kHalfBox;
    c.observer.count = 12;
    c.transport.sourceDt = 100.0;
    c.transport.duration = 1.0e8;            // nothing times out
    c.transport.photonsPerCell = 10;
    c.transport.generations = 1;
    c.transport.useCellVelocities = true;
    c.transport.ddmc = ddmc;
    c.transport.ddmcMinCellOpticalDepth = 3.0;
    c.transport.ddmcCompressionMaxDynamicalTimes = std::numeric_limits<double>::infinity();
    c.transport.randomWalk = false;
    c.transport.compton.enabled = false;
    c.polarization.enabled = true;
    c.photosphere.enabled = false;
    c.fluxSource.enabled = true;
    c.fluxSource.thermalizationTau = 5.0;
    c.fluxSource.constructionRays = 256;
    c.fluxSource.ddmcFaceOpticalDepth = 3.0;
    c.fluxSource.surfaceMode = FluxSourceSurfaceMode::MultigroupInnermost;
    c.volumeEmission.enabled = false;
    c.opacityScaling.mode = OpacityScaleMode::None;
    c.adaptive.source.enabled = false;
    c.loadBalance.measured = false;
    return scenario;
}

std::string const kOutputDir = "imc_postprocess_moving_frame_output";

double ReadDouble(H5::H5File const& f, std::string const& path)
{
    double value = 0.0;
    f.openDataSet(path).read(&value, H5::PredType::NATIVE_DOUBLE);
    return value;
}

struct Gain
{
    double forward = 0.0;
    double grey = 0.0;
    // Compression work and transport-sink residual over emitted energy.
    double workForward = 0.0;
    double workGrey = 0.0;
    double residualForward = 0.0;
    double residualGrey = 0.0;
};

// Energy leaving the transport over energy put in, per pass.
Gain RunCase(std::string const& name, int flowSign, bool ddmc,
             std::function<void(PostProcessIMC::PostProcessConfig&)> const& mutate = {})
{
    PostProcessIMC::PostProcessScenario const scenario = MakeScenario(flowSign, ddmc);
    PostProcessIMC::PostProcessConfig config = scenario.defaults;
    if (mutate)
        mutate(config);
    config.output.stem = kOutputDir + "/" + name;
    std::cout << "\n##### case " << name << ": flow=" << flowSign
              << " beta=" << kBeta << " ddmc=" << (ddmc ? 1 : 0) << std::endl;
    PostProcessIMC::PostProcessResult const result =
        PostProcessIMC::RunPostProcess(config, scenario, PostProcessIMC::ParallelContext{0, 1});

    H5::H5File f(result.hdf5Path, H5F_ACC_RDONLY);
    auto gain = [&](std::string const& pass) {
        std::string const d = "/passes/" + pass + "/diagnostics/";
        double const emitted = ReadDouble(f, d + "emitted_energy");
        double const out = ReadDouble(f, d + "box_escape_energy") +
                           ReadDouble(f, d + "timed_out_energy") +
                           ReadDouble(f, d + "absorbed_energy") +
                           ReadDouble(f, d + "cutoff_energy");
        if (!(emitted > 0.0))
            throw UniversalError("imc_postprocess_moving_frame: case " + name + " pass " + pass + " emitted no energy");
        return out / emitted;
    };
    auto relative = [&](std::string const& pass, std::string const& what) {
        std::string const d = "/passes/" + pass + "/diagnostics/";
        return ReadDouble(f, d + what) / ReadDouble(f, d + "emitted_energy");
    };
    Gain g;
    g.forward = gain("forward");
    g.grey = gain("grey");
    g.workForward = relative("forward", "compression_work");
    g.workGrey = relative("grey", "compression_work");
    g.residualForward = relative("forward", "transport_sink_residual");
    g.residualGrey = relative("grey", "transport_sink_residual");
    std::cout << "CASE " << name << " gain_forward=" << g.forward
              << " gain_grey=" << g.grey
              << " work_forward=" << g.workForward << " work_grey=" << g.workGrey
              << " residual_forward=" << g.residualForward
              << " residual_grey=" << g.residualGrey << std::endl;
    return g;
}

} // namespace

int main()
{
    std::filesystem::remove_all(kOutputDir);
    std::filesystem::create_directories(kOutputDir);
    std::ofstream metrics("imc_postprocess_moving_frame_metrics.txt");
    metrics << std::setprecision(10);
    bool pass = true;
    auto check = [&](bool ok, std::string const& what) {
        std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
        pass = pass && ok;
    };

    try {
        Gain imcOut, imcIn, ddmcOut, ddmcIn;
        for (bool const ddmc : {false, true}) {
            std::string const mode = ddmc ? "ddmc" : "imc";
            Gain const still = RunCase(mode + "_static", 0, ddmc);
            Gain const out = RunCase(mode + "_outflow", +1, ddmc);
            Gain const in = RunCase(mode + "_inflow", -1, ddmc);
            (ddmc ? ddmcOut : imcOut) = out;
            (ddmc ? ddmcIn : imcIn) = in;
            for (auto const& [pass_name, s, o, i] : {
                     std::tuple<std::string, Gain const*, Gain const*, Gain const*>{"forward", &still, &out, &in},
                     std::tuple<std::string, Gain const*, Gain const*, Gain const*>{"grey", &still, &out, &in}}) {
                bool const forward = pass_name == "forward";
                auto work = [&](Gain const* g) { return forward ? g->workForward : g->workGrey; };
                auto residual = [&](Gain const* g) { return forward ? g->residualForward : g->residualGrey; };
                std::string const tag = mode + "_" + pass_name;
                metrics << tag << "_work_static " << work(s) << "\n"
                        << tag << "_work_outflow " << work(o) << "\n"
                        << tag << "_work_inflow " << work(i) << "\n"
                        << tag << "_residual_outflow " << residual(o) << "\n"
                        << tag << "_residual_inflow " << residual(i) << "\n";
                check(work(s) == 0.0, tag + ": no compression work in the static envelope");
                if (ddmc) {
                    check(work(o) < 0.0, tag + ": DDMC expansion work is negative (" + std::to_string(work(o)) + ")");
                    check(work(i) > 0.0, tag + ": DDMC compression work is positive (" + std::to_string(work(i)) + ")");
                } else {
                    check(work(o) == 0.0 && work(i) == 0.0, tag + ": no DDMC compression work without DDMC");
                }
            }
            for (auto const& [pass_name, s, o, i] : {
                     std::tuple<std::string, double, double, double>{"forward", still.forward, out.forward, in.forward},
                     std::tuple<std::string, double, double, double>{"grey", still.grey, out.grey, in.grey}}) {
                std::string const tag = mode + "_" + pass_name;
                metrics << tag << "_static " << s << "\n"
                        << tag << "_outflow " << o << "\n"
                        << tag << "_inflow " << i << "\n";
                check(std::abs(s - 1.0) < 1e-6,
                      tag + ": static envelope conserves energy (G-1=" + std::to_string(s - 1.0) + ")");
                check(o < 1.0,
                      tag + ": outflow loses energy to the gas (G=" + std::to_string(o) + ")");
                check(i > 1.0,
                      tag + ": inflow gains energy from the gas (G=" + std::to_string(i) + ")");
            }
        }
        // The compression clock in the DDMC inflow: no shift with a limit of 0,
        // a partial one with 0.25 local dynamical times, the full one with no
        // limit (ddmcIn above).
        {
            auto limited = [](double maxTimes) {
                return [maxTimes](PostProcessIMC::PostProcessConfig& c) {
                    c.transport.ddmcCompressionMaxDynamicalTimes = maxTimes;
                };
            };
            Gain const off = RunCase("ddmc_inflow_clock0", -1, true, limited(0.0));
            Gain const quarter = RunCase("ddmc_inflow_clock025", -1, true, limited(0.25));
            for (auto const& [pass_name, w0, wq, wInf] : {
                     std::tuple<std::string, double, double, double>{"forward", off.workForward, quarter.workForward, ddmcIn.workForward},
                     std::tuple<std::string, double, double, double>{"grey", off.workGrey, quarter.workGrey, ddmcIn.workGrey}}) {
                std::string const tag = "ddmc_inflow_" + pass_name;
                metrics << tag << "_work_clock0 " << w0 << "\n"
                        << tag << "_work_clock025 " << wq << "\n"
                        << tag << "_work_unlimited " << wInf << "\n";
                check(w0 == 0.0, tag + ": no compression work with a limit of 0");
                check(wq > 0.0 && wq < wInf,
                      tag + ": the 0.25 limit gives part of the unlimited work (" +
                      std::to_string(wq) + " of " + std::to_string(wInf) + ")");
            }
        }
        // Without polarization DDMC runs through the shared AdvanceDDMC kernel
        // instead of the host engine; its energy bookkeeping must close too.
        {
            Gain const kernelStill = RunCase("ddmc_static_kernel", 0, true,
                [](PostProcessIMC::PostProcessConfig& c) { c.polarization.enabled = false; });
            metrics << "ddmc_kernel_forward_static " << kernelStill.forward << "\n"
                    << "ddmc_kernel_grey_static " << kernelStill.grey << "\n";
            check(std::abs(kernelStill.forward - 1.0) < 1e-6,
                  "ddmc_kernel_forward: static envelope conserves energy (G-1=" + std::to_string(kernelStill.forward - 1.0) + ")");
            check(std::abs(kernelStill.grey - 1.0) < 1e-6,
                  "ddmc_kernel_grey: static envelope conserves energy (G-1=" + std::to_string(kernelStill.grey - 1.0) + ")");
        }
        // DDMC carries radiation with the gas (advection leaks) and does the
        // work term, so it reproduces IMC up to the first-order upwind
        // smearing of the advection, ~1.5 beta tau_cell of the diffusion
        // (0.15 here). Inflow, where radiation is pushed against its gradient,
        // shows it most: G_ddmc - G_imc = -0.10 on this 12^3 mesh and -0.05
        // on an 18^3 one. The gate is relative to the energy change itself.
        for (auto const& [tag, imc, ddmc] : {
                 std::tuple<std::string, double, double>{"outflow_forward", imcOut.forward, ddmcOut.forward},
                 std::tuple<std::string, double, double>{"outflow_grey", imcOut.grey, ddmcOut.grey},
                 std::tuple<std::string, double, double>{"inflow_forward", imcIn.forward, ddmcIn.forward},
                 std::tuple<std::string, double, double>{"inflow_grey", imcIn.grey, ddmcIn.grey}}) {
            metrics << "ddmc_minus_imc_" << tag << " " << (ddmc - imc) << "\n";
            check(std::abs(ddmc - imc) <= 0.2 * std::abs(imc - 1.0),
                  "DDMC matches IMC within 20% of the energy change, " + tag +
                  " (G_ddmc - G_imc=" + std::to_string(ddmc - imc) +
                  ", G_imc - 1=" + std::to_string(imc - 1.0) + ")");
        }
    } catch (UniversalError const& eo) {
        std::cout << "FAIL exception: " << eo.getErrorMessage() << std::endl;
        pass = false;
    } catch (std::exception const& e) {
        std::cout << "FAIL exception: " << e.what() << std::endl;
        pass = false;
    }

    metrics << "pass " << (pass ? 1 : 0) << "\n";
    std::cout << (pass ? "imc_postprocess_moving_frame PASSED" : "imc_postprocess_moving_frame FAILED") << std::endl;
    return pass ? 0 : 1;
}
