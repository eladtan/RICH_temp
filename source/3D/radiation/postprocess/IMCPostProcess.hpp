#ifndef IMC_POST_PROCESS_HPP
#define IMC_POST_PROCESS_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/output/Snapshot3D.hpp"
#include "Radiation/OpacityCalculator.hpp"

class EquationOfState;
class RadiationIMC;
class SphericalObserver;
class Tessellation3D;
struct ComputationalCell3D;

namespace PostProcessIMC {

// Multigroup absorption normalization. Auto (default) follows the snapshot:
// none for a multigroup snapshot (the hydro's own group opacities), planck
// for a grey one or one whose group count differs from the table; see
// ResolveSnapshotRadiation.
enum class OpacityScaleMode { None, Rosseland, Planck, Auto };

//! \brief Monte Carlo particle-exchange engine.  TwoSided has the smaller memory
//! footprint; Rdma is usually faster where the fabric supports it; Auto tries
//! RDMA and falls back to TwoSided, reporting which it used.
enum class MonteCarloCommunication { Auto, Rdma, TwoSided };
// Where the flux-source (CER) surface sits. Grey: the grey effective optical
// depth reaches the target. MultigroupInnermost: each energy group's own
// effective depth reaches the target and the deepest of those radii is taken
// per direction, so every group is thermalized at the surface and a Planck
// spectrum there is exact; the transport then handles each group's own
// thermalization layer.
enum class FluxSourceSurfaceMode { Grey, MultigroupInnermost };

// Temperature of the Planck spectrum given to packets emitted at, or
// re-emitted by, a CER face (multigroup): the radiation temperature
// (E_rad / a)^(1/4) of the face's outer cell, or that cell's gas temperature.
// Radiation is the default: the face carries the flux of the thermalized
// radiation field, and hot optically thin gas next to the surface (T_gas up to
// ~1e8 K under T_rad ~1e5 K in snap_109) does not set its spectrum.
enum class FluxSourceSpectrumTemperature { Radiation, Gas };

// Luminosity of a CER face: the hydro's FLD flux across that face (two-point
// gradient between the cells' centres of mass, both cells' diffusion
// coefficients at the face temperature ((T_1^4 + T_2^4) / 2)^(1/4), harmonic
// mean, flux limiter on the face gradient; Diffusion.cpp), or the arithmetic
// mean of the two cells' cell-centred FLD fluxes (the original formula; in
// snap_109 it gives 0.85 of the hydro's positive face luminosity, 3.86e42
// against 4.56e42 erg/s). HydroMultigroup: the hydro's multigroup FLD flux
// across the face (MultigroupDiffusion.cpp: group energy rho E_g, group
// diffusion coefficient at the face temperature, harmonic mean, flux limiter
// per group); each group's outward flux is that group's face luminosity, so it
// sets both the luminosity and the spectrum of the face (the grey pass emits
// the sum over groups). Needs the snapshot's group energies.
// Auto (default) is HydroMultigroup for a multigroup snapshot and Hydro
// otherwise (ResolveSnapshotRadiation).
enum class FluxSourceFaceFlux { Hydro, CellAverage, HydroMultigroup, Auto };

// A snapshot without radiation (Erad and E_g zero everywhere) is given LTE
// radiation before the post-process. ConserveEnergy (Auto): each cell's gas
// and radiation share its snapshot internal energy at one temperature,
// rho e_snap = rho e_gas(T) + a T^4. GasTemperature: T_rad = T_gas (adds
// energy). Setting either for a snapshot with radiation is an error.
enum class MissingRadiation { Auto, ConserveEnergy, GasTemperature };

// How the cooling limit caps a hot cell's net emission at the hydro's
// per-step rate. Temperature: the cell is cooled to the temperature at which
// its net emission equals the cap (spectrum and opacity at that temperature).
// Emission: the cell keeps its snapshot temperature (spectrum and opacity) and
// its emission is scaled so that the net emission equals the cap, as the
// hydro's Fleck step emits from the real temperature at a reduced rate
// (default). Cold cells (net absorption capped) are heated in both modes.
enum class CoolingLimitMode { Temperature, Emission };

// Command-line spelling of option values, shared by the parser, the logs and
// the error messages.
std::string OptionValueName(OpacityScaleMode mode);
std::string OptionValueName(FluxSourceFaceFlux mode);
std::string OptionValueName(MissingRadiation mode);

struct PostProcessConfig
{
    struct Input
    {
        std::string snapshot;
        std::string multigroupOpacityDirectory;
        std::string greyOpacityDirectory;
        std::string eosDirectory;
        MissingRadiation missingRadiation = MissingRadiation::Auto;
    } input;

    struct Output
    {
        std::string stem = "tde_postprocess_output";
        bool writeVtk = true;
    } output;

    // Continue an earlier result: load its generation statistics, learned
    // allocation state and load-balance costs, skip the burn-in and probe
    // generations, and add transport.generations more final generations to
    // each pass. The physics settings must match the file.
    struct Restart
    {
        std::string from;
        // Allow output.stem to name the file being continued.
        bool inPlace = false;
    } restart;

    struct Observer
    {
        Vector3D center = Vector3D(0.0, 0.0, 0.0);
        double radius = 5e14;
        size_t count = 256;
    } observer;

    struct Transport
    {
        double sourceDt = 1.0;
        double duration = 0.0;
        size_t photonsPerCell = 100;
        size_t generations = 1;
        bool ddmc = true;
        bool randomWalk = true;
        // Per-group cell optical depth above which DDMC takes over from
        // explicit IMC. Densmore's asymptotic boundary treatment holds down to
        // a few; 15 is very conservative and leaves mid-depth cells to
        // explicit random walks.
        double ddmcMinCellOpticalDepth = 15.0;
        // Moving-interface weight correction on IMC -> DDMC entry (STORM's
        // ddmcUseMovingInterfaceCorrection). Only acts with cell velocities.
        bool ddmcMovingInterfaceCorrection = true;
        // Advection of DDMC radiation with the gas (STORM's ddmcAdvection).
        // Only acts with cell velocities.
        bool ddmcAdvection = true;
        // Cells whose advection Peclet number |v| (L/2) / D exceeds this stay
        // in IMC (STORM's ddmcMaxCellAdvectionPeclet); infinity disables it.
        double ddmcMaxCellAdvectionPeclet = std::numeric_limits<double>::infinity();
        // Frozen-flow limit on the DDMC comoving-frame compression shift
        // (STORM's postProcess.compressionMaxDynamicalTimes): each packet's
        // clock advances at max(|div v|, |v| / max(r, cell size)) on every
        // step, with r measured from the observer center, and the shift of
        // its frequency and weight stops once the clock reaches this value.
        // 1 allows the compression of one local dynamical time, the most the
        // frozen snapshot can justify; 0 turns the shift off; infinity is the
        // unlimited shift.
        double ddmcCompressionMaxDynamicalTimes = 1.0;
        // Weight window (importance splitting): a packet that starts a step
        // in a cell whose measured escape probability is p splits into equal
        // copies when heavier than w_max / min(1, p / splitEscapeReference),
        // w_max = splitObserverFraction x (last generation's escaping energy)
        // / (number of observers). 0 disables it. p is the cell's escaped
        // energy over its emitted energy, accumulated over the final
        // generations.
        double splitObserverFraction = 0.0;
        double splitEscapeReference = 0.1;
        size_t splitMaxCopies = 1000;
        // At most this many copies per rank per generation (288 B each).
        size_t splitCopyBudget = 1000000;
        bool useCellVelocities = true;
        MonteCarloCommunication communication = MonteCarloCommunication::Auto;

        struct Compton
        {
            bool enabled = false;
            size_t matrixSamples = 200000;
            bool angleDependent = true;
        } compton;
    } transport;

    struct Polarization
    {
        bool enabled = true;
        int manualScatteringsAfterAcceleration = 128;
        double depolarizationScatterings = 0.5;
        std::string acceleratedClosure = "damped_last_scatterings";
    } polarization;

    struct Photosphere
    {
        bool enabled = true;
    } photosphere;

    struct FluxSource
    {
        bool enabled = false;
        double thermalizationTau = 10.0;
        size_t constructionRays = 0;
        double ddmcFaceOpticalDepth = 5.0;
        FluxSourceSurfaceMode surfaceMode = FluxSourceSurfaceMode::Grey;
        // Make inside any outside region not connected, through outside cells,
        // to the exterior beyond the outermost surface radius. Such a region
        // is walled in by thermalizing faces on every side and traps packets.
        bool removeEnclosedOutside = true;
        // Raise any direction's surface radius to at least this fraction of
        // the median radius of its 8 nearest resolved directions, so a single
        // direction cannot dip far below its neighbours. 0 disables it. A
        // raised direction no longer sits at thermalizationTau, and the run
        // warns when that happens.
        double smoothingFloor = 0.5;
        // Never let smoothing raise a direction above its own photosphere:
        // the deepest radius, along the same construction ray, where the
        // total optical depth of a channel the surface is built from reaches
        // 2/3. A direction without one is not raised.
        bool smoothingPhotosphereCap = true;
        // A cell is inside only if it lies below the surface of its nearest
        // construction ray and of that ray's mappingNeighbours nearest
        // resolved rays, so a cell between rays never takes a neighbour's
        // larger radius. 0 uses the nearest ray alone.
        size_t mappingNeighbours = 6;
        // The mapping never lowers a ray below the radius where its own
        // tau_eff reaches mappingMaxDepthFactor x thermalizationTau, so it
        // cannot expose gas far below the thermalization depth (hot,
        // opaque cells whose gross emission dwarfs the escaping light).
        // 0 = no floor.
        double mappingMaxDepthFactor = 3.0;
        FluxSourceSpectrumTemperature spectrumTemperature = FluxSourceSpectrumTemperature::Radiation;
        FluxSourceFaceFlux faceFlux = FluxSourceFaceFlux::Auto;
    } fluxSource;

    // Thermal emission from every cell outside the flux-source surface at its
    // snapshot temperature (Fleck factor 1, no temperature update): the
    // instantaneous emissivity of the frozen state, on top of the face sources.
    struct VolumeEmission
    {
        bool enabled = true;
        // A cell emits packets only if its gross luminosity is at least this
        // fraction of the escaping luminosity (last generation's observer
        // crossings; the face-source flux before the first generation).
        double cutoffFraction = 1e-10;
        // Burn-in generations include a random subset of emitting cells sized
        // so that about this many volume packets are launched per generation.
        size_t burninPacketsTarget = 2000000;
        // true: a cell emits only in the groups whose own thermalization
        // surface lies inside it (radial escape gate). false: every cell
        // outside the deep surface emits its full spectrum and the learned
        // allocation decides which cells matter from the crossings it sees.
        bool gateGroups = false;
        // true: burn-in gives every emitting cell exactly newPhotonsPerCell
        // packets instead of an energy-proportional count, so thin cells are
        // explored even when thick cells hold nearly all the emitted energy.
        bool burninExact = true;
        // Exploration packets of a cell left at zero by the allocation are
        // split so that none carries more than this fraction of the previous
        // generation's escaping energy. 0 = one packet per cell.
        double explorationWeightFraction = 0.005;
        // Grey pass fraction; negative means the same as
        // explorationWeightFraction. The grey escaping luminosity is ~5x lower
        // than the MG one, so the same fraction gives a ~5x smaller cap.
        double explorationWeightFractionGrey = 0.005;
        // true: in the probe and final generations the same weight bound also
        // applies to cells the allocation gave packets: each gets at least
        // ceil(E / bound) packets, above learnedMaxPhotons if needed. Without
        // it, a learned cell that emits far more than the escaping luminosity
        // but gets few packets sends rare packets carrying several per cent of
        // a generation's escaping energy.
        bool capLearnedWeight = true;
        // Neyman budget and bounds for learned volume cells (face cells keep
        // adaptive.source.learned*).
        size_t learnedPhotonsPerCellBudget = 1600;
        // Grey pass budget; 0 means the same as learnedPhotonsPerCellBudget.
        size_t learnedPhotonsPerCellBudgetGrey = 500;
        size_t learnedMinPhotons = 1;
        size_t learnedMaxPhotons = 50000;
        // Limit each outside cell's net emission to what the hydro's Fleck
        // step removes per hydro time, (aT^4 - E) V / (beta t_hydro). A cell
        // whose cooling time is far shorter than the hydro step stays too hot
        // in the hydro (the Fleck factor spreads its cooling over several
        // steps), so its snapshot temperature overstates the emission by about
        // t_hydro / t_cool. The cell's temperature is lowered to the value that
        // gives the limited net emission, or (coolingLimitMode Emission) its
        // emission is scaled at its own temperature; transport keeps Fleck
        // factor 1.
        bool coolingLimit = true;
        // Courant number for t_hydro (RICH CourantFriedrichsLewy, face
        // velocities from the gas as on a Lagrangian mesh).
        double coolingLimitCfl = 0.25;
        // Explicit hydro time step in seconds; 0 uses the Courant step.
        double coolingLimitDt = 0.0;
        CoolingLimitMode coolingLimitMode = CoolingLimitMode::Emission;
    } volumeEmission;

    struct OpacityScaling
    {
        OpacityScaleMode mode = OpacityScaleMode::Auto;
        // Bounds on the per-cell scale factor alpha (multigroup absorption
        // multiplied to match the grey mean). 0 means unbounded. Cells whose
        // ratio falls outside are clamped, so the multigroup opacity there no
        // longer reproduces the grey mean but keeps the table's own magnitude.
        double alphaMin = 0.0;
        double alphaMax = 0.0;
    } opacityScaling;

    struct Adaptive
    {
        struct Source
        {
            bool enabled = false;
            size_t burninGenerations = 3;
            double strength = 0.95;
            double ema = 0.5;
            double minEscapedFraction = 1e-12;
            double maxFactor = 1000.0;
            size_t burninPhotonMultiplier = 2;
            double learnedReserveFraction = 0.25;
            double learnedMinFactor = 20.0;
            // Final-phase allocation. Each learned cell receives a packet
            // count proportional to score^scorePower (0.5 is the Neyman
            // optimum for the variance-fraction score), clamped to
            // [learnedMinPhotons, learnedMaxPhotons], with a total budget of
            // learnedPhotonsPerCellBudget packets per learned cell.
            size_t learnedMinPhotons = 10;
            size_t learnedMaxPhotons = 5000;
            size_t learnedPhotonsPerCellBudget = 100;
            double scorePower = 0.5;
            double weightScoreFraction = 0.85;
            // Final-budget ramp between the learned-only probe and the final
            // generations, so the packet count (and memory) grows in steps on
            // a balanced decomposition: ramp stage k gives the learned cells
            // rampStartPhotons * rampFactor^k packets per cell (at most each
            // cell type's own budget, at least 1), raises the weight bounds of
            // the extra packets by the same ratio, and is followed by a load
            // balance on the packets the next stage's plan will create. The
            // ramp ends when the full budgets are reached. Ramp generations
            // only learn; they are not part of the result. 0 disables it.
            double rampStartPhotons = 1.0;
            double rampFactor = 2.0;
        } source;

        struct Observer
        {
            bool equity = true;
            double extraBudgetFraction = 2.0;
            double targetEffectivePackets = 1e6;
            double targetPolarizationSnr = 10.0;
            // When > 0 the observer deficit is sigma_P / targetPolarizationSigma
            // instead of the SNR ratio, so unpolarized observers are not
            // chased for an SNR they cannot reach.
            double targetPolarizationSigma = 0.0;
            // Divide raw deficits by their median before clamping so the
            // clamp never flattens the whole map to maxDeficit.
            bool normalizeDeficits = true;
            double maxDeficit = 100.0;
            double deficitEma = 0.8;
        } observer;

        struct Group
        {
            bool quality = false;
            bool sourceCells = false;
            bool frequencySampling = false;
            bool history = true;
            double targetEffectivePackets = 1e4;
            double targetPolarizationSnr = 10.0;
            double maxDeficit = 100.0;
            size_t minCrossings = 3;
            double minLuminosity = 0.0;
            double minLuminosityFractionOfGroupMax = 0.01;
            double ineligiblePriorityCap = 2.0;
            double retainPriorityFloor = 3.0;
            std::string luminosityNormalization = "mixed";
            double luminosityGlobalWeight = 0.5;
            double luminosityPower = 1.0;
            double polarizationPower = 1.0;
            double luminosityWeight = 0.5;
            double polarizationWeight = 0.5;
            double polarizationFloor = 0.02;
            double historyEma = 0.35;
            double latestWeight = 0.25;
            double cumulativeWeight = 0.50;
            double emaWeight = 0.25;
            double scoreEma = 0.35;
            double strength = 0.75;
            double pdfFloor = 0.02;
            double maxBias = 100.0;
            double maxWeightCorrection = 100.0;
            size_t maxLocalStats = 200000;
            size_t statMinCount = 1;
            double statPriorityKeep = 2.0;
            bool fallbackToIntegratedOnOverflow = true;
        } group;
    } adaptive;

    struct LoadBalance
    {
        bool measured = true;
        double weightCompression = -1.0;
        double imbalanceThreshold = 2.0;
        size_t cooldownGenerations = 2;
        size_t maxRebalances = 6;
        // Rebalance on the measured costs after every finalPeriod-th final
        // generation (besides final steps 1 and 2). 0 keeps the built-in
        // period: 10, or 50 for a grey pass without the flux source.
        size_t finalPeriod = 0;
        // Cap a cell's measured cost at clampFactor times the mean cost of the
        // cells that did work (true) or of all cells (false). With the work in
        // ~2% of the cells, the all-cell cap hid ~30x of the hot cells' cost
        // and left a 34x rank imbalance on snap_109.
        bool clampToActiveMean = true;
        // 0 keeps the built-in factor (30).
        double clampFactor = 0.0;
    } loadBalance;

    struct Diagnostics
    {
        bool verboseAdaptive = false;
    } diagnostics;
};

struct ParallelContext
{
    int rank = 0;
    int size = 1;
};

struct SourceContext
{
    Tessellation3D& tessellation;
    std::vector<ComputationalCell3D>& cells;
    OpacityCalculator& multigroupOpacity;
    OpacityCalculator& greyOpacity;
    SphericalObserver& observer;
    RadiationIMC& physics;
    ParallelContext parallel;
};

struct PostProcessFactories
{
    std::function<Snapshot3D(PostProcessConfig::Input const&, ParallelContext const&)> loadSnapshot;
    std::function<void(Snapshot3D&, ParallelContext const&)> transformSnapshot;
    std::function<std::shared_ptr<EquationOfState>(PostProcessConfig::Input const&)> createEquationOfState;
    std::function<std::shared_ptr<OpacityCalculator>(PostProcessConfig::Input const&)> createMultigroupOpacity;
    std::function<std::shared_ptr<OpacityCalculator>(PostProcessConfig::Input const&)> createGreyOpacity;
    std::function<void(OpacityCalculator&, std::unordered_map<size_t, double>)> applyOpacityScaleFactors;
    std::function<std::shared_ptr<SphericalObserver>(
        PostProcessConfig::Observer const&, std::vector<double> const&)> createObserver;
    std::function<void(SourceContext&, PostProcessConfig const&)> configureSource;
};

struct PostProcessScenario
{
    std::string name;
    PostProcessConfig defaults;
    PostProcessFactories factories;
};

struct PostProcessPassResult
{
    bool ran = false;
    bool usesVelocity = false;
    bool usesDDMC = false;
    bool usesPolarization = false;
    bool usesCompton = false;
    double sourceLuminosity = 0.0;
    double emittedLuminosity = 0.0;
    double crossingLuminosity = 0.0;
    double crossingLuminosityStderr = 0.0;
    double emittedEnergy = 0.0;
    double timedOutFraction = 0.0;
    double luminosityWeightedPolarizationDegree = 0.0;
    uint64_t polarizedObserverCount = 0;
    uint64_t splitCopies = 0;   // weight-window copies over the pass
    // Packets added by volume-emission.cap-learned-weight over the pass, the
    // cells it raised and all cells given packets, summed over generations.
    uint64_t weightCapExtraPhotons = 0;
    uint64_t weightCappedCells = 0;
    uint64_t allocatedCells = 0;
    // Packets of the learned-only probe generation: predicted by the load
    // balance before it (0 when that balance did not run, e.g. serial) and
    // created.
    uint64_t probePredictedPackets = 0;
    uint64_t probePackets = 0;
};

// Volume-emission cooling limit applied before transport (global values).
struct PostProcessCoolingLimitResult
{
    bool applied = false;
    double hydroTime = 0.0;
    bool hydroTimeFromCourant = false;
    uint64_t outsideCells = 0;
    uint64_t hotLimitedCells = 0;
    uint64_t coldLimitedCells = 0;
    double netEmissionBefore = 0.0;
    double netEmissionAfter = 0.0;
};

struct PostProcessResult
{
    PostProcessPassResult forward;
    PostProcessPassResult grey;
    PostProcessCoolingLimitResult coolingLimit;
    std::string hdf5Path;
    std::string vtkPath;
};

PostProcessScenario MakeStaSnapshotScenario(std::string name);
PostProcessResult RunPostProcess(
    PostProcessConfig config,
    PostProcessScenario const& scenario,
    ParallelContext parallel);
int RunPostProcessMain(int argc, char* argv[], PostProcessScenario scenario);

} // namespace PostProcessIMC

#endif // IMC_POST_PROCESS_HPP
