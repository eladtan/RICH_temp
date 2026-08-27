#ifndef IMC_POST_PROCESS_HPP
#define IMC_POST_PROCESS_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
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

enum class OpacityScaleMode { None, Rosseland, Planck };
enum class MonteCarloCommunication { TwoSided, Rdma };

struct PostProcessConfig
{
    struct Input
    {
        std::string snapshot;
        std::string multigroupOpacityDirectory;
        std::string greyOpacityDirectory;
        std::string eosDirectory;
    } input;

    struct Output
    {
        std::string stem = "tde_postprocess_output";
        bool writeVtk = true;
    } output;

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
        size_t generations = 1; // Independent generations included in statistics.
        MonteCarloCommunication communication = MonteCarloCommunication::Rdma;
        bool ddmc = true;
        bool randomWalk = true;
        bool useCellVelocities = true;

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
        double thermalizationTau = 5.0;
        size_t constructionRays = 0;
        double ddmcFaceOpticalDepth = 5.0;
    } fluxSource;

    struct OpacityScaling
    {
        OpacityScaleMode mode = OpacityScaleMode::Planck;
    } opacityScaling;

    struct Adaptive
    {
        struct Source
        {
            bool enabled = false;
            size_t burninGenerations = 21; // Learning generations excluded from statistics.
            double strength = 0.95;
            double ema = 0.5;
            double minEscapedFraction = 1e-12;
            double maxFactor = 1000.0;
            size_t burninPhotonMultiplier = 2;
            double learnedReserveFraction = 0.25;
            double learnedMinFactor = 20.0;
            size_t learnedMinPhotons = 100;
            size_t learnedMaxPhotons = 5000;
            double scorePower = 2.0;
            double weightScoreFraction = 0.85;
        } source;

        struct Observer
        {
            bool equity = true;
            double extraBudgetFraction = 2.0;
            double targetEffectivePackets = 1e6;
            double targetPolarizationSnr = 10.0;
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
};

struct PostProcessResult
{
    PostProcessPassResult forward;
    PostProcessPassResult grey;
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
