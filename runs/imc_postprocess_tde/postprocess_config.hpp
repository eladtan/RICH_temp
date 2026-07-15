#ifndef IMC_POSTPROCESS_TDE_CONFIG_HPP
#define IMC_POSTPROCESS_TDE_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "source/newtonian/three_dimensional/computational_cell.hpp"

namespace imc_postprocess_tde {

enum class OpacityScaleMode { None, Rosseland, Planck };

struct Config
{
    std::string inputPath = "/home/elads/TDEMG/R0.47M0.5BH1e+06beta1S50n1.5Compton/snap_full_136.h5";
    std::string outputPath = "tde_postprocess_output.h5";
    std::string vtkOutput;
    std::string opacityDir = "/home/elads/RICH/data/STA/MG/";
    std::string greyOpacityDir;
    std::string eosDir = "/home/elads/RICH/data/EOS/";
    double radius = 5e14;
    size_t nObservers = 256;
    double sourceDt = 1.0;
    double transportTime = 0.0;
    Vector3D center = Vector3D(0, 0, 0);
    size_t photonsPerCell = 100;
    bool compton = false;
    size_t comptonSamples = 200000;
    bool comptonAngleDependent = true;
    size_t nGenerations = 1;
    bool ddmc = true;
    bool randomWalk = true;
    bool useCellVelocities = true;
    bool polarization = true;
    bool photosphere = true;
    bool fluxSourceCompare = false;
    double fluxSourceThermalizationTau = 5.0;
    double fluxSourceDDMCFaceOpticalDepth = 5.0;
    int polarizationManualScatterings = 128;
    double polarizationDepolarizationScatterings = 0.5;
    std::string polarizationClosure = "damped_last_scatterings";
    bool measuredLoadBalance = true;
    OpacityScaleMode opacityScaleMode = OpacityScaleMode::Planck;
    bool adaptiveSourceCells = false;
    size_t adaptiveSourceBurnin = 3;
    double adaptiveSourceStrength = 0.95;
    double adaptiveSourceEma = 0.5;
    double adaptiveSourceMinEscapedFrac = 1e-12;
    double adaptiveSourceMaxFactor = 1000.0;
    size_t adaptiveSourceBurninPhotonMultiplier = 2;
    double adaptiveSourceLearnedReserveFrac = 0.25;
    double adaptiveSourceLearnedMinFactor = 20.0;
    size_t adaptiveSourceLearnedMinPhotons = 100;
    size_t adaptiveSourceLearnedMaxPhotons = 5000;
    double adaptiveSourceScorePower = 2.0;
    double adaptiveSourceWeightScoreFrac = 0.85;
    bool adaptiveObserverEquity = true;
    double adaptiveObserverExtraBudgetFrac = 2;
    double adaptiveObserverTargetNeff = 1000000;
    double adaptiveObserverTargetPolSnr = 10.0;
    double adaptiveObserverDeficitMax = 100.0;
    double adaptiveObserverDeficitEma = 0.8;
    double measuredLBWeightCompression = -1.0;
    double adaptiveLBImbalanceThreshold = 2.0;
    size_t adaptiveLBCooldownGenerations = 2;
    size_t adaptiveLBMaxRebalances = 6;

    bool adaptiveGroupQuality = false;
    bool adaptiveGroupSourceCells = false;
    bool adaptiveGroupFrequencySampling = false;
    bool adaptiveGroupHistory = true;
    double adaptiveGroupTargetNeff = 1e4;
    double adaptiveGroupTargetPolSnr = 10.0;
    double adaptiveGroupDeficitMax = 100.0;
    size_t adaptiveGroupMinCrossings = 3;
    double adaptiveGroupMinLuminosity = 0.0;
    double adaptiveGroupMinLuminosityFracOfGroupMax = 0.01;
    double adaptiveGroupIneligiblePriorityCap = 2.0;
    double adaptiveGroupRetainPriorityFloor = 3.0;
    std::string adaptiveGroupLuminosityNormalization = "mixed";
    double adaptiveGroupLuminosityGlobalWeight = 0.5;
    double adaptiveGroupLuminosityPower = 1.0;
    double adaptiveGroupPolarizationPower = 1.0;
    double adaptiveGroupLuminosityWeight = 0.5;
    double adaptiveGroupPolarizationWeight = 0.5;
    double adaptiveGroupPolarizationFloor = 0.02;
    double adaptiveGroupHistoryEma = 0.35;
    double adaptiveGroupLatestWeight = 0.25;
    double adaptiveGroupCumulativeWeight = 0.50;
    double adaptiveGroupEmaWeight = 0.25;
    double adaptiveGroupScoreEma = 0.35;
    double adaptiveGroupStrength = 0.75;
    double adaptiveGroupPdfFloor = 0.02;
    double adaptiveGroupMaxBias = 100.0;
    double adaptiveGroupMaxWeightCorrection = 100.0;
    size_t adaptiveGroupMaxLocalStats = 200000;
    size_t adaptiveGroupStatMinCount = 1;
    double adaptiveGroupStatPriorityKeep = 2.0;
    bool adaptiveGroupFallbackToIntegratedOnOverflow = true;
    bool adaptiveDiagnosticsVerbose = false;

};


void printUsage(int rank);
std::string ReplaceExtension(std::string const& path, std::string const& newExt);
std::string InsertSuffixBeforeExtension(std::string const& path, std::string const& suffix);
std::string BaseVtkOutputPath(Config const& cfg);
std::string GreyVtkOutputPath(Config const& cfg);
bool parseArgs(int argc, char* argv[], Config& cfg, int rank);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_CONFIG_HPP
