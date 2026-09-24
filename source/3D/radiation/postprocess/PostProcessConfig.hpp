#ifndef IMC_POSTPROCESS_TDE_CONFIG_HPP
#define IMC_POSTPROCESS_TDE_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "IMCPostProcess.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"

namespace imc_postprocess_tde {

using OpacityScaleMode = PostProcessIMC::OpacityScaleMode;
using FluxSourceSurfaceMode = PostProcessIMC::FluxSourceSurfaceMode;

//! \brief Monte Carlo particle-exchange engine selection
enum class CommunicationMode
{
    Auto,      //!< try RDMA, fall back to two-sided MPI
    Rdma,      //!< force RDMA
    TwoSided   //!< force two-sided MPI point-to-point
};

struct Config
{
    std::string inputPath;
    std::string outputPath = "tde_postprocess_output.h5";
    std::string vtkOutput;
    bool writeVtk = true;
    std::string opacityDir;
    std::string greyOpacityDir;
    std::string eosDir;
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
    double ddmcMinCellOpticalDepth = 15.0;
    bool useCellVelocities = true;
    CommunicationMode communication = CommunicationMode::Auto;
    bool polarization = true;
    bool photosphere = true;
    bool fluxSourceCompare = false;
    double fluxSourceThermalizationTau = 5.0;
    size_t fluxSourceRays = 0; // 0 means use nObservers (backward compatible)
    double fluxSourceDDMCFaceOpticalDepth = 5.0;
    FluxSourceSurfaceMode fluxSourceSurfaceMode = FluxSourceSurfaceMode::Grey;
    bool volumeEmissionEnabled = false;
    double volumeEmissionCutoffFraction = 1e-10;
    size_t volumeEmissionBurninPacketsTarget = 2000000;
    bool volumeEmissionGateGroups = true;
    bool volumeEmissionBurninExact = true;
    double volumeEmissionExplorationWeightFraction = 0.0;
    size_t volumeEmissionLearnedPhotonsPerCellBudget = 10;
    size_t volumeEmissionLearnedPhotonsPerCellBudgetGrey = 0;  // 0 = same as MG
    size_t volumeEmissionLearnedMinPhotons = 1;
    size_t volumeEmissionLearnedMaxPhotons = 5000;
    int polarizationManualScatterings = 128;
    double polarizationDepolarizationScatterings = 0.5;
    std::string polarizationClosure = "damped_last_scatterings";
    bool measuredLoadBalance = true;
    OpacityScaleMode opacityScaleMode = OpacityScaleMode::Planck;
    double opacityScaleAlphaMin = 0.0;  // 0 = unbounded
    double opacityScaleAlphaMax = 0.0;  // 0 = unbounded
    bool adaptiveSourceCells = false;
    size_t adaptiveSourceBurnin = 3;
    double adaptiveSourceStrength = 0.95;
    double adaptiveSourceEma = 0.5;
    double adaptiveSourceMinEscapedFrac = 1e-12;
    double adaptiveSourceMaxFactor = 1000.0;
    size_t adaptiveSourceBurninPhotonMultiplier = 2;
    double adaptiveSourceLearnedReserveFrac = 0.25;
    double adaptiveSourceLearnedMinFactor = 20.0;
    size_t adaptiveSourceLearnedMinPhotons = 10;
    size_t adaptiveSourceLearnedMaxPhotons = 5000;
    size_t adaptiveSourceLearnedPhotonsPerCellBudget = 100;
    double adaptiveSourceScorePower = 0.5;
    double adaptiveSourceWeightScoreFrac = 0.85;
    bool adaptiveObserverEquity = true;
    double adaptiveObserverExtraBudgetFrac = 2;
    double adaptiveObserverTargetNeff = 1000000;
    double adaptiveObserverTargetPolSnr = 10.0;
    double adaptiveObserverTargetPolSigma = 0.0;
    bool adaptiveObserverNormalizeDeficits = true;
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


std::string ReplaceExtension(std::string const& path, std::string const& newExt);
std::string InsertSuffixBeforeExtension(std::string const& path, std::string const& suffix);
std::string BaseVtkOutputPath(Config const& cfg);
std::string GreyVtkOutputPath(Config const& cfg);
bool ValidateConfig(Config& cfg, int rank);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_CONFIG_HPP
