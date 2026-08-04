#ifndef IMC_POST_PROCESS_CONTROL_HPP
#define IMC_POST_PROCESS_CONTROL_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

struct IMCPostProcessExternalSource
{
    size_t faceIndex = std::numeric_limits<size_t>::max();
    size_t cellID = std::numeric_limits<size_t>::max();
    size_t interiorCellID = std::numeric_limits<size_t>::max();
    Vector3D location = Vector3D(0.0, 0.0, 0.0);
    Vector3D outwardNormal = Vector3D(0.0, 0.0, 1.0);
    double luminosity = 0.0;
};

struct IMCSourceAllocationSummary
{
    bool adaptiveEnabled = false;
    size_t totalPhotons = 0;
    size_t sourceCells = 0;
    size_t boostedCells = 0;
    size_t learnedCells = 0;
    size_t learnedBoostedCells = 0;
    size_t learnedPhotons = 0;
    size_t learnedExtraPhotons = 0;
    size_t minPhotons = 0;
    size_t maxPhotons = 0;
    size_t learnedMinPhotons = 0;
    size_t learnedMaxPhotons = 0;
    double adaptiveScoreSum = 0.0;
    double adaptiveScoreP05 = 0.0;
    double adaptiveScoreP50 = 0.0;
    double adaptiveScoreP95 = 0.0;
    double adaptiveScoreMax = 0.0;
    double adaptiveScoreSpanLow = 0.0;
    double adaptiveScoreSpanHigh = 0.0;
    size_t learnedPhotonsAtLeast1000 = 0;
    size_t learnedPhotonsAtLeast2000 = 0;
};

struct IMCGroupSamplingDiagnostics
{
    size_t totalSampled = 0;
    size_t cellsWithGroupScores = 0;
    double weightCorrectionMin = 1.0;
    double weightCorrectionMax = 1.0;
    double weightCorrectionSum = 0.0;
    size_t weightCorrectionCount = 0;
    size_t weightCorrectionCapped = 0;
    size_t weightCorrectionFallback = 0;
    size_t invalidPdfFallback = 0;
    size_t invalidPdfFallbackPackets = 0;
    double sampledEnergy = 0.0;
    double cappedEnergy = 0.0;
    double cappedEnergyFraction = 0.0;
    bool estimatorPotentiallyBiased = false;
};

struct IMCPostProcessControl
{
    struct AdaptiveCells
    {
        bool enabled = false;
        std::unordered_map<size_t, double> scores;
        double strength = 0.0;
        double maxFactor = 1.0;
        double learnedReserveFraction = 0.0;
        double learnedMinFactor = 1.0;
        double observerBudgetMultiplier = 1.0;
        size_t learnedMinPhotons = 0;
        size_t learnedMaxPhotons = 0;
        double scorePower = 1.0;
    } adaptiveCells;

    struct AdaptiveGroups
    {
        bool enabled = false;
        std::unordered_map<size_t, std::array<double, ENERGY_GROUPS_NUM>> scores;
        double strength = 0.0;
        double pdfFloor = 0.0;
        double maxBias = 1.0;
        double maxWeightCorrection = 1.0;
    } adaptiveGroups;

    struct Emission
    {
        bool enabled = false;
        bool useLearnedScores = false;
        bool includeUniformBase = true;
        size_t baseMultiplier = 1;
        size_t learnedBoostFactor = 20;
        size_t learnedExtraBudget = 0;
    } emission;

    std::vector<IMCPostProcessExternalSource> externalSources;
};

struct IMCPostProcessGenerationDiagnostics
{
    IMCSourceAllocationSummary sourceAllocation;
    IMCGroupSamplingDiagnostics groupSampling;
    std::vector<size_t> sourcePhotonsPerCell;
};

#endif // IMC_POST_PROCESS_CONTROL_HPP
