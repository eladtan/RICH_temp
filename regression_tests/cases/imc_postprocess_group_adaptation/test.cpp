// Analytic source records exercise production scoring, MPI totals and overflow.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "source/3D/radiation/postprocess/AdaptiveStatistics.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {
using namespace imc_postprocess_tde;
int Rank = 0;
int Ranks = 1;
int Failures = 0;

void Check(bool Passed, std::string const& Label) {
    if (!Passed)
        ++Failures;
    if (Rank == 0)
        std::cout << (Passed ? "PASS " : "FAIL ") << Label << '\n';
}

bool Close(double A, double B) {
    return std::isfinite(A) && std::abs(A - B) <= 1e-12 * std::max(1.0, std::abs(B));
}

SphericalObserver::SourceCellGroupEscapeStat Record(size_t Cell, size_t Group, double Energy, double WeightSq) {
    SphericalObserver::SourceCellGroupEscapeStat Stat;
    Stat.cellID = Cell;
    Stat.observerIndex = 0;
    Stat.groupIndex = Group;
    Stat.energy = Energy;
    Stat.weightSq = WeightSq;
    Stat.count = 8;
    return Stat;
}

void TestSourceScores() {
    Config Settings;
    Settings.adaptiveGroupSourceCells = true;
    Settings.adaptiveGroupMaxLocalStats = 10;
    Settings.adaptiveGroupScoreEma = 1.0;
    ObserverGroupQualityDiagnostics Quality;
    Quality.enabled = true;
    Quality.observerCount = 1;
    Quality.groupCount = 2;
    Quality.predictedPriority = {{1.0, 2.0}};
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> Records;
    if (Rank == 0)
        Records.push_back(Record(10, 0, 2.0, 1.0));
    if (Rank == (Ranks > 1 ? 1 : 0))
        Records.push_back(Record(11, 0, 8.0, 9.0));

    for (bool Polarized : {false, true}) {
        Quality.polarizationMode = Polarized;
        for (double EnergyScale : {1.0, 1e-40, 1e20}) {
            auto Scaled = Records;
            for (auto& Stat : Scaled) {
                Stat.energy *= EnergyScale;
                Stat.weightSq *= EnergyScale * EnergyScale;
            }
            AdaptiveGroupSourceState State;
            UpdateAdaptiveSourceGroupScores(Scaled, Quality, Settings, State, Rank, Ranks);
            // E fractions are 2/10,8/10; kw^2 fractions are 1/10,9/10.
            Check(State.cellScoreFromGroups.size() == 2 &&
                  Close(State.cellScoreFromGroups.at(10), Polarized ? 0.115 : 0.15) &&
                  Close(State.cellScoreFromGroups.at(11), Polarized ? 0.885 : 0.85),
                  "global energy/weight-square fractions and energy-unit invariance");
        }
    }

    Quality.polarizationMode = true;
    auto Invalid = Records;
    if (Rank == 0) {
        Invalid.push_back(Record(20, 0, 0.0, 100.0));
        Invalid.push_back(Record(21, 0, -1.0, 100.0));
        Invalid.push_back(Record(22, 0, std::numeric_limits<double>::quiet_NaN(), 100.0));
        Invalid.push_back(Record(23, 0, std::numeric_limits<double>::infinity(), 100.0));
    }
    AdaptiveGroupSourceState Filtered;
    UpdateAdaptiveSourceGroupScores(Invalid, Quality, Settings, Filtered, Rank, Ranks);
    Check(Filtered.cellScoreFromGroups.size() == 2 && Close(Filtered.cellScoreFromGroups.at(10), 0.115) &&
          Close(Filtered.cellScoreFromGroups.at(11), 0.885), "invalid energy does not contaminate global totals");

    auto MissingWeightSq = Records;
    for (auto& Stat : MissingWeightSq)
        Stat.weightSq = 0.0;
    AdaptiveGroupSourceState Fallback;
    UpdateAdaptiveSourceGroupScores(MissingWeightSq, Quality, Settings, Fallback, Rank, Ranks);
    Check(Close(Fallback.cellScoreFromGroups.at(10), 0.03 + 0.85 / 17.0),
          "missing weight-square uses energy-square fallback");

    Settings.adaptiveGroupScoreEma = 0.25;
    Quality.predictedPriority[0][0] = 3.0;
    AdaptiveGroupSourceState Weighted;
    UpdateAdaptiveSourceGroupScores(Records, Quality, Settings, Weighted, Rank, Ranks);
    Check(Close(Weighted.cellScoreFromGroups.at(10), 0.25 * 3.0 * 0.115) &&
          Close(Weighted.cellScoreFromGroups.at(11), 0.25 * 3.0 * 0.885), "priority and EMA scale source scores");

    Settings.adaptiveGroupScoreEma = 1.0;
    Quality.predictedPriority[0][0] = 1.0;
    Records.clear();
    if (Rank == 0)
        Records = {Record(30, 0, 100.0, 10000.0), Record(31, 1, 1.0, 1.0)};
    AdaptiveGroupSourceState TwoGroups;
    UpdateAdaptiveSourceGroupScores(Records, Quality, Settings, TwoGroups, Rank, Ranks);
    Check(Close(TwoGroups.cellScoreFromGroups.at(30), 1.0) &&
          Close(TwoGroups.cellScoreFromGroups.at(31), 2.0), "each group uses its own normalization");

    Records.clear();
    if (Rank == 0)
        Records = {Record(10, 0, 2.0, 1.0), Record(11, 0, 8.0, 9.0)};
    Settings.adaptiveGroupMaxLocalStats = 1;
    AdaptiveGroupSourceState Capped;
    auto Summary = UpdateAdaptiveSourceGroupScores(Records, Quality, Settings, Capped, Rank, Ranks);
    Check(Capped.cellScoreFromGroups.size() == 1 && Close(Capped.cellScoreFromGroups.at(11), 0.885),
          "legacy retention keeps bright record but scoring totals include pruned records");
    Check(Summary.capExceeded && Summary.totalStatsDropped == 1 && Close(Summary.keptEnergyFraction, 0.8),
          "overflow diagnostics preserve dropped count and retained energy");
    Settings.adaptiveGroupFallbackToIntegratedOnOverflow = true;
    auto IntegratedFallback = Capped;
    Summary = UpdateAdaptiveSourceGroupScores(Records, Quality, Settings, IntegratedFallback, Rank, Ranks);
    Check(Summary.fallbackToIntegratedPath && IntegratedFallback.scoreByCellGroup.empty() &&
          IntegratedFallback.cellScoreFromGroups.empty(), "overflow fallback clears group scores on all ranks");
}
} // namespace

int main(int Argc, char* Argv[]) {
#ifdef RICH_MPI
    MPI_Init(&Argc, &Argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ranks);
#else
    (void)Argc;
    (void)Argv;
#endif
    TestSourceScores();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (Rank == 0) {
        std::ofstream Metrics("imc_postprocess_group_adaptation_metrics.txt");
        Metrics << "failures " << Failures << "\npass " << (Failures == 0 ? 1 : 0) << '\n';
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return Failures == 0 ? 0 : 1;
}
