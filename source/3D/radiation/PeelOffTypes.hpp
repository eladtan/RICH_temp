#ifndef PEEL_OFF_TYPES_HPP
#define PEEL_OFF_TYPES_HPP

#include <array>
#include <cstddef>
#include <string>

enum class PeelOffEventKind
{
    SOURCE_EMISSION = 0,
    RESOLVED_ELASTIC_SCATTER = 1,
    RESOLVED_EFFECTIVE_SCATTER = 2,
    RW_CLOSURE = 3,
    RW_UPSCATTER = 4,
    DDMC_LEAK = 5,
    DDMC_UPSCATTER = 6,
    COUNT = 7
};

static constexpr size_t NumPeelOffKinds =
    static_cast<size_t>(PeelOffEventKind::COUNT);

inline const char* peelOffKindName(PeelOffEventKind kind)
{
    switch (kind)
    {
        case PeelOffEventKind::SOURCE_EMISSION:            return "source_emission";
        case PeelOffEventKind::RESOLVED_ELASTIC_SCATTER:   return "resolved_elastic_scatter";
        case PeelOffEventKind::RESOLVED_EFFECTIVE_SCATTER: return "resolved_effective_scatter";
        case PeelOffEventKind::RW_CLOSURE:                 return "rw_closure";
        case PeelOffEventKind::RW_UPSCATTER:               return "rw_upscatter";
        case PeelOffEventKind::DDMC_LEAK:                  return "ddmc_leak";
        case PeelOffEventKind::DDMC_UPSCATTER:             return "ddmc_upscatter";
        default:                                           return "unknown";
    }
}

// Counter schema v5: precise enough for distributed MPI ray debugging.
// rayFailed is the aggregate failure counter (sum of all failure sub-categories).
// Invariants (global, after MPI reduction to rank 0; per-rank counters in
// DistributedExact mode do NOT satisfy these because a ray can start on one
// rank and complete/fail on another):
//   directionsConsidered = phaseAccepted + phaseRejected
//   raysStarted = raysCompleted + rayFailed
//   rayFailed = tauClipped + noExitFace + maxCellsExceeded + invalidState
//             + unsupportedBoundary + lostRemoteCell
//             + distributedExchangeLimitExceeded + mpiBoundaryRejected
//   recorded <= raysCompleted
//   mpiBoundaryCrossings >= raysCrossedMpiBoundary
//
// sourceExitClassFailed is NOT part of the ray-level invariants. It counts
// source-level rejections where face classification failed before any
// observer-direction ray was created (e.g. DDMC leak on an unsupported face).
static constexpr int PeelOffCounterSchemaVersion = 5;

struct PeelOffCounters
{
    std::array<unsigned long long, NumPeelOffKinds> directionsConsidered{};
    std::array<unsigned long long, NumPeelOffKinds> phaseAccepted{};
    std::array<unsigned long long, NumPeelOffKinds> phaseRejected{};
    std::array<unsigned long long, NumPeelOffKinds> observerMissed{};
    std::array<unsigned long long, NumPeelOffKinds> timeRejected{};

    std::array<unsigned long long, NumPeelOffKinds> raysStarted{};
    std::array<unsigned long long, NumPeelOffKinds> raysCompleted{};
    std::array<unsigned long long, NumPeelOffKinds> recorded{};

    std::array<unsigned long long, NumPeelOffKinds> tauClipped{};
    std::array<unsigned long long, NumPeelOffKinds> rayFailed{};
    std::array<unsigned long long, NumPeelOffKinds> noExitFace{};
    std::array<unsigned long long, NumPeelOffKinds> maxCellsExceeded{};
    std::array<unsigned long long, NumPeelOffKinds> unsupportedBoundary{};
    std::array<unsigned long long, NumPeelOffKinds> lostRemoteCell{};
    std::array<unsigned long long, NumPeelOffKinds> distributedExchangeLimitExceeded{};
    std::array<unsigned long long, NumPeelOffKinds> mpiBoundaryRejected{};
    std::array<unsigned long long, NumPeelOffKinds> invalidState{};

    std::array<unsigned long long, NumPeelOffKinds> raysCrossedMpiBoundary{};
    std::array<unsigned long long, NumPeelOffKinds> mpiBoundaryCrossings{};

    std::array<unsigned long long, NumPeelOffKinds> physicalVacuumExits{};

    // Source-level rejection: face classification failed before any ray was
    // created. NOT included in the rayFailed invariant (no ray was started).
    std::array<unsigned long long, NumPeelOffKinds> sourceExitClassFailed{};

    void reset()
    {
        directionsConsidered = {};
        phaseAccepted = {};
        phaseRejected = {};
        observerMissed = {};
        timeRejected = {};
        raysStarted = {};
        raysCompleted = {};
        recorded = {};
        tauClipped = {};
        rayFailed = {};
        noExitFace = {};
        maxCellsExceeded = {};
        unsupportedBoundary = {};
        lostRemoteCell = {};
        distributedExchangeLimitExceeded = {};
        mpiBoundaryRejected = {};
        invalidState = {};
        raysCrossedMpiBoundary = {};
        mpiBoundaryCrossings = {};
        physicalVacuumExits = {};
        sourceExitClassFailed = {};
    }

    unsigned long long totalDirectionsConsidered() const
    {
        unsigned long long s = 0;
        for (auto v : directionsConsidered) s += v;
        return s;
    }
    unsigned long long totalRaysStarted() const
    {
        unsigned long long s = 0;
        for (auto v : raysStarted) s += v;
        return s;
    }
    unsigned long long totalRecorded() const
    {
        unsigned long long s = 0;
        for (auto v : recorded) s += v;
        return s;
    }
    unsigned long long totalTauClipped() const
    {
        unsigned long long s = 0;
        for (auto v : tauClipped) s += v;
        return s;
    }
    unsigned long long totalRayFailed() const
    {
        unsigned long long s = 0;
        for (auto v : rayFailed) s += v;
        return s;
    }

    bool validateInvariants(std::string& msg) const
    {
        for (size_t k = 0; k < NumPeelOffKinds; ++k)
        {
            if (directionsConsidered[k] != phaseAccepted[k] + phaseRejected[k])
            {
                msg = "directionsConsidered != phaseAccepted + phaseRejected for kind " + std::to_string(k);
                return false;
            }
            if (raysStarted[k] != raysCompleted[k] + rayFailed[k])
            {
                msg = "raysStarted != raysCompleted + rayFailed for kind " + std::to_string(k);
                return false;
            }
            unsigned long long failSum = tauClipped[k] + noExitFace[k]
                + maxCellsExceeded[k] + invalidState[k] + unsupportedBoundary[k]
                + lostRemoteCell[k] + distributedExchangeLimitExceeded[k]
                + mpiBoundaryRejected[k];
            if (rayFailed[k] != failSum)
            {
                msg = "rayFailed != failure subcategory sum for kind " + std::to_string(k);
                return false;
            }
            if (recorded[k] > raysCompleted[k])
            {
                msg = "recorded > raysCompleted for kind " + std::to_string(k);
                return false;
            }
            if (mpiBoundaryCrossings[k] < raysCrossedMpiBoundary[k])
            {
                msg = "mpiBoundaryCrossings < raysCrossedMpiBoundary for kind " + std::to_string(k);
                return false;
            }
        }
        return true;
    }
};

#endif // PEEL_OFF_TYPES_HPP
