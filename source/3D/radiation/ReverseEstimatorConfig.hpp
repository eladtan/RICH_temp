#ifndef REVERSE_ESTIMATOR_CONFIG_HPP
#define REVERSE_ESTIMATOR_CONFIG_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

enum class PostProcessEstimatorMode
{
    Forward,
    Reverse,
    Both
};

enum class ReverseFollowForward
{
    On,
    Off,
    FollowForward
};

enum class ReverseDDMCPolarizationClosure
{
    Synthetic,
    Depolarize,
    ExplicitK
};

enum class ReverseMultigroupDDMCMode
{
    Grey,
    ExactGroup,
    PGRWCollapsed
};

struct ThermalSampleResult
{
    enum class FailureReason
    {
        None,
        NoExactSampler,
        EmptyHighEnergyCDF,
        BadTemperature,
        ZeroApproxWeight
    };

    bool ok = false;
    bool exactForward = false;
    bool approximate = false;
    double nuCo = 0.0;
    size_t sampledGroup = 0;
    FailureReason failure = FailureReason::None;
};

struct ReverseEstimatorConfig
{
    size_t packetsPerObserver = 10000;
    size_t packetsPerObserverGroup = 0;
    double weightCutoff = 1e-8;
    size_t maxEvents = 500000;
    size_t maxDDMCSteps = 100000;
    uint64_t seed = 42;
    std::string outputPrefix = "reverse_tally";
    double progressIntervalSec = 5.0;

    ReverseFollowForward polarization = ReverseFollowForward::FollowForward;
    ReverseFollowForward velocity = ReverseFollowForward::FollowForward;
    ReverseFollowForward ddmc = ReverseFollowForward::FollowForward;

    double ddmcMinCellOpticalDepth = 15.0;
    double ddmcMinParticleOpticalDepth = 5.0;
    bool ddmcObserverExclusion = true;
    bool ddmcPhotosphereExclusion = true;
    double ddmcPhotosphereOpticalDepth = 5.0;

    ReverseDDMCPolarizationClosure ddmcPolClosure = ReverseDDMCPolarizationClosure::Synthetic;
    int ddmcManualScatterings = 4;
    double ddmcDepolarizationScatterings = 2.0;

    ReverseMultigroupDDMCMode multigroupDDMCMode = ReverseMultigroupDDMCMode::PGRWCollapsed;

    bool allowApproximateThermalUpscatter = false;
    bool allowDefaultFleckOne = false;

    size_t debugScatterOrders = 0;

    PostProcessEstimatorMode estimatorMode = PostProcessEstimatorMode::Forward;

    bool resolvePolarization(bool forwardPolarization) const
    {
        switch (polarization)
        {
        case ReverseFollowForward::On:  return true;
        case ReverseFollowForward::Off: return false;
        case ReverseFollowForward::FollowForward: return forwardPolarization;
        }
        return false;
    }

    bool resolveVelocity(bool forwardVelocity) const
    {
        switch (velocity)
        {
        case ReverseFollowForward::On:  return true;
        case ReverseFollowForward::Off: return false;
        case ReverseFollowForward::FollowForward: return forwardVelocity;
        }
        return false;
    }

    bool resolveDDMC(bool forwardDDMC) const
    {
        switch (ddmc)
        {
        case ReverseFollowForward::On:  return true;
        case ReverseFollowForward::Off: return false;
        case ReverseFollowForward::FollowForward: return forwardDDMC;
        }
        return false;
    }
};

inline size_t ResolveReverseLBPilotPacketsPerObserverGroup(
    size_t productionPacketsPerObserverGroup,
    size_t requestedPilotPacketsPerObserverGroup)
{
    if (productionPacketsPerObserverGroup <= 1)
        return 0;

    size_t pilot = requestedPilotPacketsPerObserverGroup;
    if (pilot == 0)
        pilot = (productionPacketsPerObserverGroup + 99) / 100;

    return std::max<size_t>(
        1, std::min(pilot, productionPacketsPerObserverGroup - 1));
}

namespace ReverseOutputStrings
{
inline constexpr char ResolvedOnlyLegacyAlias[] =
    "resolved_only_legacy_alias";

inline constexpr char GroupISemantics[] =
    "group_N/I is resolved_only; group_N/I_total is resolved+collapsed";

inline constexpr char PGRWOutputSemanticsMetadataPath[] =
    "/metadata/pgrw_output_semantics";

inline constexpr char PGRWEstimatorGroupOutputModeMetadataPath[] =
    "/metadata/pgrw_estimator_group_output_mode";

inline constexpr char SigmaSampleSpace[] =
    "launched_packet_count; nonzero_packet_count is NOT effective_sample_size";

inline constexpr char FleckForwardVectorShared[] =
    "reverse received exact forward IMC Fleck vector";

inline constexpr char FleckHelperNotForward[] =
    "reverse recomputed Fleck factors with helper; not the exact forward vector";
}

#endif // REVERSE_ESTIMATOR_CONFIG_HPP
