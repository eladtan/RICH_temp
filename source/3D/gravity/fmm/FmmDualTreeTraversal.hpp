#ifndef FMM_DUAL_TREE_TRAVERSAL_HPP
#define FMM_DUAL_TREE_TRAVERSAL_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"

struct FmmLocalInteractionPlan
{
    struct M2LPair
    {
        std::uint32_t targetNode = 0;
        std::uint32_t sourceNode = 0;
        std::uint32_t geometryIndex = 0;
        float admissibleRadiusSum = 0.0f;
    };

    struct P2PPair
    {
        std::uint32_t targetNode = 0;
        std::uint32_t sourceNode = 0;
    };

    std::vector<M2LPair> m2lPairs;
    std::vector<P2PPair> p2pPairs;
    std::vector<FmmM2LOperatorCache::PreparedGeometry> geometries;
    std::uint64_t rejectedSameNode = 0;
    std::uint64_t rejectedOverlap = 0;
    std::uint64_t rejectedRatio = 0;
    std::size_t maxTraversalStack = 0;
    bool initialized = false;

    void clear();
    std::size_t bytesOwned() const;
};

class FmmDualTreeTraversal
{
public:
    static void buildLocalPlan(const FmmTree& tree,
                               double thetaCritical,
                               FmmLocalInteractionPlan& plan);

    static bool localPlanReusable(const FmmTree& tree,
                                  const FmmLocalInteractionPlan& plan);

    static void runLocalPlan(const FmmTree& tree,
                             const FmmLocalInteractionPlan& plan,
                             const std::vector<Vector3D>& positions,
                             const std::vector<double>& masses,
                             const FmmTaylorExpansion& layout,
                             const std::vector<double>& multipoles,
                             std::vector<double>& locals,
                             std::vector<Vector3D>& acceleration,
                             std::vector<double>* positiveKernelPotential,
                             FmmM2LOperatorCache& operatorCache,
                             std::size_t maxOperatorCacheBytes,
                             FmmSolveStats& stats);

    static void run(const FmmTree& targetTree,
                    const FmmTree& sourceTree,
                    const std::vector<Vector3D>& targetPositions,
                    const std::vector<Vector3D>& sourcePositions,
                    const std::vector<double>& sourceMasses,
                    const FmmTaylorExpansion& layout,
                    const std::vector<double>& sourceMultipoles,
                    std::vector<double>& targetLocals,
                    bool sameParticleSet,
                    double thetaCritical,
                    std::vector<Vector3D>& acceleration,
                    std::vector<double>* positiveKernelPotential,
                    FmmM2LOperatorCache& operatorCache,
                    std::size_t maxOperatorCacheBytes,
                    FmmSolveStats& stats);
};

#endif // FMM_DUAL_TREE_TRAVERSAL_HPP
