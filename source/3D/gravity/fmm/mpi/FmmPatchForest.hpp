#ifndef FMM_PATCH_FOREST_HPP
#define FMM_PATCH_FOREST_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <vector>

#include "3D/gravity/fmm/FmmConfig.hpp"
#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/mpi/FmmDistributedOptions.hpp"
#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
#include "3D/gravity/fmm/mpi/FmmLocalPatch.hpp"
#include "3D/gravity/fmm/mpi/FmmPackets.hpp"

struct FmmPatchForestChange
{
    bool patchSetChanged = false;
    bool patchGeometryChanged = false;
    bool structuralTopologyChanged = false;
    bool occupancyChanged = false;
    bool countOnlyChanged = false;
    std::size_t createdPatches = 0;
    std::size_t removedPatches = 0;
    std::size_t matchedPatchIds = 0;
    std::uint64_t persistentLeafSplits = 0;
    std::uint64_t persistentSubtreeMerges = 0;
    std::uint64_t persistentEmptyLeaves = 0;
    std::size_t overfullPatches = 0;
    std::size_t reusedPatches = 0;
    std::size_t reusedLocalPlans = 0;
    std::size_t rebuiltLocalPlans = 0;
    std::size_t nodeGeometryExpansionPatches = 0;
    std::size_t retainedBytes = 0;
    std::size_t releasedBytes = 0;
};

struct FmmPatchForestDiagnostics
{
    std::size_t patchCount = 0;
    std::size_t fixedLevelPatchCount = 0;
    std::size_t particlesPerPatchMin = 0;
    std::size_t particlesPerPatchMedian = 0;
    std::size_t particlesPerPatchP95 = 0;
    std::size_t particlesPerPatchMax = 0;
    std::vector<std::size_t> levelHistogram;
    std::size_t copiedInputBytes = 0;
    std::size_t patchTreeBytes = 0;
    std::size_t createdPatches = 0;
    std::size_t removedPatches = 0;
    std::size_t matchedPatchIds = 0;
    std::size_t overfullPatches = 0;
    std::size_t reusedPatches = 0;
    std::size_t reusedLocalPlans = 0;
    std::size_t rebuiltLocalPlans = 0;
    std::size_t nodeGeometryExpansionPatches = 0;
    std::size_t retainedBytes = 0;
    std::size_t releasedBytes = 0;
    double largestPatchRadius = 0.0;
};

class FmmPatchForest
{
public:
    FmmPatchForest();

    FmmPatchForestChange prepare(const std::vector<Vector3D>& positions,
                                 const std::vector<double>& masses,
                                 const std::vector<std::uint64_t>& cellIds,
                                 const Vector3D& domainLower,
                                 const Vector3D& domainUpper,
                                 const FmmGravityOptions& gravityOptions,
                                 const FmmDistributedOptions& distributedOptions,
                                 int ownerRank);

    // Reuse patch objects by stable patch ID.  Existing trees are refit with
    // persistent split/merge hysteresis and local interaction plans are kept
    // whenever their conservative geometry contract remains valid.
    FmmPatchForestChange preparePersistent(
        const std::vector<Vector3D>& positions,
        const std::vector<double>& masses,
        const std::vector<std::uint64_t>& cellIds,
        const Vector3D& domainLower,
        const Vector3D& domainUpper,
        const FmmGravityOptions& gravityOptions,
        const FmmDistributedOptions& distributedOptions,
        int ownerRank);

    void buildUpward(const FmmTaylorExpansion& layout);
    void clearLocals(const FmmTaylorExpansion& layout);
    void executeLocalSelf(const FmmTaylorExpansion& layout,
                          FmmM2LOperatorCache& operatorCache,
                          std::size_t maxOperatorCacheBytes,
                          FmmSolveStats& stats);
    void executeDirectCrossPatch(const FmmTaylorExpansion& layout,
                                 FmmSolveStats& stats);
    void applyDownward(const FmmTaylorExpansion& layout);

    void scatterAcceleration(std::vector<Vector3D>& acceleration) const;
    void scatterPotential(std::vector<double>& potential) const;

    std::vector<FmmPatchRootDescriptor> descriptors(int ownerRank,
                                                    std::uint64_t topologyEpoch) const;

    const std::vector<FmmLocalPatch>& patches() const { return patches_; }
    std::vector<FmmLocalPatch>& patches() { return patches_; }
    std::size_t findPatch(std::uint64_t patchId) const;

    std::uint64_t geometryHash() const { return geometryHash_; }
    std::uint64_t structuralHash() const { return structuralHash_; }
    std::uint64_t occupancyHash() const { return occupancyHash_; }

    const FmmPatchForestDiagnostics& diagnostics() const { return diagnostics_; }
    const FmmGlobalDyadicLattice& lattice() const { return lattice_; }

private:
    struct PatchBucket
    {
        std::uint64_t patchId = 0;
        std::vector<std::size_t> inputIndices;
    };

    void sortPatchBuckets(std::vector<PatchBucket>& buckets) const;
    void sortIndicesWithinPatch(FmmLocalPatch& patch) const;
    void buildPatchObjects(const std::vector<PatchBucket>& buckets,
                           const std::vector<Vector3D>& positions,
                           const std::vector<double>& masses,
                           const std::vector<std::uint64_t>& cellIds,
                           int ownerRank);
    std::vector<PatchBucket> partitionParticles(
        const std::vector<Vector3D>& positions,
        const FmmDistributedOptions& distributedOptions,
        std::size_t& overfullPatches,
        std::size_t& fixedLevelPatchCount) const;
    std::vector<PatchBucket> adaptiveRefine(
        std::uint64_t patchId,
        int level,
        std::vector<std::size_t> inputIndices,
        const std::vector<Vector3D>& positions,
        const FmmDistributedOptions& distributedOptions,
        std::size_t& overfullPatches,
        std::size_t& projectedPatchCount) const;
    void validatePrepareInputs(const std::vector<Vector3D>& positions,
                               const std::vector<double>& masses,
                               const std::vector<std::uint64_t>& cellIds,
                               const Vector3D& domainLower,
                               const Vector3D& domainUpper,
                               const FmmGravityOptions& gravityOptions,
                               const FmmDistributedOptions& distributedOptions,
                               int ownerRank) const;
    FmmPatchForestChange compareWithPrevious() const;
    void updateDiagnostics(const FmmPatchForestChange& change,
                           std::size_t fixedLevelPatchCount);
    void updateHashes();

    FmmGlobalDyadicLattice lattice_;
    FmmGravityOptions gravityOptions_;
    FmmDistributedOptions distributedOptions_;
    std::vector<FmmLocalPatch> patches_;
    std::vector<FmmLocalPatch> previousPatches_;
    std::size_t inputCount_ = 0;
    std::size_t fixedLevelPatchCount_ = 0;
    FmmPatchForestDiagnostics diagnostics_;
    std::uint64_t geometryHash_ = 0;
    std::uint64_t structuralHash_ = 0;
    std::uint64_t occupancyHash_ = 0;
};

#endif // RICH_MPI

#endif // FMM_PATCH_FOREST_HPP
