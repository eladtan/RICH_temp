#ifndef FMM_LOCAL_PATCH_HPP
#define FMM_LOCAL_PATCH_HPP

#ifdef RICH_MPI

#include <cstdint>
#include <vector>

#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmRootGeometry.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchKey.hpp"

struct FmmLocalPatch
{
    FmmPatchKey key;
    FmmRootGeometry root;

    std::vector<std::size_t> inputIndices;
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> cellIds;

    FmmTree tree;
    FmmLocalInteractionPlan localPlan;
    std::vector<double> multipoles;
    std::vector<double> locals;
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;

    // topologyHash is the wire-visible invalidation token. Persistent patch
    // mode uses a monotonic generation; the nonpersistent builder uses the
    // planning-envelope hash so communication remains conservative.
    // structuralTreeHash and structuralSignature describe only spatial keys,
    // child masks, leaf state, and depth. nodeGeometryHash records tight current
    // geometry, while geometryEnvelopeHash and geometryEnvelopeGeneration
    // describe the retained conservative planning geometry. None includes leaf
    // occupancy.
    std::uint64_t topologyHash = 0;
    std::uint64_t structuralTreeHash = 0;
    std::uint64_t nodeGeometryHash = 0;
    std::uint64_t geometryEnvelopeHash = 0;
    std::uint64_t geometryEnvelopeGeneration = 0;
    std::vector<std::uint64_t> structuralSignature;
    std::vector<std::uint64_t> occupancySignature;

    bool rootGeometryChanged = false;
    bool leafTopologyChanged = false;
    bool nodeGeometryChanged = false;
    bool geometryEnvelopeChanged = false;
    bool leafOccupancyChanged = false;
    bool localPlanReused = false;
    bool persistentTreeRefit = false;
};

#endif // RICH_MPI

#endif // FMM_LOCAL_PATCH_HPP
