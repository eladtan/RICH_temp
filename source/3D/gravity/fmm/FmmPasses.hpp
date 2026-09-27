#ifndef FMM_PASSES_HPP
#define FMM_PASSES_HPP

#include <cstdint>
#include <vector>

#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"

namespace FmmPasses
{
void allocate(FmmTree& tree,
              const FmmTaylorExpansion& layout,
              std::vector<double>& multipoles,
              std::vector<double>& locals);

void upward(const FmmTree& tree,
            const std::vector<Vector3D>& positions,
            const std::vector<double>& masses,
            const FmmTaylorExpansion& layout,
            std::vector<double>& multipoles);

// With nodeTargets (one flag per tree node, from markTargetNodes), only the
// subtrees holding a target particle receive locals and leaf evaluations;
// the other particles keep whatever acceleration they had.  prunedLeaves
// counts the occupied leaves skipped that way.
void downward(const FmmTree& tree,
              const std::vector<Vector3D>& positions,
              const FmmTaylorExpansion& layout,
              std::vector<double>& locals,
              std::vector<Vector3D>& acceleration,
              std::vector<double>* positiveKernelPotential,
              const std::vector<unsigned char>* nodeTargets = nullptr,
              std::uint64_t* prunedLeaves = nullptr);

// Flags every node with a target particle (particleTargets, indexed like the
// tree's particles) in its subtree.  Returns the number of flagged nodes.
std::uint64_t markTargetNodes(const FmmTree& tree,
                              const std::vector<unsigned char>& particleTargets,
                              std::vector<unsigned char>& nodeTargets);

void updateTreeStats(const FmmTree& tree, FmmSolveStats& stats);
}

#endif // FMM_PASSES_HPP
