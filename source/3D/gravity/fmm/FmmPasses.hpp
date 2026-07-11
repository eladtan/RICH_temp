#ifndef FMM_PASSES_HPP
#define FMM_PASSES_HPP

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

void downward(const FmmTree& tree,
              const std::vector<Vector3D>& positions,
              const FmmTaylorExpansion& layout,
              std::vector<double>& locals,
              std::vector<Vector3D>& acceleration,
              std::vector<double>* positiveKernelPotential);

void updateTreeStats(const FmmTree& tree, FmmSolveStats& stats);
}

#endif // FMM_PASSES_HPP
