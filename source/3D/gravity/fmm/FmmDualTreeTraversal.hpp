#ifndef FMM_DUAL_TREE_TRAVERSAL_HPP
#define FMM_DUAL_TREE_TRAVERSAL_HPP

#include <vector>

#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"

class FmmDualTreeTraversal
{
public:
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
                    FmmSolveStats& stats);
};

#endif // FMM_DUAL_TREE_TRAVERSAL_HPP
