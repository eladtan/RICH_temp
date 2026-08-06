#include "3D/gravity/fmm/FmmPasses.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "3D/gravity/fmm/FmmKernels.hpp"
#include "misc/universal_error.hpp"

void FmmPasses::allocate(FmmTree& tree,
                         const FmmTaylorExpansion& layout,
                         std::vector<double>& multipoles,
                         std::vector<double>& locals)
{
    tree.assignExpansionOffsets(layout.coefficientCount());
    if(tree.nodes().size() > std::numeric_limits<std::size_t>::max() /
                           layout.coefficientCount())
        throw UniversalError("FmmPasses::allocate: expansion storage overflow");
    const std::size_t expansionSize = tree.nodes().size() * layout.coefficientCount();
    multipoles.assign(expansionSize, 0.0);
    locals.assign(expansionSize, 0.0);
}

void FmmPasses::upward(const FmmTree& tree,
                       const std::vector<Vector3D>& positions,
                       const std::vector<double>& masses,
                       const FmmTaylorExpansion& layout,
                       std::vector<double>& multipoles)
{
    if(positions.size() != masses.size())
        throw UniversalError("FmmPasses::upward: position/mass size mismatch");
    if(layout.coefficientCount() == 0 ||
       tree.nodes().size() > std::numeric_limits<std::size_t>::max() /
                               layout.coefficientCount())
        throw UniversalError("FmmPasses::upward: expansion storage overflow");
    const std::size_t expected = tree.nodes().size() * layout.coefficientCount();
    if(multipoles.size() != expected || tree.particleOrder().size() != positions.size())
        throw UniversalError("FmmPasses::upward: inconsistent tree or expansion storage");
    for(std::size_t nodeIndex : tree.postOrder())
    {
        const FmmNode& node = tree.nodes()[nodeIndex];
        if(node.particleCount() == 0)
            continue;
        if(node.isLeaf())
        {
            FmmKernels::accumulateP2M(node, positions, masses, tree.particleOrder(),
                                      layout, multipoles);
        }
        else
        {
            for(int octant = 0; octant < 8; ++octant)
            {
                const std::size_t child = tree.childIndex(node, octant);
                if(child != std::numeric_limits<std::size_t>::max() &&
                   tree.nodes()[child].particleCount() != 0)
                    FmmKernels::translateM2M(tree.nodes()[child], node,
                                             layout, multipoles);
            }
        }
    }
}

void FmmPasses::downward(const FmmTree& tree,
                         const std::vector<Vector3D>& positions,
                         const FmmTaylorExpansion& layout,
                         std::vector<double>& locals,
                         std::vector<Vector3D>& acceleration,
                         std::vector<double>* positiveKernelPotential)
{
    if(layout.coefficientCount() == 0 ||
       tree.nodes().size() > std::numeric_limits<std::size_t>::max() /
                               layout.coefficientCount())
        throw UniversalError("FmmPasses::downward: expansion storage overflow");
    const std::size_t expected = tree.nodes().size() * layout.coefficientCount();
    if(locals.size() != expected || tree.particleOrder().size() != positions.size() ||
       acceleration.size() != positions.size() ||
       (positiveKernelPotential != nullptr &&
        positiveKernelPotential->size() != positions.size()))
        throw UniversalError("FmmPasses::downward: inconsistent tree or output storage");
    for(std::size_t nodeIndex : tree.preOrder())
    {
        const FmmNode& node = tree.nodes()[nodeIndex];
        if(node.particleCount() == 0)
            continue;
        if(node.parent != std::numeric_limits<std::size_t>::max())
            FmmKernels::translateL2L(tree.nodes()[node.parent], node,
                                     layout, locals);
        if(node.isLeaf())
            FmmKernels::evaluateL2P(node, positions, tree.particleOrder(), layout,
                                    locals, acceleration, positiveKernelPotential);
    }
}

void FmmPasses::updateTreeStats(const FmmTree& tree, FmmSolveStats& stats)
{
    stats.nodeCount = tree.nodes().size();
    stats.leafCount = tree.leafCount();
    stats.maxDepth = tree.maxDepth();
    std::size_t leafParticles = 0;
    const double cubeCornerFactor = std::sqrt(3.0);
    for(const FmmNode& node : tree.nodes())
    {
        if(node.isLeaf())
        {
            leafParticles += node.particleCount();
            stats.maxLeafOccupancy = std::max(stats.maxLeafOccupancy,
                                              node.particleCount());
        }
        if(node.halfSize > 0.0)
            stats.maxRadiusRatio = std::max(stats.maxRadiusRatio,
                node.radius / (cubeCornerFactor * node.halfSize));
    }
    stats.averageLeafOccupancy = stats.leafCount == 0 ? 0.0 :
        static_cast<double>(leafParticles) / static_cast<double>(stats.leafCount);
}
