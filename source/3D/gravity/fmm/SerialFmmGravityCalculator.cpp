#include "3D/gravity/fmm/SerialFmmGravityCalculator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmKernels.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "misc/universal_error.hpp"

namespace
{
typedef std::chrono::steady_clock Clock;

double elapsedSeconds(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool finiteVector(const Vector3D& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
}

SerialFmmGravityCalculator::SerialFmmGravityCalculator(FmmGravityOptions options):
    options_(options)
{
    validateOptions();
}

void SerialFmmGravityCalculator::solve(const std::vector<Vector3D>& positions,
                                       const std::vector<double>& masses,
                                       const Vector3D& domainLower,
                                       const Vector3D& domainUpper,
                                       std::vector<Vector3D>& acceleration,
                                       std::vector<double>* positiveKernelPotential)
{
    validateOptions();
    validateInputs(positions, masses);
    if(options_.computePotential != (positiveKernelPotential != nullptr))
        throw UniversalError("SerialFmmGravityCalculator::solve: computePotential must match the potential output pointer");

    const Clock::time_point totalStart = Clock::now();
    stats_ = FmmSolveStats();
    stats_.particleCount = positions.size();
    long double totalMass = 0.0L;
    for(double mass : masses)
        totalMass += static_cast<long double>(mass);
    stats_.totalMass = static_cast<double>(totalMass);
    if(!std::isfinite(stats_.totalMass))
        throw UniversalError("SerialFmmGravityCalculator::solve: total mass is not finite");

    const Clock::time_point buildStart = Clock::now();
    tree_.build(positions, domainLower, domainUpper, options_);
    stats_.buildSeconds = elapsedSeconds(buildStart);
    stats_.nodeCount = tree_.nodes().size();
    stats_.leafCount = tree_.leafCount();
    stats_.maxDepth = tree_.maxDepth();

    acceleration.assign(positions.size(), Vector3D());
    if(positiveKernelPotential != nullptr)
        positiveKernelPotential->assign(positions.size(), 0.0);

    if(positions.empty())
    {
        multipoles_.clear();
        locals_.clear();
        stats_.bytesOwned = tree_.bytesOwned() +
            multipoles_.capacity() * sizeof(double) +
            locals_.capacity() * sizeof(double);
        stats_.totalSeconds = elapsedSeconds(totalStart);
        return;
    }

    const FmmTaylorExpansion layout(options_.expansionOrder);
    tree_.assignExpansionOffsets(layout.coefficientCount());
    if(tree_.nodes().size() > std::numeric_limits<std::size_t>::max() /
                               layout.coefficientCount())
        throw UniversalError("SerialFmmGravityCalculator::solve: expansion storage overflow");
    const std::size_t expansionSize = tree_.nodes().size() * layout.coefficientCount();
    multipoles_.assign(expansionSize, 0.0);
    locals_.assign(expansionSize, 0.0);

    std::size_t leafParticles = 0;
    const double cubeCornerFactor = std::sqrt(3.0);
    for(const FmmNode& node : tree_.nodes())
    {
        if(node.isLeaf())
        {
            leafParticles += node.particleCount();
            stats_.maxLeafOccupancy = std::max(stats_.maxLeafOccupancy,
                                               node.particleCount());
        }
        if(node.halfSize > 0.0)
            stats_.maxRadiusRatio = std::max(stats_.maxRadiusRatio,
                node.radius / (cubeCornerFactor * node.halfSize));
    }
    stats_.averageLeafOccupancy = stats_.leafCount == 0 ? 0.0 :
        static_cast<double>(leafParticles) / static_cast<double>(stats_.leafCount);

    const Clock::time_point upwardStart = Clock::now();
    for(std::size_t nodeIndex : tree_.postOrder())
    {
        const FmmNode& node = tree_.nodes()[nodeIndex];
        if(node.isLeaf())
        {
            FmmKernels::accumulateP2M(node, positions, masses, tree_.particleOrder(),
                                      layout, multipoles_);
        }
        else
        {
            for(int octant = 0; octant < 8; ++octant)
            {
                const std::size_t child = tree_.childIndex(node, octant);
                if(child != std::numeric_limits<std::size_t>::max())
                    FmmKernels::translateM2M(tree_.nodes()[child], node,
                                             layout, multipoles_);
            }
        }
    }
    stats_.upwardSeconds = elapsedSeconds(upwardStart);
    stats_.rootMass = multipoles_[tree_.nodes()[0].multipoleOffset +
                                  layout.index(0, 0, 0)];
    if(!std::isfinite(stats_.rootMass))
        throw UniversalError("SerialFmmGravityCalculator::solve: non-finite root mass");
    const double massTolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(stats_.totalMass));
    if(std::abs(stats_.rootMass - stats_.totalMass) > massTolerance)
        throw UniversalError("SerialFmmGravityCalculator::solve: root multipole mass mismatch");

    const Clock::time_point interactionStart = Clock::now();
    FmmDualTreeTraversal::run(tree_, tree_, positions, positions, masses, layout,
                              multipoles_, locals_, true, options_.thetaCritical,
                              acceleration, positiveKernelPotential, stats_);
    stats_.interactionSeconds = elapsedSeconds(interactionStart);

    const Clock::time_point downwardStart = Clock::now();
    for(std::size_t nodeIndex : tree_.preOrder())
    {
        const FmmNode& node = tree_.nodes()[nodeIndex];
        if(node.parent != std::numeric_limits<std::size_t>::max())
            FmmKernels::translateL2L(tree_.nodes()[node.parent], node,
                                     layout, locals_);
        if(node.isLeaf())
            FmmKernels::evaluateL2P(node, positions, tree_.particleOrder(), layout,
                                    locals_, acceleration, positiveKernelPotential);
    }
    stats_.downwardSeconds = elapsedSeconds(downwardStart);

    stats_.bytesOwned = tree_.bytesOwned() +
        multipoles_.capacity() * sizeof(double) +
        locals_.capacity() * sizeof(double) +
        stats_.maxTraversalStack * 2 * sizeof(std::size_t);

    if(options_.validateFinite)
    {
        for(std::size_t i = 0; i < acceleration.size(); ++i)
        {
            if(!finiteVector(acceleration[i]) ||
               (positiveKernelPotential != nullptr &&
                !std::isfinite((*positiveKernelPotential)[i])))
            {
                UniversalError eo("SerialFmmGravityCalculator::solve: non-finite output");
                eo.addEntry("particle", i);
                throw eo;
            }
        }
    }

    stats_.totalSeconds = elapsedSeconds(totalStart);
}

const FmmSolveStats& SerialFmmGravityCalculator::stats() const noexcept
{
    return stats_;
}

void SerialFmmGravityCalculator::validateOptions() const
{
    if(options_.expansionOrder < 1 || options_.expansionOrder > FMM_MAX_ORDER)
        throw UniversalError("SerialFmmGravityCalculator: expansionOrder outside supported range");
    if(!(options_.thetaCritical > 0.0) || options_.thetaCritical > 1.0 ||
       !std::isfinite(options_.thetaCritical))
        throw UniversalError("SerialFmmGravityCalculator: thetaCritical must be finite and in (0,1]");
    if(options_.leafCapacity == 0)
        throw UniversalError("SerialFmmGravityCalculator: leafCapacity must be positive");
    if(options_.maxDepth <= 0 || options_.maxDepth > FMM_MAX_TREE_DEPTH)
        throw UniversalError("SerialFmmGravityCalculator: maxDepth outside supported range");
}

void SerialFmmGravityCalculator::validateInputs(const std::vector<Vector3D>& positions,
                                                const std::vector<double>& masses) const
{
    if(positions.size() != masses.size())
        throw UniversalError("SerialFmmGravityCalculator::solve: positions/masses size mismatch");
    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        if(!finiteVector(positions[i]) || !std::isfinite(masses[i]))
        {
            UniversalError eo("SerialFmmGravityCalculator::solve: non-finite input");
            eo.addEntry("particle", i);
            throw eo;
        }
    }
}
