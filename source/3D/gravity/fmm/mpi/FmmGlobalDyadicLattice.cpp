#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "misc/universal_error.hpp"

namespace
{
constexpr int CommonLatticeBits = 50;

bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}
}

FmmGlobalDyadicLattice FmmGlobalDyadicLattice::fromDomain(
    const Vector3D& domainLower,
    const Vector3D& domainUpper)
{
    FmmGlobalDyadicLattice result;
    result.globalRoot_ = FmmRootGeometry::fromDomain(
        domainLower, domainUpper, true);
    result.quantum_ = std::ldexp(result.globalRoot_.halfSize, -CommonLatticeBits);
    if(!(result.quantum_ > 0.0) || !std::isfinite(result.quantum_))
        throw UniversalError(
            "FmmGlobalDyadicLattice::fromDomain: invalid lattice quantum");
    return result;
}

int FmmGlobalDyadicLattice::patchLevel(std::uint64_t patchId) const
{
    if(patchId == 0)
        return -1;
    int level = 0;
    while(patchId > 1)
    {
        ++level;
        patchId >>= 3;
    }
    return level;
}

std::uint64_t FmmGlobalDyadicLattice::parentPatchId(std::uint64_t patchId) const
{
    if(patchId <= 1)
        return 0;
    return patchId >> 3;
}

std::uint64_t FmmGlobalDyadicLattice::childPatchId(std::uint64_t patchId,
                                                   int octant) const
{
    if(patchId == 0 || octant < 0 || octant > 7)
        throw UniversalError("FmmGlobalDyadicLattice::childPatchId: invalid input");
    if(patchLevel(patchId) >= FMM_MAX_TREE_DEPTH)
        throw UniversalError("FmmGlobalDyadicLattice::childPatchId: maximum patch level exceeded");
    if(patchId > (std::numeric_limits<std::uint64_t>::max() >> 3))
        throw UniversalError("FmmGlobalDyadicLattice::childPatchId: patch id overflow");
    return (patchId << 3) | static_cast<std::uint64_t>(octant);
}

bool FmmGlobalDyadicLattice::validateParentChild(std::uint64_t parentPatchId,
                                                 std::uint64_t childPatchId,
                                                 int octant) const
{
    if(parentPatchId == 0 || childPatchId == 0 || octant < 0 || octant > 7)
        return false;
    return childPatchId == ((parentPatchId << 3) |
                            static_cast<std::uint64_t>(octant));
}

void FmmGlobalDyadicLattice::pointToCellIndices(const Vector3D& point,
                                                int level,
                                                int& ix,
                                                int& iy,
                                                int& iz) const
{
    if(!finiteVector(point))
        throw UniversalError(
            "FmmGlobalDyadicLattice::pointToCellIndices: non-finite point");
    if(level < 0 || level > FMM_MAX_TREE_DEPTH)
        throw UniversalError(
            "FmmGlobalDyadicLattice::pointToCellIndices: invalid level");
    if(level == 0)
    {
        ix = iy = iz = 0;
        return;
    }

    const Vector3D lower = globalRoot_.lower();
    const Vector3D upper = globalRoot_.upper();
    const int cellCount = 1 << level;
    const double width = (2.0 * globalRoot_.halfSize) /
        static_cast<double>(cellCount);

    auto axisIndex = [&](double coordinate,
                         double lowerBound,
                         double upperBound) -> int {
        if(coordinate <= lowerBound)
            return 0;
        if(coordinate >= upperBound)
            return cellCount - 1;
        const double relative = coordinate - lowerBound;
        int index = static_cast<int>(std::floor(relative / width));
        if(index >= cellCount)
            index = cellCount - 1;
        if(index < 0)
            index = 0;
        return index;
    };

    ix = axisIndex(point.x, lower.x, upper.x);
    iy = axisIndex(point.y, lower.y, upper.y);
    iz = axisIndex(point.z, lower.z, upper.z);
}

std::uint64_t FmmGlobalDyadicLattice::patchIdFromCellIndices(int level,
                                                             int ix,
                                                             int iy,
                                                             int iz) const
{
    if(level < 0 || level > FMM_MAX_TREE_DEPTH)
        throw UniversalError(
            "FmmGlobalDyadicLattice::patchIdFromCellIndices: invalid level");
    if(wouldOverflowLevel(level))
        throw UniversalError(
            "FmmGlobalDyadicLattice::patchIdFromCellIndices: patch id overflow");
    if(level == 0)
        return 1;

    const int cellCount = 1 << level;
    if(ix < 0 || iy < 0 || iz < 0 ||
       ix >= cellCount || iy >= cellCount || iz >= cellCount)
        throw UniversalError(
            "FmmGlobalDyadicLattice::patchIdFromCellIndices: cell out of range");

    std::uint64_t patchId = 1;
    for(int bit = level - 1; bit >= 0; --bit)
    {
        const int mask = 1 << bit;
        const int octant = ((ix & mask) ? 4 : 0) |
                           ((iy & mask) ? 2 : 0) |
                           ((iz & mask) ? 1 : 0);
        if(patchId > (std::numeric_limits<std::uint64_t>::max() >> 3))
            throw UniversalError(
                "FmmGlobalDyadicLattice::patchIdFromCellIndices: patch id overflow");
        patchId = (patchId << 3) | static_cast<std::uint64_t>(octant);
    }
    return patchId;
}

std::uint64_t FmmGlobalDyadicLattice::patchIdAtLevel(const Vector3D& point,
                                                     int level) const
{
    int ix = 0;
    int iy = 0;
    int iz = 0;
    pointToCellIndices(point, level, ix, iy, iz);
    return patchIdFromCellIndices(level, ix, iy, iz);
}

void FmmGlobalDyadicLattice::decodeOctantPath(std::uint64_t patchId,
                                              int& level,
                                              int octants[FMM_MAX_TREE_DEPTH]) const
{
    level = 0;
    while(patchId > 1)
    {
        if(level >= FMM_MAX_TREE_DEPTH)
            throw UniversalError(
                "FmmGlobalDyadicLattice: patch id exceeds maximum depth");
        octants[level] = static_cast<int>(patchId & 7u);
        ++level;
        patchId >>= 3;
    }
    std::reverse(octants, octants + level);
}

void FmmGlobalDyadicLattice::patchCenterAndHalf(std::uint64_t patchId,
                                                Vector3D& center,
                                                double& halfSize) const
{
    if(patchId == 0)
        throw UniversalError(
            "FmmGlobalDyadicLattice::patchCenterAndHalf: invalid patch id");

    int level = 0;
    int octants[FMM_MAX_TREE_DEPTH] = {};
    if(patchId != 1)
        decodeOctantPath(patchId, level, octants);

    center = globalRoot_.center;
    halfSize = globalRoot_.halfSize;
    for(int i = 0; i < level; ++i)
    {
        halfSize *= 0.5;
        const double quarter = 0.5 * halfSize;
        const int octant = octants[i];
        center.x += (octant & 4) ? quarter : -quarter;
        center.y += (octant & 2) ? quarter : -quarter;
        center.z += (octant & 1) ? quarter : -quarter;
    }
}

int FmmGlobalDyadicLattice::octantForPoint(std::uint64_t patchId,
                                           const Vector3D& point) const
{
    Vector3D center;
    double halfSize = 0.0;
    patchCenterAndHalf(patchId, center, halfSize);
    return (point.x >= center.x ? 4 : 0) |
           (point.y >= center.y ? 2 : 0) |
           (point.z >= center.z ? 1 : 0);
}

FmmRootGeometry FmmGlobalDyadicLattice::patchRootGeometry(
    std::uint64_t patchId) const
{
    const int level = patchLevel(patchId);
    if(level < 0 || level > FMM_MAX_TREE_DEPTH)
        throw UniversalError(
            "FmmGlobalDyadicLattice::patchRootGeometry: invalid patch id");

    Vector3D center;
    double halfSize = 0.0;
    patchCenterAndHalf(patchId, center, halfSize);

    FmmRootGeometry result;
    result.center = center;
    result.halfSize = halfSize;
    result.active = true;
    result.latticeId = globalRoot_.latticeId;
    result.latticeAligned = 1;

    std::int64_t latticeCenterX = globalRoot_.latticeCenterX;
    std::int64_t latticeCenterY = globalRoot_.latticeCenterY;
    std::int64_t latticeCenterZ = globalRoot_.latticeCenterZ;
    std::uint64_t latticeHalfUnits = globalRoot_.latticeHalfUnits;

    if(patchId != 1)
    {
        int pathLevel = 0;
        int octants[FMM_MAX_TREE_DEPTH] = {};
        decodeOctantPath(patchId, pathLevel, octants);
        for(int i = 0; i < pathLevel; ++i)
        {
            if(latticeHalfUnits == 0)
                throw UniversalError(
                    "FmmGlobalDyadicLattice::patchRootGeometry: invalid lattice half units");
            latticeHalfUnits >>= 1;
            const std::int64_t delta = static_cast<std::int64_t>(latticeHalfUnits);
            const int octant = octants[i];
            latticeCenterX += (octant & 4) ? delta : -delta;
            latticeCenterY += (octant & 2) ? delta : -delta;
            latticeCenterZ += (octant & 1) ? delta : -delta;
        }
    }

    result.latticeCenterX = latticeCenterX;
    result.latticeCenterY = latticeCenterY;
    result.latticeCenterZ = latticeCenterZ;
    result.latticeHalfUnits = latticeHalfUnits;
    result.validate();
    return result;
}

bool FmmGlobalDyadicLattice::wouldOverflowLevel(int level) const
{
    if(level < 0 || level > FMM_MAX_TREE_DEPTH)
        return true;
    return level > 0 && (3 * level + 1) > 64;
}

bool FmmGlobalDyadicLattice::contains(const Vector3D& point,
                                      std::uint64_t patchId) const
{
    Vector3D center;
    double halfSize = 0.0;
    patchCenterAndHalf(patchId, center, halfSize);
    return point.x >= center.x - halfSize && point.x <= center.x + halfSize &&
           point.y >= center.y - halfSize && point.y <= center.y + halfSize &&
           point.z >= center.z - halfSize && point.z <= center.z + halfSize;
}
