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

FmmDecodedPatchId FmmGlobalDyadicLattice::decodePatchId(std::uint64_t patchId)
{
    if(patchId == 0)
        throw UniversalError("FmmGlobalDyadicLattice: invalid patch id zero");

    FmmDecodedPatchId decoded;
    while(patchId != 1)
    {
        if(patchId == 0 || decoded.level >= FMM_MAX_TREE_DEPTH)
            throw UniversalError("FmmGlobalDyadicLattice: malformed patch id");
        decoded.octants[static_cast<std::size_t>(decoded.level)] =
            static_cast<unsigned>(patchId & 7u);
        ++decoded.level;
        patchId >>= 3;
    }
    std::reverse(decoded.octants.begin(),
                 decoded.octants.begin() + decoded.level);
    return decoded;
}

bool FmmGlobalDyadicLattice::isValidPatchId(std::uint64_t patchId)
{
    if(patchId == 0)
        return false;
    try
    {
        decodePatchId(patchId);
        return true;
    }
    catch(const UniversalError&)
    {
        return false;
    }
}

void FmmGlobalDyadicLattice::validatePatchId(std::uint64_t patchId)
{
    decodePatchId(patchId);
}

int FmmGlobalDyadicLattice::patchLevel(std::uint64_t patchId) const
{
    return decodePatchId(patchId).level;
}

std::uint64_t FmmGlobalDyadicLattice::parentPatchId(std::uint64_t patchId) const
{
    validatePatchId(patchId);
    if(patchId <= 1)
        return 0;
    return patchId >> 3;
}

std::uint64_t FmmGlobalDyadicLattice::childPatchId(std::uint64_t patchId,
                                                   int octant) const
{
    validatePatchId(patchId);
    if(octant < 0 || octant > 7)
        throw UniversalError("FmmGlobalDyadicLattice::childPatchId: invalid octant");
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
    if(!isValidPatchId(parentPatchId) || !isValidPatchId(childPatchId) ||
       octant < 0 || octant > 7)
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

void FmmGlobalDyadicLattice::latticeMetadataForPatch(
    std::uint64_t patchId,
    std::int64_t& latticeCenterX,
    std::int64_t& latticeCenterY,
    std::int64_t& latticeCenterZ,
    std::uint64_t& latticeHalfUnits) const
{
    validatePatchId(patchId);
    const FmmDecodedPatchId decoded = decodePatchId(patchId);

    latticeCenterX = globalRoot_.latticeCenterX;
    latticeCenterY = globalRoot_.latticeCenterY;
    latticeCenterZ = globalRoot_.latticeCenterZ;
    latticeHalfUnits = globalRoot_.latticeHalfUnits;

    for(int i = 0; i < decoded.level; ++i)
    {
        if(latticeHalfUnits == 0)
            throw UniversalError(
                "FmmGlobalDyadicLattice: invalid lattice half units");
        latticeHalfUnits >>= 1;
        const std::int64_t delta = static_cast<std::int64_t>(latticeHalfUnits);
        const unsigned octant = decoded.octants[static_cast<std::size_t>(i)];
        latticeCenterX += (octant & 4u) ? delta : -delta;
        latticeCenterY += (octant & 2u) ? delta : -delta;
        latticeCenterZ += (octant & 1u) ? delta : -delta;
    }
}

void FmmGlobalDyadicLattice::patchCenterAndHalf(std::uint64_t patchId,
                                                Vector3D& center,
                                                double& halfSize) const
{
    std::int64_t latticeCenterX = 0;
    std::int64_t latticeCenterY = 0;
    std::int64_t latticeCenterZ = 0;
    std::uint64_t latticeHalfUnits = 0;
    latticeMetadataForPatch(patchId, latticeCenterX, latticeCenterY,
                          latticeCenterZ, latticeHalfUnits);

    halfSize = static_cast<double>(latticeHalfUnits) * quantum_;
    center = Vector3D(
        globalRoot_.center.x + static_cast<double>(latticeCenterX) * quantum_,
        globalRoot_.center.y + static_cast<double>(latticeCenterY) * quantum_,
        globalRoot_.center.z + static_cast<double>(latticeCenterZ) * quantum_);
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
    validatePatchId(patchId);

    Vector3D center;
    double halfSize = 0.0;
    patchCenterAndHalf(patchId, center, halfSize);

    std::int64_t latticeCenterX = 0;
    std::int64_t latticeCenterY = 0;
    std::int64_t latticeCenterZ = 0;
    std::uint64_t latticeHalfUnits = 0;
    latticeMetadataForPatch(patchId, latticeCenterX, latticeCenterY,
                          latticeCenterZ, latticeHalfUnits);

    FmmRootGeometry result;
    result.center = center;
    result.halfSize = halfSize;
    result.active = true;
    result.latticeId = globalRoot_.latticeId;
    result.latticeAligned = 1;
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
