#ifndef FMM_ROOT_GEOMETRY_HPP
#define FMM_ROOT_GEOMETRY_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/FmmConfig.hpp"
#include "misc/universal_error.hpp"

struct FmmRootGeometry
{
    Vector3D center;
    double halfSize = 0.0;
    bool active = false;

    // A lattice-aligned cube uses one common binary quantum per global domain:
    //   center = globalCenter + latticeCenter * quantum.
    // latticeHalfUnits is an integer multiple of quantum and is divisible by
    // 2^maxDepth, so every descendant center remains exactly representable in
    // the same integer coordinate system without enlarging the local root.
    std::uint64_t latticeId = 0;
    std::int64_t latticeCenterX = 0;
    std::int64_t latticeCenterY = 0;
    std::int64_t latticeCenterZ = 0;
    std::uint64_t latticeHalfUnits = 0;
    int latticeAligned = 0;

    Vector3D lower() const
    {
        return Vector3D(center.x - halfSize, center.y - halfSize, center.z - halfSize);
    }

    Vector3D upper() const
    {
        return Vector3D(center.x + halfSize, center.y + halfSize, center.z + halfSize);
    }

    bool contains(const Vector3D& point) const
    {
        return active &&
            point.x >= center.x - halfSize && point.x <= center.x + halfSize &&
            point.y >= center.y - halfSize && point.y <= center.y + halfSize &&
            point.z >= center.z - halfSize && point.z <= center.z + halfSize;
    }

    static FmmRootGeometry fromDomain(const Vector3D& lower,
                                      const Vector3D& upper,
                                      bool addLegacyPadding)
    {
        const Vector3D extent = checkedExtent(lower, upper,
            "FmmRootGeometry::fromDomain: domain bounds must be finite and strictly ordered");
        double half = 0.5 * std::max(extent.x, std::max(extent.y, extent.z));
        if(addLegacyPadding)
            half = checkedPaddedHalfSize(half, 1.0);
        FmmRootGeometry result;
        result.center = checkedLegacyCenter(lower, upper, extent,
            "FmmRootGeometry::fromDomain: domain center overflow");
        result.halfSize = half;
        result.active = true;
        result.latticeId = latticeHash(result.center, result.halfSize);
        result.latticeHalfUnits = std::uint64_t(1) << CommonLatticeBits;
        result.latticeAligned = 1;
        result.validate();
        return result;
    }

    static FmmRootGeometry containingPoints(const std::vector<Vector3D>& points,
                                            const Vector3D& domainLower,
                                            const Vector3D& domainUpper,
                                            double slackFactor)
    {
        const Vector3D domainExtent = checkedExtent(domainLower, domainUpper,
            "FmmRootGeometry::containingPoints: invalid domain");
        if(points.empty())
            return FmmRootGeometry();
        if(!(slackFactor >= 1.0) || !std::isfinite(slackFactor))
            throw UniversalError("FmmRootGeometry::containingPoints: invalid slack factor");

        Vector3D lower = points.front();
        Vector3D upper = points.front();
        if(!finiteVector(lower))
            throw UniversalError("FmmRootGeometry::containingPoints: non-finite point");
        for(const Vector3D& point : points)
        {
            if(!finiteVector(point))
                throw UniversalError("FmmRootGeometry::containingPoints: non-finite point");
            lower.x = std::min(lower.x, point.x);
            lower.y = std::min(lower.y, point.y);
            lower.z = std::min(lower.z, point.z);
            upper.x = std::max(upper.x, point.x);
            upper.y = std::max(upper.y, point.y);
            upper.z = std::max(upper.z, point.z);
        }

        const Vector3D extent = checkedNonnegativeExtent(lower, upper,
            "FmmRootGeometry::containingPoints: point extent overflow");
        const double domainScale = std::max(domainExtent.x,
            std::max(domainExtent.y, domainExtent.z));
        const double minimumHalf = 64.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, domainScale);
        const double pointScale = 0.5 *
            std::max(extent.x, std::max(extent.y, extent.z));
        if(pointScale > std::numeric_limits<double>::max() / slackFactor)
            throw UniversalError("FmmRootGeometry::containingPoints: slack expansion overflow");

        FmmRootGeometry result;
        result.center = checkedCenter(lower, extent,
            "FmmRootGeometry::containingPoints: point center overflow");
        result.halfSize = std::max(minimumHalf, slackFactor * pointScale);
        result.halfSize = checkedPaddedHalfSize(result.halfSize, 1.0);
        result.active = true;
        result.validate();
        return result;
    }

    static FmmRootGeometry containingPointsOnDyadicLattice(
        const std::vector<Vector3D>& points,
        const Vector3D& domainLower,
        const Vector3D& domainUpper,
        double slackFactor,
        int maxDepth)
    {
        if(points.empty())
            return FmmRootGeometry();
        if(maxDepth < 0 || maxDepth > FMM_MAX_TREE_DEPTH ||
           maxDepth >= CommonLatticeBits)
            throw UniversalError(
                "FmmRootGeometry::containingPointsOnDyadicLattice: invalid maximum depth");

        const FmmRootGeometry desired = containingPoints(
            points, domainLower, domainUpper, slackFactor);
        const FmmRootGeometry global = fromDomain(
            domainLower, domainUpper, true);
        const double quantum = std::ldexp(global.halfSize, -CommonLatticeBits);
        if(!(quantum > 0.0) || !std::isfinite(quantum))
            throw UniversalError(
                "FmmRootGeometry::containingPointsOnDyadicLattice: invalid lattice quantum");

        std::int64_t ix = 0;
        std::int64_t iy = 0;
        std::int64_t iz = 0;
        const Vector3D snappedCenter(
            snapCoordinate(desired.center.x, global.center.x, quantum, ix),
            snapCoordinate(desired.center.y, global.center.y, quantum, iy),
            snapCoordinate(desired.center.z, global.center.z, quantum, iz));

        const Vector3D desiredLower = desired.lower();
        const Vector3D desiredUpper = desired.upper();
        long double requiredHalf = 0.0L;
        for(int axis = 0; axis < 3; ++axis)
        {
            const double centerValue = axis == 0 ? snappedCenter.x :
                                       (axis == 1 ? snappedCenter.y : snappedCenter.z);
            const double lowerValue = axis == 0 ? desiredLower.x :
                                      (axis == 1 ? desiredLower.y : desiredLower.z);
            const double upperValue = axis == 0 ? desiredUpper.x :
                                      (axis == 1 ? desiredUpper.y : desiredUpper.z);
            requiredHalf = std::max(requiredHalf,
                std::max(std::abs(static_cast<long double>(centerValue) - lowerValue),
                         std::abs(static_cast<long double>(upperValue) - centerValue)));
        }

        const std::uint64_t depthAlignment = std::uint64_t(1) << maxDepth;
        const long double rawUnits = std::ceil(requiredHalf /
            static_cast<long double>(quantum));
        if(!(rawUnits > 0.0L) ||
           rawUnits > static_cast<long double>(
               std::numeric_limits<std::uint64_t>::max() - depthAlignment))
            throw UniversalError(
                "FmmRootGeometry::containingPointsOnDyadicLattice: half-size overflow");
        const std::uint64_t units = static_cast<std::uint64_t>(rawUnits);
        std::uint64_t roundedUnits =
            ((units + depthAlignment - 1) / depthAlignment) * depthAlignment;
        double half = static_cast<double>(
            static_cast<long double>(roundedUnits) * quantum);

        FmmRootGeometry result;
        result.center = snappedCenter;
        result.halfSize = half;
        result.active = true;
        result.latticeId = global.latticeId;
        result.latticeCenterX = ix;
        result.latticeCenterY = iy;
        result.latticeCenterZ = iz;
        result.latticeHalfUnits = roundedUnits;
        result.latticeAligned = 1;
        if(!result.contains(desiredLower) || !result.contains(desiredUpper))
        {
            if(roundedUnits >
               std::numeric_limits<std::uint64_t>::max() - depthAlignment)
                throw UniversalError(
                    "FmmRootGeometry::containingPointsOnDyadicLattice: containment growth overflow");
            roundedUnits += depthAlignment;
            half = static_cast<double>(
                static_cast<long double>(roundedUnits) * quantum);
            result.halfSize = half;
            result.latticeHalfUnits = roundedUnits;
        }
        if(!result.contains(desiredLower) || !result.contains(desiredUpper))
            throw UniversalError(
                "FmmRootGeometry::containingPointsOnDyadicLattice: rounded root lost containment");
        result.validate();
        return result;
    }

    void validate() const
    {
        if(!active)
        {
            if(latticeAligned != 0 || latticeId != 0 ||
               latticeCenterX != 0 || latticeCenterY != 0 ||
               latticeCenterZ != 0 || latticeHalfUnits != 0)
                throw UniversalError(
                    "FmmRootGeometry: inactive root has lattice metadata");
            return;
        }
        if(!finiteVector(center) || !(halfSize > 0.0) ||
           !std::isfinite(halfSize) ||
           !std::isfinite(center.x - halfSize) ||
           !std::isfinite(center.x + halfSize) ||
           !std::isfinite(center.y - halfSize) ||
           !std::isfinite(center.y + halfSize) ||
           !std::isfinite(center.z - halfSize) ||
           !std::isfinite(center.z + halfSize))
            throw UniversalError("FmmRootGeometry: invalid active root");
        if(latticeAligned != 0 && latticeAligned != 1)
            throw UniversalError("FmmRootGeometry: invalid lattice flag");
        if(latticeAligned != 0 &&
           (latticeId == 0 || latticeHalfUnits == 0 ||
            latticeHalfUnits > static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())))
            throw UniversalError("FmmRootGeometry: invalid lattice metadata");
        if(latticeAligned == 0 &&
           (latticeId != 0 || latticeCenterX != 0 ||
            latticeCenterY != 0 || latticeCenterZ != 0 ||
            latticeHalfUnits != 0))
            throw UniversalError("FmmRootGeometry: unaligned root has lattice metadata");
    }

private:
    static constexpr int CommonLatticeBits = 50;

    static std::uint64_t latticeHash(const Vector3D& center, double halfSize)
    {
        const double values[4] = {center.x, center.y, center.z, halfSize};
        std::uint64_t hash = 1469598103934665603ull;
        for(double value : values)
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(double));
            for(int byte = 0; byte < 8; ++byte)
            {
                hash ^= (bits >> (8 * byte)) & 0xffu;
                hash *= 1099511628211ull;
            }
        }
        hash ^= static_cast<std::uint64_t>(CommonLatticeBits);
        hash *= 1099511628211ull;
        return hash == 0 ? 1 : hash;
    }

    static double snapCoordinate(double value,
                                 double globalCenter,
                                 double quantum,
                                 std::int64_t& index)
    {
        const long double scaled =
            (static_cast<long double>(value) - globalCenter) /
            static_cast<long double>(quantum);
        if(!std::isfinite(static_cast<double>(scaled)) ||
           scaled < static_cast<long double>(
               std::numeric_limits<std::int64_t>::min() + 1) ||
           scaled > static_cast<long double>(
               std::numeric_limits<std::int64_t>::max() - 1))
            throw UniversalError(
                "FmmRootGeometry::containingPointsOnDyadicLattice: center overflow");
        index = static_cast<std::int64_t>(std::llround(scaled));
        const long double snapped = static_cast<long double>(globalCenter) +
            static_cast<long double>(index) * quantum;
        const double result = static_cast<double>(snapped);
        if(!std::isfinite(result))
            throw UniversalError(
                "FmmRootGeometry::containingPointsOnDyadicLattice: snapped center overflow");
        return result;
    }

    static bool finiteVector(const Vector3D& value)
    {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
               std::isfinite(value.z);
    }

    static Vector3D checkedExtent(const Vector3D& lower,
                                  const Vector3D& upper,
                                  const char* message)
    {
        if(!finiteVector(lower) || !finiteVector(upper) ||
           !(lower.x < upper.x) || !(lower.y < upper.y) ||
           !(lower.z < upper.z))
            throw UniversalError(message);
        const Vector3D extent(upper.x - lower.x, upper.y - lower.y,
                              upper.z - lower.z);
        if(!finiteVector(extent) || !(extent.x > 0.0) ||
           !(extent.y > 0.0) || !(extent.z > 0.0))
            throw UniversalError(message);
        return extent;
    }

    static Vector3D checkedNonnegativeExtent(const Vector3D& lower,
                                             const Vector3D& upper,
                                             const char* message)
    {
        const Vector3D extent(upper.x - lower.x, upper.y - lower.y,
                              upper.z - lower.z);
        if(!finiteVector(extent) || extent.x < 0.0 ||
           extent.y < 0.0 || extent.z < 0.0)
            throw UniversalError(message);
        return extent;
    }

    static Vector3D checkedCenter(const Vector3D& lower,
                                  const Vector3D& extent,
                                  const char* message)
    {
        const Vector3D result(lower.x + 0.5 * extent.x,
                              lower.y + 0.5 * extent.y,
                              lower.z + 0.5 * extent.z);
        if(!finiteVector(result))
            throw UniversalError(message);
        return result;
    }

    static Vector3D checkedLegacyCenter(const Vector3D& lower,
                                        const Vector3D& upper,
                                        const Vector3D& extent,
                                        const char* message)
    {
        const Vector3D sum(lower.x + upper.x, lower.y + upper.y,
                           lower.z + upper.z);
        if(finiteVector(sum))
            return Vector3D(0.5 * sum.x, 0.5 * sum.y, 0.5 * sum.z);
        return checkedCenter(lower, extent, message);
    }

    static double checkedPaddedHalfSize(double halfSize, double scaleFloor)
    {
        if(!(halfSize > 0.0) || !std::isfinite(halfSize))
            throw UniversalError("FmmRootGeometry: invalid half size");
        const double padding = 16.0 * std::numeric_limits<double>::epsilon() *
            std::max(scaleFloor, halfSize);
        if(padding > std::numeric_limits<double>::max() - halfSize)
            throw UniversalError("FmmRootGeometry: padding overflow");
        return halfSize + padding;
    }
};

#endif // FMM_ROOT_GEOMETRY_HPP
