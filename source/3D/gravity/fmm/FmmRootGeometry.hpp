#ifndef FMM_ROOT_GEOMETRY_HPP
#define FMM_ROOT_GEOMETRY_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "misc/universal_error.hpp"

struct FmmRootGeometry
{
    Vector3D center;
    double halfSize = 0.0;
    bool active = false;

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

    void validate() const
    {
        if(!active)
            return;
        if(!finiteVector(center) || !(halfSize > 0.0) ||
           !std::isfinite(halfSize) ||
           !std::isfinite(center.x - halfSize) ||
           !std::isfinite(center.x + halfSize) ||
           !std::isfinite(center.y - halfSize) ||
           !std::isfinite(center.y + halfSize) ||
           !std::isfinite(center.z - halfSize) ||
           !std::isfinite(center.z + halfSize))
            throw UniversalError("FmmRootGeometry: invalid active root");
    }

private:
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
