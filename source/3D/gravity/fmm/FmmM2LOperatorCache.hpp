#ifndef FMM_M2L_OPERATOR_CACHE_HPP
#define FMM_M2L_OPERATOR_CACHE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "3D/gravity/fmm/FmmKernels.hpp"
#include "misc/universal_error.hpp"

// A byte-bounded cache of scale-free M2L operators.  Nodes carrying the same
// lattice id use an exact primitive integer direction key, so translations at
// different octree levels share one operator.  Other geometries use exact bits
// of a max-norm-normalized direction.  The omitted physical length is restored
// analytically by FmmKernels::translateM2L.
class FmmM2LOperatorCache
{
public:
    struct Lookup
    {
        const std::vector<double>* coefficients = nullptr;
        double inverseScale = 1.0;
        bool integerKey = false;
    };

    FmmM2LOperatorCache():
        maxEntries_(0), configuredMaxEntries_(0), termCount_(0),
        budgetBytes_(0), hits_(0), misses_(0), bypasses_(0),
        integerKeyHits_(0), integerKeyMisses_(0) {}

    void clear()
    {
        std::unordered_map<Key, std::vector<double>, KeyHash>().swap(entries_);
        maxEntries_ = 0;
        configuredMaxEntries_ = 0;
        termCount_ = 0;
        budgetBytes_ = 0;
        beginPhase();
    }

    void configure(std::size_t maxBytes,
                   std::size_t termCount,
                   std::size_t entryHint)
    {
        if(termCount == 0)
            throw UniversalError(
                "FmmM2LOperatorCache::configure: zero operator size");

        const std::size_t estimatedEntryBytes = estimatedBytesPerEntry(termCount);
        std::size_t maxEntries = 0;
        if(estimatedEntryBytes != 0 &&
           estimatedEntryBytes != std::numeric_limits<std::size_t>::max())
        {
            maxEntries = std::min(entryHint, maxBytes / estimatedEntryBytes);
        }

        if(termCount_ == termCount && configuredMaxEntries_ == maxEntries &&
           budgetBytes_ == maxBytes)
            return;

        std::unordered_map<Key, std::vector<double>, KeyHash>().swap(entries_);
        maxEntries_ = maxEntries;
        configuredMaxEntries_ = maxEntries;
        termCount_ = termCount;
        budgetBytes_ = maxBytes;
        if(maxEntries_ != 0)
        {
            entries_.reserve(maxEntries_);
            if(bytesOwned() > budgetBytes_)
            {
                std::unordered_map<Key, std::vector<double>, KeyHash>().swap(entries_);
                maxEntries_ = 0;
            }
        }
    }

    void beginPhase()
    {
        hits_ = 0;
        misses_ = 0;
        bypasses_ = 0;
        integerKeyHits_ = 0;
        integerKeyMisses_ = 0;
    }

    Lookup get(const FmmNode& source,
               const FmmNode& target,
               const FmmTaylorExpansion& layout,
               std::vector<double>& derivativeScratch,
               std::vector<double>& uncachedOperator)
    {
        if(layout.m2lTerms().size() != termCount_)
            throw UniversalError(
                "FmmM2LOperatorCache::get: cache/layout size mismatch");

        const CanonicalGeometry geometry = canonicalGeometry(source, target);
        const auto found = entries_.find(geometry.key);
        if(found != entries_.end())
        {
            ++hits_;
            if(geometry.integerKey)
                ++integerKeyHits_;
            return Lookup{&found->second, geometry.inverseScale,
                          geometry.integerKey};
        }

        ++misses_;
        if(geometry.integerKey)
            ++integerKeyMisses_;
        if(entries_.size() < maxEntries_)
        {
            auto inserted = entries_.emplace(geometry.key, std::vector<double>());
            if(!inserted.second)
                throw UniversalError(
                    "FmmM2LOperatorCache::get: duplicate insertion");
            FmmKernels::computeM2LOperator(
                geometry.direction, layout, derivativeScratch,
                inserted.first->second);

            if(bytesOwned() <= budgetBytes_)
                return Lookup{&inserted.first->second, geometry.inverseScale,
                              geometry.integerKey};

            // Preserve the just-computed operator for this translation, remove
            // it from persistent storage, and stop growing if allocator/hash
            // overhead has exhausted the byte budget.
            uncachedOperator.swap(inserted.first->second);
            entries_.erase(inserted.first);
            if(bytesOwned() > budgetBytes_)
            {
                std::unordered_map<Key, std::vector<double>, KeyHash>().swap(entries_);
                maxEntries_ = 0;
            }
            else
            {
                maxEntries_ = entries_.size();
            }
            ++bypasses_;
            return Lookup{&uncachedOperator, geometry.inverseScale,
                          geometry.integerKey};
        }

        FmmKernels::computeM2LOperator(
            geometry.direction, layout, derivativeScratch, uncachedOperator);
        ++bypasses_;
        return Lookup{&uncachedOperator, geometry.inverseScale,
                      geometry.integerKey};
    }

    std::size_t bytesOwned() const
    {
        const std::size_t mapEntry =
            sizeof(typename EntryMap::value_type) + 2 * sizeof(void*);
        std::size_t result = entries_.empty() && entries_.bucket_count() <= 1 ?
            0 : saturatingMultiply(entries_.bucket_count(), sizeof(void*));
        result = saturatingAdd(result,
            saturatingMultiply(entries_.size(), mapEntry));
        for(const auto& entry : entries_)
        {
            result = saturatingAdd(result,
                saturatingMultiply(entry.second.capacity(), sizeof(double)));
        }
        return result;
    }

    std::size_t entries() const { return entries_.size(); }
    std::size_t maxEntries() const { return maxEntries_; }
    std::size_t budgetBytes() const { return budgetBytes_; }
    std::uint64_t hits() const { return hits_; }
    std::uint64_t misses() const { return misses_; }
    std::uint64_t bypasses() const { return bypasses_; }
    std::uint64_t integerKeyHits() const { return integerKeyHits_; }
    std::uint64_t integerKeyMisses() const { return integerKeyMisses_; }

private:
    enum : std::uint64_t
    {
        ExactDirectionKey = 0,
        IntegerDirectionKey = 1
    };

    struct Key
    {
        std::uint64_t x = 0;
        std::uint64_t y = 0;
        std::uint64_t z = 0;
        std::uint64_t kind = ExactDirectionKey;

        bool operator==(const Key& other) const
        {
            return x == other.x && y == other.y && z == other.z &&
                   kind == other.kind;
        }
    };

    struct KeyHash
    {
        std::size_t operator()(const Key& key) const
        {
            std::size_t result = std::hash<std::uint64_t>()(key.x);
            result ^= std::hash<std::uint64_t>()(key.y) + 0x9e3779b9u +
                (result << 6u) + (result >> 2u);
            result ^= std::hash<std::uint64_t>()(key.z) + 0x9e3779b9u +
                (result << 6u) + (result >> 2u);
            result ^= std::hash<std::uint64_t>()(key.kind) + 0x9e3779b9u +
                (result << 6u) + (result >> 2u);
            return result;
        }
    };

    struct CanonicalGeometry
    {
        Key key;
        Vector3D direction;
        double inverseScale = 1.0;
        bool integerKey = false;
    };

    using EntryMap = std::unordered_map<Key, std::vector<double>, KeyHash>;

    static std::uint64_t unsignedMagnitude(std::int64_t value)
    {
        return value < 0 ?
            static_cast<std::uint64_t>(-(value + 1)) + 1u :
            static_cast<std::uint64_t>(value);
    }

    static bool checkedDifference(std::int64_t target,
                                  std::int64_t source,
                                  std::int64_t& result)
    {
        if((source > 0 && target <
            std::numeric_limits<std::int64_t>::min() + source) ||
           (source < 0 && target >
            std::numeric_limits<std::int64_t>::max() + source))
            return false;
        result = target - source;
        return true;
    }

    static bool latticeDifference(const FmmNode& source,
                                  const FmmNode& target,
                                  std::int64_t& dx,
                                  std::int64_t& dy,
                                  std::int64_t& dz)
    {
        if(source.latticeAligned == 0 || target.latticeAligned == 0 ||
           source.latticeId == 0 || source.latticeId != target.latticeId)
            return false;
        return checkedDifference(target.latticeCenterX,
                                 source.latticeCenterX, dx) &&
               checkedDifference(target.latticeCenterY,
                                 source.latticeCenterY, dy) &&
               checkedDifference(target.latticeCenterZ,
                                 source.latticeCenterZ, dz);
    }

    static CanonicalGeometry canonicalGeometry(const FmmNode& source,
                                                const FmmNode& target)
    {
        const Vector3D displacement = target.center - source.center;
        std::int64_t dx = 0;
        std::int64_t dy = 0;
        std::int64_t dz = 0;
        if(latticeDifference(source, target, dx, dy, dz))
        {
            const std::uint64_t gcd = std::gcd(unsignedMagnitude(dx),
                std::gcd(unsignedMagnitude(dy), unsignedMagnitude(dz)));
            if(gcd == 0)
                throw UniversalError(
                    "FmmM2LOperatorCache::get: coincident lattice centers");
            if(gcd <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
            {
                const std::int64_t cx = dx / static_cast<std::int64_t>(gcd);
                const std::int64_t cy = dy / static_cast<std::int64_t>(gcd);
                const std::int64_t cz = dz / static_cast<std::int64_t>(gcd);
                const std::uint64_t primitiveMax = std::max(
                    unsignedMagnitude(cx),
                    std::max(unsignedMagnitude(cy), unsignedMagnitude(cz)));
                const double displacementMax = std::max(std::abs(displacement.x),
                    std::max(std::abs(displacement.y), std::abs(displacement.z)));
                const double physicalScale = displacementMax /
                    static_cast<double>(primitiveMax);
                if(!(physicalScale > 0.0) || !std::isfinite(physicalScale))
                    throw UniversalError(
                        "FmmM2LOperatorCache::get: invalid lattice scale");

                const double tolerance = 128.0 *
                    std::numeric_limits<double>::epsilon() *
                    std::max(1.0, displacementMax);
                const bool consistent =
                    std::abs(displacement.x - physicalScale *
                        static_cast<double>(cx)) <= tolerance &&
                    std::abs(displacement.y - physicalScale *
                        static_cast<double>(cy)) <= tolerance &&
                    std::abs(displacement.z - physicalScale *
                        static_cast<double>(cz)) <= tolerance;
                if(!consistent)
                    return canonicalGeometryWithoutLattice(displacement);
                CanonicalGeometry result;
                result.key.x = static_cast<std::uint64_t>(cx);
                result.key.y = static_cast<std::uint64_t>(cy);
                result.key.z = static_cast<std::uint64_t>(cz);
                result.key.kind = IntegerDirectionKey;
                result.direction = Vector3D(static_cast<double>(cx),
                                            static_cast<double>(cy),
                                            static_cast<double>(cz));
                result.inverseScale = 1.0 / physicalScale;
                result.integerKey = true;
                return result;
            }
        }

        return canonicalGeometryWithoutLattice(displacement);
    }

    static CanonicalGeometry canonicalGeometryWithoutLattice(
        const Vector3D& displacement)
    {
        const double scale = std::max(std::abs(displacement.x),
            std::max(std::abs(displacement.y), std::abs(displacement.z)));
        if(!(scale > 0.0) || !std::isfinite(scale))
            throw UniversalError(
                "FmmM2LOperatorCache::get: invalid center separation");
        CanonicalGeometry result;
        result.direction = Vector3D(displacement.x / scale,
                                    displacement.y / scale,
                                    displacement.z / scale);
        if(result.direction.x == 0.0) result.direction.x = 0.0;
        if(result.direction.y == 0.0) result.direction.y = 0.0;
        if(result.direction.z == 0.0) result.direction.z = 0.0;
        std::memcpy(&result.key.x, &result.direction.x, sizeof(double));
        std::memcpy(&result.key.y, &result.direction.y, sizeof(double));
        std::memcpy(&result.key.z, &result.direction.z, sizeof(double));
        result.key.kind = ExactDirectionKey;
        result.inverseScale = 1.0 / scale;
        result.integerKey = false;
        return result;
    }

    static std::size_t saturatingAdd(std::size_t first, std::size_t second)
    {
        return second > std::numeric_limits<std::size_t>::max() - first ?
            std::numeric_limits<std::size_t>::max() : first + second;
    }

    static std::size_t saturatingMultiply(std::size_t first, std::size_t second)
    {
        return first != 0 &&
               second > std::numeric_limits<std::size_t>::max() / first ?
            std::numeric_limits<std::size_t>::max() : first * second;
    }

    static std::size_t estimatedBytesPerEntry(std::size_t termCount)
    {
        const std::size_t mapEntry =
            sizeof(typename EntryMap::value_type) + 4 * sizeof(void*);
        return saturatingAdd(mapEntry,
            saturatingMultiply(termCount, sizeof(double)));
    }

    EntryMap entries_;
    std::size_t maxEntries_;
    std::size_t configuredMaxEntries_;
    std::size_t termCount_;
    std::size_t budgetBytes_;
    std::uint64_t hits_;
    std::uint64_t misses_;
    std::uint64_t bypasses_;
    std::uint64_t integerKeyHits_;
    std::uint64_t integerKeyMisses_;
};

#endif // FMM_M2L_OPERATOR_CACHE_HPP
