#ifndef FMM_M2L_OPERATOR_CACHE_HPP
#define FMM_M2L_OPERATOR_CACHE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include "3D/gravity/fmm/FmmKernels.hpp"
#include "misc/universal_error.hpp"

// A byte-bounded exact-displacement M2L cache.  Exact keys preserve the
// existing numerical result.  Once the cache reaches its budget, additional
// operators are computed into one reusable scratch vector instead of growing
// persistent storage without bound.
class FmmM2LOperatorCache
{
public:
    FmmM2LOperatorCache():
        maxEntries_(0), configuredMaxEntries_(0), termCount_(0),
        budgetBytes_(0), hits_(0), misses_(0), bypasses_(0) {}

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
    }

    const std::vector<double>& get(
        const Vector3D& displacement,
        const FmmTaylorExpansion& layout,
        std::vector<double>& derivativeScratch,
        std::vector<double>& uncachedOperator)
    {
        if(layout.m2lTerms().size() != termCount_)
            throw UniversalError(
                "FmmM2LOperatorCache::get: cache/layout size mismatch");

        const Key key = makeKey(displacement);
        const auto found = entries_.find(key);
        if(found != entries_.end())
        {
            ++hits_;
            return found->second;
        }

        ++misses_;
        if(entries_.size() < maxEntries_)
        {
            auto inserted = entries_.emplace(key, std::vector<double>());
            if(!inserted.second)
                throw UniversalError(
                    "FmmM2LOperatorCache::get: duplicate insertion");
            FmmKernels::computeM2LOperator(
                displacement, layout, derivativeScratch, inserted.first->second);

            if(bytesOwned() <= budgetBytes_)
                return inserted.first->second;

            // An implementation-specific hash/node allocation exceeded the
            // conservative estimate.  Preserve the just-computed operator for
            // this translation, remove it from persistent storage, and disable
            // further insertions if the retained bucket array itself is too big.
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
            return uncachedOperator;
        }

        FmmKernels::computeM2LOperator(
            displacement, layout, derivativeScratch, uncachedOperator);
        ++bypasses_;
        return uncachedOperator;
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

private:
    struct Key
    {
        std::uint64_t x = 0;
        std::uint64_t y = 0;
        std::uint64_t z = 0;

        bool operator==(const Key& other) const
        {
            return x == other.x && y == other.y && z == other.z;
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
            return result;
        }
    };

    using EntryMap = std::unordered_map<Key, std::vector<double>, KeyHash>;

    static Key makeKey(const Vector3D& displacement)
    {
        Key key;
        std::memcpy(&key.x, &displacement.x, sizeof(double));
        std::memcpy(&key.y, &displacement.y, sizeof(double));
        std::memcpy(&key.z, &displacement.z, sizeof(double));
        return key;
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
};

#endif // FMM_M2L_OPERATOR_CACHE_HPP
