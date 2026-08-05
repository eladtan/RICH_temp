#ifndef FMM_PATCH_KEY_HPP
#define FMM_PATCH_KEY_HPP

#ifdef RICH_MPI

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <tuple>

// Legacy one-patch-per-rank compatibility mode uses patch ID 1.
static constexpr std::uint64_t FMM_COMPAT_PATCH_ID = 1u;

struct FmmPatchKey
{
    int ownerRank = -1;
    std::uint64_t patchId = 0;

    bool valid() const
    {
        return ownerRank >= 0 && patchId != 0;
    }
};

inline bool operator==(const FmmPatchKey& first, const FmmPatchKey& second)
{
    return first.ownerRank == second.ownerRank &&
           first.patchId == second.patchId;
}

inline bool operator!=(const FmmPatchKey& first, const FmmPatchKey& second)
{
    return !(first == second);
}

inline bool operator<(const FmmPatchKey& first, const FmmPatchKey& second)
{
    return std::tie(first.ownerRank, first.patchId) <
           std::tie(second.ownerRank, second.patchId);
}

struct FmmPatchKeyHash
{
    std::size_t operator()(const FmmPatchKey& key) const noexcept
    {
        const std::uint64_t rankPart =
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.ownerRank));
        return static_cast<std::size_t>(rankPart ^ (key.patchId + 0x9e3779b97f4a7c15ull +
            (rankPart << 6) + (rankPart >> 2)));
    }
};

struct FmmRemoteNodeKey
{
    FmmPatchKey patch;
    std::uint64_t spatialKey = 0;

    bool valid() const
    {
        return patch.valid() && spatialKey != 0;
    }
};

inline bool operator==(const FmmRemoteNodeKey& first,
                       const FmmRemoteNodeKey& second)
{
    return first.patch == second.patch &&
           first.spatialKey == second.spatialKey;
}

inline bool operator!=(const FmmRemoteNodeKey& first,
                       const FmmRemoteNodeKey& second)
{
    return !(first == second);
}

inline bool operator<(const FmmRemoteNodeKey& first,
                      const FmmRemoteNodeKey& second)
{
    return std::tie(first.patch, first.spatialKey) <
           std::tie(second.patch, second.spatialKey);
}

struct FmmRemoteNodeKeyHash
{
    std::size_t operator()(const FmmRemoteNodeKey& key) const noexcept
    {
        FmmPatchKeyHash patchHash;
        return patchHash(key.patch) ^
               (static_cast<std::size_t>(key.spatialKey) +
                0x9e3779b97f4a7c15ull + (patchHash(key.patch) << 6));
    }
};

inline FmmPatchKey fmmCompatPatchKey(int ownerRank)
{
    FmmPatchKey key;
    key.ownerRank = ownerRank;
    key.patchId = FMM_COMPAT_PATCH_ID;
    return key;
}

inline std::string fmmPatchKeyString(const FmmPatchKey& key)
{
    return "rank=" + std::to_string(key.ownerRank) +
           " patch=" + std::to_string(key.patchId);
}

inline std::ostream& operator<<(std::ostream& stream, const FmmPatchKey& key)
{
    return stream << fmmPatchKeyString(key);
}

static_assert(sizeof(FmmPatchKey) == 16,
              "FmmPatchKey must remain wire-compatible");

#endif // RICH_MPI

#endif // FMM_PATCH_KEY_HPP
