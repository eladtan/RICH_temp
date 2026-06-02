#ifndef FACE_EXIT_INFO_HPP
#define FACE_EXIT_INFO_HPP

#include <cstddef>
#include <limits>

enum class FaceExitKind
{
    LocalRealCell,
    PhysicalVacuumBoundary,
    RemoteRankBoundary,
    UnsupportedBoundary,
    InvalidOrUnknown
};

struct FaceExitInfo
{
    FaceExitKind kind = FaceExitKind::InvalidOrUnknown;
    size_t nextLocalCell = std::numeric_limits<size_t>::max();
    int remoteRank = -1;
    size_t remoteLocalIndex = std::numeric_limits<size_t>::max();
    // Diagnostic fields populated on UnsupportedBoundary for debugging
    size_t faceId = std::numeric_limits<size_t>::max();
    size_t fromCellId = std::numeric_limits<size_t>::max();
    size_t nextCellId = std::numeric_limits<size_t>::max();
};

inline const char* faceExitKindName(FaceExitKind k)
{
    switch (k)
    {
        case FaceExitKind::LocalRealCell:          return "LocalRealCell";
        case FaceExitKind::PhysicalVacuumBoundary: return "PhysicalVacuumBoundary";
        case FaceExitKind::RemoteRankBoundary:     return "RemoteRankBoundary";
        case FaceExitKind::UnsupportedBoundary:    return "UnsupportedBoundary";
        case FaceExitKind::InvalidOrUnknown:       return "InvalidOrUnknown";
        default:                                   return "Unknown";
    }
}

#endif // FACE_EXIT_INFO_HPP
