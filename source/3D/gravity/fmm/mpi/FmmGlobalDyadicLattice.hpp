#ifndef FMM_GLOBAL_DYADIC_LATTICE_HPP
#define FMM_GLOBAL_DYADIC_LATTICE_HPP

#include <array>
#include <cstdint>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/FmmConfig.hpp"
#include "3D/gravity/fmm/FmmRootGeometry.hpp"

struct FmmDecodedPatchId
{
    int level = 0;
    std::array<unsigned, FMM_MAX_TREE_DEPTH> octants{};
};

// Canonical global dyadic lattice shared by every rank.  Patch IDs encode an
// octant path from the domain root with root path key 1.
class FmmGlobalDyadicLattice
{
public:
    static FmmGlobalDyadicLattice fromDomain(const Vector3D& domainLower,
                                             const Vector3D& domainUpper);

    const FmmRootGeometry& globalRoot() const { return globalRoot_; }
    double quantum() const { return quantum_; }

    static FmmDecodedPatchId decodePatchId(std::uint64_t patchId);
    static bool isValidPatchId(std::uint64_t patchId);
    static void validatePatchId(std::uint64_t patchId);

    int patchLevel(std::uint64_t patchId) const;
    std::uint64_t parentPatchId(std::uint64_t patchId) const;
    std::uint64_t childPatchId(std::uint64_t patchId, int octant) const;
    bool validateParentChild(std::uint64_t parentPatchId,
                             std::uint64_t childPatchId,
                             int octant) const;

    void pointToCellIndices(const Vector3D& point,
                            int level,
                            int& ix,
                            int& iy,
                            int& iz) const;

    std::uint64_t patchIdAtLevel(const Vector3D& point, int level) const;
    std::uint64_t patchIdFromCellIndices(int level,
                                         int ix,
                                         int iy,
                                         int iz) const;

    int octantForPoint(std::uint64_t patchId, const Vector3D& point) const;
    FmmRootGeometry patchRootGeometry(std::uint64_t patchId) const;

    bool wouldOverflowLevel(int level) const;
    bool contains(const Vector3D& point, std::uint64_t patchId) const;

private:
    void patchCenterAndHalf(std::uint64_t patchId,
                            Vector3D& center,
                            double& halfSize) const;
    void latticeMetadataForPatch(std::uint64_t patchId,
                                 std::int64_t& latticeCenterX,
                                 std::int64_t& latticeCenterY,
                                 std::int64_t& latticeCenterZ,
                                 std::uint64_t& latticeHalfUnits) const;

    FmmRootGeometry globalRoot_;
    double quantum_ = 0.0;
};

#endif // FMM_GLOBAL_DYADIC_LATTICE_HPP
