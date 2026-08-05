#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include "source/3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
#include "source/3D/gravity/fmm/mpi/FmmPatchKey.hpp"

namespace
{
bool testValidity()
{
    FmmPatchKey invalid;
    if(invalid.valid())
        return false;
    FmmPatchKey key{-1, 1};
    if(key.valid())
        return false;
    key.ownerRank = 0;
    if(!key.valid())
        return false;
    key.patchId = 0;
    if(key.valid())
        return false;
    key.patchId = FMM_COMPAT_PATCH_ID;
    if(!key.valid())
        return false;
    FmmPatchKey large{std::numeric_limits<int>::max(), 0xdeadbeefcafebabeull};
    if(!large.valid())
        return false;
    return true;
}

bool testOrdering()
{
    const FmmPatchKey a{1, 2};
    const FmmPatchKey b{1, 3};
    const FmmPatchKey c{2, 1};
    if(!(a < b) || !(a < c) || !(b < c))
        return false;
    if(a == b || b == a)
        return false;
    return true;
}

bool testHashLookup()
{
    std::unordered_map<FmmPatchKey, int, FmmPatchKeyHash> table;
    table[FmmPatchKey{0, 1}] = 11;
    table[FmmPatchKey{3, 42}] = 42;
    table[FmmPatchKey{3, 43}] = 43;
    if(table.size() != 3)
        return false;
    if(table[FmmPatchKey{3, 42}] != 42)
        return false;
    return true;
}

bool testRemoteNodeKey()
{
    const FmmRemoteNodeKey first{{1, 7}, 9};
    const FmmRemoteNodeKey second{{1, 7}, 10};
    if(!(first < second))
        return false;
    if(first == second)
        return false;
    std::unordered_map<FmmRemoteNodeKey, int, FmmRemoteNodeKeyHash> table;
    table[first] = 1;
    table[second] = 2;
    return table.size() == 2;
}

bool testMalformedPatchIds()
{
    if(FmmGlobalDyadicLattice::isValidPatchId(0))
        return false;
    for(std::uint64_t malformed = 2; malformed <= 7; ++malformed)
    {
        if(FmmGlobalDyadicLattice::isValidPatchId(malformed))
            return false;
    }
    if(!FmmGlobalDyadicLattice::isValidPatchId(1))
        return false;
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(Vector3D(-1, -1, -1),
                                           Vector3D(1, 1, 1));
    const std::uint64_t child = lattice.childPatchId(1, 0);
    if(!FmmGlobalDyadicLattice::isValidPatchId(child))
        return false;
    return true;
}
}

int main()
{
    const bool pass = testValidity() && testOrdering() && testHashLookup() &&
                      testRemoteNodeKey() && testMalformedPatchIds();
    std::ofstream output("fmm_patch_key_metrics.txt");
    output << "pass " << (pass ? 1 : 0) << "\n";
    std::cout << "fmm_patch_key pass=" << pass << std::endl;
    return pass ? 0 : 1;
}
