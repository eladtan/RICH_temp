#ifndef IMC_POSTPROCESS_TDE_SNAPSHOT_RADIATION_HPP
#define IMC_POSTPROCESS_TDE_SNAPSHOT_RADIATION_HPP

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "PostProcessConfig.hpp"

#include "source/newtonian/common/equation_of_state.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"

namespace imc_postprocess_tde {

// What radiation the snapshot carries, which decides the opacity treatment:
//  - Multigroup: group energies E_g with as many groups as the opacity table
//    (a multigroup hydro run: its own group opacities, unscaled);
//  - GroupMismatch: group energies with a different group count (they cannot
//    be used per group; only the total E_rad is);
//  - Grey: E_rad only (a grey hydro run: group opacities normalized to the
//    grey table);
//  - None: no radiation anywhere (LTE radiation is built, BuildLteRadiation).
enum class SnapshotRadiationType { Multigroup, GroupMismatch, Grey, None };

std::string SnapshotRadiationTypeName(SnapshotRadiationType type);

struct SnapshotRadiationInfo
{
    SnapshotRadiationType type = SnapshotRadiationType::None;
    // Eg_k datasets in the snapshot file; for a snapshot not read from a file
    // (scenarios that build it), the table's group count when any E_g is
    // nonzero, else 0.
    size_t snapshotGroups = 0;
    size_t tableGroups = 0;
    bool fromFile = false;
    // Largest |E_rad - sum E_g| / E_rad over cells with group energies.
    double groupSumMismatch = 0.0;
};

// Number of consecutive Eg_0, Eg_1, ... datasets in a Snapshot3D HDF5 file
// (in /rank0 for per-rank files, else at the root); npos when `path` is not
// an HDF5 file (a scenario that builds its snapshot). Rank 0 reads the file;
// collective.
constexpr size_t kNotASnapshotFile = std::numeric_limits<size_t>::max();
size_t CountSnapshotGroupDatasets(std::string const& path, int rank);

// Classifies the snapshot from its cells (this rank's; collective) and the
// group count of its file (kNotASnapshotFile for a built snapshot). Throws
// when E_g has data but E_rad does not, and when a multigroup snapshot's
// E_rad differs from the sum of its E_g by more than kGroupSumTolerance (a
// multigroup hydro writes E_rad as the groups' sum; snap_429 agrees to 3e-5,
// so the tolerance catches wrong units or stale fields, not roundoff).
constexpr double kGroupSumTolerance = 1e-3;
SnapshotRadiationInfo ClassifySnapshotRadiation(
    std::vector<ComputationalCell3D> const& cells,
    size_t fileGroups,
    size_t tableGroups);

struct SnapshotRadiationRequest
{
    OpacityScaleMode scaling = OpacityScaleMode::Auto;
    FluxSourceFaceFlux faceFlux = FluxSourceFaceFlux::Auto;
    MissingRadiation missingRadiation = MissingRadiation::Auto;
    bool fluxSource = false;
};

struct SnapshotRadiationResolution
{
    OpacityScaleMode scaling = OpacityScaleMode::Planck;
    FluxSourceFaceFlux faceFlux = FluxSourceFaceFlux::Hydro;
    // ConserveEnergy or GasTemperature when LTE radiation is to be built
    // (type None), Auto otherwise.
    MissingRadiation lteRadiation = MissingRadiation::Auto;
    bool groupEnergyUsable = false;
};

// The settings the snapshot implies, with the user's explicit choices where
// they are allowed:
//   type           scaling (auto)          face flux (auto)   explicit choices
//   Multigroup     none                    hydro-multigroup   scaling must be none; any face flux
//   Grey           planck (or rosseland)   hydro              none / hydro-multigroup refused
//   GroupMismatch  planck                  hydro              scaling free; hydro-multigroup refused
//   None (LTE)     planck                  hydro              free
// missing-radiation may be set only for type None. Throws UniversalError
// naming the snapshot type and the setting to use. Pure.
SnapshotRadiationResolution ResolveSnapshotRadiation(
    SnapshotRadiationInfo const& info,
    SnapshotRadiationRequest const& request);

struct LteRadiationSummary
{
    size_t cells = 0;
    double sumRelativeChange = 0.0;     // sum of (T - T_snap) / T_snap
    double minRelativeChange = 0.0;     // min (T - T_snap) / T_snap
    double maxRelativeChange = 0.0;
    double maxEnergyError = 0.0;        // max |rho e_gas + aT^4 - rho e_snap| / (rho e_snap)
};

// Builds LTE radiation in every cell (type None): E_rad = a T^4 / rho and,
// with group edges (erg) of more than one group, E_g its Planck split at T.
// ConserveEnergy solves rho e_snap = rho e_gas(rho, T) + a T^4 for T (bisection
// in log T below the snapshot temperature) and sets the temperature, internal
// energy and pressure; GasTemperature keeps T. Throws with the cell's state
// when the EOS cannot be inverted. Local (no communication).
LteRadiationSummary BuildLteRadiation(
    std::vector<ComputationalCell3D>& cells,
    EquationOfState const& eos,
    MissingRadiation mode,
    std::vector<double> const& groupEdges);

} // namespace imc_postprocess_tde
#endif
