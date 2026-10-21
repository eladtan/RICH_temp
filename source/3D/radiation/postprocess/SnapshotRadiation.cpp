#include "SnapshotRadiation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>

#include "source/Radiation/planck_integral/planck_integral.hpp"
#include "source/misc/universal_error.hpp"
#include "source/utils/hdf5/HDF5Reader.hpp"
#include "source/Radiation/conj_grad_solve.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace imc_postprocess_tde {
namespace {

[[noreturn]] void Refuse(SnapshotRadiationInfo const& info, std::string const& option,
                         std::string const& value, std::string const& reason)
{
    UniversalError eo("--" + option + " " + value + " does not fit this snapshot: " + reason);
    eo.addEntry("Snapshot radiation", SnapshotRadiationTypeName(info.type));
    eo.addEntry("Snapshot groups", info.snapshotGroups);
    eo.addEntry("Opacity table groups", info.tableGroups);
    throw eo;
}

} // namespace

std::string SnapshotRadiationTypeName(SnapshotRadiationType type)
{
    switch(type)
    {
    case SnapshotRadiationType::Multigroup: return "multigroup";
    case SnapshotRadiationType::GroupMismatch: return "group-mismatch";
    case SnapshotRadiationType::Grey: return "grey";
    case SnapshotRadiationType::None: return "none";
    }
    return "?";
}

size_t CountSnapshotGroupDatasets(std::string const& path, int rank)
{
    // Rank 0 reads; a failure there is broadcast so every rank throws rather
    // than the others waiting in the broadcast.
    uint64_t count = static_cast<uint64_t>(kNotASnapshotFile);
    std::string failure;
    if(rank == 0)
    {
        try
        {
            std::error_code ec;
            if(!path.empty() && std::filesystem::is_regular_file(path, ec))
            {
                HDF5Reader reader(path);
                std::string const prefix = reader.Exists("/rank0") ? "/rank0" : "";
                count = 0;
                while(reader.Exists(prefix + "/Eg_" + std::to_string(count)))
                    ++count;
            }
        }
        catch(UniversalError const& eo)
        {
            failure = eo.getErrorMessage();
        }
        catch(std::exception const& e)
        {
            failure = e.what();
        }
    }
#ifdef RICH_MPI
    uint64_t failed = failure.empty() ? 0 : 1;
    uint64_t header[2] = {count, failed};
    MPI_Bcast(header, 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    count = header[0];
    failed = header[1];
    if(failed != 0 && rank != 0)
        failure = "see rank 0";
#endif
    if(!failure.empty())
    {
        UniversalError eo("Cannot count the group datasets (Eg_k) of the snapshot file: " + failure);
        eo.addEntry("Snapshot", path);
        throw eo;
    }
    return static_cast<size_t>(count);
}

SnapshotRadiationInfo ClassifySnapshotRadiation(
    std::vector<ComputationalCell3D> const& cells,
    size_t fileGroups,
    size_t tableGroups)
{
    // [cells with E_rad > 0, cells with any E_g > 0, cells with E_g but no E_rad]
    unsigned long long counts[3] = {0, 0, 0};
    // Largest |E_rad - sum E_g| / E_rad over cells with group energies: a
    // multigroup hydro writes E_rad as the groups' sum.
    double worstSum = 0.0;
    unsigned long long worstID = 0;
    for(ComputationalCell3D const& cell : cells)
    {
        bool const erad = cell.Erad > 0.0;
        bool groups = false;
        double groupSum = 0.0;
        for(double const value : cell.Eg)
        {
            groups = groups || value > 0.0;
            groupSum += value;
        }
        counts[0] += erad ? 1 : 0;
        counts[1] += groups ? 1 : 0;
        counts[2] += (groups && !erad) ? 1 : 0;
        if(groups && erad)
        {
            double const mismatch = std::abs(cell.Erad - groupSum) / cell.Erad;
            if(!(mismatch <= worstSum))
            {
                worstSum = mismatch;
                worstID = static_cast<unsigned long long>(cell.ID);
            }
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    {
        struct { double value; int rank; } local{worstSum, 0}, global{0.0, 0};
        MPI_Comm_rank(MPI_COMM_WORLD, &local.rank);
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        MPI_Bcast(&worstID, 1, MPI_UNSIGNED_LONG_LONG, global.rank, MPI_COMM_WORLD);
        worstSum = global.value;
    }
#endif
    SnapshotRadiationInfo info;
    info.tableGroups = tableGroups;
    info.fromFile = fileGroups != kNotASnapshotFile;
    info.snapshotGroups = info.fromFile ? fileGroups : (counts[1] > 0 ? tableGroups : 0);
    info.groupSumMismatch = worstSum;
    if(counts[2] > 0)
    {
        UniversalError eo("Snapshot has group energies E_g but no E_rad in some cells; "
                          "E_rad must be the radiation energy the groups sum to");
        eo.addEntry("Cells with E_g but no E_rad", counts[2]);
        throw eo;
    }
    if(counts[1] > 0)
        info.type = info.snapshotGroups == tableGroups ? SnapshotRadiationType::Multigroup
                                                       : SnapshotRadiationType::GroupMismatch;
    else if(counts[0] > 0)
        info.type = SnapshotRadiationType::Grey;
    else
        info.type = SnapshotRadiationType::None;
    if(info.type == SnapshotRadiationType::Multigroup && !(worstSum <= kGroupSumTolerance))
    {
        UniversalError eo("Multigroup snapshot: E_rad is not the sum of the group energies E_g, "
                          "so the grey and the per-group radiation fields disagree");
        eo.addEntry("Largest |E_rad - sum E_g| / E_rad", worstSum);
        eo.addEntry("Cell ID", worstID);
        eo.addEntry("Tolerance", kGroupSumTolerance);
        throw eo;
    }
    return info;
}

SnapshotRadiationResolution ResolveSnapshotRadiation(
    SnapshotRadiationInfo const& info,
    SnapshotRadiationRequest const& request)
{
    SnapshotRadiationResolution result;
    bool const explicitScaling = request.scaling != OpacityScaleMode::Auto;
    bool const explicitFaces = request.faceFlux != FluxSourceFaceFlux::Auto;

    if(info.type != SnapshotRadiationType::None &&
       request.missingRadiation != MissingRadiation::Auto)
        Refuse(info, "input.missing-radiation",
               PostProcessIMC::OptionValueName(request.missingRadiation),
               "the snapshot already has radiation; the option is only for a snapshot without any");

    switch(info.type)
    {
    case SnapshotRadiationType::Multigroup:
        if(explicitScaling && request.scaling != OpacityScaleMode::None)
            Refuse(info, "opacity-scaling.mode", PostProcessIMC::OptionValueName(request.scaling),
                   "a multigroup snapshot was made with the group opacities themselves; "
                   "use none (or leave it auto)");
        // The grey face fluxes use E_rad, the groups' sum: a coarser choice,
        // not a mismatch, so they stay allowed.
        result.scaling = OpacityScaleMode::None;
        result.faceFlux = explicitFaces ? request.faceFlux : FluxSourceFaceFlux::HydroMultigroup;
        result.groupEnergyUsable = true;
        break;
    case SnapshotRadiationType::Grey:
        if(request.scaling == OpacityScaleMode::None)
            Refuse(info, "opacity-scaling.mode", "none",
                   "a grey snapshot was made with the grey opacity; the group opacities must be "
                   "normalized to it: use planck or rosseland (or leave it auto)");
        if(request.faceFlux == FluxSourceFaceFlux::HydroMultigroup)
            Refuse(info, "flux-source.face-flux", "hydro-multigroup",
                   "a grey snapshot has no group energies; use hydro or cell-average (or leave it auto)");
        result.scaling = explicitScaling ? request.scaling : OpacityScaleMode::Planck;
        result.faceFlux = explicitFaces ? request.faceFlux : FluxSourceFaceFlux::Hydro;
        break;
    case SnapshotRadiationType::GroupMismatch:
        if(request.faceFlux == FluxSourceFaceFlux::HydroMultigroup)
            Refuse(info, "flux-source.face-flux", "hydro-multigroup",
                   "the snapshot's groups differ from the opacity table's, so its group energies "
                   "cannot be used per group; use hydro or cell-average (or leave it auto)");
        result.scaling = explicitScaling ? request.scaling : OpacityScaleMode::Planck;
        result.faceFlux = explicitFaces ? request.faceFlux : FluxSourceFaceFlux::Hydro;
        break;
    case SnapshotRadiationType::None:
        result.lteRadiation = request.missingRadiation == MissingRadiation::Auto
            ? MissingRadiation::ConserveEnergy : request.missingRadiation;
        result.scaling = explicitScaling ? request.scaling : OpacityScaleMode::Planck;
        result.faceFlux = explicitFaces ? request.faceFlux : FluxSourceFaceFlux::Hydro;
        result.groupEnergyUsable = info.tableGroups > 1;
        break;
    }
    if(!request.fluxSource && !explicitFaces)
        result.faceFlux = FluxSourceFaceFlux::Hydro;
    return result;
}

LteRadiationSummary BuildLteRadiation(
    std::vector<ComputationalCell3D>& cells,
    EquationOfState const& eos,
    MissingRadiation mode,
    std::vector<double> const& groupEdges)
{
    if(mode != MissingRadiation::ConserveEnergy && mode != MissingRadiation::GasTemperature)
        throw UniversalError("BuildLteRadiation needs conserve-energy or gas-temperature");
    double const a = CG::radiation_constant;
    size_t const groups = groupEdges.size() > 1 ? groupEdges.size() - 1 : 0;
    LteRadiationSummary summary;
    summary.minRelativeChange = 0.0;
    summary.maxRelativeChange = 0.0;
    std::vector<std::string> const& names = ComputationalCell3D::tracerNames;
    for(ComputationalCell3D& cell : cells)
    {
        double const rho = cell.density;
        double const snapshotT = cell.temperature;
        double const snapshotE = cell.internal_energy;
        auto cellError = [&](std::string const& what) {
            UniversalError eo("LTE radiation: " + what);
            eo.addEntry("Cell ID", cell.ID);
            eo.addEntry("Density", rho);
            eo.addEntry("Snapshot temperature", snapshotT);
            eo.addEntry("Snapshot internal energy", snapshotE);
            return eo;
        };
        if(!(rho > 0.0) || !std::isfinite(rho))
            throw cellError("non-positive density");
        double T = snapshotT;
        if(mode == MissingRadiation::ConserveEnergy)
        {
            if(!(snapshotE > 0.0) || !std::isfinite(snapshotE))
                throw cellError("non-positive internal energy");
            // g(T) = rho e_gas(T) + a T^4 - rho e_snap rises with T. The upper
            // end holds all the energy in the gas: g >= 0 there.
            double const target = rho * snapshotE;
            auto g = [&](double temperature) {
                return rho * eos.dT2e(rho, temperature, cell.tracers, names) +
                       a * std::pow(temperature, 4) - target;
            };
            double hi = eos.de2T(rho, snapshotE, cell.tracers, names);
            if(snapshotT > 0.0 && std::isfinite(snapshotT))
                hi = std::max(hi, snapshotT);
            if(!(hi > 0.0) || !std::isfinite(hi))
                throw cellError("no positive temperature for the snapshot energy");
            if(g(hi) < 0.0)
                throw cellError("the EOS gives less than the snapshot energy at its own temperature");
            double lo = 0.5 * hi;
            int halvings = 0;
            while(g(lo) >= 0.0)
            {
                lo *= 0.5;
                if(++halvings > 200)
                    throw cellError("no temperature below the snapshot one where the gas holds less energy");
            }
            for(int iteration = 0; iteration < 200 && hi - lo > 1e-13 * hi; ++iteration)
            {
                double const mid = std::sqrt(lo * hi);
                (g(mid) < 0.0 ? lo : hi) = mid;
            }
            T = 0.5 * (lo + hi);
            double const gasEnergy = eos.dT2e(rho, T, cell.tracers, names);
            summary.maxEnergyError = std::max(summary.maxEnergyError,
                std::abs(rho * gasEnergy + a * std::pow(T, 4) - target) / target);
            cell.temperature = T;
            cell.internal_energy = gasEnergy;
            cell.pressure = eos.de2p(rho, gasEnergy, cell.tracers, names);
        }
        else if(!(T > 0.0) || !std::isfinite(T))
            throw cellError("non-positive snapshot temperature");

        double const radiationDensity = a * std::pow(T, 4);
        cell.Erad = radiationDensity / rho;
        if(groups > 1)
        {
            if(cell.Eg.size() != groups)
                throw cellError("cell group count differs from the opacity table's");
            double total = 0.0;
            for(size_t g = 0; g < groups; ++g)
            {
                cell.Eg[g] = planck_integral::planck_energy_density_group_integral(
                    groupEdges[g], groupEdges[g + 1], T);
                total += cell.Eg[g];
            }
            if(!(total > 0.0) || !std::isfinite(total))
                throw cellError("the groups hold none of the Planck spectrum at the LTE temperature");
            for(size_t g = 0; g < groups; ++g)
                cell.Eg[g] *= cell.Erad / total;
        }
        double const change = snapshotT > 0.0 ? (T - snapshotT) / snapshotT : 0.0;
        summary.minRelativeChange = summary.cells == 0 ? change : std::min(summary.minRelativeChange, change);
        summary.maxRelativeChange = summary.cells == 0 ? change : std::max(summary.maxRelativeChange, change);
        summary.sumRelativeChange += change;
        ++summary.cells;
    }
    return summary;
}

} // namespace imc_postprocess_tde
