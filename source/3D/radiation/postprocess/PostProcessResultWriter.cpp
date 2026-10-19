#include "PostProcessResultWriter.hpp"

#include "PostProcessCommandLine.hpp"
#include "source/3D/radiation/StateHdf5IO.hpp"

#include "utils/hdf5/HDF5Writer.hpp"

#include <H5Cpp.h>
#include <H5Lpublic.h>
#include <H5Opublic.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace imc_postprocess_tde
{
namespace
{

std::string GreyHdf5ScratchPath(Config const& config)
{
    return InsertSuffixBeforeExtension(config.outputPath, ".grey-pass-scratch");
}

bool LinkExists(hid_t file, std::string const& path)
{
    return H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0;
}

void CreateGroup(hid_t file, std::string const& path)
{
    if (LinkExists(file, path))
        return;
    hid_t property = H5Pcreate(H5P_LINK_CREATE);
    if (property < 0)
        throw UniversalError("Could not create HDF5 link property list");
    H5Pset_create_intermediate_group(property, 1);
    hid_t group = H5Gcreate2(
        file, path.c_str(), property, H5P_DEFAULT, H5P_DEFAULT);
    H5Pclose(property);
    if (group < 0)
        throw UniversalError("Could not create HDF5 group " + path);
    H5Gclose(group);
}

void CopyIfPresent(
    hid_t source, std::string const& sourcePath,
    hid_t destination, std::string const& destinationPath)
{
    if (!LinkExists(source, sourcePath))
        return;
    if (H5Ocopy(source, sourcePath.c_str(), destination,
                destinationPath.c_str(), H5P_DEFAULT, H5P_DEFAULT) < 0)
        throw UniversalError(
            "Could not copy HDF5 object " + sourcePath + " to " +
            destinationPath);
}

void DeleteIfPresent(hid_t file, std::string const& path)
{
    if (LinkExists(file, path) &&
        H5Ldelete(file, path.c_str(), H5P_DEFAULT) < 0)
        throw UniversalError("Could not remove legacy HDF5 link " + path);
}

std::string DatasetPath(std::string optionName)
{
    for (char& character : optionName) {
        if (character == '.')
            character = '/';
    }
    return "/config/" + optionName;
}

bool StartsWith(std::string const& value, std::string const& prefix)
{
    return value.compare(0, prefix.size(), prefix) == 0;
}

std::string PrefixedField(std::string const& name, std::string const& pass)
{
    if (StartsWith(name, "photosphere_"))
        return name;
    if (StartsWith(name, "fld_"))
        return name;
    if (StartsWith(name, "log10_fld_"))
        return "fld_log10_" + name.substr(10);
    return pass + "_" + name;
}

bool ReadLines(std::string const& path, std::vector<std::string>& lines)
{
    std::ifstream input(path);
    if (!input.is_open())
        return false;
    std::string line;
    while (std::getline(input, line))
        lines.push_back(line);
    return true;
}

std::string ScalarName(std::string const& line)
{
    if (!StartsWith(line, "SCALARS "))
        return std::string();
    std::istringstream stream(line);
    std::string marker, name;
    stream >> marker >> name;
    return name;
}

std::string RenameScalar(
    std::string const& line, std::string const& renamed)
{
    std::istringstream input(line);
    std::string marker, oldName;
    input >> marker >> oldName;
    std::string remainder;
    std::getline(input, remainder);
    return marker + " " + renamed + remainder;
}

size_t PointCount(std::vector<std::string> const& lines)
{
    for (std::string const& line : lines) {
        if (StartsWith(line, "POINT_DATA ")) {
            std::istringstream stream(line.substr(11));
            size_t count = 0;
            stream >> count;
            return count;
        }
    }
    throw UniversalError("Observer VTK is missing POINT_DATA");
}

void MergeVtk(Config const& config)
{
    std::string const forwardPath = BaseVtkOutputPath(config);
    std::string const greyPath = GreyVtkOutputPath(config);
    if (forwardPath.empty())
        return;

    std::vector<std::string> forwardLines, greyLines;
    if (!ReadLines(forwardPath, forwardLines) ||
        !ReadLines(greyPath, greyLines))
        throw UniversalError("Could not read observer VTK pass outputs");
    size_t const pointCount = PointCount(forwardLines);

    std::string const scratch = forwardPath + ".bundle-scratch";
    std::ofstream output(scratch);
    if (!output.is_open())
        throw UniversalError("Could not create combined observer VTK");

    for (std::string const& line : forwardLines) {
        std::string const name = ScalarName(line);
        output << (name.empty() ? line : RenameScalar(
            line, PrefixedField(name, "forward"))) << "\n";
    }

    bool inPointData = false;
    for (size_t i = 0; i < greyLines.size();) {
        if (!inPointData) {
            if (StartsWith(greyLines[i], "POINT_DATA "))
                inPointData = true;
            ++i;
            continue;
        }
        std::string const name = ScalarName(greyLines[i]);
        if (name.empty()) {
            ++i;
            continue;
        }
        bool const shared = StartsWith(name, "fld_") ||
            StartsWith(name, "log10_fld_") ||
            StartsWith(name, "photosphere_");
        if (!shared) {
            output << RenameScalar(
                greyLines[i], PrefixedField(name, "grey")) << "\n";
            if (i + 1 >= greyLines.size())
                throw UniversalError("Truncated grey observer VTK scalar");
            output << greyLines[i + 1] << "\n";
            for (size_t p = 0; p < pointCount; ++p) {
                if (i + 2 + p >= greyLines.size())
                    throw UniversalError("Truncated grey observer VTK values");
                output << greyLines[i + 2 + p] << "\n";
            }
        }
        i += pointCount + 2;
    }
    output.close();
    if (std::rename(scratch.c_str(), forwardPath.c_str()) != 0) {
        std::remove(scratch.c_str());
        throw UniversalError("Could not install combined observer VTK");
    }
    std::remove(greyPath.c_str());
}

void PromoteFile(std::string const& source, std::string const& destination)
{
    if (source == destination)
        return;
    if (std::rename(source.c_str(), destination.c_str()) != 0)
        throw UniversalError(
            "Could not promote post-process output " + source + " to " +
            destination);
}

std::string RestartSegmentPath(size_t index)
{
    char name[32];
    std::snprintf(name, sizeof(name), "segment_%04zu", index);
    return std::string("/restart/history/") + name;
}

// Copies the run segments of the file being continued and returns how many
// there were (0 for a fresh run).
size_t CopyRestartHistory(hid_t dst, std::string const& restartFrom)
{
    CreateGroup(dst, "/restart/history");
    if (restartFrom.empty())
        return 0;
    try {
        H5::H5File source(restartFrom, H5F_ACC_RDONLY);
        hid_t const src = source.getId();
        if (!LinkExists(src, "/restart/segment_count"))
            throw UniversalError("Restart file " + restartFrom + " has no run history");
        unsigned long long stored = 0;
        source.openDataSet("/restart/segment_count").read(&stored, H5::PredType::NATIVE_ULLONG);
        size_t const segments = static_cast<size_t>(stored);
        for (size_t k = 0; k < segments; ++k) {
            if (!LinkExists(src, RestartSegmentPath(k)))
                throw UniversalError("Restart file history is missing " + RestartSegmentPath(k));
            CopyIfPresent(src, RestartSegmentPath(k), dst, RestartSegmentPath(k));
        }
        return segments;
    } catch (H5::Exception const& e) {
        // HDF5's C++ exceptions do not derive from std::exception.
        throw UniversalError("Could not copy the run history of " + restartFrom +
                             ": " + e.getFuncName() + ": " + e.getDetailMsg());
    }
}

void MergeHdf5(
    Config const& config,
    std::string const& finalOutputPath,
    PostProcessIMC::PostProcessConfig const& effectiveConfig,
    std::string const& scenarioName,
    PostprocessRuntime const& runtime,
    ForwardPostprocessResult const& forward,
    ForwardPostprocessResult const& grey)
{
    std::string const scratchPath = GreyHdf5ScratchPath(config);
    size_t previousSegments = 0;
    {
        H5::H5File destination(config.outputPath, H5F_ACC_RDWR);
        H5::H5File source(scratchPath, H5F_ACC_RDONLY);
        hid_t const dst = destination.getId();
        hid_t const src = source.getId();
        CreateGroup(dst, "/passes/forward");
        CreateGroup(dst, "/passes/grey");
        CreateGroup(dst, "/photosphere");
        for (char const* group : {"/tally", "/adaptive", "/diagnostics",
                                  "/statistics_state", "/learned_state", "/load_balance"}) {
            CopyIfPresent(dst, group, dst, std::string("/passes/forward") + group);
            CopyIfPresent(src, group, dst, std::string("/passes/grey") + group);
            DeleteIfPresent(dst, group);
        }
        previousSegments = CopyRestartHistory(dst, config.restartFrom);
    }

    HDF5Writer writer(config.outputPath, false);
    writer.WriteElement("/metadata/schema_version", 1);
    writer.WriteElement("/metadata/scenario", scenarioName);
    writer.WriteElement("/metadata/snapshot", effectiveConfig.input.snapshot);
    writer.WriteElement("/metadata/snapshot_time", runtime.snapshotTime);
    writer.WriteElement("/metadata/snapshot_cycle", runtime.snapshotCycle);
    writer.WriteElement("/metadata/mpi_size", runtime.mpiSize);
#ifdef MONTECARLO_POLARIZATION
    writer.WriteElement("/metadata/build/polarization", 1);
#else
    writer.WriteElement("/metadata/build/polarization", 0);
#endif
#ifdef RICH_IMC_DDMC_ENABLED
    writer.WriteElement("/metadata/build/ddmc", 1);
#else
    writer.WriteElement("/metadata/build/ddmc", 0);
#endif
    writer.WriteElement("/metadata/build/energy_groups", static_cast<int>(ENERGY_GROUPS_NUM));

    // This run's segment of the result history.
    std::ostringstream configText;
    for (auto const& entry : PostProcessIMC::EffectiveConfigEntries(effectiveConfig))
        configText << entry.first << " = " << entry.second << "\n";
    char const* jobID = std::getenv("SLURM_JOB_ID");
    std::string const segment = RestartSegmentPath(previousSegments);
    writer.WriteElement(segment + "/job_id", std::string(jobID != nullptr ? jobID : "none"));
    writer.WriteElement(segment + "/mpi_size", runtime.mpiSize);
    writer.WriteElement(segment + "/restart_from", config.restartFrom);
    writer.WriteElement(segment + "/output", finalOutputPath);
    writer.WriteElement(segment + "/build_time", std::string(__DATE__ " " __TIME__));
    writer.WriteElement(segment + "/forward_generations_added", forward.finalGenerationsAdded);
    writer.WriteElement(segment + "/forward_generations_total", forward.finalGenerationsTotal);
    writer.WriteElement(segment + "/grey_generations_added", grey.finalGenerationsAdded);
    writer.WriteElement(segment + "/grey_generations_total", grey.finalGenerationsTotal);
    writer.WriteElement(segment + "/effective_config", configText.str());
    writer.WriteElement("/restart/schema_version", kRestartSchemaVersion);
    writer.WriteElement("/restart/segment_count", previousSegments + 1);
    std::vector<std::pair<std::string, std::string>> const entries =
        PostProcessIMC::EffectiveConfigEntries(effectiveConfig);
    for (auto const& entry : entries)
        writer.WriteElement(DatasetPath(entry.first), entry.second);

    writer.WriteElement("/comparison/fld_total_luminosity", runtime.totalFldLuminosity);
    writer.WriteElement("/comparison/fld_observer_luminosity", runtime.fldLuminosity);
    writer.WriteElement("/comparison/source_surface/tau", runtime.fluxSourceTau);
    writer.WriteElement("/comparison/source_surface/directly_resolved_fraction", runtime.fluxSourceDirectlyResolvedFraction);
    writer.WriteElement("/comparison/source_surface/boundary_face_count", runtime.fluxSourceBoundaryFaceCount);
    writer.WriteElement("/comparison/source_surface/emitting_face_count", runtime.fluxSourceEmittingFaceCount);
    writer.WriteElement("/comparison/source_surface/injected_luminosity", runtime.fluxSourceInjectedLuminosity);
    writer.WriteElement("/comparison/source_surface/net_luminosity", runtime.fluxSourceNetLuminosity);
    writer.WriteElement("/comparison/source_surface/inward_luminosity", runtime.fluxSourceInwardLuminosity);
    writer.WriteElement("/comparison/source_surface/smoothed_directions",
                        runtime.fluxSourceSmoothedDirections.size());
    writer.WriteElement("/comparison/source_surface/enclosed_cells_removed",
                        runtime.fluxSourceEnclosedCellIDs.size());
    writer.WriteElement("/comparison/source_surface/enclosed_wall_luminosity_removed",
                        runtime.fluxSourceEnclosedLuminosity);
    if (runtime.fluxSourceEnabled) {
        // The surface per construction direction, as in the .source-surface.h5
        // file, so it travels with the result. Arrays are stored flat next to
        // a <name>_shape dataset; group matrices are [group][direction].
        std::string const s = "/comparison/source_surface/";
        std::vector<double> dx, dy, dz;
        for (Vector3D const& d : runtime.fluxSourceDirections) {
            dx.push_back(d.x);
            dy.push_back(d.y);
            dz.push_back(d.z);
        }
        state_hdf5::WriteVector(writer, s + "directions/x", dx);
        state_hdf5::WriteVector(writer, s + "directions/y", dy);
        state_hdf5::WriteVector(writer, s + "directions/z", dz);
        writer.WriteElement(s + "mapping_neighbours", runtime.fluxSourceMappingNeighbours);
        state_hdf5::WriteVector(writer, s + "radius", runtime.fluxSourceRadius);
        state_hdf5::WriteVector(writer, s + "radius_before_mapping", runtime.fluxSourceSmoothedRadius);
        state_hdf5::WriteVector(writer, s + "radius_before_smoothing", runtime.fluxSourceRawRadius);
        state_hdf5::WriteVector(writer, s + "resolved", runtime.fluxSourceRadiusDirectlyResolved);
        state_hdf5::WriteVector(writer, s + "photosphere_radius", runtime.fluxSourceSurfacePhotosphere);
        state_hdf5::WriteVector(writer, s + "smoothed_direction_indices", runtime.fluxSourceSmoothedDirections);
        state_hdf5::WriteVector(writer, s + "capped_direction_indices", runtime.fluxSourceCappedDirections);
        state_hdf5::WriteVector(writer, s + "mapped_direction_indices", runtime.fluxSourceMappedDirections);
        writer.WriteElement(s + "mapping_floor_tau", runtime.fluxSourceMappingFloorTau);
        state_hdf5::WriteVector(writer, s + "mapping_floor_radius", runtime.fluxSourceMappingFloor);
        state_hdf5::WriteVector(writer, s + "floor_held_direction_indices", runtime.fluxSourceFloorHeldDirections);
        state_hdf5::WriteVector(writer, s + "grey/radius", runtime.fluxSourceGreyRadius);
        state_hdf5::WriteVector(writer, s + "grey/resolved", runtime.fluxSourceGreyResolved);
        state_hdf5::WriteVector(writer, s + "grey/photosphere_radius", runtime.fluxSourceGreyPhotosphere);
        state_hdf5::WriteVector(writer, s + "grey/photosphere_resolved", runtime.fluxSourceGreyPhotosphereResolved);
        state_hdf5::WriteMatrix(writer, s + "groups/radius", runtime.fluxSourceGroupRadius);
        state_hdf5::WriteMatrix(writer, s + "groups/resolved", runtime.fluxSourceGroupResolved);
        state_hdf5::WriteMatrix(writer, s + "groups/photosphere_radius", runtime.fluxSourceGroupPhotosphere);
        state_hdf5::WriteMatrix(writer, s + "groups/photosphere_resolved", runtime.fluxSourceGroupPhotosphereResolved);
        auto const& check = runtime.fluxSourcePhotosphereCheck;
        if (check.computed) {
            writer.WriteElement(s + "photosphere_check/boundary_faces", check.faces);
            writer.WriteElement(s + "photosphere_check/face_luminosity", check.luminosity);
            writer.WriteElement(s + "photosphere_check/faces_above_deepest", check.facesAboveDeepest);
            writer.WriteElement(s + "photosphere_check/luminosity_above_deepest", check.luminosityAboveDeepest);
            writer.WriteElement(s + "photosphere_check/faces_transparent_deepest", check.facesTransparentDeepest);
            writer.WriteElement(s + "photosphere_check/luminosity_transparent_deepest",
                                check.luminosityTransparentDeepest);
            // Entry 0 is grey, then one per group in mg-innermost mode.
            state_hdf5::WriteVector(writer, s + "photosphere_check/faces_above", check.facesAbove);
            state_hdf5::WriteVector(writer, s + "photosphere_check/luminosity_above", check.luminosityAbove);
            state_hdf5::WriteVector(writer, s + "photosphere_check/faces_transparent", check.facesTransparent);
            state_hdf5::WriteVector(writer, s + "photosphere_check/luminosity_transparent",
                                    check.luminosityTransparent);
        }
    }
    writer.WriteElement("/comparison/forward_crossing_luminosity", forward.crossingLuminosity);
    writer.WriteElement("/comparison/grey_crossing_luminosity", grey.crossingLuminosity);
    writer.WriteElement("/comparison/forward_emitted_luminosity", forward.emittedLuminosity);
    writer.WriteElement("/comparison/grey_emitted_luminosity", grey.emittedLuminosity);
    writer.WriteElement("/comparison/forward_timed_out_fraction", forward.timedOutFraction);
    writer.WriteElement("/comparison/grey_timed_out_fraction", grey.timedOutFraction);
    double const ratio = (grey.crossingLuminosity > 0.0)
        ? forward.crossingLuminosity / grey.crossingLuminosity : 0.0;
    writer.WriteElement("/comparison/forward_to_grey_luminosity_ratio", ratio);
    writer.WriteElement("/comparison/forward_polarization_degree", forward.luminosityWeightedPolarizationDegree);
    writer.WriteElement("/comparison/grey_polarization_degree", grey.luminosityWeightedPolarizationDegree);
    writer.WriteElement("/diagnostics/source_allocation/forward_source_luminosity", forward.sourceLuminosity);
    writer.WriteElement("/diagnostics/source_allocation/grey_source_luminosity", grey.sourceLuminosity);
    writer.WriteElement("/diagnostics/estimator_warning", std::string("none"));
    writer.WriteElement("/passes/forward/summary/ran", forward.ran ? 1 : 0);
    writer.WriteElement("/passes/forward/summary/crossing_luminosity", forward.crossingLuminosity);
    writer.WriteElement("/passes/forward/summary/crossing_luminosity_stderr", forward.crossingLuminosityStderr);
    writer.WriteElement("/passes/grey/summary/ran", grey.ran ? 1 : 0);
    writer.WriteElement("/passes/grey/summary/crossing_luminosity", grey.crossingLuminosity);
    writer.WriteElement("/passes/grey/summary/crossing_luminosity_stderr", grey.crossingLuminosityStderr);
    writer.Close();

    std::remove(scratchPath.c_str());
}

} // namespace

Config MakePassOutputConfig(Config const& finalConfig)
{
    Config passConfig = finalConfig;
    passConfig.outputPath = InsertSuffixBeforeExtension(
        finalConfig.outputPath, ".postprocess-scratch");
    std::string const finalVtk = BaseVtkOutputPath(finalConfig);
    passConfig.vtkOutput = finalVtk.empty() ? std::string() :
        InsertSuffixBeforeExtension(finalVtk, ".postprocess-scratch");
    return passConfig;
}

void WriteGreyPassScratch(
    Config const& config,
    SphericalObserver const& observer,
    SphericalObserver::Diagnostics const& diagnostics,
    PassRestartState const& restartState)
{
    observer.writeHDF5(GreyHdf5ScratchPath(config), diagnostics);
    WritePassRestartState(GreyHdf5ScratchPath(config), restartState, observer);
}

void FinalizeResultBundle(
    Config const& finalConfig,
    Config const& passConfig,
    PostProcessIMC::PostProcessConfig const& effectiveConfig,
    std::string const& scenarioName,
    PostprocessRuntime const& runtime,
    ForwardPostprocessResult const& forward,
    ForwardPostprocessResult const& grey)
{
    if (runtime.rank != 0)
        return;
    MergeHdf5(
        passConfig, finalConfig.outputPath, effectiveConfig, scenarioName,
        runtime, forward, grey);
    MergeVtk(passConfig);
    PromoteFile(passConfig.outputPath, finalConfig.outputPath);
    std::string const passVtk = BaseVtkOutputPath(passConfig);
    std::string const finalVtk = BaseVtkOutputPath(finalConfig);
    if (!passVtk.empty())
        PromoteFile(passVtk, finalVtk);
}

} // namespace imc_postprocess_tde
