#!/usr/bin/env python3
"""Apply the FMM V6 experiment to the exact feat/fmm-bounded-let-waves base.

This deliberately does not parse or apply the previously malformed patch.
It verifies the complete Git blob hash of every target file, performs exact
single-occurrence text replacements in memory, writes the files only after all
replacements validate, and generates a fresh Git-native patch.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Dict

EXPECTED_BRANCH = "feat/fmm-bounded-let-waves"
EXPECTED_HEAD = "616d5293c478c62b137946d7e02ccafd75963fcc"
OUTPUT_PATCH = "fmm_tde_v6_compact_payload_corrected.patch"

EXPECTED_BLOBS: Dict[str, str] = {
    "source/3D/gravity/fmm/mpi/FmmPackets.hpp": "dd5488f7946da88bba0e2f889cbf3d003c86888a",
    "source/3D/gravity/fmm/mpi/FmmPatchLetPlan.cpp": "9453fc0be839519e881abc57881345eb09f87a16",
    "source/3D/gravity/fmm/mpi/FmmDistributedOptions.hpp": "132933314f10524a0dfcf0c6ed49fe041d9ac192",
    "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp": "a18ed0b9d1917c07481ea3acf7893a8686450b0d",
    "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.cpp": "d3efd30457ff4df8730b1468fd7ad5da965492a9",
    "runs/M05R05MBH1e5ComptonGrayFrom148/test.cpp": "d65685fac3c489b046c67b0b32531be0bd349ad4",
    "regression_tests/cases/fmm_patch_let_mpi/test.cpp": "ae13c9b8c1372a10621e128a17900b10ee11f31d",
}


def run(*args: str, capture: bool = True) -> str:
    result = subprocess.run(
        list(args),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{detail}")
    return (result.stdout or "").strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one old-text match, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> int:
    root = Path(run("git", "rev-parse", "--show-toplevel"))
    os.chdir(root)

    branch = run("git", "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"wrong branch: expected {EXPECTED_BRANCH}, found {branch or '<detached>'}"
        )
    head = run("git", "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise RuntimeError(
            f"wrong HEAD: expected {EXPECTED_HEAD}, found {head}"
        )

    paths = list(EXPECTED_BLOBS)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *paths], check=False
    )
    if staged.returncode != 0:
        raise RuntimeError("one or more target files have staged changes")

    originals: Dict[str, str] = {}
    transformed: Dict[str, str] = {}
    for relative, expected in EXPECTED_BLOBS.items():
        path = root / relative
        actual = run("git", "hash-object", str(path))
        if actual != expected:
            raise RuntimeError(
                f"preimage mismatch for {relative}: expected {expected}, found {actual}"
            )
        originals[relative] = path.read_text(encoding="utf-8")
        transformed[relative] = originals[relative]
        print(f"PREIMAGE_OK {actual} {relative}")

    p = "source/3D/gravity/fmm/mpi/FmmPackets.hpp"
    t = transformed[p]
    t = replace_once(
        t,
        "static constexpr std::uint16_t FMM_MPI_PACKET_VERSION = 6u;",
        "static constexpr std::uint16_t FMM_MPI_PACKET_VERSION = 7u;",
        "packet version",
    )
    t = replace_once(
        t,
        """// Patch LET P2P traffic never needs body identity: source and target patches
// are owned by different MPI ranks, and the patch interaction plan already
// identifies the source leaf.  Keep only the values used by the force kernel.
struct FmmPatchWireParticle
{
    double position[3] = {0.0, 0.0, 0.0};
    double mass = 0.0;

    Vector3D positionVector() const
    {
        return Vector3D(position[0], position[1], position[2]);
    }
};""",
        """// Patch LET P2P traffic never needs body identity: source and target patches
// are owned by different MPI ranks, and the source leaf geometry is already
// present in the retained LET plan. Quantize each coordinate relative to that
// leaf cube while preserving mass in binary64. The TDE accepts a mean force
// error of 1e-2; this representation has a coordinate step of at most
// leaf_half_size / 32767 and reduces the particle record from 32 to 16 bytes.
struct FmmPatchWireParticle
{
    std::int16_t offset[3] = {0, 0, 0};
    std::uint16_t reserved = 0;
    double mass = 0.0;
};""",
        "compact particle structure",
    )
    t = replace_once(
        t,
        """static_assert(sizeof(FmmPatchWireParticle) == 32,
              "Distributed patch FMM wire particle has unsupported padding");""",
        """static_assert(sizeof(FmmPatchWireParticle) == 16,
              "Distributed patch FMM compact wire particle has unsupported padding");""",
        "compact particle size assertion",
    )
    transformed[p] = t

    p = "source/3D/gravity/fmm/mpi/FmmPatchLetPlan.cpp"
    t = transformed[p]
    t = replace_once(
        t,
        """void addDirectRemoteParticles(FmmLocalPatch& targetPatch,
                              const FmmNode& targetNode,""",
        """void addDirectRemoteParticles(FmmLocalPatch& targetPatch,
                              const FmmNode& sourceNode,
                              const FmmNode& targetNode,""",
        "direct-particle signature",
    )
    t = replace_once(
        t,
        """    const std::vector<std::size_t>& targetOrder =
        targetPatch.tree.particleOrder();
    for(std::size_t k = targetNode.particleBegin;""",
        """    const std::vector<std::size_t>& targetOrder =
        targetPatch.tree.particleOrder();
    if(!(sourceNode.halfSize > 0.0) ||
       !std::isfinite(sourceNode.halfSize))
        throw UniversalError(
            "FmmPatchLetPlan::execute: invalid compact particle scale");
    const double offsetScale = sourceNode.halfSize / 32767.0;
    for(std::size_t k = targetNode.particleBegin;""",
        "compact decode scale",
    )
    t = replace_once(
        t,
        """            FmmPatchWireParticle source;
            std::memcpy(&source,
                        payload + sourceIndex * sizeof(FmmPatchWireParticle),
                        sizeof(FmmPatchWireParticle));
            if(!finiteVector(source.positionVector()) ||
               !std::isfinite(source.mass))
                throw UniversalError(
                    "FmmPatchLetPlan::execute: malformed remote particle");
            const Vector3D delta = target - source.positionVector();
            const long double r2 =
                static_cast<long double>(delta.x) * delta.x +
                static_cast<long double>(delta.y) * delta.y +
                static_cast<long double>(delta.z) * delta.z;
            if(r2 == 0.0L)
                continue;
            const long double invR = 1.0L / std::sqrt(r2);
            const long double factor =
                static_cast<long double>(source.mass) * invR * invR * invR;
            ax -= factor * delta.x;
            ay -= factor * delta.y;
            az -= factor * delta.z;
            potential += static_cast<long double>(source.mass) * invR;""",
        """            FmmPatchWireParticle source;
            std::memcpy(&source,
                        payload + sourceIndex * sizeof(FmmPatchWireParticle),
                        sizeof(FmmPatchWireParticle));
            const Vector3D sourcePosition(
                sourceNode.center.x +
                    static_cast<double>(source.offset[0]) * offsetScale,
                sourceNode.center.y +
                    static_cast<double>(source.offset[1]) * offsetScale,
                sourceNode.center.z +
                    static_cast<double>(source.offset[2]) * offsetScale);
            const double sourceMass = source.mass;
            if(!finiteVector(sourcePosition) || !std::isfinite(sourceMass))
                throw UniversalError(
                    "FmmPatchLetPlan::execute: malformed remote particle");
            const Vector3D delta = target - sourcePosition;
            const long double r2 =
                static_cast<long double>(delta.x) * delta.x +
                static_cast<long double>(delta.y) * delta.y +
                static_cast<long double>(delta.z) * delta.z;
            if(r2 == 0.0L)
                continue;
            const long double invR = 1.0L / std::sqrt(r2);
            const long double factor =
                static_cast<long double>(sourceMass) * invR * invR * invR;
            ax -= factor * delta.x;
            ay -= factor * delta.y;
            az -= factor * delta.z;
            potential += static_cast<long double>(sourceMass) * invR;""",
        "compact particle decode",
    )
    t = replace_once(
        t,
        """    const Clock::time_point compactionStart = Clock::now();
    std::set<FmmRemoteNodeKey> retainedDescriptors;
    for(const PendingInteraction& interaction : terminalM2L)
        retainedDescriptors.insert(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(const PendingInteraction& interaction : terminalP2P)
        retainedDescriptors.insert(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(const PendingInteraction& interaction : terminalM2P)
        retainedDescriptors.insert(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});""",
        """    const Clock::time_point compactionStart = Clock::now();
    const std::size_t terminalCount = checkedAdd(
        checkedAdd(terminalM2L.size(), terminalP2P.size(),
                   "FmmPatchLetPlan::build: terminal count overflow"),
        terminalM2P.size(),
        "FmmPatchLetPlan::build: terminal count overflow");
    std::vector<FmmRemoteNodeKey> retainedDescriptors;
    retainedDescriptors.reserve(terminalCount);
    for(const PendingInteraction& interaction : terminalM2L)
        retainedDescriptors.push_back(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(const PendingInteraction& interaction : terminalP2P)
        retainedDescriptors.push_back(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(const PendingInteraction& interaction : terminalM2P)
        retainedDescriptors.push_back(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    std::sort(retainedDescriptors.begin(), retainedDescriptors.end());
    retainedDescriptors.erase(
        std::unique(retainedDescriptors.begin(), retainedDescriptors.end()),
        retainedDescriptors.end());""",
        "flat retained-descriptor construction",
    )
    t = replace_once(
        t,
        """            if(retainedDescriptors.count(FmmRemoteNodeKey{
                   patchIt->first, nodeIt->first}) == 0)""",
        """            if(!std::binary_search(
                   retainedDescriptors.begin(), retainedDescriptors.end(),
                   FmmRemoteNodeKey{patchIt->first, nodeIt->first}))""",
        "flat retained-descriptor lookup",
    )
    t = replace_once(
        t,
        """    std::vector<SourceIdentity> uniqueSources;
    const std::size_t terminalCount = checkedAdd(
        checkedAdd(terminalM2L.size(), terminalP2P.size(),
                   "FmmPatchLetPlan::build: terminal count overflow"),
        terminalM2P.size(),
        "FmmPatchLetPlan::build: terminal count overflow");
    uniqueSources.reserve(terminalCount);""",
        """    std::vector<SourceIdentity> uniqueSources;
    uniqueSources.reserve(terminalCount);""",
        "move terminal-count calculation",
    )
    t = replace_once(
        t,
        """                const std::size_t body = patch.tree.particleOrder()[k];
                FmmPatchWireParticle particle;
                particle.position[0] = patch.positions[body].x;
                particle.position[1] = patch.positions[body].y;
                particle.position[2] = patch.positions[body].z;
                particle.mass = patch.masses[body];
                FmmPacketIO::appendPod(buffer, particle);""",
        """                const std::size_t body = patch.tree.particleOrder()[k];
                const Vector3D offset = patch.positions[body] - node.center;
                const double tolerance =
                    64.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, node.halfSize);
                if(!(node.halfSize > 0.0) ||
                   !std::isfinite(node.halfSize) ||
                   !finiteVector(offset) ||
                   std::abs(offset.x) > node.halfSize + tolerance ||
                   std::abs(offset.y) > node.halfSize + tolerance ||
                   std::abs(offset.z) > node.halfSize + tolerance ||
                   !std::isfinite(patch.masses[body]))
                    throw UniversalError(
                        "FmmPatchLetPlan::executeWave: particle outside compact source leaf");
                const auto quantizeOffset = [&](double value) {
                    const double normalized = std::max(
                        -1.0, std::min(1.0, value / node.halfSize));
                    const long quantized = std::lround(normalized * 32767.0);
                    return static_cast<std::int16_t>(quantized);
                };
                FmmPatchWireParticle particle;
                particle.offset[0] = quantizeOffset(offset.x);
                particle.offset[1] = quantizeOffset(offset.y);
                particle.offset[2] = quantizeOffset(offset.z);
                particle.mass = patch.masses[body];
                FmmPacketIO::appendPod(buffer, particle);""",
        "compact particle encode",
    )
    t = replace_once(
        t,
        """        addDirectRemoteParticles(targetPatch, targetNode,
                                 payload->view.data,
                                 payload->view.count,
                                 stats);""",
        """        addDirectRemoteParticles(targetPatch, source.sourceNode,
                                 targetNode,
                                 payload->view.data,
                                 payload->view.count,
                                 stats);""",
        "direct-particle call",
    )
    transformed[p] = t

    p = "source/3D/gravity/fmm/mpi/FmmDistributedOptions.hpp"
    t = transformed[p]
    t = replace_once(
        t,
        """    bool rebuildTopologyEverySolve = false;
    bool reuseInteractionPlansAcrossLeafCountChanges = true;""",
        """    bool rebuildTopologyEverySolve = false;
    // Emit the detailed fmm_solve_trace line without relying on a process
    // environment variable. The legacy RICH_FMM_TRACE switch remains supported
    // for existing runs.
    bool emitSolveTrace = false;
    bool reuseInteractionPlansAcrossLeafCountChanges = true;""",
        "trace option",
    )
    transformed[p] = t

    p = "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"
    t = transformed[p]
    t = replace_once(
        t,
        """private:
    double G_;
#ifdef RICH_MPI""",
        """private:
    double G_;
    bool traceEnabled_;
#ifdef RICH_MPI""",
        "trace member",
    )
    transformed[p] = t

    p = "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.cpp"
    t = transformed[p]
    t = replace_once(
        t,
        """void traceFmmSolve(const FmmSolveStats& stats)
{
    if(!fmmTraceEnabled())""",
        """void traceFmmSolve(const FmmSolveStats& stats, bool explicitlyEnabled)
{
    if(!explicitlyEnabled && !fmmTraceEnabled())""",
        "trace function",
    )
    t = replace_once(
        t,
        """#endif
    calculator_(validateAccelerationOptions(options))""",
        """#endif
    traceEnabled_(false),
    calculator_(validateAccelerationOptions(options))""",
        "default trace initialization",
    )
    t = replace_once(
        t,
        """    G_(validateDistributedGravityConstant(G)),
    calculator_(validateAccelerationOptions(options), distributedOptions)
{}""",
        """    G_(validateDistributedGravityConstant(G)),
    traceEnabled_(distributedOptions.emitSolveTrace),
    calculator_(validateAccelerationOptions(options), distributedOptions)
{}""",
        "distributed trace initialization",
    )
    t = replace_once(
        t,
        "traceFmmSolve(calculator_.stats());",
        "traceFmmSolve(calculator_.stats(), traceEnabled_);",
        "trace call",
    )
    transformed[p] = t

    p = "runs/M05R05MBH1e5ComptonGrayFrom148/test.cpp"
    t = transformed[p]
    t = replace_once(
        t,
        """\t// Leaf 16 made the persistent full-octant local trees exceed node memory on
\t// this restart.  Leaf 64 is the measured memory-safe operating point; remote
\t// near-field reduction is handled by M2P and the adaptive patch hierarchy.
\tfmmOptions.leafCapacity = 64;
\tfmmOptions.persistentRadiusSlackFactor = 1.02;
\tFmmDistributedOptions fmmDistributed;
\t// A fixed level-7 forest creates about 169k global process leaves for this
\t// snapshot.  Start one level coarser and split only patches containing more
\t// than 2048 particles.  Dense regions may still reach level 7, while sparse
\t// outer material no longer pays one process leaf and one set of LET metadata
\t// for every occupied level-7 cube.
\tfmmDistributed.minimumPatchLevel = 6;
\tfmmDistributed.maximumPatchLevel = 7;
\tfmmDistributed.targetParticlesPerPatch = 2048;""",
        """\t// Leaf 16 made the persistent full-octant local trees exceed node memory on
\t// this restart.  Leaf 64 is the measured memory-safe operating point; remote
\t// near-field reduction is handled by M2P with the fixed level-7 hierarchy.
\tfmmOptions.leafCapacity = 64;
\tfmmOptions.persistentRadiusSlackFactor = 1.02;
\tFmmDistributedOptions fmmDistributed;
\t// V5's adaptive level-6/7 forest reduced patch count but increased remote
\t// P2P blocks, traffic, waves, LET-plan storage, and total solve time. Restore
\t// the measured-better fixed level-7 geometry before changing the wire format.
\tfmmDistributed.minimumPatchLevel = 7;
\tfmmDistributed.maximumPatchLevel = 7;
\tfmmDistributed.targetParticlesPerPatch = 0;""",
        "fixed level-7 TDE configuration",
    )
    t = replace_once(
        t,
        """\tfmmDistributed.letParticlePayloadSlackFactor = 1.10;
\tfmmDistributed.letParticlePayloadSlackCount = 8;""",
        """\tfmmDistributed.letParticlePayloadSlackFactor = 1.10;
\tfmmDistributed.letParticlePayloadSlackCount = 8;
\tfmmDistributed.emitSolveTrace = true;""",
        "enable code trace",
    )
    t = replace_once(
        t,
        """\tfmmDistributed.maxTargetPatchesPerWave =
\t\tfmmDistributed.maxLocalPatchCount;
\tfmmDistributed.maxLetWaveBytes =
\t\tstatic_cast<std::size_t>(128) * 1024u * 1024u;""",
        """\tfmmDistributed.maxTargetPatchesPerWave =
\t\tfmmDistributed.maxLocalPatchCount;
\t// Compact 16-byte particles reduce the fixed-level-7 payload enough to use
\t// fewer, larger waves while remaining below the measured leaf-64 memory
\t// envelope.
\tfmmDistributed.maxLetWaveBytes =
\t\tstatic_cast<std::size_t>(256) * 1024u * 1024u;""",
        "TDE wave size",
    )
    transformed[p] = t

    p = "regression_tests/cases/fmm_patch_let_mpi/test.cpp"
    t = transformed[p]
    t = replace_once(
        t,
        """double solveError(DistributedFmmGravityCalculator& solver,
                  const std::vector<Body>& localBodies,
                  const std::vector<Body>& globalBodies,
                  FmmSolveStats& stats)""",
        """double solveError(DistributedFmmGravityCalculator& solver,
                  const std::vector<Body>& localBodies,
                  const std::vector<Body>& globalBodies,
                  FmmSolveStats& stats,
                  double& relativeAccelerationErrorSum,
                  unsigned long long& relativeAccelerationErrorCount)""",
        "regression solve signature",
    )
    t = replace_once(
        t,
        """        const Vector3D reference =
            directAcceleration(localBodies[i], globalBodies);
        maximum = std::max(maximum,
            norm(acceleration[i] - reference) /
            std::max(1.0, norm(reference)));""",
        """        const Vector3D reference =
            directAcceleration(localBodies[i], globalBodies);
        const double relativeAccelerationError =
            norm(acceleration[i] - reference) /
            std::max(1.0, norm(reference));
        maximum = std::max(maximum, relativeAccelerationError);
        relativeAccelerationErrorSum += relativeAccelerationError;
        ++relativeAccelerationErrorCount;""",
        "regression acceleration accumulation",
    )
    t = replace_once(
        t,
        """    DistributedFmmGravityCalculator solver(options, distributed);
    double localMaximumError = 0.0;
    std::size_t maximumWaveCount = 0;""",
        """    DistributedFmmGravityCalculator solver(options, distributed);
    double localMaximumError = 0.0;
    double localRelativeAccelerationErrorSum = 0.0;
    unsigned long long localRelativeAccelerationErrorCount = 0;
    std::size_t maximumWaveCount = 0;""",
        "regression local mean state",
    )
    t = replace_once(
        t,
        """        localMaximumError = std::max(
            localMaximumError, solveError(solver, local, global, stats));""",
        """        localMaximumError = std::max(
            localMaximumError, solveError(
                solver, local, global, stats,
                localRelativeAccelerationErrorSum,
                localRelativeAccelerationErrorCount));""",
        "regression solve call",
    )
    t = replace_once(
        t,
        """    double globalMaximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    unsigned long long localWaves =""",
        """    double globalMaximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double globalRelativeAccelerationErrorSum = 0.0;
    unsigned long long globalRelativeAccelerationErrorCount = 0;
    MPI_Allreduce(&localRelativeAccelerationErrorSum,
                  &globalRelativeAccelerationErrorSum, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localRelativeAccelerationErrorCount,
                  &globalRelativeAccelerationErrorCount, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    const double globalMeanRelativeAccelerationError =
        globalRelativeAccelerationErrorCount == 0 ? 0.0 :
        globalRelativeAccelerationErrorSum /
            static_cast<double>(globalRelativeAccelerationErrorCount);
    unsigned long long localWaves =""",
        "regression global mean reduction",
    )
    t = replace_once(
        t,
        """    const bool pass = globalMaximumError < 0.08 && globalMassPass != 0 &&
                      globalWaves > 1 &&""",
        """    const bool pass = globalMaximumError < 0.08 && globalMassPass != 0 &&
                      globalMeanRelativeAccelerationError < 0.01 &&
                      globalWaves > 1 &&""",
        "regression mean gate",
    )
    t = replace_once(
        t,
        """        output << "max_relative_error " << globalMaximumError << "\\n";
        output << "max_wave_count " << globalWaves << "\\n";""",
        """        output << "max_relative_error " << globalMaximumError << "\\n";
        output << "mean_relative_acceleration_error "
               << globalMeanRelativeAccelerationError << "\\n";
        output << "max_wave_count " << globalWaves << "\\n";""",
        "regression metrics output",
    )
    t = replace_once(
        t,
        """        std::cout << "fmm_patch_let_mpi pass=" << pass
                  << " error=" << globalMaximumError
                  << " waves=" << globalWaves << std::endl;""",
        """        std::cout << "fmm_patch_let_mpi pass=" << pass
                  << " error=" << globalMaximumError
                  << " mean_acceleration_error="
                  << globalMeanRelativeAccelerationError
                  << " waves=" << globalWaves << std::endl;""",
        "regression console output",
    )
    transformed[p] = t

    for relative, text in transformed.items():
        if text == originals[relative]:
            raise RuntimeError(f"no changes generated for {relative}")

    try:
        for relative, text in transformed.items():
            path = root / relative
            temporary = path.with_name(path.name + ".fmm_v6_tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)

        run("git", "diff", "--check", "--", *paths)

        required = {
            "source/3D/gravity/fmm/mpi/FmmPackets.hpp": [
                "FMM_MPI_PACKET_VERSION = 7u",
                "sizeof(FmmPatchWireParticle) == 16",
            ],
            "source/3D/gravity/fmm/mpi/FmmDistributedOptions.hpp": [
                "bool emitSolveTrace = false",
            ],
            "runs/M05R05MBH1e5ComptonGrayFrom148/test.cpp": [
                "fmmDistributed.minimumPatchLevel = 7",
                "fmmDistributed.maximumPatchLevel = 7",
                "fmmDistributed.targetParticlesPerPatch = 0",
                "fmmDistributed.emitSolveTrace = true",
                "static_cast<std::size_t>(256) * 1024u * 1024u",
            ],
            "regression_tests/cases/fmm_patch_let_mpi/test.cpp": [
                "mean_relative_acceleration_error",
                "globalMeanRelativeAccelerationError < 0.01",
            ],
        }
        for relative, needles in required.items():
            text = (root / relative).read_text(encoding="utf-8")
            for needle in needles:
                if needle not in text:
                    raise RuntimeError(f"postcondition missing in {relative}: {needle}")

        patch_path = root / OUTPUT_PATCH
        with patch_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                ["git", "diff", "--full-index", "--binary", "--", *paths],
                text=True,
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"failed to generate corrected patch: {result.stderr.strip()}")
        if patch_path.stat().st_size == 0:
            raise RuntimeError("generated corrected patch is empty")

        run("git", "apply", "--check", "--cached", str(patch_path))
        run("git", "apply", "--check", "--reverse", str(patch_path))
    except Exception:
        for relative, text in originals.items():
            (root / relative).write_text(text, encoding="utf-8")
        raise

    print(f"CORRECTED_PATCH {root / OUTPUT_PATCH}")
    print("CORRECTED_PATCH_FORWARD_CHECK_OK")
    print("CORRECTED_PATCH_REVERSE_CHECK_OK")
    print("TRANSFORM_APPLIED_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
