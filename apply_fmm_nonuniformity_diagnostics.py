#!/usr/bin/env python3
"""Add one-run nonuniformity diagnostics to the current V6-no-compact tree.

The transformation is intentionally applied on top of the user's current
working tree. It requires the known branch and base commit, but preserves all
existing uncommitted V6 changes.

Diagnostics are enabled at runtime with:
    RICH_FMM_NONUNIFORM_DIAGNOSTICS=1

They execute after stats.totalSeconds is finalized and therefore do not alter
fmm_solve_trace phase timings.
"""
from __future__ import annotations

import difflib
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_BRANCH = "feat/fmm-bounded-let-waves"
EXPECTED_HEAD = "616d5293c478c62b137946d7e02ccafd75963fcc"

PLAN_HPP = Path("source/3D/gravity/fmm/mpi/FmmPatchLetPlan.hpp")
PLAN_CPP = Path("source/3D/gravity/fmm/mpi/FmmPatchLetPlan.cpp")
SOLVER_CPP = Path("source/3D/gravity/fmm/mpi/FmmPatchDistributedSolver.cpp")
PACKETS_HPP = Path("source/3D/gravity/fmm/mpi/FmmPackets.hpp")
OUTPUT_PATCH = Path("fmm_nonuniformity_diagnostics_incremental.patch")
FULL_PATCH = Path("fmm_nonuniformity_diagnostics_full_source.patch")


def run(*args: str) -> str:
    result = subprocess.run(
        list(args), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n{detail}"
        )
    return (result.stdout or "").strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def make_patch(before: dict[Path, str], after: dict[Path, str]) -> str:
    pieces: list[str] = []
    for path in (PLAN_HPP, PLAN_CPP, SOLVER_CPP):
        old = before[path].splitlines(keepends=True)
        new = after[path].splitlines(keepends=True)
        body = list(difflib.unified_diff(
            old, new,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=5,
        ))
        if body:
            pieces.append(f"diff --git a/{path} b/{path}\n")
            pieces.extend(body)
    return "".join(pieces)


def main() -> int:
    root = Path(run("git", "rev-parse", "--show-toplevel"))
    os.chdir(root)

    branch = run("git", "branch", "--show-current")
    head = run("git", "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"wrong branch: expected {EXPECTED_BRANCH}, found {branch or '<detached>'}"
        )
    if head != EXPECTED_HEAD:
        raise RuntimeError(
            f"wrong HEAD: expected {EXPECTED_HEAD}, found {head}"
        )

    before = {
        PLAN_HPP: PLAN_HPP.read_text(encoding="utf-8"),
        PLAN_CPP: PLAN_CPP.read_text(encoding="utf-8"),
        SOLVER_CPP: SOLVER_CPP.read_text(encoding="utf-8"),
    }

    for path, text in before.items():
        if "fmm_nonuniform_" in text or "emitNonuniformityDiagnostics" in text:
            raise RuntimeError(f"diagnostics already appear in {path}")

    # Confirm the expected V6-no-compact working tree before editing.
    combined = "\n".join(before.values())
    required = (
        "FmmPatchLetPlan::execute(",
        "stats.totalSeconds = elapsed(totalStart);",
        "std::size_t bytesOwned() const;",
    )
    for token in required:
        if token not in combined:
            raise RuntimeError(f"required source token missing: {token}")

    packets = PACKETS_HPP.read_text(encoding="utf-8")
    packet_required = (
        "static constexpr std::uint16_t FMM_MPI_PACKET_VERSION = 6u;",
        "struct FmmPatchWireParticle",
        "sizeof(FmmPatchWireParticle) == 32",
    )
    for token in packet_required:
        if token not in packets:
            raise RuntimeError(
                f"expected V6-no-compact packet token missing from {PACKETS_HPP}: {token}"
            )

    hpp = before[PLAN_HPP]
    cpp = before[PLAN_CPP]
    solver = before[SOLVER_CPP]

    hpp = replace_once(
        hpp,
        """    std::size_t bytesOwned() const;
""",
        """    // Emits exact source-fanout and payload-duplication diagnostics.
    // The method is collective on the LET communicator and is a no-op unless
    // RICH_FMM_NONUNIFORM_DIAGNOSTICS is enabled.
    void emitNonuniformityDiagnostics(const FmmPatchForest& forest,
                                      std::uint64_t call) const;

    std::size_t bytesOwned() const;
""",
        "LET diagnostic declaration",
    )

    cpp = replace_once(
        cpp,
        """#include <cmath>
#include <cstring>
#include <limits>
""",
        """#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
""",
        "LET diagnostic includes",
    )

    cpp = replace_once(
        cpp,
        """double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

""",
        """double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool nonuniformDiagnosticsEnabled()
{
    const char* value = std::getenv("RICH_FMM_NONUNIFORM_DIAGNOSTICS");
    return value != nullptr && value[0] != '\\0' &&
        !(value[0] == '0' && value[1] == '\\0');
}

""",
        "LET diagnostic enable helper",
    )

    let_method = r'''void FmmPatchLetPlan::emitNonuniformityDiagnostics(
    const FmmPatchForest& forest,
    std::uint64_t call) const
{
    if(!nonuniformDiagnosticsEnabled())
        return;

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    typedef std::tuple<std::uint64_t, std::uint64_t, int> DiagnosticSource;
    std::map<DiagnosticSource, std::size_t> fanoutBySource;

    unsigned long long localTotals[10] = {};
    // 0 all subscriptions, 1 particle subscriptions, 2 multipole subscriptions,
    // 3 source records, 4 particle source records, 5 multipole source records,
    // 6 record-header bytes, 7 particle payload bytes,
    // 8 multipole payload bytes, 9 complete estimated payload bytes.
    for(const auto& peerEntry : subscriptionsReceived_)
    {
        for(const FmmSubscription& subscription : peerEntry.second)
        {
            ++localTotals[0];
            if(subscription.kind ==
               static_cast<int>(FmmSubscriptionKind::Particles))
                ++localTotals[1];
            else
                ++localTotals[2];

            ++fanoutBySource[DiagnosticSource(
                subscription.patchId, subscription.spatialKey,
                subscription.kind)];

            const std::size_t patchIndex = forest.findPatch(
                subscription.patchId);
            if(patchIndex == std::numeric_limits<std::size_t>::max() ||
               patchIndex >= localNodeByPatch_.size())
                throw UniversalError(
                    "FmmPatchLetPlan diagnostics: missing subscribed patch");
            const auto nodeIt = localNodeByPatch_[patchIndex].find(
                subscription.spatialKey);
            if(nodeIt == localNodeByPatch_[patchIndex].end() ||
               nodeIt->second >= forest.patches()[patchIndex].tree.nodes().size())
                throw UniversalError(
                    "FmmPatchLetPlan diagnostics: missing subscribed node");
            const FmmNode& node = forest.patches()[patchIndex].tree.nodes()[
                nodeIt->second];

            const unsigned long long headerBytes =
                static_cast<unsigned long long>(sizeof(FmmPayloadRecordHeader));
            localTotals[6] += headerBytes;
            unsigned long long payloadBytes = 0;
            if(subscription.kind ==
               static_cast<int>(FmmSubscriptionKind::Particles))
            {
                payloadBytes = static_cast<unsigned long long>(
                    node.particleCount()) * static_cast<unsigned long long>(
                        sizeof(FmmPatchWireParticle));
                localTotals[7] += payloadBytes;
            }
            else if(subscription.kind ==
                    static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                payloadBytes = static_cast<unsigned long long>(
                    multipoleCoefficientCount_) *
                    static_cast<unsigned long long>(sizeof(double));
                localTotals[8] += payloadBytes;
            }
            else
            {
                throw UniversalError(
                    "FmmPatchLetPlan diagnostics: invalid subscription kind");
            }
            localTotals[9] += headerBytes + payloadBytes;
        }
    }

    std::vector<unsigned long long> localHistogram(
        static_cast<std::size_t>(size) + 1, 0);
    for(const auto& entry : fanoutBySource)
    {
        const std::size_t fanout = entry.second;
        if(fanout == 0 || fanout > static_cast<std::size_t>(size))
            throw UniversalError(
                "FmmPatchLetPlan diagnostics: invalid source fanout");
        ++localHistogram[fanout];
        ++localTotals[3];
        if(std::get<2>(entry.first) ==
           static_cast<int>(FmmSubscriptionKind::Particles))
            ++localTotals[4];
        else
            ++localTotals[5];
    }

    unsigned long long globalTotals[10] = {};
    std::vector<unsigned long long> globalHistogram(
        static_cast<std::size_t>(size) + 1, 0);
    MPI_Reduce(localTotals, globalTotals, 10, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, comm_);
    MPI_Reduce(localHistogram.data(), globalHistogram.data(), size + 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm_);

    const unsigned long long localRankValues[4] = {
        static_cast<unsigned long long>(subscriptionsReceived_.size()),
        localTotals[0], localTotals[7], localTotals[8]};
    std::vector<unsigned long long> gatheredRankValues;
    if(rank == 0)
        gatheredRankValues.resize(static_cast<std::size_t>(size) * 4);
    MPI_Gather(localRankValues, 4, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? gatheredRankValues.data() : nullptr, 4,
               MPI_UNSIGNED_LONG_LONG, 0, comm_);

    if(rank != 0)
        return;

    const auto histogramPercentile = [&](double fraction) {
        unsigned long long total = 0;
        for(unsigned long long value : globalHistogram)
            total += value;
        if(total == 0)
            return std::size_t(0);
        const unsigned long long threshold = static_cast<unsigned long long>(
            std::ceil(fraction * static_cast<double>(total)));
        unsigned long long running = 0;
        for(std::size_t i = 0; i < globalHistogram.size(); ++i)
        {
            running += globalHistogram[i];
            if(running >= std::max<unsigned long long>(1, threshold))
                return i;
        }
        return globalHistogram.size() - 1;
    };
    const auto gatheredPercentile = [&](int column, double fraction) {
        std::vector<unsigned long long> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredRankValues[
                static_cast<std::size_t>(4 * r + column)]);
        std::sort(values.begin(), values.end());
        const std::size_t index = std::min(
            values.size() - 1, static_cast<std::size_t>(
                std::ceil(fraction * static_cast<double>(values.size()))) - 1);
        return values[index];
    };

    std::fprintf(stdout,
        "fmm_nonuniform_let call=%llu source_records=%llu "
        "particle_sources=%llu multipole_sources=%llu subscriptions=%llu "
        "particle_subscriptions=%llu multipole_subscriptions=%llu "
        "fanout_p50=%zu fanout_p95=%zu fanout_p99=%zu fanout_max=%zu "
        "requester_peers_p50=%llu requester_peers_p95=%llu "
        "requester_peers_p99=%llu requester_peers_max=%llu "
        "header_bytes=%llu particle_payload_bytes=%llu "
        "multipole_payload_bytes=%llu estimated_payload_bytes=%llu\n",
        static_cast<unsigned long long>(call),
        globalTotals[3], globalTotals[4], globalTotals[5], globalTotals[0],
        globalTotals[1], globalTotals[2],
        histogramPercentile(0.50), histogramPercentile(0.95),
        histogramPercentile(0.99), histogramPercentile(1.0),
        gatheredPercentile(0, 0.50), gatheredPercentile(0, 0.95),
        gatheredPercentile(0, 0.99), gatheredPercentile(0, 1.0),
        globalTotals[6], globalTotals[7], globalTotals[8], globalTotals[9]);
    std::fflush(stdout);
}

'''

    cpp = replace_once(
        cpp,
        """std::size_t FmmPatchLetPlan::bytesOwned() const
""",
        let_method + "std::size_t FmmPatchLetPlan::bytesOwned() const\n",
        "LET diagnostic implementation",
    )

    solver = replace_once(
        solver,
        """#include <cmath>
#include <limits>
#include <set>
""",
        """#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
""",
        "solver diagnostic includes",
    )

    solver = replace_once(
        solver,
        """#include "3D/gravity/fmm/mpi/FmmDescriptorGather.hpp"
""",
        """#include "3D/gravity/fmm/mpi/FmmDescriptorGather.hpp"
#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
""",
        "solver lattice include",
    )

    solver_helper = r'''
bool nonuniformDiagnosticsEnabled()
{
    const char* value = std::getenv("RICH_FMM_NONUNIFORM_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' &&
        !(value[0] == '0' && value[1] == '\0');
}

template<typename T>
T diagnosticPercentile(std::vector<T> values, double fraction)
{
    if(values.empty())
        return T();
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1, static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(values.size()))) - 1);
    return values[index];
}

template<typename T>
long double diagnosticMean(const std::vector<T>& values)
{
    if(values.empty())
        return 0.0L;
    long double sum = 0.0L;
    for(const T& value : values)
        sum += static_cast<long double>(value);
    return sum / static_cast<long double>(values.size());
}

void printUnsignedRankMetric(std::uint64_t call,
                             const char* name,
                             const std::vector<unsigned long long>& values)
{
    if(values.empty())
        return;
    const unsigned long long minimum = *std::min_element(
        values.begin(), values.end());
    const unsigned long long maximum = *std::max_element(
        values.begin(), values.end());
    const long double mean = diagnosticMean(values);
    std::fprintf(stdout,
        "fmm_nonuniform_rank_metric call=%llu metric=%s min=%llu "
        "mean=%.9Le p50=%llu p95=%llu p99=%llu max=%llu "
        "max_over_mean=%.9Le\n",
        static_cast<unsigned long long>(call), name, minimum, mean,
        diagnosticPercentile(values, 0.50),
        diagnosticPercentile(values, 0.95),
        diagnosticPercentile(values, 0.99), maximum,
        mean > 0.0L ? static_cast<long double>(maximum) / mean : 0.0L);
}

void printDoubleRankMetric(std::uint64_t call,
                           const char* name,
                           const std::vector<double>& values)
{
    if(values.empty())
        return;
    const double minimum = *std::min_element(values.begin(), values.end());
    const double maximum = *std::max_element(values.begin(), values.end());
    const long double mean = diagnosticMean(values);
    std::fprintf(stdout,
        "fmm_nonuniform_rank_metric call=%llu metric=%s min=%.9e "
        "mean=%.9Le p50=%.9e p95=%.9e p99=%.9e max=%.9e "
        "max_over_mean=%.9Le\n",
        static_cast<unsigned long long>(call), name, minimum, mean,
        diagnosticPercentile(values, 0.50),
        diagnosticPercentile(values, 0.95),
        diagnosticPercentile(values, 0.99), maximum,
        mean > 0.0L ? static_cast<long double>(maximum) / mean : 0.0L);
}

std::uint64_t diagnosticCellKey(int ix, int iy, int iz)
{
    return (static_cast<std::uint64_t>(ix) << 42u) |
           (static_cast<std::uint64_t>(iy) << 21u) |
           static_cast<std::uint64_t>(iz);
}

void emitRankGeometryDiagnostics(const std::vector<Vector3D>& positions,
                                 const FmmGlobalDyadicLattice& lattice,
                                 int level,
                                 std::uint64_t call,
                                 int rank,
                                 int size,
                                 const MPI_Comm& comm)
{
    std::unordered_set<std::uint64_t> occupied;
    int minIndex[3] = {std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max()};
    int maxIndex[3] = {-1, -1, -1};
    for(const Vector3D& point : positions)
    {
        int ix = 0;
        int iy = 0;
        int iz = 0;
        lattice.pointToCellIndices(point, level, ix, iy, iz);
        occupied.insert(diagnosticCellKey(ix, iy, iz));
        minIndex[0] = std::min(minIndex[0], ix);
        minIndex[1] = std::min(minIndex[1], iy);
        minIndex[2] = std::min(minIndex[2], iz);
        maxIndex[0] = std::max(maxIndex[0], ix);
        maxIndex[1] = std::max(maxIndex[1], iy);
        maxIndex[2] = std::max(maxIndex[2], iz);
    }

    std::unordered_set<std::uint64_t> remaining = occupied;
    std::size_t components = 0;
    const int axisCells = 1 << level;
    std::vector<std::uint64_t> stack;
    while(!remaining.empty())
    {
        ++components;
        stack.clear();
        stack.push_back(*remaining.begin());
        remaining.erase(stack.back());
        while(!stack.empty())
        {
            const std::uint64_t key = stack.back();
            stack.pop_back();
            const int ix = static_cast<int>((key >> 42u) & 0x1fffffu);
            const int iy = static_cast<int>((key >> 21u) & 0x1fffffu);
            const int iz = static_cast<int>(key & 0x1fffffu);
            const int neighbor[6][3] = {
                {ix - 1, iy, iz}, {ix + 1, iy, iz},
                {ix, iy - 1, iz}, {ix, iy + 1, iz},
                {ix, iy, iz - 1}, {ix, iy, iz + 1}};
            for(const auto& item : neighbor)
            {
                if(item[0] < 0 || item[0] >= axisCells ||
                   item[1] < 0 || item[1] >= axisCells ||
                   item[2] < 0 || item[2] >= axisCells)
                    continue;
                const std::uint64_t neighborKey = diagnosticCellKey(
                    item[0], item[1], item[2]);
                const auto found = remaining.find(neighborKey);
                if(found != remaining.end())
                {
                    stack.push_back(neighborKey);
                    remaining.erase(found);
                }
            }
        }
    }

    unsigned long long boundingCells = 0;
    if(!occupied.empty())
    {
        const unsigned long long nx = static_cast<unsigned long long>(
            maxIndex[0] - minIndex[0] + 1);
        const unsigned long long ny = static_cast<unsigned long long>(
            maxIndex[1] - minIndex[1] + 1);
        const unsigned long long nz = static_cast<unsigned long long>(
            maxIndex[2] - minIndex[2] + 1);
        boundingCells = nx * ny * nz;
    }
    const double fill = boundingCells == 0 ? 0.0 :
        static_cast<double>(occupied.size()) /
        static_cast<double>(boundingCells);

    const unsigned long long localUnsigned[4] = {
        static_cast<unsigned long long>(positions.size()),
        static_cast<unsigned long long>(occupied.size()),
        static_cast<unsigned long long>(components), boundingCells};
    const double localDouble[1] = {fill};
    std::vector<unsigned long long> gatheredUnsigned;
    std::vector<double> gatheredDouble;
    if(rank == 0)
    {
        gatheredUnsigned.resize(static_cast<std::size_t>(size) * 4);
        gatheredDouble.resize(static_cast<std::size_t>(size));
    }
    MPI_Gather(localUnsigned, 4, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? gatheredUnsigned.data() : nullptr, 4,
               MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Gather(localDouble, 1, MPI_DOUBLE,
               rank == 0 ? gatheredDouble.data() : nullptr, 1,
               MPI_DOUBLE, 0, comm);
    if(rank != 0)
        return;

    const char* names[4] = {
        "rank_particles", "rank_occupied_spatial_cells",
        "rank_spatial_components", "rank_spatial_bbox_cells"};
    for(int metric = 0; metric < 4; ++metric)
    {
        std::vector<unsigned long long> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredUnsigned[
                static_cast<std::size_t>(4 * r + metric)]);
        printUnsignedRankMetric(call, names[metric], values);
    }
    printDoubleRankMetric(call, "rank_spatial_fill_fraction", gatheredDouble);
}

void emitSpatialDiagnostics(const std::vector<Vector3D>& positions,
                            const Vector3D& domainLower,
                            const Vector3D& domainUpper,
                            int diagnosticLevel,
                            std::uint64_t call,
                            int rank,
                            int size,
                            const MPI_Comm& comm)
{
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(domainLower, domainUpper);
    constexpr unsigned long long kMaximumGatheredEntries = 5000000ull;
    std::map<int, unsigned long long> uniqueByLevel;

    const int firstLevel = std::max(4, diagnosticLevel - 2);
    const int lastLevel = std::min(FMM_MAX_TREE_DEPTH,
                                   diagnosticLevel + 3);
    for(int level = firstLevel; level <= lastLevel; ++level)
    {
        std::map<std::uint64_t, unsigned long long> localCounts;
        for(const Vector3D& point : positions)
            ++localCounts[lattice.patchIdAtLevel(point, level)];

        const unsigned long long localEntryCount =
            static_cast<unsigned long long>(localCounts.size());
        std::vector<unsigned long long> entryCounts;
        if(rank == 0)
            entryCounts.resize(static_cast<std::size_t>(size));
        MPI_Gather(&localEntryCount, 1, MPI_UNSIGNED_LONG_LONG,
                   rank == 0 ? entryCounts.data() : nullptr, 1,
                   MPI_UNSIGNED_LONG_LONG, 0, comm);

        int exact = 1;
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        unsigned long long ownerTagged = 0;
        if(rank == 0)
        {
            receiveCounts.resize(static_cast<std::size_t>(size));
            displacements.resize(static_cast<std::size_t>(size));
            unsigned long long packedOffset = 0;
            for(int r = 0; r < size; ++r)
            {
                ownerTagged += entryCounts[static_cast<std::size_t>(r)];
                const unsigned long long packed =
                    2ull * entryCounts[static_cast<std::size_t>(r)];
                if(entryCounts[static_cast<std::size_t>(r)] >
                       kMaximumGatheredEntries ||
                   packed > static_cast<unsigned long long>(
                       std::numeric_limits<int>::max()) ||
                   packedOffset + packed > static_cast<unsigned long long>(
                       std::numeric_limits<int>::max()))
                    exact = 0;
                if(exact != 0)
                {
                    receiveCounts[static_cast<std::size_t>(r)] =
                        static_cast<int>(packed);
                    displacements[static_cast<std::size_t>(r)] =
                        static_cast<int>(packedOffset);
                }
                packedOffset += packed;
            }
            if(ownerTagged > kMaximumGatheredEntries)
                exact = 0;
        }
        MPI_Bcast(&exact, 1, MPI_INT, 0, comm);
        if(exact == 0)
        {
            if(rank == 0)
            {
                std::fprintf(stdout,
                    "fmm_nonuniform_spatial call=%llu level=%d "
                    "owner_tagged=%llu exact=0 reason=gather_cap\n",
                    static_cast<unsigned long long>(call), level, ownerTagged);
                std::fflush(stdout);
            }
            continue;
        }

        std::vector<unsigned long long> localPacked;
        localPacked.reserve(2 * localCounts.size());
        for(const auto& entry : localCounts)
        {
            localPacked.push_back(static_cast<unsigned long long>(entry.first));
            localPacked.push_back(entry.second);
        }
        std::vector<unsigned long long> gathered;
        if(rank == 0)
            gathered.resize(static_cast<std::size_t>(2ull * ownerTagged));
        MPI_Gatherv(localPacked.data(), static_cast<int>(localPacked.size()),
                    MPI_UNSIGNED_LONG_LONG,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_UNSIGNED_LONG_LONG, 0, comm);
        if(rank != 0)
            continue;

        std::vector<std::pair<unsigned long long, unsigned long long>> entries;
        entries.reserve(static_cast<std::size_t>(ownerTagged));
        for(std::size_t i = 0; i < gathered.size(); i += 2)
            entries.emplace_back(gathered[i], gathered[i + 1]);
        std::sort(entries.begin(), entries.end());

        std::vector<unsigned long long> multiplicities;
        std::vector<unsigned long long> particlesPerPatch;
        std::vector<double> dominantFractions;
        unsigned long long totalParticles = 0;
        unsigned long long sharedParticles = 0;
        unsigned long long sharedPatches = 0;
        long double crossOwnerPairs = 0.0L;
        std::size_t cursor = 0;
        while(cursor < entries.size())
        {
            const unsigned long long id = entries[cursor].first;
            unsigned long long total = 0;
            unsigned long long largest = 0;
            unsigned long long localPairs = 0;
            unsigned long long multiplicity = 0;
            while(cursor < entries.size() && entries[cursor].first == id)
            {
                const unsigned long long count = entries[cursor].second;
                total += count;
                largest = std::max(largest, count);
                localPairs += count * (count - 1ull) / 2ull;
                ++multiplicity;
                ++cursor;
            }
            multiplicities.push_back(multiplicity);
            particlesPerPatch.push_back(total);
            dominantFractions.push_back(total == 0 ? 0.0 :
                static_cast<double>(largest) / static_cast<double>(total));
            totalParticles += total;
            if(multiplicity > 1)
            {
                ++sharedPatches;
                sharedParticles += total;
            }
            const long double allPairs = static_cast<long double>(total) *
                static_cast<long double>(total - 1ull) / 2.0L;
            crossOwnerPairs += allPairs - static_cast<long double>(localPairs);
        }

        const unsigned long long unique =
            static_cast<unsigned long long>(multiplicities.size());
        uniqueByLevel[level] = unique;
        const double duplication = unique == 0 ? 0.0 :
            static_cast<double>(ownerTagged) / static_cast<double>(unique);
        const double sharedPatchFraction = unique == 0 ? 0.0 :
            static_cast<double>(sharedPatches) / static_cast<double>(unique);
        const double sharedParticleFraction = totalParticles == 0 ? 0.0 :
            static_cast<double>(sharedParticles) /
            static_cast<double>(totalParticles);

        std::fprintf(stdout,
            "fmm_nonuniform_spatial call=%llu level=%d exact=1 "
            "owner_tagged=%llu unique=%llu duplication=%.9e "
            "shared_patch_fraction=%.9e shared_particle_fraction=%.9e "
            "owner_mult_p50=%llu owner_mult_p95=%llu owner_mult_p99=%llu "
            "owner_mult_max=%llu particles_p50=%llu particles_p95=%llu "
            "particles_p99=%llu particles_max=%llu "
            "dominant_owner_fraction_p05=%.9e "
            "dominant_owner_fraction_p50=%.9e "
            "cross_owner_same_patch_pairs=%.9Le\n",
            static_cast<unsigned long long>(call), level, ownerTagged, unique,
            duplication, sharedPatchFraction, sharedParticleFraction,
            diagnosticPercentile(multiplicities, 0.50),
            diagnosticPercentile(multiplicities, 0.95),
            diagnosticPercentile(multiplicities, 0.99),
            diagnosticPercentile(multiplicities, 1.0),
            diagnosticPercentile(particlesPerPatch, 0.50),
            diagnosticPercentile(particlesPerPatch, 0.95),
            diagnosticPercentile(particlesPerPatch, 0.99),
            diagnosticPercentile(particlesPerPatch, 1.0),
            diagnosticPercentile(dominantFractions, 0.05),
            diagnosticPercentile(dominantFractions, 0.50),
            crossOwnerPairs);
        std::fflush(stdout);
    }

    if(rank == 0)
    {
        for(auto current = uniqueByLevel.begin(); current != uniqueByLevel.end();
            ++current)
        {
            auto next = current;
            ++next;
            if(next == uniqueByLevel.end() || current->second == 0 ||
               next->second == 0)
                continue;
            const double dimension = std::log(
                static_cast<double>(next->second) /
                static_cast<double>(current->second)) / std::log(2.0);
            std::fprintf(stdout,
                "fmm_nonuniform_dimension call=%llu level_lo=%d level_hi=%d "
                "unique_lo=%llu unique_hi=%llu effective_dimension=%.9e\n",
                static_cast<unsigned long long>(call), current->first,
                next->first, current->second, next->second, dimension);
        }
        std::fflush(stdout);
    }

    emitRankGeometryDiagnostics(positions, lattice, diagnosticLevel, call,
                                rank, size, comm);
}

void emitRankWorkDiagnostics(const FmmSolveStats& stats,
                             std::uint64_t call,
                             int rank,
                             int size,
                             const MPI_Comm& comm)
{
    constexpr int kUnsignedCount = 12;
    const unsigned long long localUnsigned[kUnsignedCount] = {
        static_cast<unsigned long long>(stats.particleCount),
        static_cast<unsigned long long>(stats.localPatchCount),
        static_cast<unsigned long long>(stats.processOwnedNodeCount),
        static_cast<unsigned long long>(stats.processOwnedM2LCount),
        static_cast<unsigned long long>(stats.localPlannedP2PBlockCount),
        static_cast<unsigned long long>(stats.letPlannedP2PBlockCount),
        static_cast<unsigned long long>(stats.letP2PBlockCount),
        static_cast<unsigned long long>(stats.letM2PCount),
        static_cast<unsigned long long>(stats.bytesSent),
        static_cast<unsigned long long>(stats.bytesReceived),
        static_cast<unsigned long long>(stats.peakRemoteBytes),
        static_cast<unsigned long long>(stats.letPlanBytes)};
    constexpr int kDoubleCount = 10;
    const double localDouble[kDoubleCount] = {
        stats.totalSeconds, stats.topologyRebuildSeconds, stats.letPlanSeconds,
        stats.letExecuteSeconds, stats.letExchangeSeconds, stats.letP2PSeconds,
        stats.letM2PSeconds, stats.localTraversalSeconds,
        stats.processInteractionSeconds, stats.processDownwardSeconds};

    std::vector<unsigned long long> gatheredUnsigned;
    std::vector<double> gatheredDouble;
    if(rank == 0)
    {
        gatheredUnsigned.resize(
            static_cast<std::size_t>(size) * kUnsignedCount);
        gatheredDouble.resize(static_cast<std::size_t>(size) * kDoubleCount);
    }
    MPI_Gather(localUnsigned, kUnsignedCount, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? gatheredUnsigned.data() : nullptr,
               kUnsignedCount, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Gather(localDouble, kDoubleCount, MPI_DOUBLE,
               rank == 0 ? gatheredDouble.data() : nullptr,
               kDoubleCount, MPI_DOUBLE, 0, comm);
    if(rank != 0)
        return;

    const char* unsignedNames[kUnsignedCount] = {
        "particles", "patches", "process_owned_nodes",
        "process_owned_m2l", "local_planned_p2p_blocks",
        "let_planned_p2p_blocks", "let_active_p2p_blocks", "let_m2p",
        "bytes_sent", "bytes_received", "peak_remote_bytes",
        "let_plan_bytes"};
    for(int metric = 0; metric < kUnsignedCount; ++metric)
    {
        std::vector<unsigned long long> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredUnsigned[static_cast<std::size_t>(
                kUnsignedCount * r + metric)]);
        printUnsignedRankMetric(call, unsignedNames[metric], values);
    }

    const char* doubleNames[kDoubleCount] = {
        "total_seconds", "topology_seconds", "let_plan_seconds",
        "let_execute_seconds", "let_exchange_seconds", "let_p2p_seconds",
        "let_m2p_seconds", "local_traversal_seconds",
        "process_interaction_seconds", "process_downward_seconds"};
    for(int metric = 0; metric < kDoubleCount; ++metric)
    {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredDouble[static_cast<std::size_t>(
                kDoubleCount * r + metric)]);
        printDoubleRankMetric(call, doubleNames[metric], values);
    }
    std::fflush(stdout);
}

'''

    solver = replace_once(
        solver,
        """std::size_t saturatingMultiply(std::size_t first, std::size_t second)
{
    return first != 0 && second >
        std::numeric_limits<std::size_t>::max() / first ?
        std::numeric_limits<std::size_t>::max() : first * second;
}

""",
        """std::size_t saturatingMultiply(std::size_t first, std::size_t second)
{
    return first != 0 && second >
        std::numeric_limits<std::size_t>::max() / first ?
        std::numeric_limits<std::size_t>::max() : first * second;
}
""" + solver_helper,
        "solver diagnostic helpers",
    )

    solver = replace_once(
        solver,
        """    stats.totalSeconds = elapsed(totalStart);
}
""",
        """    stats.totalSeconds = elapsed(totalStart);

    // Keep diagnostic collectives outside the measured solve.  The resulting
    // fmm_nonuniform_* lines can therefore be compared with the ordinary
    // fmm_solve_trace without subtracting diagnostic overhead.
    if(nonuniformDiagnosticsEnabled())
    {
        static std::uint64_t diagnosticCall = 0;
        ++diagnosticCall;
        emitSpatialDiagnostics(
            positions, domainLower, domainUpper,
            distributedOptions_.minimumPatchLevel, diagnosticCall,
            rank_, size_, comm_);
        emitRankWorkDiagnostics(stats, diagnosticCall, rank_, size_, comm_);
        letPlan_.emitNonuniformityDiagnostics(forest_, diagnosticCall);
    }
}
""",
        "solver diagnostic call",
    )

    after = {PLAN_HPP: hpp, PLAN_CPP: cpp, SOLVER_CPP: solver}

    incremental = make_patch(before, after)
    if not incremental.strip():
        raise RuntimeError("generated diagnostic patch is empty")

    # Save the pre-diagnostic state of just the touched files.
    backup = Path("fmm_before_nonuniformity_diagnostics.patch")
    backup.write_text(run("git", "diff", "--", *(str(p) for p in before)) + "\n",
                      encoding="utf-8")

    for path, text in after.items():
        path.write_text(text, encoding="utf-8")

    run("git", "diff", "--check")
    OUTPUT_PATCH.write_text(incremental, encoding="utf-8")
    FULL_PATCH.write_text(run("git", "diff") + "\n", encoding="utf-8")

    # Reverse check proves that the incremental patch exactly describes the
    # edit relative to the current V6-no-compact working tree.
    run("git", "apply", "--check", "--reverse", str(OUTPUT_PATCH))

    print("NONUNIFORMITY_DIAGNOSTICS_APPLIED_OK")
    print(f"INCREMENTAL_PATCH {root / OUTPUT_PATCH}")
    print(f"FULL_SOURCE_PATCH {root / FULL_PATCH}")
    print(f"PRE_DIAGNOSTIC_BACKUP {root / backup}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
