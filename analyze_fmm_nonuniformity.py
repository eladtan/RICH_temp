#!/usr/bin/env python3
"""Parse fmm_nonuniform_* diagnostics and produce an architecture decision."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


def parse_value(text: str):
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_lines(path: Path):
    records = []
    for raw in path.read_text(errors="replace").splitlines():
        if not raw.startswith("fmm_nonuniform_"):
            continue
        fields = raw.split()
        kind = fields[0]
        values = {"kind": kind, "raw": raw}
        for token in fields[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            values[key] = parse_value(value)
        records.append(values)
    return records


def latest_call(records):
    calls = [int(r["call"]) for r in records if "call" in r]
    if not calls:
        raise RuntimeError("no fmm_nonuniform_* records found")
    return max(calls)


def find_one(records, kind, call, **criteria):
    matches = []
    for record in records:
        if record.get("kind") != kind or int(record.get("call", -1)) != call:
            continue
        if all(record.get(key) == value for key, value in criteria.items()):
            matches.append(record)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {kind} record for call={call} {criteria}, found {len(matches)}"
        )
    return matches[0]


def metric(records, call, name):
    return find_one(records, "fmm_nonuniform_rank_metric", call, metric=name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--call", type=int)
    args = parser.parse_args()

    records = parse_lines(args.log)
    call = args.call if args.call is not None else latest_call(records)
    spatial7 = find_one(records, "fmm_nonuniform_spatial", call, level=7)
    let = find_one(records, "fmm_nonuniform_let", call)

    duplication = float(spatial7.get("duplication", 0.0))
    shared_particles = float(spatial7.get("shared_particle_fraction", 0.0))
    shared_patches = float(spatial7.get("shared_patch_fraction", 0.0))
    dominant_p05 = float(spatial7.get("dominant_owner_fraction_p05", 1.0))
    process_nodes = metric(records, call, "process_owned_nodes")
    process_m2l = metric(records, call, "process_owned_m2l")
    bytes_sent = metric(records, call, "bytes_sent")
    let_p2p = metric(records, call, "let_active_p2p_blocks")
    components = metric(records, call, "rank_spatial_components")
    fill = metric(records, call, "rank_spatial_fill_fraction")

    node_imbalance = float(process_nodes.get("max_over_mean", 0.0))
    m2l_imbalance = float(process_m2l.get("max_over_mean", 0.0))
    byte_imbalance = float(bytes_sent.get("max_over_mean", 0.0))
    p2p_imbalance = float(let_p2p.get("max_over_mean", 0.0))
    fanout_p95 = int(let.get("fanout_p95", 0))
    fanout_max = int(let.get("fanout_max", 0))
    component_p95 = int(components.get("p95", 0))
    fill_p50 = float(fill.get("p50", 0.0))

    unique_owner_score = 0
    if duplication >= 1.10:
        unique_owner_score += 2
    if shared_particles >= 0.20:
        unique_owner_score += 2
    if shared_patches >= 0.20:
        unique_owner_score += 1
    if dominant_p05 <= 0.70:
        unique_owner_score += 1

    process_balance_score = 0
    if node_imbalance >= 4.0:
        process_balance_score += 2
    if m2l_imbalance >= 4.0:
        process_balance_score += 2

    node_sharing_score = 0
    if fanout_p95 >= 16:
        node_sharing_score += 1
    if fanout_max >= 128:
        node_sharing_score += 2
    if byte_imbalance >= 2.0:
        node_sharing_score += 1

    adaptive_score = 0
    if p2p_imbalance >= 2.0:
        adaptive_score += 1
    if component_p95 >= 4:
        adaptive_score += 1
    if fill_p50 <= 0.05:
        adaptive_score += 1

    print(f"Selected diagnostic call: {call}")
    print()
    print("Measured structure")
    print(f"  level-7 owner-tagged / unique: {duplication:.4f}")
    print(f"  shared-patch fraction:          {shared_patches:.4f}")
    print(f"  shared-particle fraction:       {shared_particles:.4f}")
    print(f"  dominant-owner p05:             {dominant_p05:.4f}")
    print(f"  process-node max/mean:          {node_imbalance:.3f}")
    print(f"  process-M2L max/mean:           {m2l_imbalance:.3f}")
    print(f"  sent-byte max/mean:             {byte_imbalance:.3f}")
    print(f"  LET-P2P max/mean:               {p2p_imbalance:.3f}")
    print(f"  source fanout p95 / max:        {fanout_p95} / {fanout_max}")
    print(f"  rank components p95:            {component_p95}")
    print(f"  rank fill fraction p50:         {fill_p50:.6g}")
    print()

    recommendations = []
    if unique_owner_score >= 3:
        recommendations.append(
            "PRIMARY: build a gravity shadow decomposition with one globally unique owner per spatial patch, followed by a globally adaptive weighted Morton tree."
        )
    else:
        recommendations.append(
            "PRIMARY: retain rank-owned patches initially; level-7 ownership duplication is not large enough by itself to justify immediate full particle redistribution."
        )
    if process_balance_score >= 2:
        recommendations.append(
            "REQUIRED: decouple internal process-node ownership from child owners and assign internal work globally using weighted greedy or graph partitioning."
        )
    if adaptive_score >= 2:
        recommendations.append(
            "REQUIRED: use measured-work split/merge thresholds with hysteresis; do not use particle count alone."
        )
    if node_sharing_score >= 2:
        recommendations.append(
            "HIGH VALUE: add node-level payload sharing or FMM leader ranks so a high-fanout source crosses the network once per node rather than once per rank."
        )
    recommendations.append(
        "IN ALL CASES: separate structural topology, conservative geometry-envelope generation, payload capacity, and wave packing so ordinary motion does not trigger a global LET traversal."
    )

    print("Architecture decision")
    for item in recommendations:
        print(f"  - {item}")

    print()
    print("Scores")
    print(f"  unique-owner shadow decomposition: {unique_owner_score}/6")
    print(f"  process-owner redesign:            {process_balance_score}/4")
    print(f"  node-level source sharing:          {node_sharing_score}/4")
    print(f"  work-adaptive refinement:           {adaptive_score}/3")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
