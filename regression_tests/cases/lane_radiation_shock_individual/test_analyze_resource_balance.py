#!/usr/bin/env python3
"""Focused synthetic tests for physical-work resource-log comparison."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ANALYZER_PATH = Path(__file__).with_name("analyze_resource_balance.py")
SPEC = importlib.util.spec_from_file_location("analyze_resource_balance", ANALYZER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {ANALYZER_PATH}")
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def event_lines(
    cycle: int,
    event_time: float,
    *,
    active_cells: int = 100,
    canonical_cells: int | None = 100,
    amr: int | None = 0,
    ownership_epoch: int | None = 0,
    wall_seconds: float = 1.0,
    rss_mean_kib: float = 100.0,
    lifetime_peak_kib: float | None = 150.0,
    wall_unit: str = "seconds",
    rss_unit: str = "KiB",
    physical_metadata: bool = True,
) -> list[str]:
    lines = [
        f"INDIVIDUAL_PERF cycle={cycle} phase=event-wall unit={wall_unit} "
        f"min={wall_seconds} median={wall_seconds} mean={wall_seconds} "
        f"p95={wall_seconds} max={wall_seconds}",
        f"INDIVIDUAL_PERF cycle={cycle} phase=current-rss unit={rss_unit} "
        f"min={0.9 * rss_mean_kib} median={rss_mean_kib} mean={rss_mean_kib} "
        f"p95={1.05 * rss_mean_kib} max={1.1 * rss_mean_kib}",
    ]
    if lifetime_peak_kib is not None:
        lines.append(
            f"INDIVIDUAL_PERF cycle={cycle} phase=peak-rss unit={rss_unit} "
            f"min={0.8 * lifetime_peak_kib} median={0.9 * lifetime_peak_kib} "
            f"mean={0.9 * lifetime_peak_kib} p95={lifetime_peak_kib} "
            f"max={lifetime_peak_kib}"
        )
    if physical_metadata:
        hydro_fields = (
            "INDIVIDUAL_HYDRO_PHASE_TIMING ranks=2 "
            f"active_cells_global={active_cells}"
        )
        if canonical_cells is not None:
            hydro_fields += f" canonical_cells_global={canonical_cells}"
        balance_fields = f"INDIVIDUAL_LOAD_BALANCE_DECISION cycle={cycle}"
        if amr is not None:
            balance_fields += f" amr={amr}"
        if ownership_epoch is not None:
            balance_fields += f" ownership_epoch={ownership_epoch}"
        lines[:0] = [
            f"Individual cycle {cycle} from time 0 to {event_time} with 1 active cells",
            "MG_TIMESTEP_LIMIT mode=individual "
            f"event_time={event_time:.17g} active_cells={active_cells}",
            hydro_fields,
            balance_fields,
        ]
    return lines


def write_log(
    path: Path,
    events: list[dict[str, object]],
    *,
    final_time: float,
    final_tick: int | None,
    final_cells: int,
    evolution_wall_seconds: float,
    physical_metadata: bool = True,
) -> None:
    lines: list[str] = []
    for event in events:
        lines.extend(event_lines(physical_metadata=physical_metadata, **event))
    fingerprint = "INDIVIDUAL_RESTART_FINGERPRINT phase=final"
    if final_tick is not None:
        fingerprint += f" current_tick={final_tick}"
    fingerprint += f" cells={final_cells}"
    lines.extend(
        [
            "PROGRESS phase=final mode=partial "
            f"cycle={len(events)} time={final_time:.17g} "
            f"wall_seconds={evolution_wall_seconds:.17g}",
            fingerprint,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AnalyzeResourceBalanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def analyze(
        self,
        baseline_events: list[dict[str, object]],
        optimized_events: list[dict[str, object]],
        *,
        baseline_final_time: float = 2.0,
        optimized_final_time: float = 2.0,
        baseline_tick: int | None = 200,
        optimized_tick: int | None = 200,
        baseline_cells: int = 100,
        optimized_cells: int = 100,
        baseline_wall: float = 10.0,
        optimized_wall: float = 9.0,
        baseline_metadata: bool = True,
        optimized_metadata: bool = True,
    ) -> dict[str, object]:
        baseline = self.root / "baseline.log"
        optimized = self.root / "optimized.log"
        write_log(
            baseline,
            baseline_events,
            final_time=baseline_final_time,
            final_tick=baseline_tick,
            final_cells=baseline_cells,
            evolution_wall_seconds=baseline_wall,
            physical_metadata=baseline_metadata,
        )
        write_log(
            optimized,
            optimized_events,
            final_time=optimized_final_time,
            final_tick=optimized_tick,
            final_cells=optimized_cells,
            evolution_wall_seconds=optimized_wall,
            physical_metadata=optimized_metadata,
        )
        return ANALYZER.analyze(
            optimized_path=optimized,
            baseline_path=baseline,
            ranks=2,
            late_cycles=0,
        )

    def test_pairs_physical_time_when_cycle_ids_shift(self) -> None:
        result = self.analyze(
            [
                {"cycle": 0, "event_time": 1.0},
                {"cycle": 1, "event_time": 2.0},
            ],
            [
                {"cycle": 0, "event_time": 0.5},
                {"cycle": 1, "event_time": 1.0},
                {"cycle": 2, "event_time": 2.0},
            ],
        )

        matched = result["matched"]
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(matched["accepted_event_count"], 2)
        self.assertEqual(matched["baseline_cycle_last"], 1)
        self.assertEqual(matched["optimized_cycle_last"], 2)
        self.assertEqual(matched["cycle_count"], 2)
        self.assertEqual(matched["optimized_only_event_time_count"], 1)

    def test_work_mismatch_is_excluded(self) -> None:
        result = self.analyze(
            [
                {"cycle": 0, "event_time": 1.0, "active_cells": 100},
                {"cycle": 1, "event_time": 2.0, "active_cells": 100},
            ],
            [
                {"cycle": 0, "event_time": 1.0, "active_cells": 150},
                {"cycle": 1, "event_time": 2.0, "active_cells": 100},
            ],
        )

        matched = result["matched"]
        self.assertEqual(matched["accepted_event_count"], 1)
        self.assertEqual(matched["excluded_common_event_count"], 1)
        self.assertEqual(matched["excluded_by_reason"]["active_cells"], 1)

    def test_runtime_gate_uses_completed_endpoint(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0, "wall_seconds": 1.0}],
            [{"cycle": 0, "event_time": 2.0, "wall_seconds": 0.5}],
            baseline_wall=100.0,
            optimized_wall=103.0,
        )

        self.assertEqual(result["matched"]["event_wall_sum_ratio"], 0.5)
        self.assertAlmostEqual(
            result["whole_run"]["comparison"]["evolution_wall_ratio"], 1.03
        )
        self.assertFalse(result["gates"]["event_wall_no_more_than_2pct_slower"])

    def test_mismatched_endpoint_fails_comparability(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0}],
            [{"cycle": 0, "event_time": 3.0}],
            optimized_final_time=3.0,
            optimized_tick=300,
        )

        endpoint = result["whole_run"]["comparison"]["endpoint"]
        self.assertFalse(endpoint["compatible"])
        self.assertIn("final_tick", endpoint["reasons"])
        self.assertFalse(result["gates"]["comparable_completed_endpoints"])

    def test_same_tick_does_not_hide_mismatched_final_time(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0}],
            [{"cycle": 0, "event_time": 3.0}],
            optimized_final_time=3.0,
            optimized_tick=200,
        )

        endpoint = result["whole_run"]["comparison"]["endpoint"]
        self.assertFalse(endpoint["compatible"])
        self.assertIn("final_time", endpoint["reasons"])

    def test_true_lifetime_peak_is_gated(self) -> None:
        result = self.analyze(
            [
                {
                    "cycle": 0,
                    "event_time": 2.0,
                    "rss_mean_kib": 100.0,
                    "lifetime_peak_kib": 150.0,
                }
            ],
            [
                {
                    "cycle": 0,
                    "event_time": 2.0,
                    "rss_mean_kib": 90.0,
                    "lifetime_peak_kib": 200.0,
                }
            ],
        )

        comparison = result["whole_run"]["comparison"]
        self.assertLess(comparison["total_live_rss_mean_ratio"], 1.0)
        self.assertGreater(comparison["max_rank_lifetime_peak_rss_ratio"], 1.0)
        self.assertFalse(result["gates"]["max_rank_lifetime_peak_rss_not_higher"])

    def test_legacy_logs_never_fall_back_to_cycle_pairing(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0}],
            [{"cycle": 0, "event_time": 2.0}],
            baseline_metadata=False,
            optimized_metadata=False,
        )

        matched = result["matched"]
        self.assertFalse(matched["available"])
        self.assertEqual(matched["accepted_event_count"], 0)
        self.assertNotIn("event_wall_sum_ratio", matched)
        self.assertTrue(result["gates"]["comparable_completed_endpoints"])

    def test_duplicate_physical_times_are_not_paired(self) -> None:
        result = self.analyze(
            [
                {"cycle": 0, "event_time": 1.0},
                {"cycle": 1, "event_time": 1.0},
                {"cycle": 2, "event_time": 2.0},
            ],
            [
                {"cycle": 0, "event_time": 1.0},
                {"cycle": 1, "event_time": 2.0},
            ],
        )

        matched = result["matched"]
        self.assertEqual(matched["baseline_duplicate_event_time_count"], 1)
        self.assertEqual(matched["accepted_event_count"], 1)
        self.assertEqual(matched["event_time_first"], 2.0)

    def test_duplicate_cycle_occurrences_are_rejected(self) -> None:
        path = self.root / "duplicate-cycle.log"
        write_log(
            path,
            [
                {"cycle": 0, "event_time": 1.0},
                {"cycle": 0, "event_time": 2.0},
            ],
            final_time=2.0,
            final_tick=200,
            final_cells=100,
            evolution_wall_seconds=2.0,
        )

        with self.assertRaisesRegex(ValueError, "duplicate Individual cycle 0"):
            ANALYZER.parse_log(path)

    def test_asymmetric_missing_work_signature_is_excluded(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0}],
            [
                {
                    "cycle": 0,
                    "event_time": 2.0,
                    "canonical_cells": None,
                    "amr": None,
                    "ownership_epoch": None,
                }
            ],
        )

        matched = result["matched"]
        self.assertEqual(matched["accepted_event_count"], 0)
        self.assertEqual(matched["excluded_by_reason"]["missing_canonical_cells"], 1)
        self.assertEqual(matched["excluded_by_reason"]["missing_amr"], 1)
        self.assertEqual(matched["excluded_by_reason"]["missing_ownership_epoch"], 1)

    def test_missing_final_tick_fails_comparability(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0}],
            [{"cycle": 0, "event_time": 2.0}],
            optimized_tick=None,
        )

        endpoint = result["whole_run"]["comparison"]["endpoint"]
        self.assertFalse(endpoint["compatible"])
        self.assertIn("final_tick_missing", endpoint["reasons"])
        self.assertFalse(result["gates"]["comparable_completed_endpoints"])

    def test_units_and_physical_statistics_are_validated(self) -> None:
        cases = (
            ({"rss_unit": "MiB"}, "expected current-rss unit=KiB"),
            ({"rss_mean_kib": float("nan")}, "non-finite current-rss"),
            ({"wall_seconds": -1.0}, "non-positive event-wall"),
        )
        for index, (event_update, message) in enumerate(cases):
            with self.subTest(message=message):
                path = self.root / f"invalid-{index}.log"
                event = {"cycle": 0, "event_time": 2.0, **event_update}
                write_log(
                    path,
                    [event],
                    final_time=2.0,
                    final_tick=200,
                    final_cells=100,
                    evolution_wall_seconds=2.0,
                )
                with self.assertRaisesRegex(ValueError, message):
                    ANALYZER.parse_log(path)

    def test_missing_lifetime_peak_evidence_fails_gate(self) -> None:
        result = self.analyze(
            [
                {"cycle": 0, "event_time": 1.0},
                {"cycle": 1, "event_time": 2.0},
            ],
            [
                {"cycle": 0, "event_time": 1.0},
                {
                    "cycle": 1,
                    "event_time": 2.0,
                    "lifetime_peak_kib": None,
                }
            ],
        )

        self.assertIsNone(
            result["whole_run"]["comparison"]["max_rank_lifetime_peak_rss_ratio"]
        )
        self.assertFalse(result["gates"]["lifetime_peak_rss_evidence_available"])
        self.assertFalse(result["gates"]["max_rank_lifetime_peak_rss_not_higher"])
        self.assertFalse(result["pass"])

    def test_negative_endpoint_runtime_fails_gate(self) -> None:
        result = self.analyze(
            [{"cycle": 0, "event_time": 2.0}],
            [{"cycle": 0, "event_time": 2.0}],
            optimized_wall=-1.0,
        )

        comparison = result["whole_run"]["comparison"]
        self.assertFalse(comparison["endpoint"]["compatible"])
        self.assertIsNone(comparison["evolution_wall_ratio"])
        self.assertFalse(result["gates"]["event_wall_no_more_than_2pct_slower"])


if __name__ == "__main__":
    unittest.main()
