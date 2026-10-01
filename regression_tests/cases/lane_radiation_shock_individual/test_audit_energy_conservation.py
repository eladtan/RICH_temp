#!/usr/bin/env python3
"""Focused compatibility tests for radiation rejection records."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


AUDITOR_PATH = Path(__file__).with_name("audit_energy_conservation.py")
SPEC = importlib.util.spec_from_file_location("audit_energy_conservation", AUDITOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {AUDITOR_PATH}")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class AuditEnergyConservationLogTest(unittest.TestCase):
    def parse(self, line: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(line + "\n", encoding="utf-8")
            return AUDITOR.parse_log(path)

    def test_reads_colored_rich_retry_block(self) -> None:
        report = self.parse(
            "\x1b[1;33mRICH_RETRY\x1b[0m mode=individual cycle=4 "
            "physics=radiation attempt=1\n"
            "  \x1b[33mattempt\x1b[0m | active_cells=8 "
            "| active_bins=[bin=0,count=8,dt=0.25] "
            "| attempted_dt_min=0.25 | attempted_dt_max=0.25\n"
            "  \x1b[31mfailure\x1b[0m | retry_s=0.5 "
            "| reason=negative_energy | cell=17\n"
        )
        rejection = report["first_rejection"]
        self.assertIsNotNone(rejection)
        self.assertEqual(rejection["cycle"], 4)
        self.assertEqual(rejection["attempted_dt_min"], 0.25)
        self.assertEqual(rejection["reason"], "negative_energy")
        self.assertEqual(rejection["cell"], 17)

    def test_incomplete_retry_block_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete RICH_RETRY block"):
            self.parse(
                "RICH_RETRY mode=individual cycle=4 physics=radiation "
                "attempt=1\n"
                "  attempt | active_cells=8 "
                "| active_bins=[bin=0,count=8,dt=0.25] "
                "| attempted_dt_min=0.25 | attempted_dt_max=0.25"
            )

    def test_out_of_order_retry_subject_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected retry failure line"):
            self.parse(
                "RICH_RETRY mode=individual cycle=4 physics=radiation "
                "attempt=1\n"
                "  failure | retry_s=0.5 | reason=negative_energy | cell=17"
            )

    def test_historical_rejection_is_not_a_runtime_retry(self) -> None:
        report = self.parse(
            "INDIVIDUAL_RADIATION_REJECTION cycle=4 dt=0.25 reason=negative"
        )
        self.assertIsNone(report["first_rejection"])

    def test_interleaved_retry_block_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "interleaved RICH_RETRY block"):
            self.parse(
                "RICH_RETRY mode=individual cycle=4 physics=radiation "
                "attempt=1\n"
                "UNRELATED_DIAGNOSTIC value=1\n"
                "  attempt | active_cells=8 "
                "| active_bins=[bin=0,count=8,dt=0.25] "
                "| attempted_dt_min=0.25 | attempted_dt_max=0.25\n"
                "  failure | retry_s=0.5 | reason=negative_energy | cell=17\n"
            )

    def test_missing_blank_retry_terminator_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing blank line after RICH_RETRY"):
            self.parse(
                "RICH_RETRY mode=individual cycle=4 physics=radiation "
                "attempt=1\n"
                "  attempt | active_cells=8 "
                "| active_bins=[bin=0,count=8,dt=0.25] "
                "| attempted_dt_min=0.25 | attempted_dt_max=0.25\n"
                "  failure | retry_s=0.5 | reason=negative_energy | cell=17"
            )


if __name__ == "__main__":
    unittest.main()
