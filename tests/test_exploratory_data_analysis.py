"""Focused tests for the hypothesis-driven EDA helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "exploratory_data_analysis.py"
SPEC = importlib.util.spec_from_file_location("exploratory_data_analysis", MODULE_PATH)
EDA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = EDA
SPEC.loader.exec_module(EDA)


class ExploratoryDataAnalysisTests(unittest.TestCase):
    def test_demographic_audit_reports_all_null_columns_without_failing(self) -> None:
        data = pd.DataFrame(
            {
                "area_median_household_income": [None, None],
                "area_poverty_rate": [None, None],
            }
        )

        audit, matched_rows = EDA.audit_demographics(
            data,
            ["area_median_household_income", "area_poverty_rate"],
        )

        self.assertEqual(matched_rows, 0)
        self.assertEqual(audit["non_null_count"].tolist(), [0, 0])
        self.assertTrue(audit["all_null"].all())

    def test_target_validation_accepts_documented_definitions(self) -> None:
        data = pd.DataFrame(
            {
                "review_stars": [1, 2, 3, 4, 5],
                "satisfied_binary": [0, 0, 0, 1, 1],
                "negative_binary": [1, 1, 0, 0, 0],
            }
        )

        self.assertEqual(EDA.validate_target_definitions(data), {"satisfied_mismatches": 0, "negative_mismatches": 0})

    def test_group_summary_includes_sample_counts_and_rates(self) -> None:
        data = pd.DataFrame(
            {
                "business_state": ["PA", "PA", "FL"],
                "satisfied_binary": [1, 0, 1],
            }
        )

        summary = EDA.group_satisfaction(data, "business_state")

        pa = summary.loc[summary["business_state"] == "PA"].iloc[0]
        self.assertEqual(pa["review_count"], 2)
        self.assertAlmostEqual(pa["satisfaction_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
