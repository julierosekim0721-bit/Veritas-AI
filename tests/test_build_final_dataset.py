"""Tests for the narrow business_id demographic merge in the final dataset build."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "build_final_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_final_dataset", MODULE_PATH)
BUILD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


FEATURES = [
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
]


def base_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "business_id": ["a", "a", "b"],
            "user_id": ["u1", "u2", "u3"],
            "review_stars": [5, 4, 2],
            "review_text": ["great", "good", "bad"],
            "review_date": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "business_name": ["A", "A", "B"],
            "business_city": ["Philadelphia", "Philadelphia", "Pittsburgh"],
            "business_state": ["PA", "PA", "PA"],
            "business_postal_code": ["19104", "19104", "99999"],
        }
    )


def enrichment_fixture() -> pd.DataFrame:
    rows = []
    for business_id, city, zcta, status, value in (
        ("a", "Philadelphia", "19104", "matched", 50000.0),
        ("b", "Pittsburgh", "99999", "zcta_not_returned", None),
    ):
        row = {
            "business_id": business_id,
            "business_city": city,
            "business_state": "PA",
            "business_postal_code": zcta,
            "business_zcta": zcta,
            "zcta_match_status": status,
            "zcta_match_reason": "area context only",
            "area_context_note": "Census area around the business; not an individual reviewer, patient, or dentist.",
            "census_geo_name": "ZCTA5 19104" if status == "matched" else None,
            "census_geo_level": "ACS ZCTA" if status == "matched" else None,
            "acs_year": 2022 if status == "matched" else None,
            "acs_dataset": "acs/acs5" if status == "matched" else None,
            "data_source": "Census ACS API" if status == "matched" else None,
            **{feature: (value if feature == "area_median_household_income" else (10.0 if value is not None else None)) for feature in FEATURES},
        }
        rows.append(row)
    return pd.DataFrame(rows)


class BuildFinalDatasetTests(unittest.TestCase):
    def _run_build(self, demographics: pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.csv"
            demographics_path = root / "demographics.csv"
            final_path = root / "final.csv"
            base_fixture().to_csv(base_path, index=False)
            demographics.to_csv(demographics_path, index=False)
            old_paths = BUILD.BASE_CSV, BUILD.DEMOGRAPHICS_CSV, BUILD.FINAL_CSV
            BUILD.BASE_CSV, BUILD.DEMOGRAPHICS_CSV, BUILD.FINAL_CSV = base_path, demographics_path, final_path
            try:
                BUILD.main()
                return pd.read_csv(final_path, dtype={"business_zcta": "string"})
            finally:
                BUILD.BASE_CSV, BUILD.DEMOGRAPHICS_CSV, BUILD.FINAL_CSV = old_paths

    def test_business_id_merge_retains_reviews_and_attaches_zcta_context(self) -> None:
        final = self._run_build(enrichment_fixture())

        self.assertEqual(len(final), 3)
        matched = final.loc[final["business_id"] == "a"]
        self.assertEqual(matched["business_zcta"].tolist(), ["19104", "19104"])
        self.assertEqual(matched["area_median_household_income"].tolist(), [50000.0, 50000.0])
        self.assertTrue(matched["area_context_note"].str.contains("not an individual reviewer").all())
        unmatched = final.loc[final["business_id"] == "b"].iloc[0]
        self.assertTrue(pd.isna(unmatched["area_median_household_income"]))
        self.assertEqual(unmatched["zcta_match_status"], "zcta_not_returned")

    def test_duplicate_enrichment_business_id_fails(self) -> None:
        duplicate = pd.concat([enrichment_fixture(), enrichment_fixture().iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            self._run_build(duplicate)


if __name__ == "__main__":
    unittest.main()
