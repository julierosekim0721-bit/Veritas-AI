"""Tests for business ZIP/ZCTA ACS demographic enrichment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
CENSUS_SPEC = importlib.util.spec_from_file_location("census_api", SRC_DIR / "census_api.py")
CENSUS_API = importlib.util.module_from_spec(CENSUS_SPEC)
assert CENSUS_SPEC.loader is not None
sys.modules[CENSUS_SPEC.name] = CENSUS_API
CENSUS_SPEC.loader.exec_module(CENSUS_API)

MODULE_PATH = SRC_DIR / "enrich_zcta_demographics.py"
SPEC = importlib.util.spec_from_file_location("enrich_zcta_demographics", MODULE_PATH)
ENRICH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ENRICH
SPEC.loader.exec_module(ENRICH)


def acs_payload() -> list[list[str]]:
    header = ENRICH.ACS_VARIABLES + ["zip code tabulation area"]
    row1 = [
        "ZCTA5 19104", "50000", "100", "20", "100", "10", "5", "3", "2",
        "50", "5", "200", "100", "40", "20", "30", "19104",
    ]
    row2 = [
        "ZCTA5 02108", "-666666666", "0", "0", "0", "0", "0", "0", "0",
        "0", "0", "0", "0", "0", "0", "0", "2108",
    ]
    return [header, row1, row2]


def catalogue_row(zcta: str = "19104") -> dict[str, object]:
    return {
        "zcta": zcta,
        "census_geo_name": f"ZCTA5 {zcta}",
        "census_geo_level": "ACS ZCTA",
        "acs_year": 2022,
        "acs_dataset": "acs/acs5",
        "data_source": "Census ACS API",
        "area_median_household_income": 50000.0,
        "area_poverty_rate": 20.0,
        "area_bachelor_plus_rate": 20.0,
        "area_unemployment_rate": 10.0,
        "area_pct_white": 50.0,
        "area_pct_black": 20.0,
        "area_pct_asian": 10.0,
        "area_pct_hispanic_or_latino": 15.0,
    }


class ZctaEnrichmentTests(unittest.TestCase):
    def test_normalize_us_postal_code(self) -> None:
        self.assertEqual(ENRICH.normalize_us_postal_code("19104"), "19104")
        self.assertEqual(ENRICH.normalize_us_postal_code("19104-1234"), "19104")
        for value in (None, "", "  ", "1910", "191041", "ABCDE", "A1A 1A1", "19104 1234"):
            with self.subTest(value=value):
                self.assertIsNone(ENRICH.normalize_us_postal_code(value))

    def test_numeric_and_safe_rate_are_nullable_and_bounded(self) -> None:
        numeric = ENRICH.numeric(pd.Series(["10", "-1", "-666666666", "bad", None]))
        self.assertEqual(numeric.iloc[0], 10.0)
        self.assertTrue(numeric.iloc[1:].isna().all())

        rates = ENRICH.safe_rate(
            pd.Series([25, 1, -1, 101]),
            pd.Series([100, 0, 10, 100]),
        )
        self.assertEqual(rates.iloc[0], 25.0)
        self.assertTrue(rates.iloc[1:].isna().all())

    def test_parse_acs_catalogue_calculates_exact_features_and_provenance(self) -> None:
        result = ENRICH.parse_acs_catalogue(acs_payload(), acs_year=2022)

        self.assertEqual(result["zcta"].tolist(), ["19104", "02108"])
        self.assertTrue(result["zcta"].is_unique)
        first = result.iloc[0]
        self.assertEqual(first["census_geo_level"], "ACS ZCTA")
        self.assertEqual(first["acs_year"], 2022)
        self.assertEqual(first["acs_dataset"], "acs/acs5")
        self.assertEqual(first["data_source"], "Census ACS API")
        self.assertEqual(first["area_median_household_income"], 50000.0)
        self.assertEqual(first["area_poverty_rate"], 20.0)
        self.assertEqual(first["area_bachelor_plus_rate"], 20.0)
        self.assertEqual(first["area_unemployment_rate"], 10.0)
        self.assertEqual(first["area_pct_white"], 50.0)
        self.assertEqual(first["area_pct_black"], 20.0)
        self.assertEqual(first["area_pct_asian"], 10.0)
        self.assertEqual(first["area_pct_hispanic_or_latino"], 15.0)
        self.assertTrue(result.iloc[1][ENRICH.AREA_FEATURE_COLUMNS].isna().all())

    def test_parse_acs_catalogue_rejects_missing_columns_and_duplicate_zctas(self) -> None:
        missing = acs_payload()
        missing[0] = missing[0][:-2] + [missing[0][-1]]
        missing[1] = missing[1][:-2] + [missing[1][-1]]
        missing[2] = missing[2][:-2] + [missing[2][-1]]
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            ENRICH.parse_acs_catalogue(missing)

        duplicate = acs_payload()
        duplicate[2][-1] = "19104"
        with self.assertRaisesRegex(ValueError, "duplicate zcta"):
            ENRICH.parse_acs_catalogue(duplicate)

    def test_fetch_acs_catalogue_uses_single_wildcard_request_and_redacts_failures(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = acs_payload()
        with patch.object(ENRICH, "get_census_api_key", return_value="secret"), patch.object(
            ENRICH.requests, "get", return_value=response
        ) as request:
            result = ENRICH.fetch_acs_catalogue(acs_year=2022)

        self.assertEqual(len(result), 2)
        request.assert_called_once()
        args, kwargs = request.call_args
        self.assertEqual(args[0], "https://api.census.gov/data/2022/acs/acs5")
        self.assertEqual(kwargs["timeout"], 30)
        self.assertEqual(kwargs["params"]["for"], "zip code tabulation area:*")
        self.assertEqual(kwargs["params"]["get"], ",".join(ENRICH.ACS_VARIABLES))

        with patch.object(ENRICH, "get_census_api_key", return_value="secret"), patch.object(
            ENRICH.requests, "get", side_effect=RuntimeError("failed https://example?key=secret")
        ) as failed_request:
            with self.assertRaises(ENRICH.CensusRequestError) as caught:
                ENRICH.fetch_acs_catalogue()
        self.assertEqual(failed_request.call_count, 2)
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("key=", str(caught.exception))

    def test_catalogue_cache_is_reused_and_refresh_is_keyless(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "acs_2022_acs5_zcta_catalogue.csv"
            expected = pd.DataFrame([catalogue_row()])
            expected.to_csv(cache_path, index=False)
            with patch.object(ENRICH, "fetch_acs_catalogue") as fetch:
                cached = ENRICH.load_zcta_catalogue(cache_path=cache_path)
            fetch.assert_not_called()
            self.assertEqual(cached["zcta"].tolist(), ["19104"])

            refreshed = expected.copy()
            refreshed.loc[0, "area_median_household_income"] = 60000
            with patch.object(ENRICH, "fetch_acs_catalogue", return_value=refreshed) as fetch:
                actual = ENRICH.load_zcta_catalogue(cache_path=cache_path, refresh=True)
            fetch.assert_called_once_with(acs_year=2022)
            self.assertEqual(actual.loc[0, "area_median_household_income"], 60000)
            self.assertNotIn("secret", str(cache_path))
            self.assertNotIn("key=", str(cache_path))

    def test_enrich_businesses_preserves_all_ids_and_match_reasons(self) -> None:
        businesses = pd.DataFrame(
            {
                "business_id": ["a", "b", "c", "d"],
                "business_city": ["Philadelphia", "Missing", "Toronto", "Nowhere"],
                "business_state": ["PA", "PA", "ON", "PA"],
                "business_postal_code": ["19104-1234", None, "A1A 1A1", "99999"],
            }
        )
        result = ENRICH.enrich_businesses(businesses, pd.DataFrame([catalogue_row()]))

        self.assertEqual(result["business_id"].tolist(), ["a", "b", "c", "d"])
        self.assertEqual(result["business_zcta"].tolist()[0], "19104")
        self.assertEqual(
            result["zcta_match_status"].tolist(),
            ["matched", "missing_postal_code", "invalid_or_non_us_postal_code", "zcta_not_returned"],
        )
        self.assertEqual(len(set(result["zcta_match_reason"])), 4)
        self.assertEqual(result.loc[0, "area_median_household_income"], 50000)
        self.assertEqual(
            result.loc[0, "area_context_note"],
            "Census area around the business; not an individual reviewer, patient, or dentist.",
        )

    def test_enrich_businesses_rejects_bad_business_ids_and_duplicate_zctas(self) -> None:
        good = pd.DataFrame(
            {
                "business_id": ["a"],
                "business_city": ["Philadelphia"],
                "business_state": ["PA"],
                "business_postal_code": ["19104"],
            }
        )
        for ids in (["a", "a"], [None]):
            bad = pd.concat([good] * len(ids), ignore_index=True)
            bad["business_id"] = ids
            with self.assertRaisesRegex(ValueError, "business_id"):
                ENRICH.enrich_businesses(bad, pd.DataFrame([catalogue_row()]))
        duplicate_catalogue = pd.DataFrame([catalogue_row(), catalogue_row()])
        with self.assertRaisesRegex(ValueError, "duplicate zcta"):
            ENRICH.enrich_businesses(good, duplicate_catalogue)

    def test_coverage_reports_separate_business_and_review_weighting(self) -> None:
        businesses = pd.DataFrame(
            {
                "business_id": ["a", "b", "c"],
                "business_city": ["Philadelphia", "Pittsburgh", "Toronto"],
                "business_state": ["PA", "PA", "ON"],
                "business_postal_code": ["19104", "99999", "A1A 1A1"],
                "business_zcta": ["19104", "99999", pd.NA],
                "zcta_match_status": ["matched", "zcta_not_returned", "invalid_or_non_us_postal_code"],
                "zcta_match_reason": ["Matched ACS ZCTA.", "ZIP candidate was not returned by ACS.", "Postal code is not a valid US ZIP candidate."],
                **{feature: [1.0 if feature == "area_median_household_income" else pd.NA, pd.NA, pd.NA] for feature in ENRICH.AREA_FEATURE_COLUMNS},
            }
        )
        reviews = pd.DataFrame({"business_id": ["a", "a", "b", "c"]})

        non_null, by_state, by_city, summary = ENRICH.build_coverage_tables(businesses, reviews)

        income = non_null.loc[non_null["feature"] == "area_median_household_income"].iloc[0]
        self.assertEqual((income["business_non_null_count"], income["business_denominator"]), (1, 3))
        self.assertEqual((income["review_non_null_count"], income["review_denominator"]), (2, 4))
        self.assertEqual(set(by_state["business_state"]), {"PA", "ON"})
        self.assertEqual(set(by_city["business_city"]), {"Philadelphia", "Pittsburgh", "Toronto"})
        self.assertEqual(summary["business_any_feature_count"], 1)
        self.assertEqual(summary["review_any_feature_count"], 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            ENRICH.write_coverage_reports(
                businesses,
                reviews,
                report_dir=report_dir,
                businesses_path=Path("input/businesses.csv"),
                reviews_path=Path("input/reviews.csv"),
                catalogue_path=Path("cache/catalogue.csv"),
            )
            report = (report_dir / "coverage_report.md").read_text()
            self.assertIn("not an individual reviewer, patient, or dentist", report)
            self.assertIn("Business-weighted coverage", report)
            self.assertIn("Review-weighted coverage", report)
            self.assertNotIn("secret", report)
            rates = pd.read_csv(report_dir / "tables/non_null_rates.csv")
            self.assertEqual(set(rates["feature"]), set(ENRICH.AREA_FEATURE_COLUMNS))
            self.assertTrue((report_dir / "tables/match_rate_by_state.csv").exists())
            self.assertTrue((report_dir / "tables/match_rate_by_city.csv").exists())

    def test_main_writes_business_and_report_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            businesses_path = root / "businesses.csv"
            reviews_path = root / "reviews.csv"
            catalogue_path = root / "zcta.csv"
            output_path = root / "business_zcta.csv"
            report_dir = root / "reports"
            pd.DataFrame(
                {
                    "business_id": ["b1", "b2"],
                    "business_city": ["Philadelphia", "Nowhere"],
                    "business_state": ["PA", "ZZ"],
                    "business_postal_code": ["19104", "99999"],
                }
            ).to_csv(businesses_path, index=False)
            pd.DataFrame({"business_id": ["b1", "b1", "b2"]}).to_csv(reviews_path, index=False)
            catalogue = pd.DataFrame([catalogue_row()])
            argv = [
                "enrich",
                "--businesses",
                str(businesses_path),
                "--reviews",
                str(reviews_path),
                "--zcta-catalogue",
                str(catalogue_path),
                "--business-output",
                str(output_path),
                "--report-dir",
                str(report_dir),
            ]
            with patch.object(sys, "argv", argv), patch.object(ENRICH, "load_zcta_catalogue", return_value=catalogue):
                ENRICH.main()
            self.assertTrue(output_path.exists())
            self.assertTrue((report_dir / "coverage_report.md").exists())
            result = pd.read_csv(output_path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.loc[0, "zcta_match_status"], "matched")


if __name__ == "__main__":
    unittest.main()
