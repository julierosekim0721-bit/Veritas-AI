"""Tests for the final grouped satisfaction model trainer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
MODULE_PATH = SRC_DIR / "train_final_grouped_satisfaction_model.py"
SPEC = importlib.util.spec_from_file_location(
    "train_final_grouped_satisfaction_model", MODULE_PATH
)
TRAIN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


def synthetic_frame(
    n_businesses: int = 8,
    reviews_per_business: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Small synthetic dataset: enough businesses and repeated tokens for TF-IDF."""
    rng = pd.DataFrame({"x": range(100)}).sample(frac=1, random_state=seed).index
    positive_words = ["great", "friendly", "clean", "painless", "caring"]
    negative_words = ["rude", "painful", "dirty", "rushed", "expensive"]
    rows = []
    idx = 0
    for b in range(n_businesses):
        business_id = f"biz_{b:03d}"
        lean = b % 2  # alternate businesses lean satisfied / not
        for r in range(reviews_per_business):
            if (r + lean) % 2 == 0:
                label = 1
                words = positive_words
            else:
                label = 0
                words = negative_words
            tokens = [words[(r + w) % len(words)] for w in range(4)]
            tokens.append("dentist")  # shared token so vocabulary is not degenerate
            text = " ".join(tokens)
            rows.append(
                {
                    "review_id": f"rev_{idx:05d}",
                    "business_id": business_id,
                    "user_id": f"user_{idx:05d}",
                    "review_stars": 5 if label == 1 else 1,
                    "satisfied_binary": label,
                    "negative_binary": 1 - label,
                    "review_text": text,
                    "review_date": "2023-05-01",
                    "review_year": 2023,
                    "review_length_chars": len(text),
                    "review_length_words": len(tokens),
                    "business_name": f"Business {b}",
                    "business_city": "Bristol",
                    "business_state": "ENG",
                    "business_postal_code": "BS1",
                    "latitude": 51.45,
                    "longitude": -2.58,
                    "business_categories": "dentist",
                    "business_review_count": 20 + b,
                    "area_median_household_income": 40000.0 + 1000 * b,
                    "area_poverty_rate": 15.0 + b,
                    "area_bachelor_plus_rate": 30.0 + b,
                    "area_unemployment_rate": 5.0 + 0.1 * b,
                    "area_pct_white": 60.0 + b,
                    "area_pct_black": 10.0 + b,
                    "area_pct_asian": 8.0 + b,
                    "area_pct_hispanic_or_latino": 5.0 + b,
                    "area_income_band": "medium",
                    "area_largest_demographic_group": "group_a",
                    "area_largest_demographic_share": 0.4,
                    "useful": idx % 3,
                    "funny": 0,
                    "cool": 0,
                    "business_stars": 4.5,
                    "business_zcta": "BS1",
                    "zcta_match_status": "match",
                    "zcta_match_reason": "ok",
                    "area_context_note": "",
                    "census_geo_name": "ZCTA BS1",
                    "census_geo_level": "zcta",
                    "acs_year": 2022,
                    "acs_dataset": "acs5",
                    "data_source": "test",
                }
            )
            idx += 1
    return pd.DataFrame(rows)


class FeatureRegistryTests(unittest.TestCase):
    def test_review_stars_cannot_enter_registry(self) -> None:
        with self.assertRaisesRegex(ValueError, "review_stars"):
            TRAIN.assert_feature_registry(["review_text", "review_stars"])

    def test_business_stars_cannot_enter_registry(self) -> None:
        with self.assertRaisesRegex(ValueError, "business_stars"):
            TRAIN.assert_feature_registry(["review_text", "business_stars"])

    def test_approved_registry_passes(self) -> None:
        TRAIN.assert_feature_registry(TRAIN.FEATURE_REGISTRY)

    def test_non_approved_columns_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-approved"):
            TRAIN.assert_feature_registry(["review_text", "latitude"])

    def test_build_models_transformers_respect_registry(self) -> None:
        models = TRAIN.build_models(seed=42, max_features=200, min_df=1)
        self.assertEqual(
            sorted(models.keys()),
            sorted(
                [
                    "majority_class_dummy",
                    "safe_metadata_logistic",
                    "demographics_logistic",
                    "tfidf_text_logistic",
                    TRAIN.FINAL_MODEL_NAME,
                ]
            ),
        )
        for name, model in models.items():
            if name == "majority_class_dummy":
                continue
            used = []
            for _, _, cols in model.named_steps["pre"].transformers:
                if cols == "drop" or cols is None:
                    continue
                used.extend(list(cols))
            self.assertNotIn("review_stars", used, msg=name)
            self.assertNotIn("business_stars", used, msg=name)
            self.assertNotIn("useful", used, msg=name)
            self.assertNotIn("cool", used, msg=name)


class ValidationTests(unittest.TestCase):
    def test_missing_required_column_raises(self) -> None:
        df = synthetic_frame().drop(columns=["review_year"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            TRAIN.validate_dataframe(df)

    def test_non_binary_target_raises(self) -> None:
        df = synthetic_frame()
        df.loc[df.index[0], TRAIN.TARGET] = 7
        with self.assertRaisesRegex(ValueError, "binary"):
            TRAIN.validate_dataframe(df)

    def test_fractional_target_raises(self) -> None:
        # Regression: fractional values were silently truncated by int casting.
        df = synthetic_frame()
        df[TRAIN.TARGET] = df[TRAIN.TARGET].astype(float)
        df.loc[df.index[0], TRAIN.TARGET] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            TRAIN.validate_dataframe(df)

    def test_string_fractional_target_raises(self) -> None:
        # Regression: CSV-like string "0.5" parsed to fractional then truncated.
        df = synthetic_frame()
        df[TRAIN.TARGET] = df[TRAIN.TARGET].astype(object)
        df.loc[df.index[0], TRAIN.TARGET] = "0.5"
        with self.assertRaisesRegex(ValueError, "binary"):
            TRAIN.validate_dataframe(df)

    def test_prepare_model_frame_rejects_fractional_target(self) -> None:
        # prepare_model_frame must not coerce a fractional target to a label;
        # validation must fire before any integer casting occurs.
        df = synthetic_frame()
        df[TRAIN.TARGET] = df[TRAIN.TARGET].astype(float)
        df.loc[df.index[0], TRAIN.TARGET] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            TRAIN.prepare_model_frame(df)

    def test_all_null_text_raises(self) -> None:
        df = synthetic_frame()
        df[TRAIN.TEXT] = None
        with self.assertRaisesRegex(ValueError, "empty after null-fill"):
            TRAIN.validate_dataframe(df)

    def test_missing_data_file_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            TRAIN.load_dataset(Path("/nonexistent/path/dataset.csv"))


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_zero_overlap(self) -> None:
        frame = TRAIN.prepare_model_frame(synthetic_frame())
        idx1 = TRAIN.make_grouped_split(frame, seed=42, test_size=0.25)
        idx2 = TRAIN.make_grouped_split(frame, seed=42, test_size=0.25)
        self.assertEqual(idx1[0].tolist(), idx2[0].tolist())
        self.assertEqual(idx1[1].tolist(), idx2[1].tolist())

        groups = frame[TRAIN.GROUP]
        train_groups = set(groups.iloc[idx1[0]])
        test_groups = set(groups.iloc[idx1[1]])
        self.assertEqual(train_groups & test_groups, set())
        self.assertGreater(len(train_groups), 0)
        self.assertGreater(len(test_groups), 0)

    def test_split_guard_both_labels(self) -> None:
        # A frame with only label 0 must trip the both-labels guard.
        frame = TRAIN.prepare_model_frame(synthetic_frame())
        frame[TRAIN.TARGET] = 0
        with self.assertRaisesRegex(ValueError, "both target labels"):
            TRAIN.make_grouped_split(frame, seed=42, test_size=0.25)

        frame_all_one = TRAIN.prepare_model_frame(synthetic_frame())
        frame_all_one[TRAIN.TARGET] = 1
        with self.assertRaisesRegex(ValueError, "both target labels"):
            TRAIN.make_grouped_split(frame_all_one, seed=42, test_size=0.25)


class TopTfidfCoefficientsTests(unittest.TestCase):
    def test_numeric_dominated_coefficients_stay_in_tfidf_block(self) -> None:
        # Regression: the function sorted the FULL coefficient vector (TF-IDF
        # block followed by numeric metadata/demographic coefficients) and then
        # indexed the TF-IDF vocabulary with full-model positions. When numeric
        # coefficients dominated, that produced out-of-bounds vocabulary access
        # or wrong terms. Ordering must use only the TF-IDF coefficient slice.
        from types import SimpleNamespace

        vocab = ["bad", "good"]
        text_transformer = SimpleNamespace(get_feature_names_out=lambda: vocab)
        pre = SimpleNamespace(named_transformers_={"text": text_transformer})

        class FakeClf:
            # 2 TF-IDF coefficients + 3 numeric coefficients that dominate.
            coef_ = [[0.1, -0.2, 50.0, -60.0, 70.0]]

        model = SimpleNamespace(named_steps={"pre": pre, "clf": FakeClf()})

        df = TRAIN.top_tfidf_coefficients(model, top_k=1)
        self.assertEqual(set(df["term"]), {"good", "bad"})
        self.assertEqual(set(df["direction"]), {"positive", "negative"})
        pos = df.loc[df["direction"] == "positive", "coefficient"]
        neg = df.loc[df["direction"] == "negative", "coefficient"]
        self.assertEqual(float(pos.iloc[0]), 0.1)    # top TF-IDF positive
        self.assertEqual(float(neg.iloc[0]), -0.2)   # top TF-IDF negative


class EndToEndSmokeTests(unittest.TestCase):
    def _run(self, tmpdir: Path, extra_args: list[str]) -> dict:
        import argparse

        data_path = tmpdir / "input.csv"
        synthetic_frame().to_csv(data_path, index=False)
        args = argparse.Namespace(
            data=str(data_path),
            model_out=str(tmpdir / "model.joblib"),
            report_dir=str(tmpdir / "reports" / "ml"),
            figures_dir=str(tmpdir / "figures" / "ml"),
            test_size=0.25,
            random_seed=42,
            max_features=500,
            min_df=1,
            top_k=5,
            overwrite=False,
        )
        result = TRAIN.run(args)
        return result

    def test_smoke_writes_all_outputs_with_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            result = self._run(tmpdir, extra_args=[])

            expected_files = [
                "model",
                "metrics_csv",
                "metrics_json",
                "manifest_json",
                "manifest_csv",
                "coefficients_csv",
                "report_md",
                "block_fig",
                "cm_fig",
            ]
            for key in expected_files:
                path = Path(result["paths"][key])
                self.assertTrue(path.is_file(), msg=key)
                self.assertGreater(path.stat().st_size, 0, msg=key)

            metrics = result["metrics"]
            self.assertEqual(
                sorted(metrics.keys()),
                sorted(
                    [
                        "majority_class_dummy",
                        "safe_metadata_logistic",
                        "demographics_logistic",
                        "tfidf_text_logistic",
                        TRAIN.FINAL_MODEL_NAME,
                    ]
                ),
            )
            for name, m in metrics.items():
                for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"):
                    self.assertTrue(
                        m[key] == m[key] and abs(m[key]) != float("inf"),
                        msg=f"{name}.{key}",
                    )
                    self.assertGreaterEqual(m[key], 0.0, msg=f"{name}.{key}")
                    self.assertLessEqual(m[key], 1.0, msg=f"{name}.{key}")
                self.assertIn(m["minority_label"], (0, 1))
                self.assertGreater(m["minority_support"], 0)
                cm = m["confusion_matrix"]
                # The confusion matrix covers exactly the test partition.
                self.assertEqual(
                    sum(sum(row) for row in cm["matrix"]),
                    result["manifest"]["rows_test"],
                    msg=name,
                )

            manifest = result["manifest"]
            self.assertEqual(manifest["business_overlap_train_test"], 0)
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(
                manifest["rows_train"] + manifest["rows_test"],
                manifest["rows_total"],
            )

            # Row-level manifest must cover every row exactly once.
            rows = pd.read_csv(Path(result["paths"]["manifest_csv"]))
            self.assertEqual(len(rows), 48)
            self.assertEqual(rows["review_id"].nunique(), 48)
            self.assertEqual(set(rows["split"].unique()), {"train", "test"})
            train_groups = set(rows.loc[rows["split"] == "train", "business_id"])
            test_groups = set(rows.loc[rows["split"] == "test", "business_id"])
            self.assertEqual(train_groups & test_groups, set())

            # Metrics JSON and CSV must agree on the comparator set.
            metrics_json = json.loads(
                Path(result["paths"]["metrics_json"]).read_text()
            )
            self.assertEqual(
                sorted(metrics_json["models"].keys()),
                sorted(metrics.keys()),
            )
            self.assertEqual(metrics_json["seed"], 42)
            self.assertEqual(metrics_json["input_sha256"], result["input_sha256"])

            coefficients = pd.read_csv(
                Path(result["paths"]["coefficients_csv"])
            )
            self.assertEqual(len(coefficients), 10)  # top 5 positive + top 5 negative
            self.assertEqual(
                set(coefficients["direction"].unique()), {"positive", "negative"}
            )

            report = Path(result["paths"]["report_md"]).read_text()
            self.assertIn("Final Grouped Satisfaction Model Report", report)
            self.assertIn(result["input_sha256"], report)
            self.assertIn(str(TRAIN.FINAL_MODEL_NAME), report)

    def test_refuses_overwrite_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self._run(tmpdir, extra_args=[])
            # Second run without --overwrite must fail loudly.
            import argparse

            args = argparse.Namespace(
                data=str(tmpdir / "input.csv"),
                model_out=str(tmpdir / "model.joblib"),
                report_dir=str(tmpdir / "reports" / "ml"),
                figures_dir=str(tmpdir / "figures" / "ml"),
                test_size=0.25,
                random_seed=42,
                max_features=500,
                min_df=1,
                top_k=5,
                overwrite=False,
            )
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                TRAIN.run(args)

    def test_overwrite_flag_allows_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self._run(tmpdir, extra_args=[])
            import argparse

            args = argparse.Namespace(
                data=str(tmpdir / "input.csv"),
                model_out=str(tmpdir / "model.joblib"),
                report_dir=str(tmpdir / "reports" / "ml"),
                figures_dir=str(tmpdir / "figures" / "ml"),
                test_size=0.25,
                random_seed=42,
                max_features=500,
                min_df=1,
                top_k=5,
                overwrite=True,
            )
            result = TRAIN.run(args)  # must not raise
            self.assertTrue(Path(result["paths"]["model"]).is_file())


if __name__ == "__main__":
    unittest.main()