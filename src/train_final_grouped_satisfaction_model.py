"""Train the final grouped-satisfaction model for Julie's Yelp dental project.

Deterministic, standalone CLI. Compares five models on one fixed
GroupShuffleSplit holdout (grouped by business_id, zero business overlap):

  1. majority_class_dummy
  2. safe_metadata_logistic
  3. demographics_logistic
  4. tfidf_text_logistic
  5. tfidf_text_metadata_demographics_logistic  (predetermined final artifact)

Leakage policy: review_stars and business_stars are mechanically excluded from
every primary model, together with IDs, target columns, outcome-adjacent vote
fields, and unapproved categorical/location fields. The feature registry below
is asserted before any transformer is built.

All preprocessing (imputer, scaler, TF-IDF) and every classifier are fitted
only on the training partition. Nothing is tuned against the held-out test
set; the fifth pipeline is the predetermined final artifact, not a post-hoc
winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "satisfied_binary"
GROUP = "business_id"
TEXT = "review_text"

REQUIRED_COLUMNS = [
    "review_id",
    GROUP,
    TEXT,
    TARGET,
    "review_length_words",
    "review_year",
    "business_review_count",
    # eight area_* demographic numeric features (enriched EDA)
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
]

DEMO_FEATURES = [
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
]

# Safe metadata only (business_review_count enters as log1p).
LOG_REVIEW_COUNT = "log_business_review_count"
SAFE_METADATA_FEATURES = ["review_length_words", "review_year", LOG_REVIEW_COUNT]

# Every column a model may consume: text + safe metadata + demographics.
FEATURE_REGISTRY = [TEXT] + SAFE_METADATA_FEATURES + DEMO_FEATURES

# Mechanically forbidden columns. review_stars is target-defining/leakage-prone
# and business_stars is a target/proxy-leakage risk; the rest are IDs, target
# columns, raw outcome-adjacent vote fields, and unapproved
# categorical/location fields.
FORBIDDEN_FEATURES = [
    "review_stars",
    "business_stars",
    "useful",
    "funny",
    "cool",
    TARGET,
    "negative_binary",
    "review_id",
    "user_id",
    GROUP,
    "business_name",
    "business_city",
    "business_state",
    "business_postal_code",
    "latitude",
    "longitude",
    "business_categories",
    "business_zcta",
    "area_income_band",
    "area_largest_demographic_group",
    "area_largest_demographic_share",
    "zcta_match_status",
    "zcta_match_reason",
    "area_context_note",
    "census_geo_name",
    "census_geo_level",
    "acs_year",
    "acs_dataset",
    "data_source",
    "review_date",
    "review_length_chars",
]

FINAL_MODEL_NAME = "tfidf_text_metadata_demographics_logistic"
MODEL_NAMES = [
    "majority_class_dummy",
    "safe_metadata_logistic",
    "demographics_logistic",
    "tfidf_text_logistic",
    FINAL_MODEL_NAME,
]

METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
]

DEFAULT_DATA = Path("data/final/dental_review_satisfaction_dataset.csv")
DEFAULT_MODEL_OUT = Path("models/final_grouped_satisfaction_model.joblib")
DEFAULT_REPORT_DIR = Path("reports/ml")
DEFAULT_FIGURES_DIR = Path("figures/ml")

OUTPUT_BASENAMES = {
    "metrics_csv": "final_grouped_satisfaction_metrics.csv",
    "metrics_json": "final_grouped_satisfaction_metrics.json",
    "manifest_json": "final_grouped_satisfaction_split_manifest.json",
    "manifest_csv": "final_grouped_satisfaction_split_manifest.csv",
    "coefficients_csv": "final_grouped_satisfaction_top_tfidf_coefficients.csv",
    "report_md": "FINAL_GROUPED_SATISFACTION_MODEL_REPORT.md",
    "block_fig": "final_grouped_satisfaction_feature_block_comparison.png",
    "cm_fig": "final_grouped_satisfaction_confusion_matrix.png",
}

LIMITATIONS = """\
- This model estimates association with the project's observed binary
  `satisfied_binary` label, not causal drivers of dental-care quality or
  patient satisfaction.
- Review text is written after the experience and can be highly
  outcome-revealing; strong TF-IDF performance is not evidence that a
  pre-visit intervention model would work.
- Area-level ACS features describe business-area context, not individual
  reviewers, patients, or dentists; no group-level or demographic claims
  about individuals are supported.
- The business-grouped holdout supports generalization to unseen businesses
  only; one fixed split is a reproducible comparison, not an uncertainty
  estimate.
- Coefficient terms and feature-block deltas are correlational; verify the
  target-label construction before making student-facing claims.
"""


def fail(message: str) -> None:
    """Raise a clear, single-line error."""
    raise ValueError(f"ERROR: {message}")


def _as_column_list(cols) -> list[str]:
    """Normalize a transformer column selection to a list of column names."""
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def assert_feature_registry(registry: list[str] | tuple[str, ...]) -> None:
    """Assert no forbidden column can enter a transformer.

    Mechanical guard: raises if any hard-excluded column (notably
    review_stars or business_stars) appears in the registry of columns that
    transformers are allowed to consume.
    """
    registry_set = set(registry)
    unexpected = sorted(registry_set - set(FEATURE_REGISTRY))
    if unexpected:
        fail("registry contains non-approved columns: " + ", ".join(unexpected))
    violations = [c for c in FORBIDDEN_FEATURES if c in registry_set]
    if violations:
        fail(
            "forbidden columns in feature registry: "
            + ", ".join(sorted(violations))
            + " (review_stars and business_stars must never enter a model)"
        )


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        fail(f"data file not found: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:  # noqa: BLE001
        fail(f"could not read CSV {data_path}: {exc}")
    return df


def validate_binary_target(df: pd.DataFrame) -> None:
    """Reject non-integral or non-binary target values before any int cast."""
    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        fail(f"target column {TARGET} contains non-numeric values")
    non_integer = y[y % 1 != 0]
    if not non_integer.empty:
        examples = ", ".join(str(v) for v in non_integer.unique()[:5])
        fail(
            f"target column {TARGET} must be binary 0/1; "
            f"found non-integral values: {examples}"
        )
    labels = set(y.dropna().astype(int).unique().tolist())
    if not labels <= {0, 1}:
        fail(f"target column {TARGET} must be binary 0/1; found {sorted(labels)}")


def validate_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(f"dataset is missing required columns: {', '.join(missing)}")

    if df.empty:
        fail("dataset is empty")

    validate_binary_target(df)
    text = df[TEXT].fillna("").astype(str).str.strip()
    if text.eq("").all():
        fail(f"text column {TEXT} is empty after null-fill; nothing to vectorize")

    for col in ("review_length_words", "review_year", "business_review_count"):
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = int(coerced.isna().sum())
        if bad == len(df):
            fail(f"metadata column {col} has no numeric values")


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the working frame with safe metadata and null-filled text."""
    assert_feature_registry(FEATURE_REGISTRY)
    validate_binary_target(df)
    work = df[list(dict.fromkeys(REQUIRED_COLUMNS + ["review_id"]))].copy()
    work[TEXT] = work[TEXT].fillna("").astype(str)
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce").astype(int)
    work["review_length_words"] = pd.to_numeric(
        work["review_length_words"], errors="coerce"
    )
    work["review_year"] = pd.to_numeric(work["review_year"], errors="coerce")
    counts = pd.to_numeric(work["business_review_count"], errors="coerce")
    work[LOG_REVIEW_COUNT] = np.log1p(counts)
    for col in DEMO_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    keep = ["review_id", TARGET, GROUP, TEXT] + SAFE_METADATA_FEATURES + DEMO_FEATURES
    return work[keep]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def make_grouped_split(
    frame: pd.DataFrame, seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """One fixed grouped holdout; deterministic and leakage-safe."""
    y = frame[TARGET]
    groups = frame[GROUP]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, y, groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        fail(
            f"train/test business overlap detected: {len(overlap)} business_id "
            "values appear in both partitions"
        )
    if set(y.iloc[train_idx].unique()) != {0, 1}:
        fail("training partition does not contain both target labels")
    if set(y.iloc[test_idx].unique()) != {0, 1}:
        fail("test partition does not contain both target labels")
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def numeric_transformer() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def tfidf_vectorizer(max_features: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
        stop_words="english",
    )


def logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    )


def build_models(seed: int, max_features: int, min_df: int) -> dict[str, object]:
    """The five declared comparators; nothing more, nothing less."""
    metadata_cols = SAFE_METADATA_FEATURES
    demo_cols = DEMO_FEATURES

    models: dict[str, object] = {
        "majority_class_dummy": DummyClassifier(strategy="most_frequent"),
        "safe_metadata_logistic": Pipeline(
            [
                (
                    "pre",
                    ColumnTransformer([("num", numeric_transformer(), metadata_cols)]),
                ),
                ("clf", logistic(seed)),
            ]
        ),
        "demographics_logistic": Pipeline(
            [
                (
                    "pre",
                    ColumnTransformer([("num", numeric_transformer(), demo_cols)]),
                ),
                ("clf", logistic(seed)),
            ]
        ),
        "tfidf_text_logistic": Pipeline(
            [
                ("pre", ColumnTransformer([("text", tfidf_vectorizer(max_features, min_df), TEXT)])),
                ("clf", logistic(seed)),
            ]
        ),
        FINAL_MODEL_NAME: Pipeline(
            [
                (
                    "pre",
                    ColumnTransformer(
                        [
                            (
                                "text",
                                tfidf_vectorizer(max_features, min_df),
                                TEXT,
                            ),
                            (
                                "num",
                                numeric_transformer(),
                                metadata_cols + demo_cols,
                            ),
                        ]
                    ),
                ),
                ("clf", logistic(seed)),
            ]
        ),
    }
    if list(models.keys()) != MODEL_NAMES:
        fail("comparator set deviates from the five declared models")
    # Mechanical leakage guard for every transformer's column selection.
    for name, model in models.items():
        if name == "majority_class_dummy":
            continue
        pre = model.named_steps["pre"]  # type: ignore[union-attr]
        used: list[str] = []
        for _, _trans, cols in pre.transformers:  # type: ignore[union-attr]
            if cols == "drop" or cols is None:
                continue
            used.extend(_as_column_list(cols))
        assert_feature_registry(used)
    return models


def model_inputs(name: str, train: pd.DataFrame, test: pd.DataFrame):
    if name == "majority_class_dummy":
        return train, test
    if name == "tfidf_text_logistic":
        return train[[TEXT]], test[[TEXT]]
    return train, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def minority_label(y_test: pd.Series) -> int:
    counts = y_test.value_counts()
    minority = counts.idxmin()
    ties = counts[counts == counts.min()].index.tolist()
    return int(min(ties))


def evaluate_model(
    name: str,
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    Xtr, Xte = model_inputs(name, train, test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xte)[:, 1]
        else:
            proba = np.full(len(y_test), 0.5)

    minority = minority_label(y_test)
    minority_support = int((y_test == minority).sum())
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "minority_label": int(minority),
        "minority_support": minority_support,
        "minority_precision": float(
            precision_score(y_test, pred, pos_label=minority, zero_division=0)
        ),
        "minority_recall": float(
            recall_score(y_test, pred, pos_label=minority, zero_division=0)
        ),
        "minority_f1": float(
            f1_score(y_test, pred, pos_label=minority, zero_division=0)
        ),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
        },
    }
    for key in METRIC_KEYS:
        if not np.isfinite(metrics[key]):
            fail(f"model {name} produced non-finite {key}")
    return metrics


def top_tfidf_coefficients(
    final_model: Pipeline, top_k: int
) -> pd.DataFrame:
    pre = final_model.named_steps["pre"]
    text_transformer = pre.named_transformers_["text"]
    clf = final_model.named_steps["clf"]
    names = text_transformer.get_feature_names_out()
    # The full coefficient vector is the TF-IDF block followed by the numeric
    # metadata/demographic block. Order and export only the TF-IDF slice so
    # dominant numeric coefficients can never leak into term selection.
    text_coefs = np.asarray(clf.coef_[0])[: len(names)]
    order = np.argsort(text_coefs)
    negatives = order[:top_k]
    positives = order[::-1][:top_k]
    rows = []
    for rank, idx in enumerate(positives, start=1):
        rows.append(
            {
                "rank": rank,
                "term": names[idx],
                "coefficient": float(text_coefs[idx]),
                "direction": "positive",
                "association": "satisfied_binary=1 (satisfied)",
            }
        )
    for rank, idx in enumerate(negatives, start=1):
        rows.append(
            {
                "rank": rank,
                "term": names[idx],
                "coefficient": float(text_coefs[idx]),
                "direction": "negative",
                "association": "satisfied_binary=0 (not satisfied)",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def resolve_output_paths(
    model_out: Path, report_dir: Path, figures_dir: Path
) -> dict[str, Path]:
    paths = {
        "model": model_out,
        "metrics_csv": report_dir / OUTPUT_BASENAMES["metrics_csv"],
        "metrics_json": report_dir / OUTPUT_BASENAMES["metrics_json"],
        "manifest_json": report_dir / OUTPUT_BASENAMES["manifest_json"],
        "manifest_csv": report_dir / OUTPUT_BASENAMES["manifest_csv"],
        "coefficients_csv": report_dir / OUTPUT_BASENAMES["coefficients_csv"],
        "report_md": report_dir / OUTPUT_BASENAMES["report_md"],
        "block_fig": figures_dir / OUTPUT_BASENAMES["block_fig"],
        "cm_fig": figures_dir / OUTPUT_BASENAMES["cm_fig"],
    }
    return paths


def check_overwrite(paths: dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        fail(
            "refusing to overwrite existing outputs (pass --overwrite to allow): "
            + ", ".join(existing)
        )


def ensure_parent_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def build_manifest(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    test_size: float,
    input_sha256: str,
    data_path: Path,
) -> tuple[dict, pd.DataFrame]:
    y = frame[TARGET]
    groups = frame[GROUP]
    summary = {
        "seed": int(seed),
        "test_size": float(test_size),
        "splitter": "GroupShuffleSplit(n_splits=1, test_size=%.2f, random_state=%d)"
        % (test_size, seed),
        "input_sha256": input_sha256,
        "data_path": str(data_path),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_idx)),
        "rows_test": int(len(test_idx)),
        "groups_total": int(groups.nunique()),
        "groups_train": int(groups.iloc[train_idx].nunique()),
        "groups_test": int(groups.iloc[test_idx].nunique()),
        "class_counts_total": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "class_counts_train": {
            str(k): int(v)
            for k, v in y.iloc[train_idx].value_counts().sort_index().items()
        },
        "class_counts_test": {
            str(k): int(v)
            for k, v in y.iloc[test_idx].value_counts().sort_index().items()
        },
        "business_overlap_train_test": 0,
    }
    order = np.concatenate([train_idx, test_idx])
    rows = pd.DataFrame(
        {
            "review_id": frame["review_id"].iloc[order].values,
            "business_id": groups.iloc[order].values,
            "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
        }
    )
    return summary, rows


def write_block_figure(metrics: dict[str, dict], path: Path) -> None:
    labels = [n for n in MODEL_NAMES]
    bal = [metrics[n]["balanced_accuracy"] for n in labels]
    macro = [metrics[n]["macro_f1"] for n in labels]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, bal, width, label="Balanced accuracy", color="#4C78A8")
    plt.bar(x + width / 2, macro, width, label="Macro-F1", color="#F58518")
    plt.axhline(0.5, color="grey", linewidth=1, linestyle="--")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Feature-block comparison on the grouped business holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_confusion_figure(
    metrics: dict[str, dict], name: str, path: Path
) -> None:
    cm = np.array(metrics[name]["confusion_matrix"]["matrix"])
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix: {name}")
    plt.colorbar()
    plt.xticks([0, 1], ["not satisfied (0)", "satisfied (1)"])
    plt.yticks([0, 1], ["not satisfied (0)", "satisfied (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def build_report(
    invocation: str,
    commit: str,
    input_sha256: str,
    manifest_summary: dict,
    metrics: dict[str, dict],
    deltas: dict[str, float],
    paths: dict[str, Path],
    versions: dict[str, str],
    feature_policy: str,
) -> str:
    table_rows = [
        "| model | accuracy | balanced acc | macro-F1 | ROC-AUC | minority label | minority P | minority R | minority F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in MODEL_NAMES:
        m = metrics[name]
        table_rows.append(
            f"| {name} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
            f"{m['macro_f1']:.4f} | {m['roc_auc']:.4f} | {m['minority_label']} "
            f"(n={m['minority_support']}) | {m['minority_precision']:.4f} | "
            f"{m['minority_recall']:.4f} | {m['minority_f1']:.4f} |"
        )
    delta_rows = "\n".join(
        f"- {key.replace('_', ' ')}: {value:+.4f}" for key, value in deltas.items()
    )
    return f"""# Final Grouped Satisfaction Model Report

## Invocation

```text
{invocation}
```

## Provenance

- Repository commit: `{commit}`
- Input data SHA-256: `{input_sha256}`
- Seed: `{manifest_summary['seed']}` (single fixed split, every comparator)
- Split: `{manifest_summary['splitter']}`
- Rows: {manifest_summary['rows_total']:,} total / {manifest_summary['rows_train']:,} train / {manifest_summary['rows_test']:,} test
- Businesses (groups): {manifest_summary['groups_total']:,} total / {manifest_summary['groups_train']:,} train / {manifest_summary['groups_test']:,} test
- Business overlap between train and test: **{manifest_summary['business_overlap_train_test']}** (grouped holdout, zero leakage by business)
- Class counts (total / train / test): {manifest_summary['class_counts_total']} / {manifest_summary['class_counts_train']} / {manifest_summary['class_counts_test']}
- Versions: python {versions['python']}, scikit-learn {versions['sklearn']}, joblib {versions['joblib']}, pandas {versions['pandas']}, numpy {versions['numpy']}

## Feature policy

{feature_policy}

## Comparator table

{chr(10).join(table_rows)}

The final artifact is the predetermined `{FINAL_MODEL_NAME}` pipeline. No
model was selected post hoc from these test metrics, and no model was refit
on all rows.

## Full model vs text-only deltas

{delta_rows}

These deltas are correlational feature-block comparisons on one fixed split,
not causal effects.

## Figures

- Feature-block comparison: `{paths['block_fig']}`
- Full-model confusion matrix: `{paths['cm_fig']}`

## Outputs

- Model artifact: `{paths['model']}`
- Metrics: `{paths['metrics_csv']}`, `{paths['metrics_json']}`
- Split manifest: `{paths['manifest_json']}`, `{paths['manifest_csv']}`
- Top TF-IDF coefficients: `{paths['coefficients_csv']}`

## Limitations

{LIMITATIONS}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    commit = git_commit()
    invocation = (
        "python src/train_final_grouped_satisfaction_model.py "
        f"--data {args.data} --model-out {args.model_out} "
        f"--report-dir {args.report_dir} --figures-dir {args.figures_dir} "
        f"--test-size {args.test_size} --random-seed {args.random_seed} "
        f"--max-features {args.max_features} --min-df {args.min_df} "
        f"--top-k {args.top_k}"
        + (" --overwrite" if args.overwrite else "")
    )

    data_path = Path(args.data)
    paths = resolve_output_paths(
        Path(args.model_out), Path(args.report_dir), Path(args.figures_dir)
    )
    check_overwrite(paths, args.overwrite)
    ensure_parent_dirs(paths)

    df = load_dataset(data_path)
    validate_dataframe(df)
    input_sha256 = sha256_of_file(data_path)
    frame = prepare_model_frame(df)

    train_idx, test_idx = make_grouped_split(
        frame, args.random_seed, args.test_size
    )
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    y_train = train[TARGET]
    y_test = test[TARGET]

    models = build_models(args.random_seed, args.max_features, args.min_df)
    metrics: dict[str, dict] = {}
    for name in MODEL_NAMES:
        metrics[name] = evaluate_model(
            name, models[name], train, test, y_train, y_test
        )

    text_only = metrics["tfidf_text_logistic"]
    full = metrics[FINAL_MODEL_NAME]
    deltas = {
        "accuracy_delta_full_vs_text_only": full["accuracy"] - text_only["accuracy"],
        "balanced_accuracy_delta_full_vs_text_only": full["balanced_accuracy"]
        - text_only["balanced_accuracy"],
        "macro_f1_delta_full_vs_text_only": full["macro_f1"]
        - text_only["macro_f1"],
        "roc_auc_delta_full_vs_text_only": full["roc_auc"] - text_only["roc_auc"],
    }

    manifest_summary, manifest_rows = build_manifest(
        frame, train_idx, test_idx, args.random_seed, args.test_size,
        input_sha256, data_path,
    )

    versions = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    feature_policy = (
        "Safe metadata: review_length_words, review_year, log1p(business_review_count). "
        "Demographics: the eight area_* numeric ACS features from the enriched EDA "
        "(median-imputed, standard-scaled). Text: review_text (null -> \"\"), TF-IDF "
        f"English stop words, 1-2 grams, max_features={args.max_features}, min_df={args.min_df}. "
        "Classifier: LogisticRegression(max_iter=1000, class_weight='balanced', "
        f"random_state={args.random_seed}). Hard-excluded from every model: review_stars, "
        "business_stars, IDs, target columns, useful/funny/cool, and unapproved "
        "categorical/location fields (mechanically asserted via the feature registry)."
    )

    # ---- write outputs ----
    metrics_rows = []
    for name in MODEL_NAMES:
        m = metrics[name]
        metrics_rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "roc_auc": m["roc_auc"],
                "minority_label": m["minority_label"],
                "minority_support": m["minority_support"],
                "minority_precision": m["minority_precision"],
                "minority_recall": m["minority_recall"],
                "minority_f1": m["minority_f1"],
            }
        )
    pd.DataFrame(metrics_rows).to_csv(paths["metrics_csv"], index=False)

    metrics_json = {
        "seed": int(args.random_seed),
        "input_sha256": input_sha256,
        "commit": commit,
        "invocation": invocation,
        "versions": versions,
        "final_model": FINAL_MODEL_NAME,
        "split_summary": manifest_summary,
        "feature_policy": feature_policy,
        "deltas_full_vs_text_only": deltas,
        "models": metrics,
    }
    paths["metrics_json"].write_text(json.dumps(metrics_json, indent=2) + "\n")

    manifest_summary_json = dict(manifest_summary)
    manifest_summary_json["input_sha256"] = input_sha256
    paths["manifest_json"].write_text(
        json.dumps(manifest_summary_json, indent=2) + "\n"
    )
    manifest_rows.to_csv(paths["manifest_csv"], index=False)

    coef_df = top_tfidf_coefficients(
        models[FINAL_MODEL_NAME],  # type: ignore[arg-type]
        args.top_k,
    )
    coef_df.to_csv(paths["coefficients_csv"], index=False)

    report = build_report(
        invocation, commit, input_sha256, manifest_summary, metrics,
        deltas, paths, versions, feature_policy,
    )
    paths["report_md"].write_text(report)

    write_block_figure(metrics, paths["block_fig"])
    write_confusion_figure(metrics, FINAL_MODEL_NAME, paths["cm_fig"])

    payload = {
        "pipeline": models[FINAL_MODEL_NAME],
        "final_model_name": FINAL_MODEL_NAME,
        "feature_registry": FEATURE_REGISTRY,
        "forbidden_features": FORBIDDEN_FEATURES,
        "target": TARGET,
        "group": GROUP,
        "text": TEXT,
        "seed": int(args.random_seed),
        "split_summary": manifest_summary,
        "input_sha256": input_sha256,
        "data_path": str(data_path),
        "commit": commit,
        "invocation": invocation,
        "versions": versions,
        "fit_note": "Pipeline fitted on the training partition only; grouped holdout untouched by fitting.",
    }
    joblib.dump(payload, paths["model"])

    for key, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"output missing or empty after run: {key} -> {path}")

    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "commit": commit,
        "input_sha256": input_sha256,
        "manifest": manifest_summary,
        "metrics": metrics,
        "deltas": deltas,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the final grouped satisfaction model: five comparators on one "
            "fixed business-grouped holdout, with leakage guards and full provenance."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to the final dataset CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(DEFAULT_MODEL_OUT),
        help="Path for the joblib model artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for metrics, manifest, coefficients and report (default: %(default)s)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(DEFAULT_FIGURES_DIR),
        help="Directory for figures (default: %(default)s)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Grouped holdout fraction (default: %(default)s)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the split and classifiers (default: %(default)s)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="TF-IDF max features (default: %(default)s)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="TF-IDF min document frequency (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k positive and negative TF-IDF coefficients to export (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files (default: refuse)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.test_size < 1.0:
        print("ERROR: --test-size must be strictly between 0 and 1", file=sys.stderr)
        return 2
    if args.max_features < 1 or args.min_df < 1 or args.top_k < 1:
        print("ERROR: --max-features, --min-df and --top-k must be >= 1", file=sys.stderr)
        return 2
    try:
        run(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())