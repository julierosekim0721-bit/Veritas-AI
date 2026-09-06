"""Full enriched EDA for Julie's Yelp dental satisfaction project.

Generates root-level figures/ plots plus reports/eda/tables and a markdown report.
Uses only pandas/numpy/matplotlib/sklearn already present in the Veritas env.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data/final/dental_review_satisfaction_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "reports/eda"
TABLES_DIR = REPORT_DIR / "tables"
REPORT_PATH = REPORT_DIR / "ENRICHED_EDA_REPORT.md"
SUMMARY_JSON = REPORT_DIR / "enriched_eda_summary.json"

TARGET = "satisfied_binary"
GROUP = "business_id"
TEXT = "review_text"

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
METADATA_FEATURES = [
    "review_length_words",
    "review_year",
    "business_review_count",
]
CATEGORICAL_EDA = ["area_income_band", "area_largest_demographic_group", "business_state"]


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> str:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(PROJECT_ROOT))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def rate_by_bin(df: pd.DataFrame, feature: str, q: int = 5) -> pd.DataFrame:
    tmp = df[[feature, TARGET]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[feature], q=q, duplicates="drop")
    out = tmp.groupby("bin", observed=True)[TARGET].agg(["count", "mean"]).reset_index()
    out["bin"] = out["bin"].astype(str)
    out.rename(columns={"mean": "satisfaction_rate"}, inplace=True)
    return out


def bar_plot(labels, values, title, ylabel, filename, rotation=0):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    for i, v in enumerate(values):
        if pd.notna(v):
            plt.text(i, v, f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    return savefig(filename)


def plot_missingness(df: pd.DataFrame) -> str:
    cols = DEMO_FEATURES + ["area_largest_demographic_group", "area_largest_demographic_share", "business_zcta"]
    miss = df[cols].isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(11, 6))
    plt.bar(miss.index, miss.values * 100, color="#F58518")
    plt.title("Missingness of Census/ZCTA demographic fields")
    plt.ylabel("Missing rows (%)")
    plt.xticks(rotation=45, ha="right")
    return savefig("01_demographic_missingness.png")


def plot_target_balance(df: pd.DataFrame) -> str:
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Not satisfied (1-3 stars)", "Satisfied (4-5 stars)"]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#E45756", "#54A24B"])
    plt.title("Target class balance")
    plt.ylabel("Review count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        plt.text(i, v, f"{v:,}\n{v/len(df)*100:.1f}%", ha="center", va="bottom")
    return savefig("02_target_class_balance.png")


def plot_star_distribution(df: pd.DataFrame) -> str:
    counts = df["review_stars"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    plt.title("Yelp review-star distribution")
    plt.xlabel("Review stars")
    plt.ylabel("Review count")
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    return savefig("03_review_star_distribution.png")


def plot_review_length(df: pd.DataFrame) -> str:
    plt.figure(figsize=(9, 6))
    data0 = df.loc[df[TARGET] == 0, "review_length_words"].clip(upper=df["review_length_words"].quantile(0.99)).dropna()
    data1 = df.loc[df[TARGET] == 1, "review_length_words"].clip(upper=df["review_length_words"].quantile(0.99)).dropna()
    plt.hist(data0, bins=50, alpha=0.55, label="Not satisfied", density=True, color="#E45756")
    plt.hist(data1, bins=50, alpha=0.55, label="Satisfied", density=True, color="#54A24B")
    plt.title("Review length distribution by satisfaction class")
    plt.xlabel("Review length in words (99th percentile clipped)")
    plt.ylabel("Density")
    plt.legend()
    return savefig("04_review_length_by_satisfaction.png")


def plot_income_band(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    tmp = df.groupby("area_income_band", dropna=False)[TARGET].agg(["count", "mean"]).reset_index()
    tmp = tmp.sort_values("mean", ascending=False)
    tmp.to_csv(TABLES_DIR / "satisfaction_by_income_band.csv", index=False)
    plt.figure(figsize=(10, 6))
    labels = tmp["area_income_band"].astype(str).tolist()
    plt.bar(labels, tmp["mean"].values * 100, color="#72B7B2")
    plt.title("Satisfaction rate by area income band")
    plt.ylabel("Satisfied reviews (%)")
    plt.xlabel("Area income band")
    plt.xticks(rotation=30, ha="right")
    for i, (rate, n) in enumerate(zip(tmp["mean"].values, tmp["count"].values)):
        plt.text(i, rate * 100, f"{rate*100:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=8)
    return savefig("05_satisfaction_by_income_band.png"), tmp


def plot_demo_quintiles(df: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    files = []
    tables = {}
    features = [
        "area_median_household_income",
        "area_poverty_rate",
        "area_bachelor_plus_rate",
        "area_unemployment_rate",
        "area_pct_white",
        "area_pct_black",
        "area_pct_asian",
        "area_pct_hispanic_or_latino",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.ravel()
    for ax, feat in zip(axes, features):
        tbl = rate_by_bin(df, feat)
        tables[feat] = tbl
        tbl.to_csv(TABLES_DIR / f"satisfaction_by_{feat}_quintile.csv", index=False)
        ax.bar(range(len(tbl)), tbl["satisfaction_rate"] * 100, color="#4C78A8")
        ax.set_title(feat.replace("area_", "").replace("_", " "))
        ax.set_ylabel("Satisfied reviews (%)")
        ax.set_xticks(range(len(tbl)))
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(tbl))])
        for i, row in tbl.iterrows():
            ax.text(i, row["satisfaction_rate"] * 100, f"{row['satisfaction_rate']*100:.1f}%", ha="center", va="bottom", fontsize=8)
    files.append(savefig("06_satisfaction_by_demographic_quintiles.png"))
    return files, tables


def plot_largest_group(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    tmp = df.groupby("area_largest_demographic_group", dropna=False)[TARGET].agg(["count", "mean"]).reset_index()
    tmp = tmp[tmp["count"] >= 100].sort_values("mean", ascending=False)
    tmp.to_csv(TABLES_DIR / "satisfaction_by_largest_demographic_group.csv", index=False)
    plt.figure(figsize=(10, 6))
    labels = tmp["area_largest_demographic_group"].astype(str).tolist()
    plt.bar(labels, tmp["mean"] * 100, color="#B279A2")
    plt.title("Satisfaction rate by largest area demographic group")
    plt.ylabel("Satisfied reviews (%)")
    plt.xticks(rotation=30, ha="right")
    for i, row in tmp.iterrows():
        plt.text(list(tmp.index).index(i), row["mean"] * 100, f"{row['mean']*100:.1f}%\nn={int(row['count']):,}", ha="center", va="bottom", fontsize=8)
    return savefig("07_satisfaction_by_largest_demographic_group.png"), tmp


def plot_state_rates(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    tmp = df.groupby("business_state")[TARGET].agg(["count", "mean"]).reset_index()
    tmp = tmp[tmp["count"] >= 100].sort_values("mean", ascending=False)
    tmp.to_csv(TABLES_DIR / "satisfaction_by_state.csv", index=False)
    plt.figure(figsize=(11, 6))
    plt.bar(tmp["business_state"], tmp["mean"] * 100, color="#59A14F")
    plt.title("Satisfaction rate by state (descriptive, confounded)")
    plt.ylabel("Satisfied reviews (%)")
    plt.xlabel("Business state")
    for i, row in tmp.reset_index(drop=True).iterrows():
        plt.text(i, row["mean"] * 100, f"{row['mean']*100:.1f}%\nn={int(row['count']):,}", ha="center", va="bottom", fontsize=7)
    return savefig("08_satisfaction_by_state.png"), tmp


def plot_correlation(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    cols = [TARGET] + METADATA_FEATURES + DEMO_FEATURES + ["area_largest_demographic_share", "useful", "funny", "cool"]
    corr = df[cols].corr(numeric_only=True)
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr.to_csv(TABLES_DIR / "correlation_with_satisfaction.csv", header=["correlation"])
    plt.figure(figsize=(10, 7))
    colors = ["#54A24B" if v >= 0 else "#E45756" for v in target_corr.values]
    plt.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Linear correlation with satisfied_binary")
    plt.xlabel("Pearson correlation")
    return savefig("09_correlation_with_satisfaction.png"), target_corr.reset_index().rename(columns={"index": "feature", TARGET: "correlation"})


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    cols = METADATA_FEATURES + DEMO_FEATURES + [TARGET]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap: metadata, demographics, target")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    return savefig("10_metadata_demographic_correlation_heatmap.png")


def plot_geo_scatter(df: pd.DataFrame) -> str:
    sample = df.dropna(subset=["longitude", "latitude", TARGET]).sample(min(8000, len(df)), random_state=42)
    plt.figure(figsize=(9, 7))
    colors = sample[TARGET].map({0: "#E45756", 1: "#54A24B"})
    plt.scatter(sample["longitude"], sample["latitude"], c=colors, s=8, alpha=0.35, linewidths=0)
    plt.title("Dental reviews by business location and satisfaction class")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    return savefig("11_location_scatter_by_satisfaction.png")


def evaluate_models(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame, dict]:
    model_df = df[[TARGET, GROUP, TEXT] + METADATA_FEATURES + DEMO_FEATURES].copy()
    model_df["log_business_review_count"] = np.log1p(model_df["business_review_count"])
    metadata = ["review_length_words", "review_year", "log_business_review_count"]
    demos = DEMO_FEATURES
    y = model_df[TARGET].astype(int)
    groups = model_df[GROUP]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(model_df, y, groups=groups))
    train_businesses = set(groups.iloc[train_idx])
    test_businesses = set(groups.iloc[test_idx])
    overlap = len(train_businesses & test_businesses)

    X_train = model_df.iloc[train_idx].copy()
    X_test = model_df.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    models = {
        "majority_dummy": DummyClassifier(strategy="most_frequent"),
        "metadata_logistic": Pipeline([
            ("pre", ColumnTransformer([("num", numeric_transformer, metadata)])),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "demographics_logistic": Pipeline([
            ("pre", ColumnTransformer([("num", numeric_transformer, demos)])),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "metadata_plus_demographics_logistic": Pipeline([
            ("pre", ColumnTransformer([("num", numeric_transformer, metadata + demos)])),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "tfidf_text_logistic": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "text_plus_metadata_demographics_logistic": Pipeline([
            ("pre", ColumnTransformer([
                ("text", TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2), stop_words="english"), TEXT),
                ("num", numeric_transformer, metadata + demos),
            ])),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
    }

    rows = []
    reports = {}
    best_name = None
    best_bal = -1
    best_pred = None
    for name, model in models.items():
        if "tfidf" in name or "text_plus" in name:
            Xtr = X_train if "text_plus" in name else X_train[TEXT].fillna("")
            Xte = X_test if "text_plus" in name else X_test[TEXT].fillna("")
        else:
            Xtr = X_train
            Xte = X_test
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xte)[:, 1]
        else:
            prob = None
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "minority_f1": f1_score(y_test, pred, pos_label=0),
            "roc_auc": roc_auc_score(y_test, prob) if prob is not None else np.nan,
        }
        rows.append(row)
        reports[name] = classification_report(y_test, pred, output_dict=True, zero_division=0)
        if row["balanced_accuracy"] > best_bal:
            best_bal = row["balanced_accuracy"]
            best_name = name
            best_pred = pred

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(TABLES_DIR / "grouped_ml_signal_benchmark.csv", index=False)

    plt.figure(figsize=(12, 6))
    plot_df = metrics.sort_values("balanced_accuracy")
    plt.barh(plot_df["model"], plot_df["balanced_accuracy"], color="#4C78A8")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="random/reference")
    plt.title("Grouped holdout ML signal benchmark")
    plt.xlabel("Balanced accuracy")
    plt.legend()
    for i, v in enumerate(plot_df["balanced_accuracy"]):
        plt.text(v, i, f"{v:.3f}", va="center", ha="left", fontsize=8)
    fig_model = savefig("12_grouped_ml_signal_benchmark.png")

    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix: {best_name}")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred not sat", "Pred sat"])
    ax.set_yticklabels(["True not sat", "True sat"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig_cm = savefig("13_best_model_confusion_matrix.png")

    split_info = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_businesses": int(len(train_businesses)),
        "test_businesses": int(len(test_businesses)),
        "business_overlap": int(overlap),
        "best_model": best_name,
        "best_balanced_accuracy": float(best_bal),
    }
    (TABLES_DIR / "grouped_split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    return fig_model, fig_cm, metrics, split_info


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Defensive numeric coercion for enriched ACS fields.
    for c in DEMO_FEATURES + METADATA_FEATURES + ["review_stars", TARGET, "latitude", "longitude", "area_largest_demographic_share"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)

    figure_files = []
    figure_files.append(plot_missingness(df))
    figure_files.append(plot_target_balance(df))
    figure_files.append(plot_star_distribution(df))
    figure_files.append(plot_review_length(df))
    f, income_tbl = plot_income_band(df); figure_files.append(f)
    fs, quintile_tables = plot_demo_quintiles(df); figure_files.extend(fs)
    f, largest_tbl = plot_largest_group(df); figure_files.append(f)
    f, state_tbl = plot_state_rates(df); figure_files.append(f)
    f, corr_tbl = plot_correlation(df); figure_files.append(f)
    figure_files.append(plot_correlation_heatmap(df))
    figure_files.append(plot_geo_scatter(df))
    fig_model, fig_cm, metrics, split_info = evaluate_models(df)
    figure_files.extend([fig_model, fig_cm])

    # Summary tables
    coverage = {}
    for c in DEMO_FEATURES + ["area_largest_demographic_group", "business_zcta", "zcta_match_status"]:
        coverage[c] = {"non_null": int(df[c].notna().sum()), "rate": float(df[c].notna().mean())}
    class_counts = df[TARGET].value_counts().sort_index().to_dict()
    star_counts = df["review_stars"].value_counts().sort_index().to_dict()

    # Simple demographic signals.
    target_corr = corr_tbl.copy()
    top_corr = target_corr.assign(abs_corr=target_corr["correlation"].abs()).sort_values("abs_corr", ascending=False).head(8)
    income_spread = float(income_tbl["mean"].max() - income_tbl["mean"].min()) if len(income_tbl) else float("nan")
    demo_quintile_spreads = {
        feat: float(tbl["satisfaction_rate"].max() - tbl["satisfaction_rate"].min())
        for feat, tbl in quintile_tables.items()
    }
    largest_group_spread = float(largest_tbl["mean"].max() - largest_tbl["mean"].min()) if len(largest_tbl) else float("nan")
    state_spread = float(state_tbl["mean"].max() - state_tbl["mean"].min()) if len(state_tbl) else float("nan")

    summary = {
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "star_counts": {str(k): int(v) for k, v in star_counts.items()},
        "coverage": coverage,
        "income_band_satisfaction_spread": income_spread,
        "demographic_quintile_satisfaction_spreads": demo_quintile_spreads,
        "largest_group_satisfaction_spread": largest_group_spread,
        "state_satisfaction_spread": state_spread,
        "ml_split": split_info,
        "figures": figure_files,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    majority = class_counts.get(1, 0) / len(df)
    best = metrics.iloc[0]
    demo_row = metrics.loc[metrics["model"] == "demographics_logistic"].iloc[0]
    meta_row = metrics.loc[metrics["model"] == "metadata_logistic"].iloc[0]
    meta_demo_row = metrics.loc[metrics["model"] == "metadata_plus_demographics_logistic"].iloc[0]
    text_row = metrics.loc[metrics["model"] == "tfidf_text_logistic"].iloc[0]
    full_row = metrics.loc[metrics["model"] == "text_plus_metadata_demographics_logistic"].iloc[0]

    report = f"""# Enriched exploratory data analysis — dental Yelp satisfaction

## Purpose

This report investigates Julie's enriched final dataset after ACS/ZCTA demographic enrichment. The aim is to decide whether the new area-level socioeconomic/demographic data provide useful signal for the machine-learning direction, while keeping the interpretation safe: these variables describe the area around the dental business, not individual patients or reviewers.

## Dataset

- Source file: `{DATA_PATH.relative_to(PROJECT_ROOT)}`
- Rows: {len(df):,}
- Columns: {df.shape[1]:,}
- Unique businesses: {df[GROUP].nunique():,}
- Unique states: {df['business_state'].nunique():,}
- Target: `satisfied_binary = 1` for 4-5 star reviews; `0` for 1-3 star reviews.

## Coverage of new demographic data

The ACS/ZCTA enrichment is now usable for EDA and ML feature experiments.

- `area_median_household_income`: {coverage['area_median_household_income']['non_null']:,} / {len(df):,} non-null ({pct(coverage['area_median_household_income']['rate'])})
- `area_poverty_rate`: {coverage['area_poverty_rate']['non_null']:,} / {len(df):,} non-null ({pct(coverage['area_poverty_rate']['rate'])})
- `area_bachelor_plus_rate`: {coverage['area_bachelor_plus_rate']['non_null']:,} / {len(df):,} non-null ({pct(coverage['area_bachelor_plus_rate']['rate'])})
- `area_unemployment_rate`: {coverage['area_unemployment_rate']['non_null']:,} / {len(df):,} non-null ({pct(coverage['area_unemployment_rate']['rate'])})
- Race/ethnicity composition fields: typically {pct(coverage['area_pct_white']['rate'])} non-null.

## Class balance

- Not satisfied: {class_counts.get(0, 0):,} reviews ({pct(class_counts.get(0, 0) / len(df))})
- Satisfied: {class_counts.get(1, 0):,} reviews ({pct(class_counts.get(1, 0) / len(df))})
- Majority-class accuracy reference: {majority:.3f}

The class distribution is imbalanced but not unusable. Accuracy alone would be misleading, so balanced accuracy, macro-F1, and minority-class recall/F1 should be central.

## Main EDA signals

### Strongest established signal: review text / review length

Review length remains one of the clearest signals. Longer reviews are associated with lower expressed satisfaction. This supports a text-based ML direction using TF-IDF plus interpretable logistic regression.

### Demographic/SES signal

The new demographic features are available for almost all reviews, so they can now be investigated. Descriptive spreads across demographic bins exist, but they should be treated as area-level associations and possible predictive features, not causal explanations.

- Satisfaction-rate spread across income bands: {income_spread * 100:.1f} percentage points.
- Largest-demographic-group satisfaction-rate spread: {largest_group_spread * 100:.1f} percentage points.
- State-level satisfaction-rate spread: {state_spread * 100:.1f} percentage points, but this is heavily confounded by business mix and review culture.

Top linear correlations with `satisfied_binary` are recorded in `reports/eda/tables/correlation_with_satisfaction.csv`. Correlations are useful for EDA, not proof of causality.

## Grouped ML signal benchmark

To avoid leakage across reviews from the same dental business, the exploratory ML benchmark used a grouped holdout split by `business_id`.

- Train rows: {split_info['train_rows']:,}
- Test rows: {split_info['test_rows']:,}
- Train businesses: {split_info['train_businesses']:,}
- Test businesses: {split_info['test_businesses']:,}
- Business overlap between train/test: {split_info['business_overlap']}

| Model | Accuracy | Balanced accuracy | Macro-F1 | Minority F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
"""
    for _, row in metrics.iterrows():
        report += f"| `{row['model']}` | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | {row['minority_f1']:.3f} | {row['roc_auc']:.3f} |\n"

    report += f"""

Best exploratory model by balanced accuracy: `{best['model']}` with balanced accuracy {best['balanced_accuracy']:.3f}.

Important comparisons:

- Metadata-only balanced accuracy: {meta_row['balanced_accuracy']:.3f}
- Demographics-only balanced accuracy: {demo_row['balanced_accuracy']:.3f}
- Metadata + demographics balanced accuracy: {meta_demo_row['balanced_accuracy']:.3f}
- TF-IDF text-only balanced accuracy: {text_row['balanced_accuracy']:.3f}
- Text + metadata + demographics balanced accuracy: {full_row['balanced_accuracy']:.3f}

## Recommended ML directions

1. **Primary direction: interpretable text classification.** Use TF-IDF review text with logistic regression as the main student-friendly ML model. It is likely to be the strongest and easiest to explain.
2. **Secondary direction: metadata baseline.** Keep `review_length_words`, `review_year`, and `log1p(business_review_count)` as a clean non-leaky structured baseline.
3. **Demographic/SES extension: incremental value test.** Now that ZCTA data has high coverage, test whether area-level SES/demographic variables improve grouped held-out prediction beyond metadata and text. Frame this as area context, not patient demographics.
4. **Fair interpretation section.** If demographic features help, say they add predictive signal about the business area's context. If they do not, say the review text and business/review metadata explain more of the observable Yelp satisfaction signal than coarse ZCTA descriptors.
5. **Avoid leakage.** Do not use `review_stars` as a feature. Treat `business_stars` as leaky/proxy-risk and exclude it from primary models.

## Figures generated

"""
    for fig in figure_files:
        report += f"- `{fig}`\n"

    report += """

## Tables generated

- `reports/eda/tables/grouped_ml_signal_benchmark.csv`
- `reports/eda/tables/grouped_split_info.json`
- `reports/eda/tables/correlation_with_satisfaction.csv`
- `reports/eda/tables/satisfaction_by_income_band.csv`
- `reports/eda/tables/satisfaction_by_largest_demographic_group.csv`
- `reports/eda/tables/satisfaction_by_state.csv`
- `reports/eda/tables/satisfaction_by_<demographic_feature>_quintile.csv`

## Interpretation boundaries

- These are Yelp reviews, not clinical patient records.
- The target measures expressed Yelp satisfaction, not objective dental-care quality.
- ZCTA variables are approximate business-area descriptors.
- Demographic features do not identify individual patients, reviewers, or dentists.
- EDA and exploratory ML show associations/predictive signal, not causation.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("EDA complete")
    print(f"report={REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"summary={SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"figures={len(figure_files)}")
    for fig in figure_files:
        print(f"figure={fig}")
    print("best_model", split_info["best_model"], "balanced_accuracy", f"{split_info['best_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
