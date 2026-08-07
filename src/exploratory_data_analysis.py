"""Generate a hypothesis-driven EDA for the final dental Yelp dataset.

Run from the repository root:
    python src/exploratory_data_analysis.py

The script intentionally does not fit predictive models. It audits whether the
original demographic hypothesis is testable, summarizes hypotheses that can be
examined now, and writes student-friendly tables, figures, and a Markdown report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/final/dental_review_satisfaction_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/eda"

DEMOGRAPHIC_COLUMNS = [
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
    "area_largest_demographic_group",
    "area_largest_demographic_share",
]

REQUIRED_COLUMNS = [
    "review_stars",
    "satisfied_binary",
    "negative_binary",
    "review_text",
    "review_year",
    "review_length_words",
    "business_city",
    "business_state",
    "business_stars",
    "business_review_count",
]

STOP_WORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "been", "but", "by", "can", "could", "did", "do",
    "for", "from", "get", "got", "had", "has", "have", "he", "her", "here",
    "him", "his", "i", "if", "in", "is", "it", "its", "just", "me", "my",
    "not", "of", "on", "or", "our", "out", "she", "so", "that", "the",
    "their", "them", "there", "they", "this", "to", "up", "very", "was",
    "we", "were", "what", "when", "with", "would", "you", "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Final dataset CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="EDA artifact directory")
    return parser.parse_args()


def require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def validate_target_definitions(data: pd.DataFrame) -> dict[str, int]:
    """Count rows inconsistent with the documented target definitions."""
    require_columns(data, ["review_stars", "satisfied_binary", "negative_binary"])
    expected_satisfied = (data["review_stars"] >= 4).astype(int)
    expected_negative = (data["review_stars"] <= 2).astype(int)
    return {
        "satisfied_mismatches": int((data["satisfied_binary"] != expected_satisfied).sum()),
        "negative_mismatches": int((data["negative_binary"] != expected_negative).sum()),
    }


def audit_demographics(
    data: pd.DataFrame, columns: Iterable[str] = DEMOGRAPHIC_COLUMNS
) -> tuple[pd.DataFrame, int]:
    """Summarize demographic availability and count rows with any matched value."""
    rows: list[dict[str, object]] = []
    available_columns: list[str] = []
    for column in columns:
        present = column in data.columns
        non_null = int(data[column].notna().sum()) if present else 0
        rows.append(
            {
                "column": column,
                "present": present,
                "non_null_count": non_null,
                "missing_count": int(len(data) - non_null),
                "missing_percent": 100.0 * (len(data) - non_null) / max(len(data), 1),
                "all_null": bool(present and non_null == 0),
            }
        )
        if present:
            available_columns.append(column)
    matched_rows = (
        int(data[available_columns].notna().any(axis=1).sum()) if available_columns else 0
    )
    return pd.DataFrame(rows), matched_rows


def group_satisfaction(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Return review counts and satisfaction rates for a categorical feature."""
    require_columns(data, [group_column, "satisfied_binary"])
    summary = (
        data.dropna(subset=[group_column])
        .groupby(group_column, observed=True)["satisfied_binary"]
        .agg(review_count="size", satisfied_reviews="sum", satisfaction_rate="mean")
        .reset_index()
    )
    return summary.sort_values(["review_count", group_column], ascending=[False, True])


def safe_correlation(data: pd.DataFrame, feature: str) -> float:
    pair = data[[feature, "satisfied_binary"]].dropna()
    if len(pair) < 2 or pair[feature].nunique() < 2:
        return float("nan")
    return float(pair[feature].corr(pair["satisfied_binary"]))


def markdown_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_table(table: pd.DataFrame, name: str, tables_dir: Path) -> None:
    """Write each summary table as both CSV and dependency-free Markdown."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / f"{name}.csv", index=False)
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(markdown_value(value) for value in row) + " |")
    (tables_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_terms(text: pd.Series, limit: int = 20) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for value in text.dropna().astype(str):
        terms = re.findall(r"[a-z][a-z']{2,}", value.lower())
        counts.update(term for term in terms if term not in STOP_WORDS)
    return pd.DataFrame(counts.most_common(limit), columns=["term", "count"])


def plot_bar(
    x: Iterable[object],
    y: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output: Path,
    rotation: int = 0,
    color: str = "#4472C4",
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar([str(value) for value in x], list(y), color=color)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.tick_params(axis="x", labelrotation=rotation)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_tables(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    target_checks = validate_target_definitions(data)
    demographic_audit, matched_rows = audit_demographics(data)

    overview = pd.DataFrame(
        [
            {"metric": "rows", "value": len(data)},
            {"metric": "columns", "value": len(data.columns)},
            {"metric": "unique_businesses", "value": data["business_id"].nunique() if "business_id" in data else np.nan},
            {"metric": "unique_cities", "value": data["business_city"].nunique()},
            {"metric": "unique_states", "value": data["business_state"].nunique()},
            {"metric": "demographically_matched_rows", "value": matched_rows},
            {"metric": "satisfied_target_mismatches", "value": target_checks["satisfied_mismatches"]},
            {"metric": "negative_target_mismatches", "value": target_checks["negative_mismatches"]},
        ]
    )

    missingness = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": data.isna().sum().values,
            "missing_percent": (100.0 * data.isna().mean()).values,
        }
    ).sort_values(["missing_percent", "column"], ascending=[False, True])

    star_distribution = (
        data.groupby("review_stars", dropna=False)["satisfied_binary"]
        .agg(review_count="size", satisfaction_rate="mean")
        .reset_index()
        .sort_values("review_stars")
    )
    star_distribution["review_percent"] = 100.0 * star_distribution["review_count"] / len(data)

    class_balance = (
        data.groupby("satisfied_binary", dropna=False)
        .size()
        .rename("review_count")
        .reset_index()
    )
    class_balance["review_percent"] = 100.0 * class_balance["review_count"] / len(data)
    class_balance["class_label"] = class_balance["satisfied_binary"].map({0: "not satisfied (1-3 stars)", 1: "satisfied (4-5 stars)"})

    numeric_summary = (
        data[["business_stars", "business_review_count", "review_length_words", "review_year"]]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
        .T.reset_index(names="feature")
    )

    business_stars = group_satisfaction(data, "business_stars").sort_values("business_stars")

    length_data = data[["review_length_words", "satisfied_binary"]].dropna().copy()
    length_data["length_group"] = pd.qcut(
        length_data["review_length_words"], q=5, duplicates="drop"
    )
    length_summary = (
        length_data.groupby("length_group", observed=True, sort=True)["satisfied_binary"]
        .agg(review_count="size", satisfaction_rate="mean")
        .reset_index()
    )
    length_summary["length_group"] = length_summary["length_group"].astype(str)

    year_summary = group_satisfaction(data, "review_year").sort_values("review_year")
    state_all = group_satisfaction(data, "business_state")
    city_all = group_satisfaction(data, "business_city")
    state_summary = state_all.head(15).copy()
    city_summary = city_all.head(20).copy()
    state_summary["sample_size_caution"] = np.where(state_summary["review_count"] < 100, "small sample (<100)", "")
    city_summary["sample_size_caution"] = np.where(city_summary["review_count"] < 100, "small sample (<100)", "")

    text_non_null = int(data["review_text"].notna().sum())
    text_non_blank = int(data["review_text"].fillna("").astype(str).str.strip().ne("").sum())
    text_readiness = pd.DataFrame(
        [
            {"metric": "non_null_review_text", "value": text_non_null},
            {"metric": "non_blank_review_text", "value": text_non_blank},
            {"metric": "text_availability_percent", "value": 100.0 * text_non_blank / len(data)},
            {"metric": "median_review_length_words", "value": data["review_length_words"].median()},
            {"metric": "mean_review_length_words", "value": data["review_length_words"].mean()},
            {"metric": "95th_percentile_review_length_words", "value": data["review_length_words"].quantile(0.95)},
        ]
    )

    relationships = pd.DataFrame(
        [
            {"hypothesis": "H1-now", "feature": "business_stars", "pearson_correlation_with_satisfaction": safe_correlation(data, "business_stars"), "interpretation_caution": "Likely target proxy because Yelp business stars aggregate review ratings."},
            {"hypothesis": "H1-now", "feature": "business_review_count", "pearson_correlation_with_satisfaction": safe_correlation(data, "business_review_count"), "interpretation_caution": "Association only; reviews are clustered within businesses."},
            {"hypothesis": "H2-now", "feature": "review_length_words", "pearson_correlation_with_satisfaction": safe_correlation(data, "review_length_words"), "interpretation_caution": "Association only; length may reflect review extremity."},
            {"hypothesis": "H2-now", "feature": "review_year", "pearson_correlation_with_satisfaction": safe_correlation(data, "review_year"), "interpretation_caution": "Temporal association may reflect platform/business composition changes."},
        ]
    )

    tables = {
        "dataset_overview": overview,
        "missingness": missingness,
        "demographic_availability": demographic_audit,
        "review_star_distribution": star_distribution,
        "satisfied_class_balance": class_balance,
        "numeric_feature_summary": numeric_summary,
        "business_stars_satisfaction": business_stars,
        "review_length_satisfaction": length_summary,
        "review_year_satisfaction": year_summary,
        "state_satisfaction": state_summary,
        "city_satisfaction": city_summary,
        "text_readiness": text_readiness,
        "top_terms": top_terms(data["review_text"]),
        "feature_relationships": relationships,
    }
    facts: dict[str, object] = {
        "matched_rows": matched_rows,
        "target_checks": target_checks,
        "text_non_blank": text_non_blank,
        "state_all": state_all,
        "city_all": city_all,
    }
    return tables, facts


def create_figures(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    stars = tables["review_star_distribution"]
    plot_bar(stars["review_stars"], stars["review_count"], "Dental review star distribution", "Review stars", "Reviews", figures_dir / "review_star_distribution.png")

    balance = tables["satisfied_class_balance"]
    plot_bar(balance["class_label"], balance["review_count"], "Satisfied target class balance", "Target class", "Reviews", figures_dir / "satisfied_class_balance.png", rotation=10, color="#70AD47")

    business = tables["business_stars_satisfaction"]
    plot_bar(business["business_stars"], business["satisfaction_rate"], "Business rating and review satisfaction", "Business stars", "Satisfaction rate", figures_dir / "business_stars_vs_satisfaction.png", rotation=45, color="#ED7D31")

    length = tables["review_length_satisfaction"]
    plot_bar(length["length_group"], length["satisfaction_rate"], "Review length groups and satisfaction", "Review length (word-count quintile)", "Satisfaction rate", figures_dir / "review_length_vs_satisfaction.png", rotation=25, color="#A5A5A5")

    states = tables["state_satisfaction"].sort_values("review_count", ascending=False).head(10)
    plot_bar(states["business_state"], states["satisfaction_rate"], "Top states by sample size: satisfaction rate", "State", "Satisfaction rate", figures_dir / "top_states_satisfaction.png", color="#5B9BD5")

    missing = tables["missingness"].query("missing_percent > 0").head(15).sort_values("missing_percent")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(missing["column"], missing["missing_percent"], color="#C00000")
    axis.set(title="Missingness in columns with missing data", xlabel="Missing values (%)", ylabel="Column")
    axis.set_xlim(0, 105)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "missingness.png", dpi=180)
    plt.close(fig)


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(
    data: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    facts: dict[str, object],
    input_path: Path,
    output_path: Path,
) -> None:
    balance = tables["satisfied_class_balance"].set_index("satisfied_binary")
    satisfied_rate = float(balance.loc[1, "review_count"] / len(data))
    stars = tables["review_star_distribution"]
    most_common_star = int(stars.loc[stars["review_count"].idxmax(), "review_stars"])
    business = tables["business_stars_satisfaction"]
    length = tables["review_length_satisfaction"]
    years = tables["review_year_satisfaction"]
    relationships = tables["feature_relationships"].set_index("feature")
    city = tables["city_satisfaction"]
    states = tables["state_satisfaction"]
    text = tables["text_readiness"].set_index("metric")["value"]
    top = tables["top_terms"].head(10)
    top_term_text = ", ".join(f"{row.term} ({row.count:,})" for row in top.itertuples())
    target_checks = facts["target_checks"]

    report = f"""# Hypothesis-Driven EDA: Yelp Dental Review Satisfaction

## Purpose and provenance

This report evaluates whether the current preprocessed dataset supports the initial question: **can socioeconomic or demographic context predict Yelp dental review satisfaction?** It was generated reproducibly by `src/exploratory_data_analysis.py` from `{input_path.relative_to(PROJECT_ROOT) if input_path.is_relative_to(PROJECT_ROOT) else input_path}`. The dataset has **{len(data):,} reviews and {len(data.columns)} columns**, covering **{data['business_id'].nunique():,} businesses**, **{data['business_city'].nunique():,} cities**, and **{data['business_state'].nunique():,} states**.

Each row is one Yelp review joined to dental-business metadata. Results are descriptive associations, not causal effects, and reviews from the same business are not statistically independent.

## Target definitions and integrity

- `satisfied_binary = 1 if review_stars >= 4 else 0`
- `negative_binary = 1 if review_stars <= 2 else 0`

The audit found **{target_checks['satisfied_mismatches']:,} satisfied-target mismatches** and **{target_checks['negative_mismatches']:,} negative-target mismatches**. `review_stars` therefore directly defines both targets and must never be used as an ordinary predictor of either target.

## H0/intake: Are demographic features available?

**Verdict: no.** The demographic availability audit found **{facts['matched_rows']:,} rows with at least one non-null area-level demographic value**. All requested socioeconomic/racial-composition measures are 100% missing. The original hypothesis cannot be tested until Census enrichment is repaired and validated. No imputation or demographic model is scientifically defensible when an entire feature is absent.

See `tables/demographic_availability.csv`, `tables/missingness.csv`, and `figures/missingness.png`.

## Class balance and review stars

- Satisfied reviews (4–5 stars): **{int(balance.loc[1, 'review_count']):,} / {len(data):,} ({percent(satisfied_rate)})**.
- Not-satisfied reviews (1–3 stars): **{int(balance.loc[0, 'review_count']):,} / {len(data):,} ({percent(1-satisfied_rate)})**.
- The most common review rating is **{most_common_star} stars**.

This imbalance means accuracy alone is not a sufficient model metric. Use a majority-class baseline, balanced accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## H1-now: Business-level context

`business_stars` has a Pearson correlation of **{relationships.loc['business_stars', 'pearson_correlation_with_satisfaction']:.3f}** with satisfaction in this review-level dataset. Across observed business-star groups, satisfaction rates range from **{percent(float(business['satisfaction_rate'].min()))} to {percent(float(business['satisfaction_rate'].max()))}**. `business_review_count` has correlation **{relationships.loc['business_review_count', 'pearson_correlation_with_satisfaction']:.3f}**.

These are useful descriptive signals, but `business_stars` is likely calculated from reviews including the current review. It is therefore a strong **target proxy/leakage risk**, not clean independent context. A later model should either exclude it, reconstruct a leave-one-review-out/prior-time business rating, or present results both with and without it. Train/test splitting should be grouped by `business_id` to prevent the same business leaking into both sets.

## H2-now: Review length and time

Review length has correlation **{relationships.loc['review_length_words', 'pearson_correlation_with_satisfaction']:.3f}** with satisfaction. Satisfaction across word-count quintiles ranges from **{percent(float(length['satisfaction_rate'].min()))} to {percent(float(length['satisfaction_rate'].max()))}**. Review year spans **{int(data['review_year'].min())}–{int(data['review_year'].max())}** and has correlation **{relationships.loc['review_year', 'pearson_correlation_with_satisfaction']:.3f}**. These are descriptive associations only; length may capture review extremity, while year may reflect changes in Yelp usage or the businesses represented.

## H3-now: Geographic variation

The highest-volume states shown in the EDA have satisfaction rates from **{percent(float(states['satisfaction_rate'].min()))} to {percent(float(states['satisfaction_rate'].max()))}**; the highest-volume cities shown range from **{percent(float(city['satisfaction_rate'].min()))} to {percent(float(city['satisfaction_rate'].max()))}**. These differences are not evidence that place causes satisfaction. They may be driven by unequal sample sizes, a small number of businesses, business mix, time, or review-selection effects. Always show counts beside rates, set a minimum sample threshold, and prefer grouped uncertainty estimates before interpreting rankings.

## Text-feature readiness

Review text is non-blank for **{int(text['non_blank_review_text']):,} rows ({float(text['text_availability_percent']):.1f}%)**. Median length is **{float(text['median_review_length_words']):.0f} words**, mean length is **{float(text['mean_review_length_words']):.1f}**, and the 95th percentile is **{float(text['95th_percentile_review_length_words']):.0f} words**. The most frequent simple non-stopword terms are: {top_term_text}.

This supports a student-friendly TF-IDF model. However, terms such as “great” or “worst” can make sentiment classification easy without addressing socioeconomic context; explain that the text model predicts expressed satisfaction rather than its causes.

## Leakage, confounds, and interpretation warnings

1. **Direct leakage:** `review_stars` defines the targets; never include it as a predictor.
2. **Aggregate target proxy:** `business_stars` aggregates review ratings and may include the focal review.
3. **Cluster leakage:** many reviews belong to the same business; use group-aware splits by `business_id`.
4. **User leakage:** repeat reviewers may appear across rows; audit `user_id` and consider a stricter grouped sensitivity analysis.
5. **Geographic confounding:** city/state rates mix business composition, sample size, year, and Yelp participation.
6. **Selection bias:** Yelp reviewers are not a random sample of dental patients.
7. **Causality:** predictive associations do not establish that demographics or business features cause satisfaction.

## Hypothesis evolution

### Initial hypothesis

Area-level socioeconomic/demographic context predicts whether a dental Yelp review is satisfied.

### EDA verdict

**Not testable with the current dataset.** Census matching produced zero usable demographic rows. The correct action is to repair enrichment, not fit a zero-row model or fill wholly absent features.

### Revised, implementable hypotheses

- **H1-now:** Business-level features (`business_review_count`, and cautiously `business_stars`) improve prediction over a majority baseline. Compare versions excluding and including the proxy-risk feature.
- **H2-now:** Review text length and review year have measurable but likely limited associations with satisfaction.
- **H3-now:** City/state differences remain after minimum-count filtering, but may shrink under group-aware validation and business controls.
- **H-text-now:** TF-IDF review text predicts expressed satisfaction better than structured non-leaky features, with interpretable positive/negative coefficients.
- **H4-next:** After Census repair, demographic features add out-of-sample value beyond non-leaky business, time, geography, and text baselines.

## Recommended student ML pipeline

1. **Data gate:** verify target definitions, demographic coverage, missingness, duplicates, and class balance.
2. **Split first:** reserve a test set grouped by `business_id`; use stratified group cross-validation if practical.
3. **Baseline:** majority-class prediction and a simple non-leaky structured baseline.
4. **Interpretable structured model:** logistic regression with scaled/log-transformed `business_review_count`, review length, year, and carefully encoded geography. Exclude `review_stars`; report `business_stars` only as a sensitivity analysis.
5. **Text model:** TF-IDF (word unigrams/bigrams, capped vocabulary) plus logistic regression. Inspect top coefficients and errors.
6. **Optional extension:** random forest or gradient boosting on structured features only, after the interpretable baseline. Do not make it the starting point.
7. **Evaluation:** balanced accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and fold-to-fold variability. Compare every model on the same held-out groups.
8. **Demographic extension:** repair postal-code/Census matching, report match rate and unmatched bias, then test whether demographic features improve grouped held-out performance. Treat this as incremental predictive value, not causality.

## Bottom line

The dataset is ready for descriptive business/time/text EDA and a careful student-level prediction pipeline, but **not yet ready for the proposed socioeconomic/demographic analysis**. The immediate scientific blocker is Census enrichment coverage; the immediate implementable ML path is grouped, leakage-aware baselines followed by interpretable structured and TF-IDF logistic regression models.

## Generated artifacts

- Tables: `reports/eda/tables/` (CSV and Markdown)
- Figures: `reports/eda/figures/`
- Detailed relationship values: `reports/eda/tables/feature_relationships.csv`
"""
    output_path.write_text(report, encoding="utf-8")


def print_summary(data: pd.DataFrame, tables: dict[str, pd.DataFrame], facts: dict[str, object], output_dir: Path) -> None:
    balance = tables["satisfied_class_balance"].set_index("satisfied_binary")
    relationships = tables["feature_relationships"]
    print("Hypothesis-driven dental Yelp EDA")
    print("=================================")
    print(f"Dataset: {len(data):,} rows x {len(data.columns)} columns")
    print(f"Satisfied: {int(balance.loc[1, 'review_count']):,} ({balance.loc[1, 'review_percent']:.1f}%)")
    print(f"Rows with demographic data: {facts['matched_rows']:,}")
    if facts["matched_rows"] == 0:
        print("H0 verdict: demographic hypothesis is NOT testable until Census enrichment is repaired.")
    print("\nCorrelations with satisfied_binary (descriptive only):")
    for row in relationships.itertuples(index=False):
        print(f"  {row.feature}: {row.pearson_correlation_with_satisfaction:.3f}")
    print(f"\nArtifacts written to: {output_dir}")
    print("Read reports/eda/EDA_REPORT.md for leakage warnings and staged hypotheses.")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Final dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    require_columns(data, REQUIRED_COLUMNS)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables, facts = build_tables(data)
    for name, table in tables.items():
        write_table(table, name, tables_dir)
    create_figures(tables, figures_dir)
    write_report(data, tables, facts, input_path, output_dir / "EDA_REPORT.md")
    print_summary(data, tables, facts, output_dir)


if __name__ == "__main__":
    main()
