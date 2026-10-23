"""
Revised analysis for Yelp dental review satisfaction.

Changes relative to the original script:
- Excludes business_stars (clinic aggregate Yelp rating) from Model 2.
- Uses clinic-clustered robust standard errors for both logistic models.
- Reports model-specific complete-case sample sizes and number of clinics.
- Exports coefficients, clustered standard errors, p-values, and confidence intervals.
- Reports satisfaction counts/rates by area income band.

Run from anywhere with:
    python src/analyze_dental_satisfaction_revised.py
or place this file in src/ and run it from the repository root.
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
# Works whether this file is saved in the repository root or in src/.
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_PATH = PROJECT_ROOT / "data/final/dental_review_satisfaction_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports/revised_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. First build data/final/"
        "dental_review_satisfaction_dataset.csv using the README pipeline."
    )

df = pd.read_csv(DATA_PATH)

required = {
    "business_id",
    "satisfied_binary",
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
    "business_review_count",
    "review_length_words",
    "review_year",
    "area_income_band",
}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

print("\nDATASET SUMMARY")
print("----------------")
print(f"Rows: {len(df):,}")
print(f"Unique clinics/businesses: {df['business_id'].nunique():,}")
print(f"Overall satisfaction rate: {df['satisfied_binary'].mean():.1%}")

# Descriptive satisfaction rates by area income band.
# These are area-level categories, not individual reviewer income.
income_summary = (
    df.dropna(subset=["area_income_band", "satisfied_binary"])
      .groupby("area_income_band", dropna=False)
      .agg(
          reviews=("satisfied_binary", "size"),
          satisfied_reviews=("satisfied_binary", "sum"),
          satisfaction_rate=("satisfied_binary", "mean"),
      )
      .reset_index()
)
income_summary["satisfaction_rate_pct"] = 100 * income_summary["satisfaction_rate"]
income_summary.to_csv(OUTPUT_DIR / "satisfaction_by_income_band.csv", index=False)
print("\nSATISFACTION BY AREA INCOME BAND")
print(income_summary.to_string(index=False, formatters={
    "satisfaction_rate": "{:.4f}".format,
    "satisfaction_rate_pct": "{:.2f}%".format,
}))

ses_vars = [
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
]

# Model 1: socioeconomic and demographic area descriptors.
model_specs = {
    "Model 1: SES + demographics": [
        "satisfied_binary", "business_id", *ses_vars
    ],
    # Important reviewer-requested model: NO business_stars aggregate rating.
    "Model 2: SES + demographics + review/business metadata (no business_stars)": [
        "satisfied_binary",
        "business_id",
        *ses_vars,
        "business_review_count",
        "review_length_words",
        "review_year",
    ],
}

all_results = []
model_summaries = []

for model_name, variables in model_specs.items():
    # Keep business_id to identify clusters, but never use it as a predictor.
    model_df = df[variables].dropna().copy()
    y = pd.to_numeric(model_df["satisfied_binary"], errors="raise").astype(int)
    cluster_groups = model_df["business_id"]
    X = model_df.drop(columns=["satisfied_binary", "business_id"]).apply(
        pd.to_numeric, errors="raise"
    )
    X = sm.add_constant(X, has_constant="add")

    # Cluster-robust covariance accounts for within-clinic dependence.
    fitted = sm.Logit(y, X).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": cluster_groups},
    )

    n_rows = len(model_df)
    n_clinics = cluster_groups.nunique()
    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Complete-case reviews: {n_rows:,}")
    print(f"Unique clinics: {n_clinics:,}")
    print(f"Cluster-robust SEs: business_id")
    print(f"Converged: {fitted.mle_retvals.get('converged', 'not reported')}")
    print(fitted.summary())

    conf = fitted.conf_int()
    for term in fitted.params.index:
        all_results.append({
            "model": model_name,
            "term": term,
            "coefficient_log_odds": fitted.params[term],
            "cluster_robust_se": fitted.bse[term],
            "p_value": fitted.pvalues[term],
            "ci_95_lower": conf.loc[term, 0],
            "ci_95_upper": conf.loc[term, 1],
            "n_reviews": n_rows,
            "n_clinics": n_clinics,
        })

    model_summaries.append(
        f"{model_name}\n"
        f"Complete-case reviews: {n_rows:,}\n"
        f"Unique clinics: {n_clinics:,}\n"
        f"Cluster-robust SEs clustered by business_id\n\n"
        f"{fitted.summary()}\n\n"
    )

results = pd.DataFrame(all_results)
results.to_csv(OUTPUT_DIR / "logistic_regression_results_clustered.csv", index=False)
(OUTPUT_DIR / "model_summaries.txt").write_text(
    "\n".join(model_summaries), encoding="utf-8"
)

print("\nSaved revised outputs to:")
print(OUTPUT_DIR)
print("- logistic_regression_results_clustered.csv")
print("- model_summaries.txt")
print("- satisfaction_by_income_band.csv")
print(
    "\nInterpretation reminder: area-level Census variables describe the "
    "practice's surrounding area, not individual patients or reviewers. "
    "These observational models estimate associations, not causation."
)
