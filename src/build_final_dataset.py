"""Build the final review-level dataset for analysis.

This is the third data-preparation step. It merges dental Yelp reviews with
business metadata and area-level demographic descriptors, then creates simple
derived columns such as binary satisfaction labels and review length.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CSV = PROJECT_ROOT / "data/interim/dental_review_business_base.csv"
DEMOGRAPHICS_CSV = PROJECT_ROOT / "data/interim/area_demographics.csv"
FINAL_CSV = PROJECT_ROOT / "data/final/dental_review_satisfaction_dataset.csv"

DEMOGRAPHIC_COLUMNS = {
    "area_pct_white": "White",
    "area_pct_black": "Black",
    "area_pct_asian": "Asian",
    "area_pct_hispanic_or_latino": "Hispanic or Latino",
}

REQUIRED_FINAL_COLUMNS = [
    "review_id",
    "business_id",
    "user_id",
    "review_stars",
    "satisfied_binary",
    "negative_binary",
    "review_text",
    "review_date",
    "review_year",
    "review_length_chars",
    "review_length_words",
    "business_name",
    "business_city",
    "business_state",
    "business_postal_code",
    "latitude",
    "longitude",
    "business_categories",
    "business_review_count",
    "area_median_household_income",
    "area_income_band",
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


def normalize_city(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bst\b", "saint", text)
    text = re.sub(
        r"\b(city|town|village|borough|municipality|cdp|balance|urban county|county)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def city_state_key(city: object, state: object) -> str:
    return f"{normalize_city(city)}|{str(state).upper().strip()}"


def income_band(value: object) -> str:
    """Convert numeric area median household income into a simple labelled band."""
    if pd.isna(value):
        return "Unknown"
    income = float(value)
    if income < 40_000:
        return "Less than $40k"
    if income < 60_000:
        return "$40k-$59k"
    if income < 80_000:
        return "$60k-$79k"
    if income < 100_000:
        return "$80k-$99k"
    return "$100k+"


def largest_demographic_group(row: pd.Series) -> pd.Series:
    """Find the largest area-level demographic share among the selected columns."""
    values = row[list(DEMOGRAPHIC_COLUMNS)].dropna()
    if values.empty:
        return pd.Series([pd.NA, pd.NA])
    column = values.astype(float).idxmax()
    return pd.Series([DEMOGRAPHIC_COLUMNS[column], values[column]])


def require_file(path: Path, upstream_hint: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}. {upstream_hint}")


def main() -> None:
    require_file(BASE_CSV, "Run src/filter_yelp_dental.py first.")
    require_file(DEMOGRAPHICS_CSV, "Run src/get_area_demographics.py first.")

    base = pd.read_csv(BASE_CSV)
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)

    # City names can be written differently across sources, so both tables get
    # the same normalized city/state key before merging.
    base["city_state_key"] = [
        city_state_key(city, state) for city, state in zip(base["business_city"], base["business_state"])
    ]
    demographics["city_state_key"] = [
        city_state_key(city, state)
        for city, state in zip(demographics["business_city"], demographics["business_state"])
    ]

    area_columns = [
        "city_state_key",
        "census_geo_name",
        "census_geo_level",
        "area_median_household_income",
        "area_poverty_rate",
        "area_bachelor_plus_rate",
        "area_unemployment_rate",
        "area_pct_white",
        "area_pct_black",
        "area_pct_asian",
        "area_pct_hispanic_or_latino",
    ]
    merged = base.merge(demographics[area_columns].drop_duplicates("city_state_key"), on="city_state_key", how="left")

    # Target labels for simple classification:
    # satisfied_binary is the main outcome, and negative_binary is useful for EDA.
    merged["review_stars"] = pd.to_numeric(merged["review_stars"], errors="coerce")
    merged["satisfied_binary"] = (merged["review_stars"] >= 4).astype("Int64")
    merged["negative_binary"] = (merged["review_stars"] <= 2).astype("Int64")

    review_dates = pd.to_datetime(merged["review_date"], errors="coerce")
    merged["review_year"] = review_dates.dt.year.astype("Int64")
    review_text = merged["review_text"].fillna("").astype(str)
    merged["review_length_chars"] = review_text.str.len()
    merged["review_length_words"] = review_text.str.split().str.len()
    merged["area_income_band"] = merged["area_median_household_income"].apply(income_band)

    # This is a descriptor of the practice's Census area, not the reviewer.
    largest = merged.apply(largest_demographic_group, axis=1)
    merged["area_largest_demographic_group"] = largest[0]
    merged["area_largest_demographic_share"] = largest[1]

    # Keep the most important columns first, while preserving any useful extra
    # Yelp columns such as useful/funny/cool review votes.
    ordered_columns = [col for col in REQUIRED_FINAL_COLUMNS if col in merged.columns]
    extra_columns = [
        col
        for col in merged.columns
        if col not in ordered_columns and col not in {"city_state_key"}
    ]
    final = merged[ordered_columns + extra_columns]

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(FINAL_CSV, index=False)
    print(f"Saved {FINAL_CSV} with {len(final):,} review rows and {len(final.columns):,} columns")


if __name__ == "__main__":
    main()
