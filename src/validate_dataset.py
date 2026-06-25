"""Print a simple validation report for the final cleaned dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CSV = PROJECT_ROOT / "data/final/dental_review_satisfaction_dataset.csv"


def main() -> None:
    if not FINAL_CSV.exists():
        print(f"Final dataset does not exist yet: {FINAL_CSV}")
        print("Run filter_yelp_dental.py, get_area_demographics.py, and build_final_dataset.py first.")
        return

    data = pd.read_csv(FINAL_CSV)

    # These summaries help catch common pipeline problems: missing columns,
    # empty merges, unexpected target distributions, or failed area matching.
    print("Dataset validation")
    print("==================")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns):,}")
    print(f"Unique businesses: {data['business_id'].nunique():,}" if "business_id" in data else "business_id missing")
    print(
        f"Unique cities: {data['business_city'].nunique():,}"
        if "business_city" in data
        else "business_city missing"
    )

    print("\nMissing values per column")
    print(data.isna().sum().sort_values(ascending=False).to_string())

    if "review_stars" in data:
        print("\nReview star distribution")
        print(data["review_stars"].value_counts(dropna=False).sort_index().to_string())

    if "satisfied_binary" in data:
        print("\nSatisfied vs not satisfied distribution")
        print(data["satisfied_binary"].value_counts(dropna=False).sort_index().to_string())

    demographic_match_columns = [
        "area_median_household_income",
        "area_poverty_rate",
        "area_bachelor_plus_rate",
        "area_unemployment_rate",
        "area_pct_white",
        "area_pct_black",
        "area_pct_asian",
        "area_pct_hispanic_or_latino",
    ]
    available_match_columns = [col for col in demographic_match_columns if col in data.columns]
    if available_match_columns:
        matched = data[available_match_columns].notna().any(axis=1).sum()
        print(f"\nRows with successful area-level demographic match: {matched:,}")
    else:
        print("\nNo area-level demographic columns found.")


if __name__ == "__main__":
    main()
