"""Enrich dental businesses with ACS 2022 ZCTA area context.

The resulting features describe the Census area around a business. They do not
describe an individual reviewer, patient, or dentist.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from census_api import DEFAULT_ACS_YEAR, get_census_api_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data/interim"
REPORT_DIR = PROJECT_ROOT / "reports/demographics"
BUSINESSES_CSV = INTERIM_DIR / "dental_businesses.csv"
REVIEWS_CSV = INTERIM_DIR / "dental_review_business_base.csv"
ZCTA_CATALOGUE_CSV = INTERIM_DIR / "zcta_area_demographics.csv"
BUSINESS_DEMOGRAPHICS_CSV = INTERIM_DIR / "business_zcta_demographics.csv"


ACS_VARIABLES = [
    "NAME",
    "B19013_001E",
    "B17001_001E",
    "B17001_002E",
    "B15003_001E",
    "B15003_022E",
    "B15003_023E",
    "B15003_024E",
    "B15003_025E",
    "B23025_003E",
    "B23025_005E",
    "B03002_001E",
    "B03002_003E",
    "B03002_004E",
    "B03002_006E",
    "B03002_012E",
]

AREA_FEATURE_COLUMNS = [
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
]

CATALOGUE_COLUMNS = [
    "zcta",
    "census_geo_name",
    "census_geo_level",
    "acs_year",
    "acs_dataset",
    "data_source",
    *AREA_FEATURE_COLUMNS,
]

BUSINESS_IDENTITY_COLUMNS = [
    "business_id",
    "business_city",
    "business_state",
    "business_postal_code",
]

MATCH_REASONS = {
    "matched": "Matched five-digit ZIP candidate to an ACS ZCTA.",
    "missing_postal_code": "Business postal code is missing.",
    "invalid_or_non_us_postal_code": "Postal code is not a valid US ZIP candidate.",
    "zcta_not_returned": "Five-digit ZIP candidate was not returned as an ACS ZCTA.",
}

AREA_CONTEXT_NOTE = "Census area around the business; not an individual reviewer, patient, or dentist."


class CensusRequestError(RuntimeError):
    """Secret-safe Census request failure."""


def normalize_us_postal_code(value: object) -> str | None:
    """Return a five-digit US ZIP candidate, accepting ZIP+4, or ``None``."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{5}", text):
        return text
    match = re.fullmatch(r"(\d{5})-\d{4}", text)
    return match.group(1) if match else None


def numeric(values: pd.Series) -> pd.Series:
    """Parse non-negative ACS estimates, replacing sentinels with missing values."""
    parsed = pd.to_numeric(values, errors="coerce").astype("Float64")
    return parsed.mask(parsed < 0)


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return nullable percentages in [0, 100] for valid positive denominators."""
    numerator_values = numeric(numerator)
    denominator_values = numeric(denominator)
    result = (numerator_values / denominator_values) * 100
    return result.mask((denominator_values <= 0) | (result < 0) | (result > 100))


def _validate_unique_zctas(catalogue: pd.DataFrame) -> None:
    if "zcta" not in catalogue.columns:
        raise ValueError("ZCTA catalogue is missing required columns: zcta")
    if catalogue["zcta"].isna().any() or catalogue["zcta"].duplicated().any():
        raise ValueError("ZCTA catalogue contains a missing or duplicate zcta")


def parse_acs_catalogue(payload: object, acs_year: int = 2022) -> pd.DataFrame:
    """Parse one nationwide ACS ZCTA response into the public output contract."""
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], list):
        raise ValueError("ACS response is not a non-empty table")
    required = ACS_VARIABLES + ["zip code tabulation area"]
    missing = [column for column in required if column not in payload[0]]
    if missing:
        raise ValueError(f"ACS response missing required columns: {', '.join(missing)}")
    raw = pd.DataFrame(payload[1:], columns=payload[0])
    raw = raw[required].copy()

    result = pd.DataFrame()
    result["zcta"] = raw["zip code tabulation area"].astype("string").str.zfill(5)
    result["census_geo_name"] = raw["NAME"].astype("string")
    result["census_geo_level"] = "ACS ZCTA"
    result["acs_year"] = acs_year
    result["acs_dataset"] = "acs/acs5"
    result["data_source"] = "Census ACS API"

    result["area_median_household_income"] = numeric(raw["B19013_001E"])
    result["area_poverty_rate"] = safe_rate(numeric(raw["B17001_002E"]), numeric(raw["B17001_001E"]))
    bachelor_plus = pd.concat(
        [numeric(raw[column]) for column in ("B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E")],
        axis=1,
    ).sum(axis=1, min_count=4)
    result["area_bachelor_plus_rate"] = safe_rate(bachelor_plus, numeric(raw["B15003_001E"]))
    result["area_unemployment_rate"] = safe_rate(numeric(raw["B23025_005E"]), numeric(raw["B23025_003E"]))
    population = numeric(raw["B03002_001E"])
    result["area_pct_white"] = safe_rate(numeric(raw["B03002_003E"]), population)
    result["area_pct_black"] = safe_rate(numeric(raw["B03002_004E"]), population)
    result["area_pct_asian"] = safe_rate(numeric(raw["B03002_006E"]), population)
    result["area_pct_hispanic_or_latino"] = safe_rate(numeric(raw["B03002_012E"]), population)
    _validate_unique_zctas(result)
    return result


def fetch_acs_catalogue(acs_year: int = DEFAULT_ACS_YEAR) -> pd.DataFrame:
    """Fetch the nationwide ACS ZCTA table with one bounded retry."""
    key = get_census_api_key()
    endpoint = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    params = {
        "get": ",".join(ACS_VARIABLES),
        "for": "zip code tabulation area:*",
        "key": key,
    }
    payload: object | None = None
    for _attempt in range(2):
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception:
            payload = None
    if payload is None:
        raise CensusRequestError(
            "ACS ZCTA request failed after two attempts; credentials and URLs were redacted."
        ) from None
    return parse_acs_catalogue(payload, acs_year=acs_year)


def load_zcta_catalogue(
    cache_path: Path,
    refresh: bool = False,
    acs_year: int = DEFAULT_ACS_YEAR,
) -> pd.DataFrame:
    """Read the keyless processed cache, or fetch and replace it."""
    if cache_path.exists() and not refresh:
        catalogue = pd.read_csv(cache_path, dtype={"zcta": "string"})
    else:
        catalogue = fetch_acs_catalogue(acs_year=acs_year)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue.to_csv(cache_path, index=False)
    missing = [column for column in CATALOGUE_COLUMNS if column not in catalogue.columns]
    if missing:
        raise ValueError(f"ZCTA catalogue is missing required columns: {', '.join(missing)}")
    catalogue = catalogue[CATALOGUE_COLUMNS].copy()
    catalogue["zcta"] = catalogue["zcta"].astype("string").str.zfill(5)
    _validate_unique_zctas(catalogue)
    return catalogue


def enrich_businesses(businesses: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Left-join one validated business row to its ACS ZCTA area context."""
    missing = [column for column in BUSINESS_IDENTITY_COLUMNS if column not in businesses.columns]
    if missing:
        raise ValueError(f"Businesses input is missing required columns: {', '.join(missing)}")
    if businesses["business_id"].isna().any() or businesses["business_id"].duplicated().any():
        raise ValueError("business_id must be present, non-null, and unique")
    _validate_unique_zctas(catalogue)

    result = businesses[BUSINESS_IDENTITY_COLUMNS].copy()
    result["business_zcta"] = result["business_postal_code"].apply(normalize_us_postal_code).astype("string")
    catalogue_copy = catalogue[CATALOGUE_COLUMNS].copy()
    catalogue_copy["zcta"] = catalogue_copy["zcta"].astype("string").str.zfill(5)
    result = result.merge(
        catalogue_copy,
        left_on="business_zcta",
        right_on="zcta",
        how="left",
        validate="many_to_one",
        sort=False,
    ).drop(columns="zcta")

    missing_postal = result["business_postal_code"].isna() | result["business_postal_code"].astype("string").str.strip().eq("")
    invalid_postal = result["business_zcta"].isna() & ~missing_postal
    returned = result["census_geo_level"].notna()
    result["zcta_match_status"] = "zcta_not_returned"
    result.loc[missing_postal, "zcta_match_status"] = "missing_postal_code"
    result.loc[invalid_postal, "zcta_match_status"] = "invalid_or_non_us_postal_code"
    result.loc[returned, "zcta_match_status"] = "matched"
    result["zcta_match_reason"] = result["zcta_match_status"].map(MATCH_REASONS)
    result["area_context_note"] = AREA_CONTEXT_NOTE

    output_columns = [
        *BUSINESS_IDENTITY_COLUMNS,
        "business_zcta",
        "zcta_match_status",
        "zcta_match_reason",
        "area_context_note",
        *[column for column in CATALOGUE_COLUMNS if column != "zcta"],
    ]
    return result[output_columns]


def _percent(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else 0.0


def _group_coverage(businesses: pd.DataFrame, reviews: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    business_groups = businesses[group_column].fillna("(missing)")
    review_groups = reviews[group_column].fillna("(missing)")
    groups = sorted(set(business_groups.astype(str)) | set(review_groups.astype(str)))
    statuses = ["missing_postal_code", "invalid_or_non_us_postal_code", "zcta_not_returned"]
    for group in groups:
        business_group = businesses.loc[business_groups.astype(str).eq(group)]
        review_group = reviews.loc[review_groups.astype(str).eq(group)]
        business_any = int(business_group[AREA_FEATURE_COLUMNS].notna().any(axis=1).sum())
        review_any = int(review_group[AREA_FEATURE_COLUMNS].notna().any(axis=1).sum())
        row: dict[str, object] = {
            group_column: group,
            "business_denominator": len(business_group),
            "business_any_feature_count": business_any,
            "business_any_feature_rate": _percent(business_any, len(business_group)),
            "business_matched_count": int(business_group["zcta_match_status"].eq("matched").sum()),
            "business_unmatched_count": int(business_group["zcta_match_status"].ne("matched").sum()),
            "review_denominator": len(review_group),
            "review_any_feature_count": review_any,
            "review_any_feature_rate": _percent(review_any, len(review_group)),
            "review_matched_count": int(review_group["zcta_match_status"].eq("matched").sum()),
            "review_unmatched_count": int(review_group["zcta_match_status"].ne("matched").sum()),
        }
        for status in statuses:
            row[f"business_{status}_count"] = int(business_group["zcta_match_status"].eq(status).sum())
            row[f"review_{status}_count"] = int(review_group["zcta_match_status"].eq(status).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_coverage_tables(
    businesses: pd.DataFrame,
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build feature and geography coverage with separate business/review weights."""
    if businesses["business_id"].isna().any() or businesses["business_id"].duplicated().any():
        raise ValueError("Business enrichment must contain unique non-null business_id values")
    if "business_id" not in reviews.columns:
        raise ValueError("Review base is missing required column: business_id")
    review_enrichment = reviews[["business_id"]].merge(
        businesses,
        on="business_id",
        how="left",
        validate="many_to_one",
    )
    non_null_rows = []
    for feature in AREA_FEATURE_COLUMNS:
        business_count = int(businesses[feature].notna().sum())
        review_count = int(review_enrichment[feature].notna().sum())
        non_null_rows.append(
            {
                "feature": feature,
                "business_non_null_count": business_count,
                "business_denominator": len(businesses),
                "business_non_null_rate": _percent(business_count, len(businesses)),
                "review_non_null_count": review_count,
                "review_denominator": len(review_enrichment),
                "review_non_null_rate": _percent(review_count, len(review_enrichment)),
            }
        )
    business_any = int(businesses[AREA_FEATURE_COLUMNS].notna().any(axis=1).sum())
    review_any = int(review_enrichment[AREA_FEATURE_COLUMNS].notna().any(axis=1).sum())
    summary = {
        "business_denominator": len(businesses),
        "business_any_feature_count": business_any,
        "review_denominator": len(review_enrichment),
        "review_any_feature_count": review_any,
    }
    return (
        pd.DataFrame(non_null_rows),
        _group_coverage(businesses, review_enrichment, "business_state"),
        _group_coverage(businesses, review_enrichment, "business_city"),
        summary,
    )


def _safe_report_path(path: Path) -> str:
    """Return a secret-safe, repo-relative path for portable reports."""
    resolved = path.resolve()
    try:
        display = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        display = path
    return re.sub(r"(?i)key=[^\s&]+", "key=[REDACTED]", str(display))


def write_coverage_reports(
    businesses: pd.DataFrame,
    reviews: pd.DataFrame,
    report_dir: Path,
    businesses_path: Path,
    reviews_path: Path,
    catalogue_path: Path,
    acs_year: int = DEFAULT_ACS_YEAR,
) -> dict[str, int]:
    """Write auditable business- and review-weighted ZCTA coverage reports."""
    non_null, by_state, by_city, summary = build_coverage_tables(businesses, reviews)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    non_null.to_csv(tables_dir / "non_null_rates.csv", index=False)
    by_state.to_csv(tables_dir / "match_rate_by_state.csv", index=False)
    by_city.to_csv(tables_dir / "match_rate_by_city.csv", index=False)

    statuses = businesses["zcta_match_status"].value_counts().to_dict()
    lines = [
        "# ZCTA demographic enrichment coverage",
        "",
        f"- Run timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Businesses input: `{_safe_report_path(businesses_path)}`",
        f"- Reviews input: `{_safe_report_path(reviews_path)}`",
        f"- Processed ZCTA catalogue: `{_safe_report_path(catalogue_path)}`",
        f"- ACS vintage/dataset: {acs_year} `acs/acs5`",
        "",
        "**Interpretation warning:** These features describe the Census area around a business, not an individual reviewer, patient, or dentist.",
        "A five-digit business ZIP is treated only as a ZCTA candidate; no missing or unreturned geography is guessed.",
        "",
        "## Postal validation and ZCTA lookup",
        "",
    ]
    for status in ("matched", "missing_postal_code", "invalid_or_non_us_postal_code", "zcta_not_returned"):
        lines.append(f"- `{status}`: {int(statuses.get(status, 0)):,}")
    business_rate = _percent(summary["business_any_feature_count"], summary["business_denominator"])
    review_rate = _percent(summary["review_any_feature_count"], summary["review_denominator"])
    lines.extend(
        [
            "",
            "## Business-weighted coverage",
            "",
            f"- Any requested area feature non-null: {summary['business_any_feature_count']:,} / {summary['business_denominator']:,} ({business_rate:.2f}%)",
            "",
            "## Review-weighted coverage",
            "",
            f"- Any requested area feature non-null: {summary['review_any_feature_count']:,} / {summary['review_denominator']:,} ({review_rate:.2f}%)",
            "",
            "## Feature missingness",
            "",
            "| Feature | Business non-null | Business denominator | Business rate | Review non-null | Review denominator | Review rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in non_null.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.business_non_null_count:,} | {row.business_denominator:,} | "
            f"{row.business_non_null_rate:.2f}% | {row.review_non_null_count:,} | "
            f"{row.review_denominator:,} | {row.review_non_null_rate:.2f}% |"
        )
    (report_dir / "coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--businesses", type=Path, default=BUSINESSES_CSV)
    parser.add_argument("--reviews", type=Path, default=REVIEWS_CSV)
    parser.add_argument("--zcta-catalogue", type=Path, default=ZCTA_CATALOGUE_CSV)
    parser.add_argument("--business-output", type=Path, default=BUSINESS_DEMOGRAPHICS_CSV)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--acs-year", type=int, default=DEFAULT_ACS_YEAR)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    businesses = pd.read_csv(args.businesses, dtype={"business_postal_code": "string"})
    reviews = pd.read_csv(args.reviews)
    catalogue = load_zcta_catalogue(args.zcta_catalogue, refresh=args.refresh_cache, acs_year=args.acs_year)
    business_enrichment = enrich_businesses(businesses, catalogue)
    args.business_output.parent.mkdir(parents=True, exist_ok=True)
    business_enrichment.to_csv(args.business_output, index=False)
    summary = write_coverage_reports(
        business_enrichment,
        reviews,
        args.report_dir,
        args.businesses,
        args.reviews,
        args.zcta_catalogue,
        acs_year=args.acs_year,
    )
    print(
        f"Saved {args.zcta_catalogue} with {len(catalogue):,} ZCTA rows; "
        f"saved {args.business_output} with {len(business_enrichment):,} business rows; "
        f"business coverage {summary['business_any_feature_count']:,}/{summary['business_denominator']:,}; "
        f"review coverage {summary['review_any_feature_count']:,}/{summary['review_denominator']:,}"
    )


if __name__ == "__main__":
    main()
