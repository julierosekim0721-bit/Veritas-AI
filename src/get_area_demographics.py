"""Add area-level socioeconomic and demographic descriptors.

This is the second data-preparation step. It starts from the filtered dental
businesses and creates one area-context row for each business city/state.

Important: these variables describe the area around the dental practice. They
are not patient-level or reviewer-level ethnicity, income, or education data.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from census_api import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data/interim"
CACHE_DIR = INTERIM_DIR / "cache"
BUSINESSES_CSV = INTERIM_DIR / "dental_businesses.csv"
OUTPUT_CSV = INTERIM_DIR / "area_demographics.csv"

ACS_VARIABLES = [
    "NAME",
    "B19013_001E",  # median household income
    "B17001_001E",  # poverty universe
    "B17001_002E",  # below poverty
    "B15003_001E",  # education universe, age 25+
    "B15003_022E",  # bachelor's
    "B15003_023E",  # master's
    "B15003_024E",  # professional
    "B15003_025E",  # doctorate
    "B23025_003E",  # civilian labor force
    "B23025_005E",  # unemployed
    "B03002_001E",  # total population
    "B03002_003E",  # non-Hispanic White alone
    "B03002_004E",  # non-Hispanic Black alone
    "B03002_006E",  # non-Hispanic Asian alone
    "B03002_012E",  # Hispanic or Latino
]

STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BUSINESSES_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--api-key", default=os.getenv("CENSUS_API_KEY"))
    parser.add_argument("--acs-year", type=int, default=2022)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--census-reporter-delay",
        type=float,
        default=0.05,
        help="Small delay between Census Reporter location lookups.",
    )
    return parser.parse_args()


def normalize_city(value: object) -> str:
    """Standardize city names so Yelp and Census names are easier to match."""
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


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values < 0)


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator * 100


def fetch_state_places(state: str, acs_year: int, api_key: str | None, refresh_cache: bool) -> pd.DataFrame:
    """Fetch all ACS place rows for a state when a Census API key is available."""
    if not api_key:
        return pd.DataFrame()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"acs{acs_year}_place_{state}.csv"
    if cache_path.exists() and not refresh_cache:
        return pd.read_csv(cache_path)

    fips = STATE_FIPS.get(state)
    if fips is None:
        print(f"Skipping unsupported state code: {state}")
        return pd.DataFrame()

    url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    params: dict[str, Any] = {
        "get": ",".join(ACS_VARIABLES),
        "for": "place:*",
        "in": f"state:{fips}",
    }
    if api_key:
        params["key"] = api_key

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Could not fetch ACS place data for {state}: {exc}")
        return pd.DataFrame()

    if len(payload) <= 1:
        print(f"No ACS place rows returned for {state}")
        return pd.DataFrame()

    raw = pd.DataFrame(payload[1:], columns=payload[0])
    places = compute_area_features(raw)
    places["business_state"] = state
    places["place_state_fips"] = fips
    places.to_csv(cache_path, index=False)
    return places


def first_estimate(data: dict[str, Any], *keys: str) -> float:
    estimates = data.get("estimate", {})
    for key in keys:
        value = estimates.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def fetch_census_reporter_location(
    city: object,
    state: object,
    latitude: object,
    longitude: object,
    refresh_cache: bool,
    delay_seconds: float,
) -> dict[str, Any]:
    """Fetch one location's ACS-style descriptors through public no-key APIs.

    The script first tries Census Reporter text search. If that fails, it uses
    the business latitude/longitude with the Census geocoder to identify the
    containing place, CDP, or county subdivision, then asks Census Reporter for
    the ACS tables for that geography.
    """
    state_text = str(state).upper().strip()
    city_text = "" if pd.isna(city) else str(city).strip()
    key = city_state_key(city_text, state_text)
    cache_path = CACHE_DIR / f"census_reporter_{re.sub(r'[^a-z0-9]+', '_', key.lower())}.json"

    # Cache one file per city/state so rerunning the script does not repeatedly
    # call public APIs for the same locations.
    if cache_path.exists() and not refresh_cache:
        try:
            import json

            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    empty = empty_output_row(city_text, state_text)
    if not city_text or not state_text:
        return empty

    try:
        search_response = requests.get(
            "https://api.censusreporter.org/1.0/geo/search",
            params={"q": f"{city_text} {state_text}"},
            timeout=30,
        )
        search_response.raise_for_status()
        results = search_response.json().get("results", [])
    except Exception as exc:
        print(f"Census Reporter search failed for {city_text}, {state_text}: {exc}")
        return empty

    wanted_norm = normalize_city(city_text)
    candidates = [row for row in results if row.get("sumlevel") == "160"]
    chosen: dict[str, Any] | None = None
    if candidates:
        exact_candidates = [
            row
            for row in candidates
            if normalize_city(str(row.get("full_name", "")).split(",")[0]) == wanted_norm
            and str(row.get("full_name", "")).upper().endswith(f", {state_text}")
        ]
        chosen = exact_candidates[0] if exact_candidates else candidates[0]

    if chosen is None:
        chosen = geocode_census_area(city_text, state_text, latitude, longitude)

    if chosen is None:
        return empty

    geo_id = chosen.get("full_geoid")
    if not geo_id:
        return empty

    try:
        data_response = requests.get(
            "https://api.censusreporter.org/1.0/data/show/latest",
            params={
                "table_ids": "B19013,B17001,B15003,B23025,B03002",
                "geo_ids": geo_id,
            },
            timeout=30,
        )
        data_response.raise_for_status()
        tables = data_response.json()["data"][geo_id]
    except Exception as exc:
        print(f"Census Reporter data fetch failed for {city_text}, {state_text}: {exc}")
        return empty

    median_income = first_estimate(tables.get("B19013", {}), "B19013001")
    poverty_total = first_estimate(tables.get("B17001", {}), "B17001001")
    poverty_below = first_estimate(tables.get("B17001", {}), "B17001002")
    education_total = first_estimate(tables.get("B15003", {}), "B15003001")
    bachelor_plus = sum(
        first_estimate(tables.get("B15003", {}), key)
        for key in ["B15003022", "B15003023", "B15003024", "B15003025"]
    )
    labor_force = first_estimate(tables.get("B23025", {}), "B23025003")
    unemployed = first_estimate(tables.get("B23025", {}), "B23025005")
    population_total = first_estimate(tables.get("B03002", {}), "B03002001")

    def pct(value: float, denominator: float) -> float:
        if pd.isna(value) or pd.isna(denominator) or denominator == 0:
            return float("nan")
        return value / denominator * 100

    row = empty_output_row(city_text, state_text)
    row.update(
        {
            "census_geo_name": chosen.get("full_name"),
            "census_geo_level": chosen.get("source_label", "Census Reporter ACS place/city"),
            "area_median_household_income": median_income,
            "area_poverty_rate": pct(poverty_below, poverty_total),
            "area_bachelor_plus_rate": pct(bachelor_plus, education_total),
            "area_unemployment_rate": pct(unemployed, labor_force),
            "area_pct_white": pct(first_estimate(tables.get("B03002", {}), "B03002003"), population_total),
            "area_pct_black": pct(first_estimate(tables.get("B03002", {}), "B03002004"), population_total),
            "area_pct_asian": pct(first_estimate(tables.get("B03002", {}), "B03002006"), population_total),
            "area_pct_hispanic_or_latino": pct(
                first_estimate(tables.get("B03002", {}), "B03002012"), population_total
            ),
        }
    )

    try:
        import json

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    except Exception:
        pass

    if delay_seconds:
        time.sleep(delay_seconds)
    return row


def geocode_census_area(city: str, state: str, latitude: object, longitude: object) -> dict[str, str] | None:
    """Use business coordinates to identify a Census geography for the practice area."""
    lat = pd.to_numeric(pd.Series([latitude]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([longitude]), errors="coerce").iloc[0]
    if pd.isna(lat) or pd.isna(lon):
        return None

    try:
        response = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
            params={
                "x": float(lon),
                "y": float(lat),
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "format": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        geographies = response.json().get("result", {}).get("geographies", {})
    except Exception as exc:
        print(f"Census coordinate geocoder failed for {city}, {state}: {exc}")
        return None

    geography_options = [
        ("Incorporated Places", "16000US", "Census geocoder incorporated place"),
        ("Census Designated Places", "16000US", "Census geocoder designated place"),
        ("County Subdivisions", "06000US", "Census geocoder county subdivision"),
    ]
    for geography_name, prefix, label in geography_options:
        rows = geographies.get(geography_name) or []
        if not rows:
            continue
        row = rows[0]
        geoid = row.get("GEOID")
        name = row.get("NAME") or row.get("BASENAME")
        if geoid and name:
            return {
                "full_geoid": f"{prefix}{geoid}",
                "full_name": f"{name}, {state}",
                "sumlevel": prefix[:3],
                "source_label": label,
            }
    return None


def compute_area_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ACS table columns into readable area-level feature columns."""
    result = pd.DataFrame()
    result["census_geo_name"] = raw["NAME"]
    result["census_geo_level"] = "ACS place/city"
    result["place_code"] = raw["place"]
    result["place_name_norm"] = raw["NAME"].str.split(",").str[0].apply(normalize_city)

    median_income = numeric(raw["B19013_001E"])
    poverty_total = numeric(raw["B17001_001E"])
    poverty_below = numeric(raw["B17001_002E"])
    education_total = numeric(raw["B15003_001E"])
    bachelor_plus = (
        numeric(raw["B15003_022E"])
        + numeric(raw["B15003_023E"])
        + numeric(raw["B15003_024E"])
        + numeric(raw["B15003_025E"])
    )
    labor_force = numeric(raw["B23025_003E"])
    unemployed = numeric(raw["B23025_005E"])
    population_total = numeric(raw["B03002_001E"])

    # Rates are percentages, not proportions. For example, 25.0 means 25%.
    result["area_median_household_income"] = median_income
    result["area_poverty_rate"] = safe_rate(poverty_below, poverty_total)
    result["area_bachelor_plus_rate"] = safe_rate(bachelor_plus, education_total)
    result["area_unemployment_rate"] = safe_rate(unemployed, labor_force)
    result["area_pct_white"] = safe_rate(numeric(raw["B03002_003E"]), population_total)
    result["area_pct_black"] = safe_rate(numeric(raw["B03002_004E"]), population_total)
    result["area_pct_asian"] = safe_rate(numeric(raw["B03002_006E"]), population_total)
    result["area_pct_hispanic_or_latino"] = safe_rate(numeric(raw["B03002_012E"]), population_total)
    return result


def match_place(city_norm: str, places: pd.DataFrame) -> pd.Series | None:
    """Match a normalized Yelp city name to an ACS place row."""
    if places.empty or not city_norm:
        return None

    exact = places.loc[places["place_name_norm"] == city_norm]
    if not exact.empty:
        return exact.iloc[0]

    starts_with = places.loc[
        places["place_name_norm"].str.startswith(city_norm + " ", na=False)
        | places["place_name_norm"].str.startswith(city_norm + "-", na=False)
    ]
    if not starts_with.empty:
        return starts_with.iloc[0]

    contained = places.loc[places["place_name_norm"].str.contains(rf"\b{re.escape(city_norm)}\b", na=False)]
    if len(contained) == 1:
        return contained.iloc[0]

    return None


def empty_output_row(city: object, state: object) -> dict[str, Any]:
    """Create a consistent output row when an area lookup cannot be completed."""
    return {
        "business_city": city,
        "business_state": state,
        "city_state_key": city_state_key(city, state),
        "census_geo_name": pd.NA,
        "census_geo_level": "unmatched ACS place/city",
        "area_median_household_income": pd.NA,
        "area_poverty_rate": pd.NA,
        "area_bachelor_plus_rate": pd.NA,
        "area_unemployment_rate": pd.NA,
        "area_pct_white": pd.NA,
        "area_pct_black": pd.NA,
        "area_pct_asian": pd.NA,
        "area_pct_hispanic_or_latino": pd.NA,
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing {args.input}. Run src/filter_yelp_dental.py first.")

    businesses = pd.read_csv(args.input)
    required = {"business_city", "business_state"}
    missing = required - set(businesses.columns)
    if missing:
        raise SystemExit(f"{args.input} is missing required columns: {sorted(missing)}")

    # We enrich each city/state once, not every business row. The median latitude
    # and longitude are used only as a fallback for coordinate-based geocoding.
    unique_locations = (
        businesses[["business_city", "business_state", "latitude", "longitude"]]
        .dropna(subset=["business_city", "business_state"])
        .groupby(["business_city", "business_state"], as_index=False)
        .agg({"latitude": "median", "longitude": "median"})
        .copy()
    )
    unique_locations["business_state"] = unique_locations["business_state"].astype(str).str.upper().str.strip()
    states = sorted(unique_locations["business_state"].dropna().unique())

    places_by_state: dict[str, pd.DataFrame] = {}
    for state in states:
        print(f"Loading ACS place-level area context for {state}")
        places_by_state[state] = fetch_state_places(state, args.acs_year, args.api_key, args.refresh_cache)

    rows: list[dict[str, Any]] = []
    matched = 0
    for location in tqdm(unique_locations.to_dict("records"), desc="Matching cities", unit="city"):
        city = location["business_city"]
        state = str(location["business_state"]).upper().strip()
        city_norm = normalize_city(city)
        match = match_place(city_norm, places_by_state.get(state, pd.DataFrame()))

        if match is None:
            # No Census API key is required for this fallback path.
            reporter_row = fetch_census_reporter_location(
                city,
                state,
                location.get("latitude"),
                location.get("longitude"),
                args.refresh_cache,
                args.census_reporter_delay,
            )
            rows.append(reporter_row)
            if pd.notna(reporter_row.get("area_median_household_income")):
                matched += 1
            continue

        row = empty_output_row(city, state)
        for column in [
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
        ]:
            row[column] = match[column]
        rows.append(row)
        matched += 1

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {matched:,}/{len(output):,} matched city/state areas")


if __name__ == "__main__":
    main()
