"""Filter the Yelp Open Dataset down to dental businesses and dental reviews.

This is the first data-preparation step. It reads the raw Yelp business JSON,
keeps only U.S. businesses whose categories look dental-related, then scans the
large Yelp review JSON in chunks so the script can run on a normal laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUSINESS_JSON = PROJECT_ROOT / "data/raw/yelp_academic_dataset_business.json"
DEFAULT_REVIEW_JSON = PROJECT_ROOT / "data/raw/yelp_academic_dataset_review.json"
INTERIM_DIR = PROJECT_ROOT / "data/interim"

DENTAL_TERMS = [
    "Dentists",
    "General Dentistry",
    "Cosmetic Dentists",
    "Orthodontists",
    "Pediatric Dentists",
    "Periodontists",
    "Endodontists",
    "Oral Surgeons",
    "Dental Hygienists",
]

US_STATES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--business-json", type=Path, default=DEFAULT_BUSINESS_JSON)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument(
        "--max-review-chunks",
        type=int,
        default=None,
        help="Process only the first N review chunks for a quick laptop test.",
    )
    parser.add_argument(
        "--review-chunk-size",
        type=int,
        default=100_000,
        help="Number of Yelp review JSON lines to read per chunk.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing input file: {path}\n"
            "Place the Yelp Open Dataset JSON files in data/raw/ before running."
        )


def has_dental_category(categories: object) -> bool:
    """Return True when a Yelp category string contains a dental category term."""
    if not isinstance(categories, str):
        return False
    lowered = categories.lower()
    return any(term.lower() in lowered for term in DENTAL_TERMS)


def load_dental_businesses(business_json: Path) -> pd.DataFrame:
    print(f"Loading businesses from {business_json}")
    businesses = pd.read_json(business_json, lines=True)
    businesses["state"] = businesses["state"].astype(str).str.upper().str.strip()

    # Keep only U.S. rows and only businesses with dental-related Yelp categories.
    # The categories field is a comma-separated text field in the Yelp JSON.
    is_us = businesses["state"].isin(US_STATES)
    is_dental = businesses["categories"].apply(has_dental_category)
    dental = businesses.loc[is_us & is_dental].copy()

    # Rename business fields so they cannot be confused with review fields later.
    # Yelp uses "stars" for both businesses and reviews, so clarity matters here.
    rename_map = {
        "name": "business_name",
        "city": "business_city",
        "state": "business_state",
        "postal_code": "business_postal_code",
        "categories": "business_categories",
        "review_count": "business_review_count",
        "stars": "business_stars",
    }
    dental = dental.rename(columns=rename_map)

    keep_columns = [
        "business_id",
        "business_name",
        "business_city",
        "business_state",
        "business_postal_code",
        "latitude",
        "longitude",
        "business_categories",
        "business_stars",
        "business_review_count",
    ]
    available_columns = [col for col in keep_columns if col in dental.columns]
    dental = dental[available_columns].drop_duplicates("business_id")

    print(f"Found {len(dental):,} U.S. dental businesses")
    return dental


def filter_reviews(
    review_json: Path,
    dental_business_ids: set[str],
    chunk_size: int,
    max_chunks: int | None,
    output_path: Path,
) -> pd.DataFrame:
    """Scan Yelp reviews and keep only reviews for the dental business IDs."""
    if output_path.exists():
        output_path.unlink()

    kept_rows = 0
    chunks_seen = 0
    filtered_chunks: list[pd.DataFrame] = []
    reader = pd.read_json(review_json, lines=True, chunksize=chunk_size)

    for chunk in tqdm(reader, desc="Filtering review chunks", unit="chunk"):
        chunks_seen += 1

        # The review file is several GB, so we never load the whole thing at once.
        # Each chunk is filtered by business_id, then appended to a CSV on disk.
        filtered = chunk.loc[chunk["business_id"].isin(dental_business_ids)].copy()
        if not filtered.empty:
            filtered = filtered.rename(
                columns={"stars": "review_stars", "text": "review_text", "date": "review_date"}
            )
            keep_columns = [
                "review_id",
                "business_id",
                "user_id",
                "review_stars",
                "review_text",
                "review_date",
                "useful",
                "funny",
                "cool",
            ]
            filtered = filtered[[col for col in keep_columns if col in filtered.columns]]
            filtered.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
            filtered_chunks.append(filtered)
            kept_rows += len(filtered)

        if max_chunks is not None and chunks_seen >= max_chunks:
            break

    print(f"Kept {kept_rows:,} dental reviews from {chunks_seen:,} chunks")
    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    require_file(args.business_json)
    require_file(args.review_json)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    businesses = load_dental_businesses(args.business_json)
    businesses_path = INTERIM_DIR / "dental_businesses.csv"
    businesses.to_csv(businesses_path, index=False)
    print(f"Saved {businesses_path}")

    reviews_path = INTERIM_DIR / "dental_reviews.csv"
    reviews = filter_reviews(
        args.review_json,
        set(businesses["business_id"]),
        args.review_chunk_size,
        args.max_review_chunks,
        reviews_path,
    )
    if reviews.empty:
        print("No matching dental reviews found. Downstream scripts will not have data to merge.")
        return
    print(f"Saved {reviews_path}")

    # This merged base file is the review-level table used by later scripts.
    # Each row is one review with its dental practice metadata attached.
    base = reviews.merge(businesses, on="business_id", how="left")
    base_path = INTERIM_DIR / "dental_review_business_base.csv"
    base.to_csv(base_path, index=False)
    print(f"Saved {base_path} with {len(base):,} rows")


if __name__ == "__main__":
    main()
