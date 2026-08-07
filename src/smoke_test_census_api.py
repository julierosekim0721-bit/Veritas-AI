"""Run a tiny U.S. Census API smoke test without printing the API key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow `python src/smoke_test_census_api.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from census_api import DEFAULT_ACS_YEAR, DEFAULT_SMOKE_ZCTA, get_census_api_key, run_census_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--zcta", default=DEFAULT_SMOKE_ZCTA, help="ZCTA to query for the one-row smoke test.")
    parser.add_argument("--acs-year", type=int, default=DEFAULT_ACS_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = get_census_api_key(args.env_file)
    summary = run_census_smoke_test(key, zcta=args.zcta, acs_year=args.acs_year)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
