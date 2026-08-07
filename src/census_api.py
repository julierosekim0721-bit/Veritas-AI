"""Small helpers for safely using the U.S. Census API.

The repository should never hard-code or commit Census API keys. These helpers
load a local `.env` file when present, prefer already-set environment variables,
and provide a tiny JSON smoke test before the full demographic pipeline runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_ACS_YEAR = 2022
DEFAULT_SMOKE_ZCTA = "19104"
DEFAULT_SMOKE_VARIABLE = "B19013_001E"


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """Parse one simple KEY=VALUE dotenv line.

    This intentionally supports only the subset needed for local credentials:
    comments, blank lines, optional `export`, and single/double quoted values.
    It avoids adding a dependency just to read one local key.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_dotenv(env_path: Path = DEFAULT_ENV_PATH) -> bool:
    """Load a local dotenv file without overwriting existing variables.

    Returns True when the file exists and was read. Existing environment values
    win over values in the file so shell/session secrets take precedence.
    """
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)
    return True


def get_census_api_key(env_path: Path = DEFAULT_ENV_PATH) -> str:
    """Return CENSUS_API_KEY from environment or local `.env`.

    Raises a clear error if the key is missing. This is intended for commands
    that require Census access, such as smoke tests and enrichment runs.
    """
    load_dotenv(env_path)
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. Request a key from "
            "https://api.census.gov/data/key_signup.html, then run "
            "export CENSUS_API_KEY='your_key_here' or create a local ignored .env file."
        )
    return key


def build_acs_smoke_test_url(key: str, zcta: str = DEFAULT_SMOKE_ZCTA, acs_year: int = DEFAULT_ACS_YEAR) -> str:
    """Build a tiny ACS URL used to verify Census API access."""
    base = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    params = {
        "get": f"NAME,{DEFAULT_SMOKE_VARIABLE}",
        "for": f"zip code tabulation area:{zcta}",
        "key": key,
    }
    return base + "?" + urllib.parse.urlencode(params, safe=":,*")


def run_census_smoke_test(
    key: str,
    zcta: str = DEFAULT_SMOKE_ZCTA,
    acs_year: int = DEFAULT_ACS_YEAR,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Call one ACS endpoint and return a secret-safe summary."""
    url = build_acs_smoke_test_url(key, zcta=zcta, acs_year=acs_year)
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        content_type = response.headers.get("content-type", "")
        body = response.read()

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "json_ok": False,
            "content_type": content_type,
            "body_prefix": body[:120].decode("utf-8", "replace").replace("\n", " "),
            "error": str(exc),
        }

    if not isinstance(payload, list) or len(payload) < 2:
        return {
            "status": "failed",
            "json_ok": True,
            "content_type": content_type,
            "rows": len(payload) if isinstance(payload, list) else 0,
            "error": "ACS response did not include a data row.",
        }

    header = payload[0]
    row = payload[1]
    return {
        "status": "ok",
        "json_ok": True,
        "content_type": content_type,
        "rows": len(payload),
        "header": header,
        "sample_name": row[0],
        "sample_income_present": bool(row[1]),
    }
