"""Tests for safe Census API configuration helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "census_api.py"
SPEC = importlib.util.spec_from_file_location("census_api", MODULE_PATH)
CENSUS_API = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CENSUS_API
SPEC.loader.exec_module(CENSUS_API)


class CensusApiConfigTests(unittest.TestCase):
    def test_load_dotenv_sets_missing_key_without_overwriting_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "CENSUS_API_KEY=from_file\n"
                "OTHER_VALUE=ignored\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"CENSUS_API_KEY": "from_env"}, clear=True):
                loaded = CENSUS_API.load_dotenv(env_path)

                self.assertTrue(loaded)
                self.assertEqual(CENSUS_API.os.environ["CENSUS_API_KEY"], "from_env")
                self.assertEqual(CENSUS_API.os.environ["OTHER_VALUE"], "ignored")

    def test_get_census_api_key_loads_from_dotenv_when_environment_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("CENSUS_API_KEY=from_file\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                key = CENSUS_API.get_census_api_key(env_path)

        self.assertEqual(key, "from_file")

    def test_get_census_api_key_raises_clear_error_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "CENSUS_API_KEY is not set"):
                    CENSUS_API.get_census_api_key(Path(tmpdir) / ".env")

    def test_build_acs_smoke_test_url_does_not_log_or_expose_key_in_returned_summary(self) -> None:
        url = CENSUS_API.build_acs_smoke_test_url("testkey", zcta="19104", acs_year=2022)

        self.assertIn("key=testkey", url)
        self.assertIn("B19013_001E", url)
        self.assertIn("zip+code+tabulation+area:19104", url)


if __name__ == "__main__":
    unittest.main()
