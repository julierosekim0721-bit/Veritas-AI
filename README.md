# Yelp Dental Review Satisfaction Dataset

This repository creates a clean review-level CSV for a student project on Yelp dental review satisfaction.

Each row in the final dataset is one Yelp review for a dental practice. The review is joined with dental business metadata and area-level socioeconomic/demographic descriptors for the practice location.

## Data Sources

### Yelp Open Dataset

Use the Yelp Open Dataset.

Official source:

```text
https://business.yelp.com/data/resources/open-dataset/
```


After downloading/extracting the Yelp data, place these two files in `data/raw/`:

```text
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

The pipeline only needs those two JSON files.

### Area-Level Demographic Data

The script `src/enrich_zcta_demographics.py` retrieves area-level socioeconomic and demographic descriptors from the U.S. Census ACS API using each dental business postal code as a ZIP/ZCTA candidate:

```text
https://api.census.gov/data.html
```

These variables describe the Census area around the dental practice. They are **not** individual patient, reviewer, or dentist demographics.

#### Census API key setup

The demographic enrichment step requires a free U.S. Census API key.

1. Request a key from the official Census signup page:

   ```text
   https://api.census.gov/data/key_signup.html
   ```

2. Fill in the form:

   ```text
   Organization Name: Student Research Project
   Email Address: your email address
   ```

   Then agree to the terms and submit the form. The key should be emailed to you.

3. Store the key locally. Do **not** paste the real key into GitHub, notebooks, reports, screenshots, or shared chat messages.

   Temporary terminal option:

   ```bash
   export CENSUS_API_KEY="paste_your_key_here"
   ```

   Recommended project option: create a local `.env` file in the repository root:

   ```text
   CENSUS_API_KEY=paste_your_key_here
   ```

   The `.env` file is ignored by Git, so it stays on your machine only.

4. Test the key before running the full demographic pipeline:

   ```bash
   python src/smoke_test_census_api.py
   ```

   A working key should print a small JSON summary like this:

   ```json
   {
     "json_ok": true,
     "rows": 2,
     "sample_name": "ZCTA5 19104",
     "sample_income_present": true,
     "status": "ok"
   }
   ```

   The smoke test does not print the API key.

If the smoke test says `CENSUS_API_KEY is not set`, check that either your terminal has the environment variable exported or your `.env` file is in the repository root.

The main enrichment script caches the processed ACS ZCTA catalogue locally so later runs can reuse it without refetching. Use `--refresh-cache` when you intentionally want to download a fresh catalogue.

## Setup

From the repository root:


```bash
conda env create -f environment.yml
conda activate veritas_ai
pip install -r requirements.txt
```

## Run The Pipeline

Run these commands from the repository root.

### 1. Filter Yelp To Dental Businesses And Reviews

```bash
python src/filter_yelp_dental.py
```

This creates:

```text
data/interim/dental_businesses.csv
data/interim/dental_reviews.csv
data/interim/dental_review_business_base.csv
```

For a quick laptop test, process only the first chunk of reviews:

```bash
python src/filter_yelp_dental.py --max-review-chunks 1
```

### 2. Add Area-Level Demographic Descriptors

First verify that the Census key works:

```bash
python src/smoke_test_census_api.py
```

Then generate the ZCTA-level demographic dataset:

```bash
python src/enrich_zcta_demographics.py --refresh-cache
```

For later reruns, you can omit `--refresh-cache` to reuse the existing processed ACS ZCTA catalogue:

```bash
python src/enrich_zcta_demographics.py
```

This creates:

```text
data/interim/zcta_area_demographics.csv
data/interim/business_zcta_demographics.csv
reports/demographics/coverage_report.md
reports/demographics/tables/non_null_rates.csv
reports/demographics/tables/match_rate_by_state.csv
reports/demographics/tables/match_rate_by_city.csv
```

What these files mean:

- `zcta_area_demographics.csv`: one row per ACS ZCTA returned by the Census API.
- `business_zcta_demographics.csv`: one row per Yelp dental business, joined to its ZCTA demographic context when available.
- `coverage_report.md`: a human-readable report showing how many businesses and reviews received demographic data.
- `non_null_rates.csv`: feature-level missingness for business-weighted and review-weighted coverage.
- `match_rate_by_state.csv` and `match_rate_by_city.csv`: geographic coverage checks.

The demographic fields are linked by `business_id` using the dental practice location. They describe the area around the dental practice, not individual patients or reviewers.

### 3. Build The Final Dataset

```bash
python src/build_final_dataset.py
```

This creates:

```text
data/final/dental_review_satisfaction_dataset.csv
```

`src/build_final_dataset.py` expects `data/interim/business_zcta_demographics.csv` and merges the ZCTA enrichment by `business_id`.

### 4. Validate The Dataset

```bash
python src/validate_dataset.py
```

This prints row counts, missing values, star distributions, target distributions, and the number of rows with area-level demographic matches.

For the current dental review dataset, a successful rebuild should report approximately:

```text
Rows: 35,132
Unique businesses: 2,498
Rows with successful area-level demographic match: 35,117
```

Small differences are possible if the raw Yelp data or Census ACS data vintage changes, but the important check is that the demographic match count is far above zero and the final row count is preserved from `data/interim/dental_review_business_base.csv`.

## Final Dataset

Final output:

```text
data/final/dental_review_satisfaction_dataset.csv
```

Important columns include:

```text
review_id
business_id
user_id
review_stars
satisfied_binary
negative_binary
review_text
review_date
review_year
review_length_chars
review_length_words
business_name
business_city
business_state
business_postal_code
latitude
longitude
business_categories
business_review_count
area_median_household_income
area_income_band
area_poverty_rate
area_bachelor_plus_rate
area_unemployment_rate
area_pct_white
area_pct_black
area_pct_asian
area_pct_hispanic_or_latino
area_largest_demographic_group
area_largest_demographic_share
```

Target definitions:

```text
satisfied_binary = 1 if review_stars >= 4, else 0
negative_binary = 1 if review_stars <= 2, else 0
```
