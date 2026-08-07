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

The script `src/get_area_demographics.py` retrieves area-level descriptors from public Census-related sources:

```text
https://api.census.gov/data.html
https://censusreporter.org/
https://geocoding.geo.census.gov/
```

If a Census API key is available, set it before running the script:

```bash
export CENSUS_API_KEY="your_key_here"
```

Alternatively, create a local `.env` file in the repository root:

```text
CENSUS_API_KEY=your_key_here
```

Do not commit `.env` or paste the key into notebooks, reports, or source code. The repository ignores `.env` by default.

Before running the full demographic pipeline, verify that Census access works:

```bash
python src/smoke_test_census_api.py
```

The smoke test prints a small JSON status summary and does not print the API key.

If no key is set, the script falls back to Census Reporter and the Census coordinate geocoder. 

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

```bash
python src/get_area_demographics.py
```

This creates:

```text
data/interim/area_demographics.csv
```

The demographic fields are linked by business location. They describe the area around the dental practice, not individual patients or reviewers.

### 3. Build The Final Dataset

```bash
python src/build_final_dataset.py
```

This creates:

```text
data/final/dental_review_satisfaction_dataset.csv
```

### 4. Validate The Dataset

```bash
python src/validate_dataset.py
```

This prints row counts, missing values, star distributions, target distributions, and the number of rows with area-level demographic matches.

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
