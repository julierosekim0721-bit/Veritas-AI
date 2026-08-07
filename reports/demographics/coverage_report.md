# ZCTA demographic enrichment coverage

- Run timestamp (UTC): 2026-08-07T16:40:59.210318+00:00
- Businesses input: `data/interim/dental_businesses.csv`
- Reviews input: `data/interim/dental_review_business_base.csv`
- Processed ZCTA catalogue: `data/interim/zcta_area_demographics.csv`
- ACS vintage/dataset: 2022 `acs/acs5`

**Interpretation warning:** These features describe the Census area around a business, not an individual reviewer, patient, or dentist.
A five-digit business ZIP is treated only as a ZCTA candidate; no missing or unreturned geography is guessed.

## Postal validation and ZCTA lookup

- `matched`: 2,498
- `missing_postal_code`: 0
- `invalid_or_non_us_postal_code`: 0
- `zcta_not_returned`: 0

## Business-weighted coverage

- Any requested area feature non-null: 2,496 / 2,498 (99.92%)

## Review-weighted coverage

- Any requested area feature non-null: 35,117 / 35,132 (99.96%)

## Feature missingness

| Feature | Business non-null | Business denominator | Business rate | Review non-null | Review denominator | Review rate |
|---|---:|---:|---:|---:|---:|---:|
| `area_median_household_income` | 2,493 | 2,498 | 99.80% | 35,087 | 35,132 | 99.87% |
| `area_poverty_rate` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
| `area_bachelor_plus_rate` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
| `area_unemployment_rate` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
| `area_pct_white` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
| `area_pct_black` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
| `area_pct_asian` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
| `area_pct_hispanic_or_latino` | 2,496 | 2,498 | 99.92% | 35,117 | 35,132 | 99.96% |
