# Enriched exploratory data analysis — dental Yelp satisfaction

## Purpose

This report investigates Julie's enriched final dataset after ACS/ZCTA demographic enrichment. The aim is to decide whether the new area-level socioeconomic/demographic data provide useful signal for the machine-learning direction, while keeping the interpretation safe: these variables describe the area around the dental business, not individual patients or reviewers.

## Dataset

- Source file: `data/final/dental_review_satisfaction_dataset.csv`
- Rows: 35,132
- Columns: 43
- Unique businesses: 2,498
- Unique states: 13
- Target: `satisfied_binary = 1` for 4-5 star reviews; `0` for 1-3 star reviews.

## Coverage of new demographic data

The ACS/ZCTA enrichment is now usable for EDA and ML feature experiments.

- `area_median_household_income`: 35,087 / 35,132 non-null (99.9%)
- `area_poverty_rate`: 35,117 / 35,132 non-null (100.0%)
- `area_bachelor_plus_rate`: 35,117 / 35,132 non-null (100.0%)
- `area_unemployment_rate`: 35,117 / 35,132 non-null (100.0%)
- Race/ethnicity composition fields: typically 100.0% non-null.

## Class balance

- Not satisfied: 11,262 reviews (32.1%)
- Satisfied: 23,870 reviews (67.9%)
- Majority-class accuracy reference: 0.679

The class distribution is imbalanced but not unusable. Accuracy alone would be misleading, so balanced accuracy, macro-F1, and minority-class recall/F1 should be central.

## Main EDA signals

### Strongest established signal: review text / review length

Review length remains one of the clearest signals. Longer reviews are associated with lower expressed satisfaction. This supports a text-based ML direction using TF-IDF plus interpretable logistic regression.

### Demographic/SES signal

The new demographic features are available for almost all reviews, so they can now be investigated. Descriptive spreads across demographic bins exist, but they should be treated as area-level associations and possible predictive features, not causal explanations.

- Satisfaction-rate spread across income bands: 16.1 percentage points.
- Largest-demographic-group satisfaction-rate spread: 7.4 percentage points.
- State-level satisfaction-rate spread: 20.7 percentage points, but this is heavily confounded by business mix and review culture.

Top linear correlations with `satisfied_binary` are recorded in `reports/eda/tables/correlation_with_satisfaction.csv`. Correlations are useful for EDA, not proof of causality.

## Grouped ML signal benchmark

To avoid leakage across reviews from the same dental business, the exploratory ML benchmark used a grouped holdout split by `business_id`.

- Train rows: 28,044
- Test rows: 7,088
- Train businesses: 1,998
- Test businesses: 500
- Business overlap between train/test: 0

| Model | Accuracy | Balanced accuracy | Macro-F1 | Minority F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `text_plus_metadata_demographics_logistic` | 0.955 | 0.955 | 0.950 | 0.935 | 0.990 |
| `tfidf_text_logistic` | 0.954 | 0.954 | 0.949 | 0.933 | 0.990 |
| `metadata_plus_demographics_logistic` | 0.694 | 0.665 | 0.663 | 0.560 | 0.714 |
| `metadata_logistic` | 0.700 | 0.663 | 0.664 | 0.553 | 0.710 |
| `demographics_logistic` | 0.524 | 0.535 | 0.514 | 0.445 | 0.546 |
| `majority_dummy` | 0.663 | 0.500 | 0.399 | 0.000 | 0.500 |


Best exploratory model by balanced accuracy: `text_plus_metadata_demographics_logistic` with balanced accuracy 0.955.

Important comparisons:

- Metadata-only balanced accuracy: 0.663
- Demographics-only balanced accuracy: 0.535
- Metadata + demographics balanced accuracy: 0.665
- TF-IDF text-only balanced accuracy: 0.954
- Text + metadata + demographics balanced accuracy: 0.955

## Recommended ML directions

1. **Primary direction: interpretable text classification.** Use TF-IDF review text with logistic regression as the main student-friendly ML model. It is likely to be the strongest and easiest to explain.
2. **Secondary direction: metadata baseline.** Keep `review_length_words`, `review_year`, and `log1p(business_review_count)` as a clean non-leaky structured baseline.
3. **Demographic/SES extension: incremental value test.** Now that ZCTA data has high coverage, test whether area-level SES/demographic variables improve grouped held-out prediction beyond metadata and text. Frame this as area context, not patient demographics.
4. **Fair interpretation section.** If demographic features help, say they add predictive signal about the business area's context. If they do not, say the review text and business/review metadata explain more of the observable Yelp satisfaction signal than coarse ZCTA descriptors.
5. **Avoid leakage.** Do not use `review_stars` as a feature. Treat `business_stars` as leaky/proxy-risk and exclude it from primary models.

## Figures generated

- `figures/01_demographic_missingness.png`
- `figures/02_target_class_balance.png`
- `figures/03_review_star_distribution.png`
- `figures/04_review_length_by_satisfaction.png`
- `figures/05_satisfaction_by_income_band.png`
- `figures/06_satisfaction_by_demographic_quintiles.png`
- `figures/07_satisfaction_by_largest_demographic_group.png`
- `figures/08_satisfaction_by_state.png`
- `figures/09_correlation_with_satisfaction.png`
- `figures/10_metadata_demographic_correlation_heatmap.png`
- `figures/11_location_scatter_by_satisfaction.png`
- `figures/12_grouped_ml_signal_benchmark.png`
- `figures/13_best_model_confusion_matrix.png`


## Tables generated

- `reports/eda/tables/grouped_ml_signal_benchmark.csv`
- `reports/eda/tables/grouped_split_info.json`
- `reports/eda/tables/correlation_with_satisfaction.csv`
- `reports/eda/tables/satisfaction_by_income_band.csv`
- `reports/eda/tables/satisfaction_by_largest_demographic_group.csv`
- `reports/eda/tables/satisfaction_by_state.csv`
- `reports/eda/tables/satisfaction_by_<demographic_feature>_quintile.csv`

## Interpretation boundaries

- These are Yelp reviews, not clinical patient records.
- The target measures expressed Yelp satisfaction, not objective dental-care quality.
- ZCTA variables are approximate business-area descriptors.
- Demographic features do not identify individual patients, reviewers, or dentists.
- EDA and exploratory ML show associations/predictive signal, not causation.
