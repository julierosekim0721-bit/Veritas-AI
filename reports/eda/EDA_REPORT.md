# Hypothesis-Driven EDA: Yelp Dental Review Satisfaction

## Purpose and provenance

This report evaluates whether the current preprocessed dataset supports the initial question: **can socioeconomic or demographic context predict Yelp dental review satisfaction?** It was generated reproducibly by `src/exploratory_data_analysis.py` from `data/final/dental_review_satisfaction_dataset.csv`. The dataset has **35,132 reviews and 36 columns**, covering **2,498 businesses**, **294 cities**, and **13 states**.

Each row is one Yelp review joined to dental-business metadata. Results are descriptive associations, not causal effects, and reviews from the same business are not statistically independent.

## Target definitions and integrity

- `satisfied_binary = 1 if review_stars >= 4 else 0`
- `negative_binary = 1 if review_stars <= 2 else 0`

The audit found **0 satisfied-target mismatches** and **0 negative-target mismatches**. `review_stars` therefore directly defines both targets and must never be used as an ordinary predictor of either target.

## H0/intake: Are demographic features available?

**Verdict: no.** The demographic availability audit found **0 rows with at least one non-null area-level demographic value**. All requested socioeconomic/racial-composition measures are 100% missing. The original hypothesis cannot be tested until Census enrichment is repaired and validated. No imputation or demographic model is scientifically defensible when an entire feature is absent.

See `tables/demographic_availability.csv`, `tables/missingness.csv`, and `figures/missingness.png`.

## Class balance and review stars

- Satisfied reviews (4–5 stars): **23,870 / 35,132 (67.9%)**.
- Not-satisfied reviews (1–3 stars): **11,262 / 35,132 (32.1%)**.
- The most common review rating is **5 stars**.

This imbalance means accuracy alone is not a sufficient model metric. Use a majority-class baseline, balanced accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## H1-now: Business-level context

`business_stars` has a Pearson correlation of **0.534** with satisfaction in this review-level dataset. Across observed business-star groups, satisfaction rates range from **1.2% to 98.3%**. `business_review_count` has correlation **0.006**.

These are useful descriptive signals, but `business_stars` is likely calculated from reviews including the current review. It is therefore a strong **target proxy/leakage risk**, not clean independent context. A later model should either exclude it, reconstruct a leave-one-review-out/prior-time business rating, or present results both with and without it. Train/test splitting should be grouped by `business_id` to prevent the same business leaking into both sets.

## H2-now: Review length and time

Review length has correlation **-0.330** with satisfaction. Satisfaction across word-count quintiles ranges from **40.5% to 85.2%**. Review year spans **2006–2022** and has correlation **-0.042**. These are descriptive associations only; length may capture review extremity, while year may reflect changes in Yelp usage or the businesses represented.

## H3-now: Geographic variation

The highest-volume states shown in the EDA have satisfaction rates from **58.4% to 79.1%**; the highest-volume cities shown range from **57.7% to 77.3%**. These differences are not evidence that place causes satisfaction. They may be driven by unequal sample sizes, a small number of businesses, business mix, time, or review-selection effects. Always show counts beside rates, set a minimum sample threshold, and prefer grouped uncertainty estimates before interpreting rankings.

## Text-feature readiness

Review text is non-blank for **35,132 rows (100.0%)**. Median length is **87 words**, mean length is **122.3**, and the 95th percentile is **350 words**. The most frequent simple non-stopword terms are: dentist (31,208), office (23,857), dental (22,855), staff (18,550), time (15,202), teeth (14,816), great (12,300), appointment (12,140), work (12,051), back (10,810).

This supports a student-friendly TF-IDF model. However, terms such as “great” or “worst” can make sentiment classification easy without addressing socioeconomic context; explain that the text model predicts expressed satisfaction rather than its causes.

## Leakage, confounds, and interpretation warnings

1. **Direct leakage:** `review_stars` defines the targets; never include it as a predictor.
2. **Aggregate target proxy:** `business_stars` aggregates review ratings and may include the focal review.
3. **Cluster leakage:** many reviews belong to the same business; use group-aware splits by `business_id`.
4. **User leakage:** repeat reviewers may appear across rows; audit `user_id` and consider a stricter grouped sensitivity analysis.
5. **Geographic confounding:** city/state rates mix business composition, sample size, year, and Yelp participation.
6. **Selection bias:** Yelp reviewers are not a random sample of dental patients.
7. **Causality:** predictive associations do not establish that demographics or business features cause satisfaction.

## Hypothesis evolution

### Initial hypothesis

Area-level socioeconomic/demographic context predicts whether a dental Yelp review is satisfied.

### EDA verdict

**Not testable with the current dataset.** Census matching produced zero usable demographic rows. The correct action is to repair enrichment, not fit a zero-row model or fill wholly absent features.

### Revised, implementable hypotheses

- **H1-now:** Business-level features (`business_review_count`, and cautiously `business_stars`) improve prediction over a majority baseline. Compare versions excluding and including the proxy-risk feature.
- **H2-now:** Review text length and review year have measurable but likely limited associations with satisfaction.
- **H3-now:** City/state differences remain after minimum-count filtering, but may shrink under group-aware validation and business controls.
- **H-text-now:** TF-IDF review text predicts expressed satisfaction better than structured non-leaky features, with interpretable positive/negative coefficients.
- **H4-next:** After Census repair, demographic features add out-of-sample value beyond non-leaky business, time, geography, and text baselines.

## Recommended student ML pipeline

1. **Data gate:** verify target definitions, demographic coverage, missingness, duplicates, and class balance.
2. **Split first:** reserve a test set grouped by `business_id`; use stratified group cross-validation if practical.
3. **Baseline:** majority-class prediction and a simple non-leaky structured baseline.
4. **Interpretable structured model:** logistic regression with scaled/log-transformed `business_review_count`, review length, year, and carefully encoded geography. Exclude `review_stars`; report `business_stars` only as a sensitivity analysis.
5. **Text model:** TF-IDF (word unigrams/bigrams, capped vocabulary) plus logistic regression. Inspect top coefficients and errors.
6. **Optional extension:** random forest or gradient boosting on structured features only, after the interpretable baseline. Do not make it the starting point.
7. **Evaluation:** balanced accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and fold-to-fold variability. Compare every model on the same held-out groups.
8. **Demographic extension:** repair postal-code/Census matching, report match rate and unmatched bias, then test whether demographic features improve grouped held-out performance. Treat this as incremental predictive value, not causality.

## Bottom line

The dataset is ready for descriptive business/time/text EDA and a careful student-level prediction pipeline, but **not yet ready for the proposed socioeconomic/demographic analysis**. The immediate scientific blocker is Census enrichment coverage; the immediate implementable ML path is grouped, leakage-aware baselines followed by interpretable structured and TF-IDF logistic regression models.

## Generated artifacts

- Tables: `reports/eda/tables/` (CSV and Markdown)
- Figures: `reports/eda/figures/`
- Detailed relationship values: `reports/eda/tables/feature_relationships.csv`
