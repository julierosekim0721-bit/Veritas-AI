# Final Grouped Satisfaction Model Report

## Invocation

```text
python src/train_final_grouped_satisfaction_model.py --data data/final/dental_review_satisfaction_dataset.csv --model-out models/final_grouped_satisfaction_model.joblib --report-dir reports/ml --figures-dir figures/ml --test-size 0.2 --random-seed 42 --max-features 5000 --min-df 5 --top-k 20 --overwrite
```

## Provenance

- Repository commit: `a83b50f1dee0425937462854acb43d96a6edd1fe`
- Input data SHA-256: `4f6100fc18d4a0ab484824f79ed6b3f9795a845adf0964b863ff04ed239012f8`
- Seed: `42` (single fixed split, every comparator)
- Split: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`
- Rows: 35,132 total / 28,044 train / 7,088 test
- Businesses (groups): 2,498 total / 1,998 train / 500 test
- Business overlap between train and test: **0** (grouped holdout, zero leakage by business)
- Class counts (total / train / test): {'0': 11262, '1': 23870} / {'0': 8873, '1': 19171} / {'0': 2389, '1': 4699}
- Versions: python 3.11.15, scikit-learn 1.9.0, joblib 1.5.3, pandas 3.0.5, numpy 2.4.6

## Feature policy

Safe metadata: review_length_words, review_year, log1p(business_review_count). Demographics: the eight area_* numeric ACS features from the enriched EDA (median-imputed, standard-scaled). Text: review_text (null -> ""), TF-IDF English stop words, 1-2 grams, max_features=5000, min_df=5. Classifier: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42). Hard-excluded from every model: review_stars, business_stars, IDs, target columns, useful/funny/cool, and unapproved categorical/location fields (mechanically asserted via the feature registry).

## Comparator table

| model | accuracy | balanced acc | macro-F1 | ROC-AUC | minority label | minority P | minority R | minority F1 |
|---|---|---|---|---|---|---|---|---|
| majority_class_dummy | 0.6630 | 0.5000 | 0.3987 | 0.5000 | 0 (n=2389) | 0.0000 | 0.0000 | 0.0000 |
| safe_metadata_logistic | 0.7003 | 0.6632 | 0.6637 | 0.7102 | 0 (n=2389) | 0.5562 | 0.5492 | 0.5527 |
| demographics_logistic | 0.5244 | 0.5345 | 0.5145 | 0.5460 | 0 (n=2389) | 0.3667 | 0.5655 | 0.4449 |
| tfidf_text_logistic | 0.9536 | 0.9538 | 0.9486 | 0.9902 | 0 (n=2389) | 0.9120 | 0.9544 | 0.9327 |
| tfidf_text_metadata_demographics_logistic | 0.9550 | 0.9550 | 0.9502 | 0.9898 | 0 (n=2389) | 0.9150 | 0.9552 | 0.9347 |

The final artifact is the predetermined `tfidf_text_metadata_demographics_logistic` pipeline. No
model was selected post hoc from these test metrics, and no model was refit
on all rows.

## Full model vs text-only deltas

- accuracy delta full vs text only: +0.0014
- balanced accuracy delta full vs text only: +0.0013
- macro f1 delta full vs text only: +0.0015
- roc auc delta full vs text only: -0.0004

These deltas are correlational feature-block comparisons on one fixed split,
not causal effects.

## Figures

- Feature-block comparison: `figures/ml/final_grouped_satisfaction_feature_block_comparison.png`
- Full-model confusion matrix: `figures/ml/final_grouped_satisfaction_confusion_matrix.png`

## Outputs

- Model artifact: `models/final_grouped_satisfaction_model.joblib`
- Metrics: `reports/ml/final_grouped_satisfaction_metrics.csv`, `reports/ml/final_grouped_satisfaction_metrics.json`
- Split manifest: `reports/ml/final_grouped_satisfaction_split_manifest.json`, `reports/ml/final_grouped_satisfaction_split_manifest.csv`
- Top TF-IDF coefficients: `reports/ml/final_grouped_satisfaction_top_tfidf_coefficients.csv`

## Limitations

- This model estimates association with the project's observed binary
  `satisfied_binary` label, not causal drivers of dental-care quality or
  patient satisfaction.
- Review text is written after the experience and can be highly
  outcome-revealing; strong TF-IDF performance is not evidence that a
  pre-visit intervention model would work.
- Area-level ACS features describe business-area context, not individual
  reviewers, patients, or dentists; no group-level or demographic claims
  about individuals are supported.
- The business-grouped holdout supports generalization to unseen businesses
  only; one fixed split is a reproducible comparison, not an uncertainty
  estimate.
- Coefficient terms and feature-block deltas are correlational; verify the
  target-label construction before making student-facing claims.

