"""
Analyze dental review satisfaction dataset.

This script:
1. Loads the final Yelp dental dataset
2. Summarizes review satisfaction
3. Calculates correlations
4. Runs logistic regression Model 1:
   socioeconomic + demographic predictors
5. Runs logistic regression Model 2:
   socioeconomic + demographic + business/review predictors
"""

import pandas as pd
import statsmodels.api as sm


# Load dataset
df = pd.read_csv(
    "data/final/dental_review_satisfaction_dataset.csv"
)

print("\nDATASET SUMMARY")
print("----------------")
print(df.shape)

print("\nSATISFACTION DISTRIBUTION")
print("-------------------------")
print(df["satisfied_binary"].value_counts())
print(df["satisfied_binary"].value_counts(normalize=True))


print("\nREVIEW STAR SUMMARY")
print("-------------------")
print(df["review_stars"].describe())

print("\nREVIEW STAR COUNTS")
print("------------------")
print(df["review_stars"].value_counts().sort_index())


# ---------------------------------------
# Correlation analysis
# ---------------------------------------

corr_vars = [
    "review_stars",
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino"
]

print("\nCORRELATION WITH REVIEW STARS")
print("-----------------------------")
print(
    df[corr_vars]
    .corr()["review_stars"]
    .sort_values()
)


# ---------------------------------------
# Model 1
# SES + demographic predictors
# ---------------------------------------

model1_vars = [
    "satisfied_binary",
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino"
]

model1_df = df[model1_vars].dropna()

X1 = model1_df.drop(columns="satisfied_binary")
y1 = model1_df["satisfied_binary"]

X1 = sm.add_constant(X1)

model1 = sm.Logit(y1, X1).fit()

print("\nMODEL 1: SES + DEMOGRAPHICS")
print("---------------------------")
print(model1.summary())


# ---------------------------------------
# Model 2
# Add business/review characteristics
# ---------------------------------------

model2_vars = [
    "satisfied_binary",
    "area_median_household_income",
    "area_poverty_rate",
    "area_bachelor_plus_rate",
    "area_unemployment_rate",
    "area_pct_white",
    "area_pct_black",
    "area_pct_asian",
    "area_pct_hispanic_or_latino",
    "business_stars",
    "business_review_count",
    "review_length_words",
    "review_year"
]

model2_df = df[model2_vars].dropna()

X2 = model2_df.drop(columns="satisfied_binary")
y2 = model2_df["satisfied_binary"]

X2 = sm.add_constant(X2)

model2 = sm.Logit(y2, X2).fit()

print("\nMODEL 2: FULL MODEL")
print("-------------------")
print(model2.summary())

# =====================================
# EXPORT REGRESSION RESULTS
# =====================================

model_results = pd.DataFrame({
    "Model 1 Coefficient": model1.params,
    "Model 1 P-value": model1.pvalues,
    "Model 2 Coefficient": model2.params,
    "Model 2 P-value": model2.pvalues
})

model_results.to_csv(
    "logistic_regression_results.csv"
)

print("Saved regression table.")

# =====================================
# FIGURE 1: Review Distribution by Star Rating
# =====================================

import matplotlib.pyplot as plt

# Count reviews by star rating
star_counts = df["review_stars"].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 5))

plt.bar(
    star_counts.index,
    star_counts.values
)

plt.xlabel("Review Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Figure 1. Distribution of Dental Reviews by Star Rating")

plt.xticks([1, 2, 3, 4, 5])

plt.grid(axis="y")

plt.savefig(
    "figure1_review_star_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Save results
with open("data/final/regression_results.txt", "w") as f:
    f.write("MODEL 1\n")
    f.write(str(model1.summary()))
    f.write("\n\nMODEL 2\n")
    f.write(str(model2.summary()))

print("\nSaved regression results.")

import numpy as np
import matplotlib.pyplot as plt

# Create range of business star ratings
stars = np.linspace(1, 5, 100)

# Use your model coefficients
coef = model2.params

# Hold other variables at their mean values
X_mean = X2.mean().to_frame().T

predictions = []

for star in stars:
    X_temp = X_mean.copy()
    X_temp["business_stars"] = star

    pred = model2.predict(X_temp)[0]
    predictions.append(pred)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(stars, predictions)

plt.xlabel("Dental Business Rating (Stars)")
plt.ylabel("Predicted Probability of Satisfied Review")
plt.title("Figure 2. Predicted Patient Satisfaction by Dental Business Rating")

plt.ylim(0, 1)
plt.grid(True)

plt.savefig("figure2_business_rating_prediction.png", dpi=300, bbox_inches="tight")

plt.show()
