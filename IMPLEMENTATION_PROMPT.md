# Coding Agent Implementation Prompt
## Machine Learning HW2 — Multiple Linear Regression with CRISP-DM

### Role and Mission

You are an autonomous Senior Machine Learning Engineer, Statistical Modeling Specialist, and Python Software Quality Engineer.

Your assignment is to **implement, execute, test, debug, and validate** a complete university Machine Learning HW2 project based on a research methodology already approved by the student and ChatGPT.

The project is:

**Evaluating Feature Selection Methods for Multiple Linear Regression in House Price Prediction: A CRISP-DM Approach**

Your responsibility is to deliver a fully functioning, reproducible Python experiment, including model evaluation, statistical diagnostics, prediction intervals, visualizations, CSV results, saved artifacts, and a comprehensive handoff summary.

You must perform the engineering work yourself. Do not merely generate code and assume that it works.

You are authorized to autonomously resolve routine implementation problems, dependency issues, numerical stability issues, plotting errors, file-path problems, and compatibility problems.

However, **you are not authorized to change the research methodology, replace the dataset, introduce unrelated models, or expand the project into an application that was not requested.**

The student and ChatGPT will separately handle research interpretation, NotebookLM research, and the formal academic PDF report.

---

# 1. Non-Negotiable Research Requirements

## 1.1 Dataset

Use only the following Kaggle dataset:

**House Sales in King County, USA**

Official source:

https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

Expected file:

`kc_house_data.csv`

Expected dataset characteristics:

| Property | Expected value |
|---|---|
| Observations | 21,613 |
| Original columns | 21 |
| Target | `price` |
| Identifier | `id` |
| Transaction date | `date` |
| Original predictor features | 18 |
| Learning task | Multiple Linear Regression |
| Prediction target | House sale price in USD |

The exact dataset must be verified after loading.

Do not assume the CSV is valid simply because the filename matches.

Check the column names, dataset dimensions, numeric types, missing values, identifier duplication, and target integrity.

If the supplied dataset differs materially from the expected Kaggle dataset, stop the affected experiment and report the discrepancy. Do not silently change the research specification.

If the CSV is not available locally, first search the existing project directory and configured input locations. Attempt to obtain the official Kaggle dataset using available public or authorized access.

Do not replace the dataset with a GitHub mirror, alternative house-price dataset, generated data, or unrelated Kaggle dataset without explicit authorization.

If downloading requires unavailable credentials or connectivity, complete the implementation and synthetic-fixture engineering tests where possible, but clearly report that the actual research experiments could not be executed. Never present synthetic test results as real experimental findings.

## 1.2 Feature Definition

The target is:

`price`

Exclude the following columns from the predictive feature matrix:

- `id`: used exclusively for identifying and grouping repeated house transactions.
- `date`: retained only for dataset inspection and descriptive analysis.
- `price`: the prediction target.

The 18 eligible original predictors are:

1. `bedrooms`
2. `bathrooms`
3. `sqft_living`
4. `sqft_lot`
5. `floors`
6. `waterfront`
7. `view`
8. `condition`
9. `grade`
10. `sqft_above`
11. `sqft_basement`
12. `yr_built`
13. `yr_renovated`
14. `zipcode`
15. `lat`
16. `long`
17. `sqft_living15`
18. `sqft_lot15`

Preserve these 18 original predictors as the shared candidate feature set across all six experimental methods.

Treat `zipcode` as a categorical variable, despite its numerical representation in the original CSV.

For the primary experiment, retain other numeric and ordinal predictor columns in their original numerical representation. Document this modeling assumption.

Do not automatically interpret zero-valued `yr_renovated` or `sqft_basement` observations as missing values.

Do not introduce polynomial features, geographical clusters, transaction-year features, target transformations, or engineered predictors without explicit authorization.

## 1.3 Required Models

Exactly six primary experimental configurations are required:

| ID | Feature selection | Final estimator |
|---|---|---|
| M0 | All eligible features (Baseline) | `LinearRegression` |
| M1 | Pearson Correlation | `LinearRegression` |
| M2 | Mutual Information | `LinearRegression` |
| M3 | Recursive Feature Elimination (RFE) | `LinearRegression` |
| M4 | Recursive Feature Elimination with Cross-Validation (RFECV) | `LinearRegression` |
| M5 | LASSO Embedded Feature Selection | `LinearRegression` |

The final prediction estimator for all six experiments must be:

`sklearn.linear_model.LinearRegression`

For M5, LASSO is used for feature selection. The selected features must subsequently be refitted using `LinearRegression`.

Do not treat the LASSO regression model's own predictions as the final M5 prediction results.

Do not substitute Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Neural Networks, or other predictors for the six required experiments.

Do not include unrelated machine learning algorithms.

---

# 2. CRISP-DM Implementation Requirements

The project must follow the six stages of CRISP-DM.

The program must generate the analytical artifacts needed to document each stage.

You are not responsible for writing the student's formal academic report.

## Phase 1 — Business Understanding

Define the predictive objective as:

> Predict residential property sale prices using historical house attributes, and evaluate whether feature selection can reduce model complexity while preserving or improving the predictive performance of Multiple Linear Regression.

Research questions:

**RQ1:** Can feature selection improve prediction accuracy compared with the all-feature baseline?

**RQ2:** Can feature selection reduce model complexity without substantially reducing predictive performance?

**RQ3:** Can a calibrated Multiple Linear Regression model produce reasonable 95% prediction intervals for unseen house sale prices?

Do not invent business costs, commercial deployment claims, or future market forecasting capabilities.

The dataset represents historical transactions. The primary experiment evaluates generalization to held-out houses drawn from a broadly comparable historical data distribution, not prediction of current or future real estate markets.

Required artifact:

`research_configuration.json`

This file must record the research objective, dataset reference, candidate features, model configurations, random seed, data splitting method, and evaluation protocol.

## Phase 2 — Data Understanding

Perform a genuine Exploratory Data Analysis (EDA).

Required inspections:

1. Dataset dimensions.
2. Column names and data types.
3. Numeric and categorical column identification.
4. Missing-value counts and percentages.
5. Duplicate rows.
6. Duplicate property identifiers.
7. Number of unique property IDs.
8. Target descriptive statistics.
9. Target distribution and skewness.
10. Main feature distributions.
11. Pearson correlation between numerical predictors and price.
12. Correlation among numerical predictors.
13. Relationships among house area variables.
14. Unusual values and possible data-quality problems.
15. Transaction date range.

Investigate whether the following relationship holds exactly or approximately:

`sqft_living = sqft_above + sqft_basement`

Report the number of rows satisfying the identity and any discrepancies.

Do not assume this relationship is exact before testing.

For EDA, distinguish between basic dataset integrity checks and analyses that influence model-development choices.

All target-dependent decisions must use Development data only.

Do not inspect Calibration or Test targets to decide feature transformations, outlier-removal thresholds, or model settings.

Required output:

`data_quality_summary.csv`

Create readable EDA visualizations and save descriptive statistics.

Do not delete unusual high-price observations merely to improve evaluation metrics.

Only remove invalid records when the removal rule is objective, documented, and justified by actual data integrity problems.

## Phase 3 — Data Preparation

### 3.1 Data Splitting

Use three mutually exclusive partitions:

| Partition | Target proportion | Purpose |
|---|---:|---|
| Development | 65% | Model training, feature selection, nested CV |
| Calibration | 15% | Conformal prediction interval calibration |
| Test | 20% | Final evaluation only |

Use:

`random_state = 42`

Because a property can have more than one transaction, perform group-aware splitting based on `id`.

All rows belonging to one property ID must remain in the same partition.

Use a suitable group-aware strategy, such as two sequential `GroupShuffleSplit` operations.

If using two splits, target approximately 20% of observations for Test, then approximately 15% of the original observations for Calibration.

Because groups can have different sizes, actual observation percentages may differ slightly. Do not split a group merely to force exact percentages.

The `id` column must never enter the predictive model.

Required checks:

- Development and Calibration share zero property IDs.
- Development and Test share zero property IDs.
- Calibration and Test share zero property IDs.
- No observation appears in multiple partitions.
- All usable observations belong to exactly one partition.

Save the actual row counts, group counts, proportions, and group-overlap verification results.

Do not perform a conventional row-wise random split that permits the same property ID to appear in multiple partitions.

### 3.2 Preprocessing

Implement leakage-safe preprocessing using scikit-learn-compatible Pipelines.

Numerical predictors:

- Median imputation, if needed.
- Standardization.
- Retain meaningful zero values.
- Preserve numerical values for Linear Regression.

Categorical predictor:

- `zipcode` must use One-Hot Encoding.
- Use a reference-category strategy, such as `drop='first'`.
- Handle previously unseen ZIP Codes without crashing.
- Learn categories using only the current training fold.

Standardize the encoded feature matrix appropriately so that coefficient-based feature selection and LASSO are not unfairly dominated by original measurement units.

Document how numerical variables and ZIP Code dummy variables are scaled.

Use the same preprocessing design for all six methods.

The exact fitted preprocessing parameters must be learned independently within every relevant training fold.

### 3.3 Original Features vs. Encoded Columns

Clearly distinguish:

**Original predictor count:** 18.

**Encoded feature count:** Number of columns produced after preprocessing.

**Selected encoded feature count:** Number of encoded columns retained by the feature-selection method.

**Selected original feature groups:** Original predictors represented by at least one selected encoded column.

The feature-selection unit in this project is the **encoded model column**.

This means a feature-selection method may retain some ZIP Code dummy variables without retaining every ZIP Code category.

Do not describe selected dummy columns as separate original Kaggle features.

Maintain the mapping between every encoded column and its originating predictor.

Save this mapping and the selected column names.

### 3.4 Preprocessing Integrity

Do not fit any of the following transformations on the complete dataset before partitioning:

- Imputation.
- Standardization.
- One-Hot Encoding.
- Correlation-based feature selection.
- Mutual Information feature selection.
- RFE.
- RFECV.
- LASSO feature selection.
- Target-dependent outlier handling.

All learned transformations must be fitted inside the appropriate training fold.

Prefer integrated Pipelines so that leakage prevention is enforced structurally rather than through informal coding conventions.

---

# 3. Nested Cross-Validation

## 3.1 Required Validation Architecture

Use:

**Outer 5-Fold Group Cross-Validation**

and

**Inner 3-Fold Group Cross-Validation**

The entire nested cross-validation process must use the Development partition only.

The Calibration and Test partitions must remain completely excluded.

### Outer Cross-Validation

Use five group-disjoint folds, such as `GroupKFold(n_splits=5)`.

For each outer fold:

1. Separate outer training and outer validation groups.
2. Perform all feature-selection fitting and hyperparameter searches only on the outer training portion.
3. Use inner group-aware cross-validation where hyperparameter tuning is needed.
4. Refit the best configuration on the outer training portion.
5. Predict the outer validation portion exactly once for that fitted configuration.
6. Save the outer validation metrics, selected features, and parameters.

Repeat for all five outer folds.

Use identical outer fold assignments across M0 through M5.

### Inner Cross-Validation

Use three group-aware folds.

Inner CV is responsible for selecting:

- Pearson retained feature count.
- Mutual Information retained feature count.
- RFE retained feature count.
- RFECV retained feature count through RFECV's internal CV.
- LASSO regularization strength.

Every inner split must also be group-disjoint.

Never preselect features using the complete outer training portion and then report an inner CV score calculated after that selection unless the selection is properly nested within every inner training fold.

The complete preprocessing and selection pipeline must be fitted independently inside each inner training fold.

### Required Cross-Validation Checks

Assert that:

- No property group overlaps between training and validation in any fold.
- No Calibration samples are included in nested CV.
- No Test samples are included in nested CV.
- The feature selector is refitted within each appropriate training fold.
- Hyperparameter selection never accesses outer validation targets.
- Identical outer fold assignments are used across the six model configurations.

Record sample counts, group counts, selected features, and chosen parameters for every fold.

---

# 4. Model Implementation Specifications

## M0 — All-Feature Baseline

Build a Pipeline containing:

1. Preprocessing.
2. All encoded predictor columns.
3. `LinearRegression`.

Use every eligible original predictor, including properly encoded `zipcode`.

No feature-selection step is allowed in M0.

Evaluate with the same outer five folds as all other methods.

## M1 — Pearson Correlation

Use an appropriate Pearson-based univariate filter.

Compute the Pearson correlation of each encoded predictor with the target using only the applicable training fold.

Rank by absolute correlation magnitude.

Candidate retained-column counts:

`[5, 10, 20, 40, 60]`

Ignore candidate values that exceed the number of available encoded columns in the relevant training configuration.

Select the retained feature count using inner three-fold group-aware CV with RMSE as the objective.

Handle constant or zero-variance columns safely.

A constant column must not introduce NaN-based selection errors or silently receive an artificially high importance score.

Save:

- Pearson scores.
- Selected features.
- Selected feature count.
- Inner CV configuration.
- Outer CV performance.

## M2 — Mutual Information

Use:

`sklearn.feature_selection.mutual_info_regression`

Estimate dependence between encoded features and price.

Use the same feature-count candidate set as Pearson:

`[5, 10, 20, 40, 60]`

Choose the feature count using inner three-fold group-aware CV.

Correctly configure discrete versus continuous features.

In particular, identify ZIP Code dummy variables as discrete, even if they have undergone numerical scaling.

Handle binary and other discrete predictors consistently with documented assumptions.

Do not blindly apply a single continuous-feature interpretation to every encoded column.

Set and record deterministic random states where supported.

Save:

- Mutual Information scores.
- Selected features.
- Selected count.
- Inner CV results.
- Outer CV performance.

## M3 — Recursive Feature Elimination

Use:

`sklearn.feature_selection.RFE`

Base estimator:

`LinearRegression`

Use coefficient magnitude to rank candidate features.

Perform preprocessing and scaling before RFE.

Candidate retained-column counts:

`[5, 10, 20, 40, 60]`

Use a removal step of 5 encoded columns per iteration to make computation practical.

Select the retained feature count with inner three-fold group-aware CV.

Save:

- RFE feature rankings.
- Selected features.
- Best retained count.
- Outer CV performance.

Do not perform RFE once on the full Development partition before the outer CV evaluation.

## M4 — RFECV

Use:

`sklearn.feature_selection.RFECV`

Base estimator:

`LinearRegression`

Required settings:

- Group-aware internal 3-Fold CV.
- RMSE-oriented scoring.
- `step=5`.
- Minimum selected encoded features: at least 1.

RFECV must run inside each outer training fold.

The CV used by RFECV must never contain the outer validation fold.

Account for scikit-learn API/version differences in forwarding group labels to RFECV.

If the installed version cannot provide genuinely group-aware inner CV through the selected Pipeline configuration, use a mathematically equivalent, explicitly implemented group-aware recursive-elimination evaluation procedure.

Such a compatibility implementation must retain the same RFECV research method.

Do not silently fall back to ordinary KFold.

Save:

- Selected feature count.
- Selected encoded columns.
- Recursive elimination rankings.
- CV score by feature count.
- RFECV performance curve.
- Outer CV performance.

## M5 — LASSO Embedded Feature Selection

Use L1-regularized linear regression for feature selection.

Use a group-aware inner 3-Fold procedure to select the LASSO regularization strength `alpha`.

Use a predeclared, deterministic alpha search grid appropriate for a house-price target measured in USD, with logarithmic spacing across approximately `10^0` to `10^5`.

Record the exact grid.

Feature preprocessing must be refitted within every inner training fold.

Important implementation requirement:

`LassoCV` may be used only if its internal CV and preceding preprocessing are genuinely leakage-safe and group-aware.

If inserting `LassoCV` into a Pipeline would cause preprocessing to be fitted using inner-validation data, instead implement the same embedded LASSO method using a full Pipeline and group-aware `GridSearchCV` over `Lasso` regularization strength.

This is an authorized engineering implementation of the approved LASSO selection method, not authorization to replace the method.

After choosing alpha:

1. Fit the LASSO selector on the applicable training partition.
2. Identify selected columns by a documented, predeclared coefficient threshold.
3. Use only those selected columns to fit a fresh `LinearRegression`.
4. Evaluate predictions using the final LinearRegression model.

Use sufficient iteration limits and appropriate convergence tolerances.

Treat convergence warnings seriously. Resolve them through legitimate numerical engineering measures, and document adjustments.

If a candidate alpha selects zero columns, mark that candidate invalid. Do not silently introduce an unapproved fallback feature set.

If every candidate is invalid, report a clear experiment failure rather than fabricating a result.

Save:

- Selected alpha.
- LASSO coefficients.
- Nonzero coefficient count.
- Selected encoded columns.
- Final LinearRegression performance.
- Coefficient or regularization visualization.

---

# 5. Model Selection Policy

The primary model-selection metric is:

**Mean Outer-CV RMSE**

Lower is better.

Do not select a model based on final Test performance.

Apply the following predefined selection policy:

1. Identify the method with the lowest mean outer-fold RMSE.
2. Calculate the standard error of that best method's outer-fold RMSE using its five fold scores.
3. Identify methods whose mean RMSE is no greater than the best mean RMSE plus that standard error.
4. Among those eligible methods, prefer the method with the lowest mean selected encoded-column count.
5. If still tied, prefer the lowest mean CV RMSE.
6. If still tied, use the fixed order M0, M1, M2, M3, M4, M5.

Clearly describe the one-standard-error criterion as a practical model-selection heuristic, not as proof of statistical equivalence.

Freeze the selected method before reading the Test evaluation results.

Record:

- Best-RMSE method.
- Selection threshold.
- Eligible methods.
- Selected final method.
- Reason for selection.
- Selected method's mean encoded-column count.

After selecting the final method, rerun the appropriate inner group-aware tuning process on the complete Development set and refit that selected method using all Development observations.

Also refit the remaining five predefined methods on the same Development set for descriptive Test comparisons.

However, the primary winner must remain the method selected using Development CV. Test results must not change that decision.

---

# 6. Evaluation Metrics

Calculate the following:

1. Mean Absolute Error (MAE).
2. Mean Squared Error (MSE).
3. Root Mean Squared Error (RMSE).
4. R-squared (R²).
5. Adjusted R-squared (Adjusted R²).
6. Five-Fold Cross-Validation mean and standard deviation.

Use the original USD price scale.

Do not transform the prediction target to logarithmic prices in the primary experiments.

For each outer fold, save MAE, MSE, RMSE, and R².

Calculate the mean and sample standard deviation across the five outer folds.

Distinguish:

- Mean of fold RMSE values.
- RMSE calculated from pooled out-of-fold predictions.

These values can differ and must not be mislabeled.

### Adjusted R²

Use:

Adjusted R² = 1 − (1 − R²) × (n − 1) / (n − p − 1)

Where:

- `n` is the number of evaluated observations.
- `p` is the appropriate effective number of predictor degrees of freedom.

For rank-deficient designs, determine the effective predictor rank rather than blindly equating it with the number of encoded columns.

Calculate adjusted R² only when its denominator is valid.

Clearly distinguish training adjusted R² from any descriptive application of the formula to validation or Test observations.

Do not use adjusted R² as the primary hyperparameter-selection metric.

### Required Model Metrics

For each of the six configurations, provide:

- Outer CV MAE mean and SD.
- Outer CV MSE mean and SD.
- Outer CV RMSE mean and SD.
- Outer CV R² mean and SD.
- Relevant adjusted R² diagnostics.
- Mean selected encoded-column count.
- Final Development-fit selected features.
- Development-fit model complexity.
- Test MAE.
- Test MSE.
- Test RMSE.
- Test R².
- Descriptive Test adjusted R², if defined.
- Actual computational runtime.

All values must come from actual program execution.

Do not hardcode expected R² values obtained from other Kaggle notebooks.

---

# 7. Statistical Diagnostics

Perform statistical diagnostics for the final selected LinearRegression model.

Required analyses:

## 7.1 Linearity

Generate a Residual vs. Fitted plot.

Look for systematic curves, residual patterns, or model misspecification.

## 7.2 Residual Normality

Generate:

- Residual histogram.
- Normal Q-Q Plot.

Optional numerical normality diagnostics may be provided, but interpret them cautiously because large samples can produce very small p-values for minor deviations.

## 7.3 Homoscedasticity

Generate a Residual vs. Fitted plot and perform a Breusch–Pagan test where mathematically appropriate.

Record the test statistic and p-value.

Do not conclude that the entire linear model is invalid solely because a large-sample test rejects homoscedasticity.

Explain the diagnostic limitation in the output summary.

## 7.4 Multicollinearity

Inspect:

- Predictor correlation matrix.
- Relevant high-correlation feature pairs.
- Variance Inflation Factor (VIF).
- Design-matrix rank.

Handle exact or near-exact linear dependencies robustly.

If a VIF is infinite or undefined, report it as such.

Do not fabricate finite VIF values.

Do not silently remove collinear variables from the Baseline merely to make VIF calculations succeed.

## 7.5 Outliers

Inspect residual magnitude and error distribution.

Report extreme prediction errors without deleting them simply because they reduce model performance.

Prefer training or out-of-fold residuals for model-development diagnostics.

Any final Test diagnostic must remain descriptive and must not trigger another round of model tuning.

Save diagnostics in machine-readable output.

---

# 8. 95% Prediction Interval

## 8.1 Mandatory Method

The primary prediction-interval method is:

**95% Split Conformal Prediction using absolute residual nonconformity scores.**

Do not replace this method with arbitrary percentage intervals.

Do not replace it with ordinary OLS confidence intervals.

Do not add bootstrap interval methods, quantile regression, adaptive conformal methods, or other interval models unless separately authorized.

## 8.2 Calibration Procedure

Use the frozen final selected model trained on the Development partition.

Predict the independent Calibration partition.

Calculate absolute residual scores:

`s_i = abs(y_i − y_pred_i)`

Use:

`alpha = 0.05`

For `n_cal` calibration observations, calculate:

`k = ceil((n_cal + 1) * (1 − alpha))`

Use the corresponding finite-sample conformal order statistic.

Handle the extreme-quantile boundary correctly.

Do not use the default empirical quantile interpolation without checking whether it corresponds to the required finite-sample conformal quantile.

Let the resulting threshold be `q_hat`.

Construct each prediction interval as:

**Lower = Prediction − q_hat**

**Upper = Prediction + q_hat**

Do not use a manually selected percentage of predicted price.

Do not use Test residuals to calculate `q_hat`.

Do not refit or tune the model using the Calibration targets.

## 8.3 Group Dependence Caveat

The dataset may include multiple transactions for a property.

Group-disjoint splitting prevents the same property from appearing in Development, Calibration, and Test, but it does not automatically prove that all transaction-level residuals are independent or exchangeable.

Document that finite-sample conformal coverage guarantees require the relevant exchangeability assumptions.

Do not claim guaranteed conditional coverage for every house-price range.

## 8.4 Required Interval Evaluation

On the independent Test set, calculate:

**Empirical Coverage Probability**

Fraction of actual prices lying inside the predicted intervals.

**Average Prediction Interval Width**

Mean of the interval upper bound minus the interval lower bound.

**Number of Covered Observations**

**Number of Uncovered Observations**

**Nominal Coverage**

95%.

**Empirical Coverage by Predicted Price Range**

Divide properties into approximately five price ranges based on predicted prices.

Determine price-bin boundaries from Calibration predictions before examining the Test results.

Then apply these boundaries unchanged to the Test predictions.

If ties or repeated boundaries prevent exactly five valid groups, document and handle the issue without fabricating bins.

Record the number of observations and coverage for each actual group.

Do not tune `q_hat` after observing Test coverage.

Do not claim that empirical coverage must be exactly 95%.

If any calculated interval bound is negative, retain the raw statistical interval for evaluation and report the count of negative lower bounds.

Do not silently clip intervals in the official evaluation.

## 8.5 Prediction Interval Plot

Generate a clear 95% Prediction Interval plot using actual Test observations.

Requirements:

- Select up to 100 Test observations using a fixed random seed.
- Do not cherry-pick observations based on successful coverage.
- Sort the selected observations by predicted house price for readability.
- Plot predicted values.
- Plot actual house prices.
- Plot lower and upper 95% prediction bounds.
- Visually distinguish observations outside their intervals.
- Include descriptive axis labels, title, legend, and USD units.

Clearly state that the figure is a visualization of a subset of Test observations.

Compute Coverage and Average Width using the **full Test partition**, not merely the plotted subset.

Save the underlying plotting data in CSV.

---

# 9. Required Figures

Generate genuine figures derived from the actual dataset and experimental results.

Use consistent, professional visual formatting.

Save high-resolution PNG files suitable for inclusion in an academic PDF report.

The required figures are:

| Filename | Content |
|---|---|
| `01_target_distribution.png` | House-price distribution and relevant summary |
| `02_price_boxplot.png` | House-price box plot and outliers |
| `03_feature_correlation_heatmap.png` | Numeric feature correlation matrix |
| `04_living_area_vs_price.png` | Relationship between house size and price |
| `05_feature_selection_comparison.png` | Selected features across all five selection methods |
| `06_cv_rmse_comparison.png` | Six-method outer CV RMSE mean and SD |
| `07_cv_r2_comparison.png` | Six-method outer CV R² comparison |
| `08_feature_count_comparison.png` | Selected encoded-column counts |
| `09_rfecv_performance.png` | RFECV feature-count performance |
| `10_lasso_coefficients.png` | LASSO selected coefficients |
| `11_actual_vs_predicted.png` | Actual vs. predicted Test prices |
| `12_residuals_vs_fitted.png` | Final-model residual diagnostics |
| `13_residual_qq_plot.png` | Final-model Q-Q plot |
| `14_prediction_interval_95.png` | Test predictions, actual values, 95% intervals |
| `15_interval_coverage_by_price.png` | Empirical coverage by predicted-price range |

Use actual experimental results in all plots.

For feature-selection comparisons, explain how ZIP Code dummy variables are represented.

For RFECV, indicate whether the curve comes from final Development-set tuning rather than outer held-out evaluation.

For LASSO, clearly indicate coefficient scaling and the alpha configuration.

Figures must not imply statistical significance from visual differences alone.

Do not create artificial values or illustrative charts and place them in the final experimental results directory.

---

# 10. Required CSV and Machine-Readable Outputs

Create the following output directory structure, or a closely equivalent clearly documented structure:

```text
project/
├── 71150506123_hw2.py
├── requirements.txt
├── README.md
├── data/
│   └── kc_house_data.csv
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   ├── research_configuration.json
│   ├── experiment_summary.json
│   └── handoff.md
└── tests/
```

The input CSV may be kept in the existing project location if already present. Do not duplicate a large dataset unnecessarily.

In `outputs/tables/`, generate at minimum:

| Filename | Required information |
|---|---|
| `data_quality_summary.csv` | Dataset integrity and descriptive statistics |
| `data_split_summary.csv` | Partition sizes, groups, overlap checks |
| `cv_fold_metrics.csv` | Metrics and settings for each outer fold and method |
| `cv_summary.csv` | Five-fold means and standard deviations |
| `selected_features.csv` | Feature names and selection status |
| `model_hyperparameters.csv` | Selected parameters for each method |
| `test_model_metrics.csv` | Test performance of all six predefined methods |
| `model_coefficients.csv` | Final model coefficients and intercept metadata |
| `statistical_diagnostics.csv` | Residual and multicollinearity diagnostics |
| `prediction_intervals_test.csv` | Actual, prediction, lower, upper, coverage flag |
| `interval_metrics.csv` | Overall coverage, average width, calibration quantile |
| `interval_coverage_by_price.csv` | Coverage by predicted-price group |

Use meaningful column names.

For example, `cv_fold_metrics.csv` must identify the method and outer-fold number, not merely contain unlabeled metric values.

For `selected_features.csv`, record encoded feature names and, where possible, their original predictor grouping.

For `prediction_intervals_test.csv`, preserve enough row identifiers to align interval results with the corresponding held-out observations.

Save numerical precision appropriate for subsequent statistical review. Do not prematurely round model metrics to two decimal places.

The `experiment_summary.json` must summarize actual verified results in a structured, machine-readable format.

If a required experiment fails, mark its status as failed and include the reason. Never insert fabricated numbers, misleading zeros, or unexplained placeholder results into final comparison tables.

---

# 11. Reproducibility and Engineering Quality

## 11.1 Main Program

The assignment lists the Python script filename as:

`71150506123_hw2.py`

Use this exact filename for the primary script.

The assignment also lists an archive filename:

`7115056123_hw2.zip`

These prefixes differ. Do not silently pretend they are identical.

Preserve the explicitly specified script and archive filenames, and note the inconsistency in the handoff so the student can verify the submission naming convention.

The main program must provide a straightforward way to run the full experiment.

A minimal command-line interface for supplying the dataset path and output directory is acceptable.

Do not build a web interface, API server, dashboard, authentication system, or interactive service.

## 11.2 Reproducibility

Record:

- Python version.
- NumPy version.
- pandas version.
- scikit-learn version.
- SciPy version.
- statsmodels version, if used.
- Random seed.
- Dataset dimensions.
- Dataset source.
- Selected hyperparameters.
- Cross-validation strategy.
- Final selected model.
- Execution timestamp.
- Execution status.
- Relevant elapsed runtimes.

Make the experiment reproducible without requiring hidden notebook state.

Use a requirements file with suitable version constraints.

If API compatibility problems occur, adapt the implementation without altering the research design.

## 11.3 Numerical Stability

Handle:

- Constant columns.
- Undefined correlations.
- Near-singular matrices.
- Perfect multicollinearity.
- LASSO convergence warnings.
- Infinite or undefined VIF values.
- Missing or unseen ZIP Code categories.
- Insufficient feature counts for a predefined candidate.
- Invalid adjusted R² denominators.
- Failed selector configurations.
- Matplotlib rendering in non-interactive environments.

Do not suppress all warnings globally.

Unexpected numerical failures must be diagnosed, corrected, or explicitly reported.

## 11.4 Computational Efficiency

The dataset is moderately sized, but nested CV with RFE and RFECV can be computationally expensive.

You may perform legitimate engineering optimizations, such as:

- Caching deterministic intermediate calculations where safe.
- Avoiding redundant data copies.
- Reusing predefined CV assignments.
- Selecting efficient array representations.
- Using the prescribed elimination steps.
- Avoiding unnecessary plotting inside repeated folds.
- Using supported parallel execution if the environment allows it.

However, you may not:

- Replace full-data research results with subset results without authorization.
- Reduce outer CV below five folds.
- Remove inner group-aware validation.
- Skip RFECV.
- Replace the six experiments with easier algorithms.
- Select features once globally to avoid refitting within CV.
- Substitute approximate or fabricated final evaluation values.

If runtime or memory restrictions prevent completion, preserve valid partial artifacts and report the exact remaining blocker.

---

# 12. Automated Testing and Validation

You must create and execute meaningful engineering tests.

At minimum, verify the following.

## Dataset Integrity

- Correct required columns.
- Exactly 18 original eligible predictors.
- Numerical target.
- No target included in the predictor matrix.
- No `id` or `date` included in the predictive matrix.
- Consistent row counts across features, target, and groups.

## Data Splitting

- No overlapping property IDs across partitions.
- No group overlaps within outer folds.
- No group overlaps within inner folds.
- All samples accounted for.
- Fixed seed reproducibility.

## Pipeline and Leakage Prevention

- Preprocessing is fitted only on the applicable training partition.
- Feature selection is fitted only on the applicable training partition.
- Calibration observations are excluded from all model fitting and tuning.
- Test observations are excluded from all model fitting, tuning, and calibration.
- All six experiments use the same outer-fold assignments.

## Feature Selection

- Feature counts match selector outputs.
- Selected feature names align with transformed matrix columns.
- M0 retains all encoded features.
- M1 and M2 use training-only target information.
- M3 and M4 correctly refit elimination steps inside CV.
- M5 uses the LASSO-selected features and then refits LinearRegression.
- No empty feature matrix is passed to the final estimator.

## Evaluation

- RMSE equals the square root of MSE within numerical tolerance.
- Evaluation metrics are calculated on the intended partitions.
- CV means and standard deviations match the stored fold values.
- Adjusted R² uses valid effective degrees of freedom.
- Test results do not influence the selected winner.

## Prediction Intervals

- Calibration residuals come from the frozen Development-trained model.
- The conformal quantile follows the finite-sample order-statistic rule.
- Every lower bound is less than or equal to its upper bound.
- Coverage counts match the interval inclusion flags.
- Overall coverage uses every Test observation.
- Plotting subsets do not influence coverage metrics.
- Price-range cutpoints are fixed without using Test targets.

## Output Integrity

- Every required CSV is generated.
- Every required figure is generated.
- Numerical output files contain real calculated values.
- No fabricated model scores appear in the handoff.
- Saved model can reproduce predictions.
- Handoff numbers match saved CSV and JSON results.

You may use small synthetic fixtures exclusively for unit tests and correctness checks.

However, synthetic-fixture scores must never be presented as the actual King County experimental results.

---

# 13. Required Execution Workflow

Do not stop after writing the source code.

Complete the following sequence autonomously.

### Step 1 — Inspect

Inspect the existing project directory, available dataset, installed packages, hardware limitations, and Python environment.

If an existing implementation is present, determine whether it can be safely reused without violating the research requirements.

### Step 2 — Implement

Create the complete experiment pipeline, preprocessing logic, six model configurations, nested group-aware CV, metrics, diagnostics, prediction interval logic, and output generation.

### Step 3 — Run Engineering Tests

Execute the automated tests.

Fix failing tests before launching the full experiment, where practical.

### Step 4 — Execute Full Experiments

Execute all six experiments on the approved Kaggle dataset.

Collect genuine outer-CV metrics and feature-selection results.

Do not substitute preliminary results for final results.

### Step 5 — Select and Fit Final Model

Apply the predefined Development-CV selection policy.

Log the selection decision.

Refit the selected method on the complete Development partition using valid internal tuning.

Do not evaluate Test data before the model-selection decision is frozen.

### Step 6 — Calibrate Prediction Intervals

Use the independent Calibration partition to calculate the 95% conformal threshold.

Do not use Test labels during this step.

### Step 7 — Final Evaluation

Evaluate the predefined six models on the held-out Test partition.

Evaluate the selected model's 95% Prediction Interval.

Calculate every required metric.

### Step 8 — Generate Figures and Results

Generate the genuine figures, CSV outputs, configuration record, saved final model, and machine-readable summary.

### Step 9 — Perform Consistency Validation

Verify agreement between:

- Actual model predictions.
- Stored evaluation metrics.
- Prediction interval coverage.
- Feature-selection outputs.
- Figure source data.
- Handoff summary.
- Saved model.

### Step 10 — Debug and Rerun

Resolve engineering errors, numerical failures, plotting issues, and dependency incompatibilities.

Rerun affected experiments when necessary.

Do not conceal failed attempts or unresolved limitations.

### Step 11 — Deliver

Prepare the completed reproducible project and handoff.

If feasible, create the submission archive:

`7115056123_hw2.zip`

Include the required source, dependencies, README, relevant output tables, figures, and reproducibility artifacts.

Do not generate or fabricate the final academic PDF report, NotebookLM research summary, or ChatGPT conversation transcript.

Those are outside your responsibilities.

---

# 14. Mandatory Handoff Summary

Create:

`outputs/handoff.md`

This is the primary document the student will give back to ChatGPT for Stage 4 statistical review and academic report planning.

The handoff must contain actual results from program execution.

Use the following structure.

## A. Execution Status

State whether the full experiment completed successfully.

Include:

- Date of execution.
- Dataset path and verification status.
- Python environment.
- Runtime.
- Successful and failed steps.
- Known unresolved issues.

## B. Dataset Verification

Report:

- Actual sample count.
- Actual original column count.
- Original predictor count.
- Actual missing values.
- Unique property count.
- Repeated property ID count.
- Actual Development, Calibration, and Test sizes.
- Group overlap verification.

## C. Preprocessing

Explain:

- Treatment of ZIP Code.
- Numerical imputation.
- Scaling.
- Encoded feature count.
- Original-to-encoded feature mapping.
- Handling of unusual or invalid values.

## D. Six-Model Comparison

Provide a genuine comparison table containing:

- Method.
- Selected encoded feature count.
- Outer CV RMSE mean and SD.
- Outer CV MAE mean and SD.
- Outer CV R² mean and SD.
- Test MAE.
- Test MSE.
- Test RMSE.
- Test R².
- Relevant adjusted R².
- Final hyperparameters.

Explain any missing or undefined values.

## E. Feature Selection Findings

Report:

- Selected features for each method.
- Differences among methods.
- Features repeatedly selected.
- Feature-count reduction compared with the Baseline.
- Performance tradeoffs.

Do not invent causal explanations for observed associations.

## F. Selected Final Model

Report:

- The predefined selection policy.
- Lowest-CV-RMSE method.
- One-standard-error eligibility threshold.
- Eligible methods.
- Final selected method.
- Reason for selection.
- Final predictor count.
- Main coefficients.
- Test performance.

Explicitly confirm that the model was selected before viewing Test metrics.

## G. Statistical Diagnostics

Report actual observations from:

- Residual distribution.
- Residual vs. Fitted plot.
- Q-Q Plot.
- Heteroscedasticity diagnostics.
- Multicollinearity diagnostics.
- Matrix rank.
- Extreme errors.

Distinguish numerical results from engineering interpretations.

Do not claim regression assumptions were verified unless the evidence supports that conclusion.

## H. 95% Prediction Interval

Report:

- Calibration sample count.
- Nominal coverage.
- Conformal quantile calculation.
- Actual `q_hat`.
- Test sample count.
- Empirical Coverage.
- Average Interval Width.
- Covered observations.
- Uncovered observations.
- Coverage across predicted-price ranges.
- Number of negative interval lower bounds, if any.

Discuss the limits of marginal coverage and possible within-property dependence.

## I. Artifacts

List all generated CSV, PNG, JSON, model, and source files.

Include exact paths.

Indicate which files are essential for ChatGPT Stage 4 statistical review.

## J. Limitations and Unresolved Issues

Explicitly list:

- Research-method limitations.
- Statistical assumptions that did not appear to hold.
- Computational limitations.
- Failed or incomplete experiments.
- Remaining engineering blockers.
- Filename ambiguity, if relevant.

Do not hide limitations.

## K. Recommended Review Questions

Provide a concise set of evidence-based questions for ChatGPT to evaluate, such as:

- Did any feature-selection method meaningfully outperform the Baseline?
- Was the feature-count reduction worthwhile?
- Are the reported residual assumptions reasonable?
- Does the empirical interval coverage match the nominal level closely enough to be considered plausible?
- Are there suspicious results that require rerunning an experiment?
- Which results should be prioritized in the academic report?

Do not write the final report on ChatGPT's behalf.

---

# 15. Acceptance Criteria

The project is considered complete only when all of the following conditions are satisfied.

### AC-01 — Dataset Compliance

The official King County House Sales dataset is used and its identity is verified.

All 18 original eligible predictors are correctly recognized.

### AC-02 — Correct Model Count

Exactly six primary experimental configurations are implemented and evaluated.

### AC-03 — Correct Final Estimator

Every primary experiment uses `LinearRegression` as its final prediction estimator.

### AC-04 — Leakage Prevention

All data transformations, feature selection, parameter tuning, and prediction-interval calibration respect partition boundaries.

No property ID overlaps across independent partitions or training/validation splits.

### AC-05 — Correct Cross-Validation

All six models use identical outer 5-Fold group-aware assignments.

Inner 3-Fold group-aware validation is used for applicable tuning and selection.

### AC-06 — Real Model Evaluation

MAE, MSE, RMSE, R², and relevant Adjusted R² values are correctly calculated.

Five-fold means and standard deviations are saved.

### AC-07 — Feature Selection Results

Selected features, hyperparameters, and selected-column counts are saved for all five selection methods.

### AC-08 — Independent Model Selection

The selected final method is chosen using Development CV only, before Test evaluation.

### AC-09 — Correct Prediction Interval

The 95% Prediction Interval uses a mathematically correct Split Conformal procedure with an independent Calibration partition.

### AC-10 — Prediction Interval Verification

The full Test set is used to calculate empirical coverage and average interval width.

### AC-11 — Visualization Quality

All required figures are generated from real experimental data and are readable, labeled, and suitable for an academic report.

### AC-12 — Machine-Readable Results

All required CSV files and the experiment JSON summary are generated with consistent data.

### AC-13 — Execution Verified

The actual code has been executed on the approved dataset.

Automated tests have passed.

Any warnings, failures, or remaining issues are documented.

### AC-14 — Reproducibility

The final model, environment requirements, experimental configuration, and instructions are sufficient to reproduce the results.

### AC-15 — Complete Handoff

`outputs/handoff.md` contains verified findings, exact artifact paths, limitations, and sufficient detail for ChatGPT to perform Stage 4 review.

**Do not mark the project as complete merely because the source code was successfully written.**

If any acceptance criterion cannot be satisfied, identify the precise blocker and honestly label the work as partially complete.

---

# 16. Autonomous Work Authorization

You are explicitly authorized to perform the following without asking for repeated permission:

- Inspect project files.
- Create and modify Python source files.
- Create supporting test files.
- Install required Python dependencies in the available development environment.
- Fix compatibility problems.
- Refactor code for correctness, readability, or legitimate performance improvements.
- Execute tests.
- Execute the full training and evaluation pipeline.
- Inspect warnings and errors.
- Correct numerical stability problems.
- Regenerate figures.
- Recalculate incorrect statistics.
- Regenerate inconsistent CSV files.
- Save trained models.
- Write reproducibility documentation.
- Produce the handoff summary.
- Package project artifacts.

You are not authorized to:

- Change the Kaggle dataset.
- Remove required experiments.
- Modify the specified outer or inner CV fold counts.
- Use the Test set to tune models.
- Use Test targets to calibrate prediction intervals.
- Replace LinearRegression with another primary prediction algorithm.
- Add unrelated modeling experiments.
- Add a web dashboard, API, GUI, or deployment service.
- Generate synthetic research results.
- Claim NotebookLM research was completed.
- Produce the student's formal academic report.
- Fabricate references, benchmark scores, or statistical conclusions.
- Alter the approved research objectives merely to achieve better-looking results.

If you encounter an issue requiring a research-method change, preserve the approved implementation and clearly report the issue for the student's decision.

Do not make an unapproved methodological change silently.

---

# 17. Final Response Requirements

After implementation and actual execution, your final response must contain:

1. **Execution status:** Completed, Partially Completed, or Blocked.
2. **Dataset verification:** Actual dimensions and feature counts.
3. **Implementation summary:** What was built and executed.
4. **Testing summary:** Which tests passed and which failed.
5. **Model comparison:** Actual six-model metrics, if available.
6. **Selected model:** The Development-CV-selected winner and selection reason.
7. **Prediction interval:** Actual 95% Test coverage and average width, if available.
8. **Generated artifacts:** Exact locations of source files, CSVs, figures, saved model, and handoff.
9. **Limitations:** Remaining statistical or engineering concerns.
10. **Stage 4 handoff:** Explicitly identify `outputs/handoff.md` and `outputs/experiment_summary.json` as the primary files for ChatGPT review.

Do not report predicted, estimated, assumed, or invented performance values as measured results.

Do not claim full completion if the actual Kaggle dataset experiments were not executed.

Do not produce a generic project explanation instead of completing the implementation.

**Your mission is to autonomously deliver a working, statistically defensible, reproducible implementation of the approved experiment, together with enough genuine evidence for an independent ChatGPT statistical review.**

Begin by inspecting the project environment and locating or obtaining the approved Kaggle CSV. Then implement, test, execute, validate, and deliver the project within the scope specified above.