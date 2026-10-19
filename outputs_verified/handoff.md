# HW2 Stage 4 handoff

## A. Execution Status
Completed all six actual King County experiments at 2026-10-19T07:55:52.470696+00:00.
Python 3.9.6; runtime 240.79 seconds.
Dataset: /Users/wujinsong/Desktop/AiOt/hw2/data/kc_house_data.csv; dimensions and schema verified, official Kaggle download; SHA256 recorded.
Automated engineering tests: **13 passed, 0 failed**. Independent saved-output validation also passed. See `engineering_test_results.txt` and `validation_results.txt`.
Numerical solver adjustments, preliminary-run exclusion and retained third-party warnings: see `engineering_notes.md`.
Consistency checks: passed: group isolation, all rows, 30 outer scores, CV aggregates, interval arithmetic, saved-model prediction equality, required outputs.

## B. Dataset Verification
{
  "rows": 21613,
  "columns": 21,
  "original_predictors": 18,
  "unique_properties": 21436,
  "repeated_id_rows": 177,
  "properties_with_repeated_ids": 176,
  "duplicate_rows": 0,
  "missing_values": 0,
  "area_identity_rows": 21613,
  "area_identity_discrepancies": 0,
  "date_min": "2014-05-02 00:00:00",
  "date_max": "2015-05-27 00:00:00",
  "development_target_skew": 4.09485493856273
}

| partition   |   rows |   groups |   proportion |   overlap_with_other_partitions |
|:------------|-------:|---------:|-------------:|--------------------------------:|
| Development |  14053 |    13932 |     0.650211 |                               0 |
| Calibration |   3236 |     3216 |     0.149725 |                               0 |
| Test        |   4324 |     4288 |     0.200065 |                               0 |

All partition group intersections and all CV train/validation group intersections are empty.
No observations removed. Integrity checks cover all rows; target-dependent EDA uses Development only.

## C. Preprocessing
18 original predictors. ZIP Code is categorical, reference-category one-hot with unknown categories ignored.
Median numeric imputation and most-frequent ZIP imputation are learned per training fold.
Every encoded column, including ZIP dummies, is standardized to mean zero and unit SD (constants remain zero).
Other numeric and ordinal predictors retain their numeric representation; meaningful zero values retained.
Encoded widths and final selected counts: `model_hyperparameters.csv`.
Complete encoded-to-original mapping and selections: `selected_features.csv`.
Feature selection operates on individual encoded columns, including individual ZIP dummies.
MI treats bedrooms, floors, waterfront, view, condition, grade and year fields as discrete; bathrooms and area/location measures as continuous.

## D. Six-Model Comparison
Sample SD uses ddof=1. CV RMSE is mean of five fold RMSEs, distinct from `pooled_oof_metrics.csv`.
Adjusted R² uses the training design's effective predictor rank; validation/Test adjusted R² is descriptive.

Outer CV comparison:

| method   |   selected_count_mean |   rmse_mean |   rmse_std |   mae_mean |   mae_std |   r2_mean |    r2_std |
|:---------|----------------------:|------------:|-----------:|-----------:|----------:|----------:|----------:|
| M0       |                  86   |      165873 |   10015.4  |    97645.9 |   2898.5  |  0.794925 | 0.0140424 |
| M1       |                  60   |      172632 |    9676.45 |   103903   |   3132.27 |  0.777826 | 0.0143928 |
| M2       |                  60   |      176418 |   11944.4  |   103610   |   4792.38 |  0.768077 | 0.0176145 |
| M3       |                  60   |      166320 |    9940.48 |    97986.9 |   2859.6  |  0.793822 | 0.0138495 |
| M4       |                  78   |      165935 |    9988.88 |    97694.8 |   2897.89 |  0.794776 | 0.0139198 |
| M5       |                  84.8 |      165883 |   10008.4  |    97664.9 |   2870.97 |  0.794901 | 0.0140235 |

Final Development fits and descriptive Test comparison (seconds are Development refit/tuning times):

| method   |   selected_count |      mae |         mse |   rmse |       r2 |   adjusted_r2 |   effective_rank |   value |   seconds |
|:---------|-----------------:|---------:|------------:|-------:|---------:|--------------:|-----------------:|--------:|----------:|
| M0       |               86 |  96943.2 | 2.64296e+10 | 162572 | 0.810772 |      0.806977 |               85 |     nan |  0.107657 |
| M1       |               60 | 103581   | 2.86473e+10 | 169255 | 0.794893 |      0.792055 |               59 |      60 |  0.337035 |
| M2       |               60 | 103976   | 3.19655e+10 | 178789 | 0.771136 |      0.767969 |               59 |      60 | 27.6448   |
| M3       |               60 |  97109.6 | 2.65435e+10 | 162922 | 0.809956 |      0.807327 |               59 |      60 |  3.48401  |
| M4       |               86 |  96943.2 | 2.64296e+10 | 162572 | 0.810772 |      0.806977 |               85 |      86 |  9.40676  |
| M5       |               85 |  96943.2 | 2.64296e+10 | 162572 | 0.810772 |      0.806977 |               85 |       1 |  4.38523  |

Full precision, MSE CV aggregates and per-fold timings remain in the CSV tables.

## E. Feature Selection Findings
Development-CV accuracy/complexity differences relative to M0 (negative RMSE difference is better):

| method   |   rmse_mean |   selected_count_mean |   cv_rmse_difference_vs_baseline |   mean_column_reduction_percent |
|:---------|------------:|----------------------:|---------------------------------:|--------------------------------:|
| M0       |      165873 |                  86   |                          0       |                         0       |
| M1       |      172632 |                  60   |                       6758.85    |                        30.2326  |
| M2       |      176418 |                  60   |                      10544.6     |                        30.2326  |
| M3       |      166320 |                  60   |                        446.35    |                        30.2326  |
| M4       |      165935 |                  78   |                         61.6315  |                         9.30233 |
| M5       |      165883 |                  84.8 |                          9.52302 |                         1.39535 |


- M0: 86 encoded columns; 18 original groups: num__bedrooms, num__bathrooms, num__sqft_living, num__sqft_lot, num__floors, num__waterfront, num__view, num__condition, num__grade, num__sqft_above, num__sqft_basement, num__yr_built, num__yr_renovated, num__lat, num__long, num__sqft_living15, num__sqft_lot15, zip__zipcode_98002, zip__zipcode_98003, zip__zipcode_98004, zip__zipcode_98005, zip__zipcode_98006, zip__zipcode_98007, zip__zipcode_98008, zip__zipcode_98010, zip__zipcode_98011, zip__zipcode_98014, zip__zipcode_98019, zip__zipcode_98022, zip__zipcode_98023, zip__zipcode_98024, zip__zipcode_98027, zip__zipcode_98028, zip__zipcode_98029, zip__zipcode_98030, zip__zipcode_98031, zip__zipcode_98032, zip__zipcode_98033, zip__zipcode_98034, zip__zipcode_98038, zip__zipcode_98039, zip__zipcode_98040, zip__zipcode_98042, zip__zipcode_98045, zip__zipcode_98052, zip__zipcode_98053, zip__zipcode_98055, zip__zipcode_98056, zip__zipcode_98058, zip__zipcode_98059, zip__zipcode_98065, zip__zipcode_98070, zip__zipcode_98072, zip__zipcode_98074, zip__zipcode_98075, zip__zipcode_98077, zip__zipcode_98092, zip__zipcode_98102, zip__zipcode_98103, zip__zipcode_98105, zip__zipcode_98106, zip__zipcode_98107, zip__zipcode_98108, zip__zipcode_98109, zip__zipcode_98112, zip__zipcode_98115, zip__zipcode_98116, zip__zipcode_98117, zip__zipcode_98118, zip__zipcode_98119, zip__zipcode_98122, zip__zipcode_98125, zip__zipcode_98126, zip__zipcode_98133, zip__zipcode_98136, zip__zipcode_98144, zip__zipcode_98146, zip__zipcode_98148, zip__zipcode_98155, zip__zipcode_98166, zip__zipcode_98168, zip__zipcode_98177, zip__zipcode_98178, zip__zipcode_98188, zip__zipcode_98198, zip__zipcode_98199.

- M1: 60 encoded columns; 17 original groups: num__bedrooms, num__bathrooms, num__sqft_living, num__sqft_lot, num__floors, num__waterfront, num__view, num__condition, num__grade, num__sqft_above, num__sqft_basement, num__yr_built, num__yr_renovated, num__lat, num__sqft_living15, num__sqft_lot15, zip__zipcode_98002, zip__zipcode_98003, zip__zipcode_98004, zip__zipcode_98005, zip__zipcode_98006, zip__zipcode_98008, zip__zipcode_98022, zip__zipcode_98023, zip__zipcode_98030, zip__zipcode_98031, zip__zipcode_98032, zip__zipcode_98033, zip__zipcode_98038, zip__zipcode_98039, zip__zipcode_98040, zip__zipcode_98042, zip__zipcode_98052, zip__zipcode_98053, zip__zipcode_98055, zip__zipcode_98056, zip__zipcode_98058, zip__zipcode_98074, zip__zipcode_98075, zip__zipcode_98092, zip__zipcode_98102, zip__zipcode_98105, zip__zipcode_98106, zip__zipcode_98108, zip__zipcode_98109, zip__zipcode_98112, zip__zipcode_98115, zip__zipcode_98118, zip__zipcode_98119, zip__zipcode_98126, zip__zipcode_98133, zip__zipcode_98146, zip__zipcode_98148, zip__zipcode_98155, zip__zipcode_98168, zip__zipcode_98177, zip__zipcode_98178, zip__zipcode_98188, zip__zipcode_98198, zip__zipcode_98199.

- M2: 60 encoded columns; 18 original groups: num__bedrooms, num__bathrooms, num__sqft_living, num__sqft_lot, num__floors, num__waterfront, num__view, num__condition, num__grade, num__sqft_above, num__sqft_basement, num__yr_built, num__yr_renovated, num__lat, num__long, num__sqft_living15, num__sqft_lot15, zip__zipcode_98002, zip__zipcode_98003, zip__zipcode_98004, zip__zipcode_98005, zip__zipcode_98006, zip__zipcode_98008, zip__zipcode_98011, zip__zipcode_98022, zip__zipcode_98023, zip__zipcode_98029, zip__zipcode_98030, zip__zipcode_98031, zip__zipcode_98032, zip__zipcode_98033, zip__zipcode_98034, zip__zipcode_98038, zip__zipcode_98040, zip__zipcode_98042, zip__zipcode_98052, zip__zipcode_98053, zip__zipcode_98055, zip__zipcode_98058, zip__zipcode_98059, zip__zipcode_98074, zip__zipcode_98075, zip__zipcode_98092, zip__zipcode_98103, zip__zipcode_98105, zip__zipcode_98106, zip__zipcode_98107, zip__zipcode_98112, zip__zipcode_98115, zip__zipcode_98116, zip__zipcode_98117, zip__zipcode_98119, zip__zipcode_98133, zip__zipcode_98146, zip__zipcode_98155, zip__zipcode_98168, zip__zipcode_98178, zip__zipcode_98188, zip__zipcode_98198, zip__zipcode_98199.

- M3: 60 encoded columns; 17 original groups: num__bedrooms, num__bathrooms, num__sqft_living, num__sqft_lot, num__floors, num__waterfront, num__view, num__condition, num__grade, num__sqft_above, num__sqft_basement, num__yr_built, num__yr_renovated, num__lat, num__long, num__sqft_living15, zip__zipcode_98004, zip__zipcode_98005, zip__zipcode_98006, zip__zipcode_98007, zip__zipcode_98008, zip__zipcode_98010, zip__zipcode_98023, zip__zipcode_98024, zip__zipcode_98027, zip__zipcode_98029, zip__zipcode_98033, zip__zipcode_98034, zip__zipcode_98038, zip__zipcode_98039, zip__zipcode_98040, zip__zipcode_98045, zip__zipcode_98052, zip__zipcode_98053, zip__zipcode_98056, zip__zipcode_98059, zip__zipcode_98065, zip__zipcode_98070, zip__zipcode_98074, zip__zipcode_98075, zip__zipcode_98102, zip__zipcode_98103, zip__zipcode_98105, zip__zipcode_98106, zip__zipcode_98107, zip__zipcode_98109, zip__zipcode_98112, zip__zipcode_98115, zip__zipcode_98116, zip__zipcode_98117, zip__zipcode_98118, zip__zipcode_98119, zip__zipcode_98122, zip__zipcode_98125, zip__zipcode_98126, zip__zipcode_98136, zip__zipcode_98144, zip__zipcode_98146, zip__zipcode_98177, zip__zipcode_98199.

- M4: 86 encoded columns; 18 original groups: num__bedrooms, num__bathrooms, num__sqft_living, num__sqft_lot, num__floors, num__waterfront, num__view, num__condition, num__grade, num__sqft_above, num__sqft_basement, num__yr_built, num__yr_renovated, num__lat, num__long, num__sqft_living15, num__sqft_lot15, zip__zipcode_98002, zip__zipcode_98003, zip__zipcode_98004, zip__zipcode_98005, zip__zipcode_98006, zip__zipcode_98007, zip__zipcode_98008, zip__zipcode_98010, zip__zipcode_98011, zip__zipcode_98014, zip__zipcode_98019, zip__zipcode_98022, zip__zipcode_98023, zip__zipcode_98024, zip__zipcode_98027, zip__zipcode_98028, zip__zipcode_98029, zip__zipcode_98030, zip__zipcode_98031, zip__zipcode_98032, zip__zipcode_98033, zip__zipcode_98034, zip__zipcode_98038, zip__zipcode_98039, zip__zipcode_98040, zip__zipcode_98042, zip__zipcode_98045, zip__zipcode_98052, zip__zipcode_98053, zip__zipcode_98055, zip__zipcode_98056, zip__zipcode_98058, zip__zipcode_98059, zip__zipcode_98065, zip__zipcode_98070, zip__zipcode_98072, zip__zipcode_98074, zip__zipcode_98075, zip__zipcode_98077, zip__zipcode_98092, zip__zipcode_98102, zip__zipcode_98103, zip__zipcode_98105, zip__zipcode_98106, zip__zipcode_98107, zip__zipcode_98108, zip__zipcode_98109, zip__zipcode_98112, zip__zipcode_98115, zip__zipcode_98116, zip__zipcode_98117, zip__zipcode_98118, zip__zipcode_98119, zip__zipcode_98122, zip__zipcode_98125, zip__zipcode_98126, zip__zipcode_98133, zip__zipcode_98136, zip__zipcode_98144, zip__zipcode_98146, zip__zipcode_98148, zip__zipcode_98155, zip__zipcode_98166, zip__zipcode_98168, zip__zipcode_98177, zip__zipcode_98178, zip__zipcode_98188, zip__zipcode_98198, zip__zipcode_98199.

- M5: 85 encoded columns; 17 original groups: num__bedrooms, num__bathrooms, num__sqft_living, num__sqft_lot, num__floors, num__waterfront, num__view, num__condition, num__grade, num__sqft_above, num__yr_built, num__yr_renovated, num__lat, num__long, num__sqft_living15, num__sqft_lot15, zip__zipcode_98002, zip__zipcode_98003, zip__zipcode_98004, zip__zipcode_98005, zip__zipcode_98006, zip__zipcode_98007, zip__zipcode_98008, zip__zipcode_98010, zip__zipcode_98011, zip__zipcode_98014, zip__zipcode_98019, zip__zipcode_98022, zip__zipcode_98023, zip__zipcode_98024, zip__zipcode_98027, zip__zipcode_98028, zip__zipcode_98029, zip__zipcode_98030, zip__zipcode_98031, zip__zipcode_98032, zip__zipcode_98033, zip__zipcode_98034, zip__zipcode_98038, zip__zipcode_98039, zip__zipcode_98040, zip__zipcode_98042, zip__zipcode_98045, zip__zipcode_98052, zip__zipcode_98053, zip__zipcode_98055, zip__zipcode_98056, zip__zipcode_98058, zip__zipcode_98059, zip__zipcode_98065, zip__zipcode_98070, zip__zipcode_98072, zip__zipcode_98074, zip__zipcode_98075, zip__zipcode_98077, zip__zipcode_98092, zip__zipcode_98102, zip__zipcode_98103, zip__zipcode_98105, zip__zipcode_98106, zip__zipcode_98107, zip__zipcode_98108, zip__zipcode_98109, zip__zipcode_98112, zip__zipcode_98115, zip__zipcode_98116, zip__zipcode_98117, zip__zipcode_98118, zip__zipcode_98119, zip__zipcode_98122, zip__zipcode_98125, zip__zipcode_98126, zip__zipcode_98133, zip__zipcode_98136, zip__zipcode_98144, zip__zipcode_98146, zip__zipcode_98148, zip__zipcode_98155, zip__zipcode_98166, zip__zipcode_98168, zip__zipcode_98177, zip__zipcode_98178, zip__zipcode_98188, zip__zipcode_98198, zip__zipcode_98199.

Selection frequency across five methods (descriptive association, no causal interpretation):
| encoded_feature    |   method_count |
|:-------------------|---------------:|
| num__bathrooms     |              5 |
| zip__zipcode_98053 |              5 |
| zip__zipcode_98008 |              5 |
| zip__zipcode_98023 |              5 |
| zip__zipcode_98033 |              5 |
| zip__zipcode_98038 |              5 |
| zip__zipcode_98040 |              5 |
| num__bedrooms      |              5 |
| zip__zipcode_98052 |              5 |
| zip__zipcode_98074 |              5 |
| zip__zipcode_98005 |              5 |
| zip__zipcode_98075 |              5 |
| zip__zipcode_98105 |              5 |
| zip__zipcode_98106 |              5 |
| zip__zipcode_98112 |              5 |
| zip__zipcode_98115 |              5 |
| zip__zipcode_98119 |              5 |
| zip__zipcode_98146 |              5 |
| zip__zipcode_98006 |              5 |
| zip__zipcode_98199 |              5 |
| zip__zipcode_98004 |              5 |
| num__yr_built      |              5 |
| num__condition     |              5 |
| num__floors        |              5 |
| num__grade         |              5 |
| num__lat           |              5 |
| num__sqft_above    |              5 |
| num__sqft_living15 |              5 |
| num__sqft_lot      |              5 |
| num__view          |              5 |
| num__waterfront    |              5 |
| num__sqft_living   |              5 |
| num__yr_renovated  |              5 |
| zip__zipcode_98092 |              4 |
| zip__zipcode_98107 |              4 |
| num__sqft_basement |              4 |
| zip__zipcode_98178 |              4 |
| zip__zipcode_98168 |              4 |
| zip__zipcode_98102 |              4 |
| zip__zipcode_98103 |              4 |
| zip__zipcode_98029 |              4 |
| num__long          |              4 |
| zip__zipcode_98109 |              4 |
| zip__zipcode_98188 |              4 |
| zip__zipcode_98116 |              4 |
| zip__zipcode_98117 |              4 |
| zip__zipcode_98118 |              4 |
| zip__zipcode_98002 |              4 |
| zip__zipcode_98126 |              4 |
| zip__zipcode_98133 |              4 |
| zip__zipcode_98155 |              4 |
| zip__zipcode_98003 |              4 |
| zip__zipcode_98177 |              4 |
| zip__zipcode_98039 |              4 |
| zip__zipcode_98022 |              4 |
| zip__zipcode_98031 |              4 |
| zip__zipcode_98034 |              4 |
| zip__zipcode_98042 |              4 |
| num__sqft_lot15    |              4 |
| zip__zipcode_98030 |              4 |
| zip__zipcode_98198 |              4 |
| zip__zipcode_98055 |              4 |
| zip__zipcode_98056 |              4 |
| zip__zipcode_98058 |              4 |
| zip__zipcode_98059 |              4 |
| zip__zipcode_98032 |              4 |
| zip__zipcode_98125 |              3 |
| zip__zipcode_98148 |              3 |
| zip__zipcode_98007 |              3 |
| zip__zipcode_98144 |              3 |
| zip__zipcode_98136 |              3 |
| zip__zipcode_98070 |              3 |
| zip__zipcode_98122 |              3 |
| zip__zipcode_98027 |              3 |
| zip__zipcode_98024 |              3 |
| zip__zipcode_98065 |              3 |
| zip__zipcode_98010 |              3 |
| zip__zipcode_98011 |              3 |
| zip__zipcode_98108 |              3 |
| zip__zipcode_98045 |              3 |
| zip__zipcode_98028 |              2 |
| zip__zipcode_98166 |              2 |
| zip__zipcode_98077 |              2 |
| zip__zipcode_98072 |              2 |
| zip__zipcode_98019 |              2 |
| zip__zipcode_98014 |              2 |

## F. Selected Final Model
{
  "best_rmse_method": "M0",
  "threshold": 170352.3781155815,
  "eligible_methods": [
    "M3",
    "M4",
    "M5",
    "M0"
  ],
  "selected_method": "M3",
  "policy": "one SE; fewest mean selected encoded columns, then RMSE, then M0..M5",
  "selected_mean_count": 60.0,
  "frozen_before_test": true
}

Selection was frozen before Test metrics. The one-standard-error rule is a practical heuristic, not proof of equivalence.
Primary model: M3. Full standardized coefficients and intercept are in `model_coefficients.csv`.
Largest absolute coefficients (including any intercept):
| encoded_feature    |   coefficient_usd_per_scaled_unit |
|:-------------------|----------------------------------:|
| intercept          |                          540030   |
| num__sqft_above    |                           92658   |
| num__sqft_living   |                           90867   |
| zip__zipcode_98004 |                           81829.9 |
| num__grade         |                           65202.8 |
| num__waterfront    |                           60694.6 |
| zip__zipcode_98112 |                           56079.3 |
| zip__zipcode_98039 |                           49153   |
| zip__zipcode_98040 |                           47744.9 |
| num__lat           |                           45055.5 |
| num__view          |                           41545.8 |
| zip__zipcode_98103 |                           34439.4 |

## G. Statistical Diagnostics
Development training residuals are descriptive post-selection diagnostics.
Effective predictor rank: 59; augmented rank 60 of 61.
Residual skew: 4.40264; excess kurtosis: 74.8482.
BP/Koenker LM: 1943.89, df=59, p=0.
P-value representation: Numerical underflow: stored 0.0 means below floating-point resolution, not an exactly zero probability.
Squared residuals are regressed on an orthonormal basis of the complete design column space, retaining intercept span.
Residual normality/homoscedasticity are not assumed verified. Small large-sample p-values are not proof that prediction is useless.
Visual inspection: residual spread widens at larger fitted prices, with substantial positive residuals at the high end. The Q-Q curve has a pronounced upper-tail departure from the normal reference line. Together with skewness, kurtosis and BP results, these observations do not support normal, constant-variance residual assumptions. No Test-driven model changes were made.
VIF includes infinite entries for the retained area variables; the selected augmented design is rank-deficient (60 of 61). The largest absolute Development residual is about USD 4.366 million. Exact rows remain in `extreme_errors_development.csv`.
Baseline retains exact dependencies. Infinite VIFs are explicitly marked; no artificial finite replacements.

## H. 95% Prediction Interval
Calibration n=3236; k=ceil((n+1)*0.95)=3076; q_hat=259210.9949 USD.
Nominal coverage=0.95; full Test n=4324; coverage=0.9435707678; average width=518421.9898 USD.
Covered=4080; uncovered=244; negative lower bounds=727 (retained).
Predicted-price bin boundaries were frozen using Calibration predictions. Duplicate cutpoints are collapsed.

| predicted_price_group    |   observations |   covered |   coverage |
|:-------------------------|---------------:|----------:|-----------:|
| (-inf, 274747.909]       |            812 |       795 |   0.979064 |
| (274747.909, 408591.884] |            868 |       864 |   0.995392 |
| (408591.884, 542585.846] |            900 |       895 |   0.994444 |
| (542585.846, 726978.564] |            795 |       771 |   0.969811 |
| (726978.564, inf]        |            949 |       755 |   0.795574 |

The highest predicted-price bin (predictions above about USD 726,979) contains 949 Test observations, with coverage 755/949 = 79.56%. This is substantially below overall coverage; it is a descriptive limitation, not a reason to retune intervals on Test data.
Intervals are marginal: conditional coverage is not guaranteed. Repeated transactions within a property can violate transaction-level exchangeability even with group-disjoint partitions.

## I. Artifacts
Primary Stage 4 files: `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/handoff.md` and `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/experiment_summary.json`.
CSV tables, figures, source and saved models:

- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/calibration.json`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/engineering_notes.md`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/engineering_test_results.txt`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/environment.json`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/experiment_summary.json`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/01_target_distribution.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/02_price_boxplot.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/03_feature_correlation_heatmap.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/04_living_area_vs_price.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/05_feature_selection_comparison.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/06_cv_rmse_comparison.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/07_cv_r2_comparison.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/08_feature_count_comparison.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/09_rfecv_performance.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/10_lasso_coefficients.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/11_actual_vs_predicted.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/12_residuals_vs_fitted.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/13_residual_qq_plot.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/14_prediction_interval_95.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/15_interval_coverage_by_price.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/16_feature_distributions.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/figures/17_residual_histogram.png`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/fold_audit.json`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/handoff.md`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/model_selection.json`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/M0.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/M1.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/M2.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/M3.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/M4.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/M5.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/models/final_model.joblib`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/research_configuration.json`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/run.log`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/calibration_scores.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/column_quality.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/cv_fold_metrics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/cv_summary.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/data_quality_summary.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/data_split_summary.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/development_descriptive_statistics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/development_feature_correlations.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/development_model_metrics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/development_price_correlations.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/development_residuals.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/extreme_errors_development.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/high_correlation_pairs.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/inner_cv_results.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/interval_coverage_by_price.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/interval_metrics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/model_coefficients.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/model_hyperparameters.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/out_of_fold_predictions.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/outer_fold_assignments.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/partition_assignments.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/pooled_oof_metrics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/prediction_interval_plot_data.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/prediction_intervals_test.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/rfecv_curve.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/selected_features.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/statistical_diagnostics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/test_model_metrics.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/test_predictions.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/unusual_values.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/tables/vif.csv`
- `/Users/wujinsong/Desktop/AiOt/hw2/outputs_verified/validation_results.txt`
- `/Users/wujinsong/Desktop/AiOt/hw2/7115056123_hw2.py`
- `/Users/wujinsong/Desktop/AiOt/hw2/experiment.py`
- `/Users/wujinsong/Desktop/AiOt/hw2/finalize_artifacts.py`
- `/Users/wujinsong/Desktop/AiOt/hw2/package_submission.py`
- `/Users/wujinsong/Desktop/AiOt/hw2/validate_outputs.py`

## J. Limitations and Unresolved Issues
- Historical generalization only
- One-SE rule is a heuristic, not proof of equivalence
- Transaction residual exchangeability is not established by group splitting; no conditional coverage guarantee
- Rank-deficient baseline retained unchanged
- Numeric and ordinal predictors modeled linearly
- RFECV equivalent implementation uses fold-local preprocessing and step-5 RFE candidate paths
- Student confirmed ID 7115056123; script and archive filenames now consistently use this ID, superseding the original prompt typo
- Development residuals show a widening spread with fitted values and a heavy positive tail; Q-Q plot strongly departs from normality.
- Selected M3 augmented design has rank 60 of 61 columns; exact multicollinearity remains.
- Highest predicted-price bin has about 79.56% Test coverage; marginal coverage does not imply conditional coverage.
- Initial unoptimized run in outputs/ is preliminary and excluded from the archive; use outputs_verified/ only.
- Test comparisons are descriptive and did not change the winner. No formal academic PDF, NotebookLM summary or conversation transcript was generated.

## K. Recommended Review Questions
- Are CV gains relative to the baseline meaningful compared with fold variability?
- Is the encoded-column reduction worth the accuracy tradeoff?
- Do residual plots and VIF results suggest limitations worth emphasizing?
- How does coverage change across the fixed predicted-price bins?
- Does within-property dependence limit the conformal interpretation?
- Which results require additional methodological authorization before rerunning?
