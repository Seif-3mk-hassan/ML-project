# Milestone 1 Report: E-Learning Student Performance Prediction

## Project Overview

This project aims to predict student assessment scores using data from the **Open University Learning Analytics Dataset (OULAD)**. The dataset contains demographic information, Virtual Learning Environment (VLE) interaction data, assessment metadata, and student registration details for students enrolled in online courses. The target variable is the **assessment score** (continuous, 0–100), making this a **regression** problem.

---

## 1. Preprocessing Techniques

### 1.1 Data Loading and Merging

The raw data is distributed across seven CSV files: `assessments.csv`, `courses.csv`, `StudentAssesments.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, and `vle.csv`. These tables were merged step-by-step using common keys (`code_module`, `code_presentation`, `id_student`, `id_assessment`) to construct a single modeling dataframe.

- **StudentAssessments** was merged with **Assessments** metadata (module, presentation, deadline date, weight, assessment type).
- **StudentInfo** (demographics) was merged using module, presentation, and student ID.
- **StudentVle** (VLE interactions) was aggregated and merged in two passes — total VLE features and per-assessment VLE features.

### 1.2 Handling Missing and Invalid Values

- The `score` column contained `'?'` as a placeholder for missing values. These were converted to NaN and the corresponding rows were dropped, as this is our target variable.
- Ordinal features (`imd_band_ord`, `highest_education_ord`, `age_band_ord`) had missing values filled with the **median** of their respective columns.
- `days_before_deadline` and `weight` were filled with 0 for missing entries.
- All remaining NaN and infinity values in feature columns were replaced with 0.

### 1.3 Encoding Categorical Variables

- **Ordinal encoding** was applied to naturally ordered variables:
  - `highest_education`: No Formal quals → Lower Than A Level → A Level → HE Qualification → Post Graduate (coded 0–4)
  - `imd_band`: Mapped from '0-10%' to '90-100%' (coded 0–10)
  - `age_band`: 0-35 → 35-55 → 55<= (coded 0–2)
- **Binary encoding** was used for:
  - `disability`: Y=1, N=0
  - `gender`: M=1, F=0
- **One-hot encoding** was applied to:
  - `assessment_type` (TMA, CMA, Exam)
  - `code_module` (7 modules)
  - `code_presentation` (4 presentations)

### 1.4 Feature Engineering

Several new features were engineered to capture richer student behavior patterns:

| Feature Category | Features Created | Description |
|---|---|---|
| **Timing** | `days_before_deadline`, `is_late`, `abs_days_from_deadline` | How early/late a submission was relative to the deadline |
| **VLE Totals** | `total_clicks`, `total_vle_interactions`, `unique_active_days`, `activity_span_days`, `avg_clicks_per_day` | Overall VLE engagement metrics aggregated per student-module-presentation |
| **Per-Assessment VLE** | `clicks_before_deadline`, `active_days_before_deadline` | VLE clicks specifically before each assessment's deadline |
| **Prior Scores** | `prior_mean_score`, `prior_min_score`, `prior_max_score`, `prior_last_score`, `prior_std_score`, `assessment_number` | Historical assessment performance within the same module (expanding window, shifted to avoid leakage) |
| **Interaction Features** | `clicks_x_days_before`, `active_ratio`, `click_intensity`, `late_x_clicks`, `edu_x_clicks`, `prior_score_x_clicks` | Cross-feature interactions capturing combined effects |
| **Log Transforms** | `total_clicks_log`, `total_vle_interactions_log`, `clicks_before_deadline_log`, `click_intensity_log`, `clicks_x_days_before_log` | Log(1+x) transforms to reduce skewness in heavily right-skewed click features |

### 1.5 Normalization

**MinMaxScaler** was applied to all features, fitted only on the training set and then applied to the test set, to prevent data leakage.

---

## 2. Dataset Analysis: How Features Relate to Each Other

The initial feature set contained **49 features**. Analysis of feature relationships revealed several important patterns:

### Key Feature Relationships

1. **VLE Click Features are Highly Correlated**: `total_clicks`, `total_vle_interactions`, and their log-transformed versions showed correlations above 0.90. This motivated our correlation pruning step — 6 features were removed due to |r| > 0.90.

2. **Prior Score Features are Strong Predictors**: `prior_mean_score` emerged as the single most important feature (importance = 56.35), confirming that a student's historical performance is the strongest predictor of future scores. Related features (`prior_max_score`, `prior_std_score`, `prior_last_score`) also ranked highly.

3. **VLE Engagement Predicts Performance**: Features like `total_clicks`, `avg_clicks_per_day`, and `click_intensity` were among the top predictors, indicating that students who interact more with the VLE tend to score higher.

4. **Timing Matters**: `days_before_deadline` and related features showed meaningful predictive power. Students who submit earlier tend to score better.

5. **Demographic Features Have Moderate Impact**: `highest_education_ord`, `imd_band_ord`, and `studied_credits` contributed to predictions but were less important than behavioral features.

### Correlation Analysis of Top Features

The correlation heatmap of the top 10 features (see `visuales/top_features_corr_heatmap_v5.png`) shows:
- `prior_mean_score` and `prior_max_score` have moderate positive correlation (~0.5-0.7)
- Click-related features cluster together but log-transforms reduce multicollinearity
- `prior_std_score` (score consistency) has a weak negative correlation with mean prior score, providing complementary information

---

## 3. Regression Techniques Used

We employed **seven** regression approaches (exceeding the minimum of two):

### 3.1 Linear Regression (Baseline)

A standard Ordinary Least Squares (OLS) regression serving as the simplest baseline. It fits a linear relationship between features and the target, assuming no interaction effects.

### 3.2 Random Forest Regressor

An ensemble of 500 decision trees (max_depth=25, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'). Each tree is trained on a bootstrap sample of the data, and the final prediction is the average of all trees. This captures non-linear relationships and feature interactions.

### 3.3 XGBoost (Extreme Gradient Boosting)

A gradient-boosted tree ensemble with 700 estimators (max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8). XGBoost builds trees sequentially, each correcting the errors of the previous ones, with L1 and L2 regularization.

### 3.4 LightGBM (Light Gradient Boosting Machine)

Another gradient boosting framework with 700 estimators (max_depth=12, learning_rate=0.05, num_leaves=127). LightGBM uses a leaf-wise growth strategy (vs XGBoost's level-wise), which can be more efficient and sometimes more accurate.

### 3.5 Stacking Ensemble

A meta-learning approach combining Random Forest, XGBoost, and LightGBM as base estimators with a **Ridge Regression** meta-learner. The base models' out-of-fold predictions (3-fold CV) are used as features for the Ridge model, allowing it to learn optimal combinations.

### 3.6 ACC@10-Optimized Blend

A weighted average of RF, XGBoost, LightGBM, and Stacking predictions, with weights optimized via grid search to maximize ACC@10 (accuracy within ±10 points). A threshold snapping technique was also applied near common grade boundaries.

### 3.7 Tuned XGBoost (Hyperparameter Tuning)

XGBoost with hyperparameters optimized via **RandomizedSearchCV** (20 iterations, 3-fold CV), searching over `n_estimators`, `max_depth`, and `learning_rate`. Best parameters found: n_estimators=700, max_depth=6, learning_rate=0.05.

---

## 4. Model Comparison and Results

### Full Results Table

| Model | R² | MAE | RMSE | ACC@10 |
|---|---|---|---|---|
| **Stacking Ensemble** | **0.4493** | 10.005 | **13.777** | 0.625 |
| ACC-Optimized Blend | 0.4484 | **9.978** | 13.788 | 0.628 |
| ACC Blend + Threshold Snap | 0.4479 | 9.981 | 13.794 | **0.634** |
| Tuned XGBoost | 0.4478 | 10.026 | 13.795 | 0.624 |
| LightGBM | 0.4426 | 10.033 | 13.860 | 0.626 |
| Random Forest | 0.4389 | 10.139 | 13.906 | 0.618 |
| XGBoost | 0.4325 | 10.077 | 13.985 | 0.624 |
| Linear Regression | 0.3011 | 11.483 | 15.520 | 0.546 |

### Key Differences Between Models

1. **Linear Regression vs Tree-Based Models**: Linear Regression achieved R²=0.301, while all tree-based models exceeded R²=0.43. This ~43% improvement demonstrates that the relationship between features and scores is substantially non-linear.

2. **Individual Trees vs Ensembles**: Random Forest (R²=0.439) uses bagging (parallel trees), while XGBoost (R²=0.433) and LightGBM (R²=0.443) use boosting (sequential correction). LightGBM slightly outperformed XGBoost, likely due to its leaf-wise growth strategy being better suited to this data distribution.

3. **Stacking vs Individual Models**: The Stacking Ensemble (R²=0.449) outperformed all individual models by combining their diverse predictions through a Ridge meta-learner. This ~1-2% improvement over the best individual model shows the value of model diversity.

4. **Hyperparameter Tuning Impact**: Tuned XGBoost (R²=0.448) significantly improved over the default XGBoost (R²=0.433), a +3.5% relative improvement, by reducing max_depth from 10 to 6 (less overfitting).

5. **Blending and Post-Processing**: The ACC@10-Optimized Blend achieved the best ACC@10 (0.634 with threshold snapping), meaning 63.4% of predictions were within ±10 points of the actual score.

6. **Overall Improvement**: The best model (Stacking Ensemble, R²=0.449) represents a **+20.1% improvement** over the previous v3 baseline (R²=0.374).

---

## 5. Feature Selection: Features Used and Discarded

### Feature Selection Pipeline (applied on training data only)

| Step | Action | Features Removed | Remaining |
|---|---|---|---|
| Start | All engineered features | — | 49 |
| Step 1 | Remove module one-hot dummies (high cardinality) | 7 | 42 |
| Step 2 | Remove features with >40% missing in training set | 0 | 42 |
| Step 3 | VarianceThreshold (remove near-constant, threshold=1e-4) | 0 | 42 |
| Step 4 | Correlation pruning (drop one from pairs with |r| > 0.90) | 6 | 36 |
| Step 5 | 5-fold CV RandomForest importance + stability (keep features useful in ≥60% of folds) | 6 | 30 |

### Final 30 Selected Features

1. `prior_mean_score` — Average of previous assessment scores
2. `weight` — Assessment weight in the module
3. `prior_last_score` — Most recent previous score
4. `activity_span_days` — Days between first and last VLE interaction
5. `assessment_number` — Sequential number of this assessment for the student
6. `edu_x_clicks` — Education level × clicks interaction
7. `active_ratio` — Active days before deadline / deadline date
8. `avg_clicks_per_day` — Average daily VLE clicks
9. `prior_std_score` — Standard deviation of prior scores (consistency)
10. `prior_min_score` — Minimum prior score
11. `days_before_deadline` — Days submitted before deadline
12. `active_days_before_deadline` — VLE active days before the assessment deadline
13. `prior_max_score` — Maximum prior score
14. `imd_band_ord` — Index of Multiple Deprivation (socioeconomic indicator)
15. `total_clicks` — Total VLE clicks
16. `total_clicks_log` — Log-transformed total clicks
17. `click_intensity_log` — Log-transformed click intensity
18. `click_intensity` — Clicks per active day before deadline
19. `clicks_before_deadline_log` — Log-transformed pre-deadline clicks
20. `clicks_before_deadline` — Total clicks before deadline
21. `late_x_clicks` — Late submission flag × total clicks
22. `studied_credits` — Number of credits studied
23. `clicks_x_days_before` — Clicks × days before deadline
24. `clicks_x_days_before_log` — Log-transformed clicks × days
25. `atype_CMA` — Computer-marked assessment flag
26. `highest_education_ord` — Highest education level (ordinal)
27. `age_band_ord` — Age band (ordinal)
28. `pres_2014J` — 2014J presentation flag
29. `gender_bin` — Gender (binary)
30. `num_of_prev_attempts` — Number of previous module attempts

### Discarded Features

- **Module dummies** (`mod_AAA`, `mod_BBB`, etc.): Removed due to high cardinality and risk of overfitting to specific modules.
- **6 highly correlated features**: Removed to reduce multicollinearity (e.g., `total_vle_interactions` was highly correlated with `total_clicks`).
- **6 low-stability features**: Features that were only important in fewer than 60% of cross-validation folds were deemed unstable and removed.

---

## 6. Dataset Split Sizes

| Set | Samples | Percentage |
|---|---|---|
| **Training** | 131,001 | 80% |
| **Testing** | 32,751 | 20% |
| **Total** | 163,752 | 100% |

The split was performed using `train_test_split` with `random_state=42` for reproducibility. A validation set was not used as a separate holdout; instead, **cross-validation** (3-fold and 5-fold) was used during model training and feature selection to estimate generalization performance while maximizing training data.

---

## 7. Further Techniques Used to Improve Results

### 7.1 Interaction Feature Engineering
Multiplicative interactions between features (e.g., `clicks × days_before_deadline`, `education_level × clicks`) captured synergistic effects that individual features alone could not represent.

### 7.2 Log Transformations
Right-skewed VLE features (click counts) were log-transformed using `log(1+x)` to reduce the influence of extreme values and improve model training stability.

### 7.3 Ensemble Blending with ACC@10 Optimization
Rather than just maximizing R², we optimized blend weights for **ACC@10** (predictions within ±10 points), which is more practically relevant for grade prediction. A grid search over weight combinations (step=0.05) found optimal weights: RF=5%, XGB=40%, LGB=20%, Stacking=35%.

### 7.4 Threshold Snapping
Predictions near common grade boundaries (40, 50, 60, 70) were snapped to the boundary if within a delta of 2.0 points, improving ACC@10 from 0.628 to 0.634.

### 7.5 Hyperparameter Tuning (RandomizedSearchCV)
XGBoost hyperparameters were tuned via RandomizedSearchCV (20 iterations × 3-fold CV = 60 fits), optimizing R². The tuned model used max_depth=6 (vs 10 default), reducing overfitting and improving generalization.

### 7.6 Feature Stability Selection
Rather than using a single train/test split for feature importance, we used 5-fold cross-validation and only retained features that were important in at least 60% of folds. This produces a more robust feature set.

---

## 8. Regression Line Plots and Visualizations

All plots are saved in the `src/visuales/` directory:

| Plot | File | Description |
|---|---|---|
| Model Comparison (Bar) | `model_comparison_v5.png` | Side-by-side bar charts of R², MAE, RMSE, and ACC@10 for all models |
| Metrics Heatmap | `metrics_heatmap_v5.png` | Color-coded heatmap of all model metrics |
| R² vs RMSE Trade-off | `r2_vs_rmse_v5.png` | Scatter plot showing each model's R²-RMSE trade-off |
| Feature Importance | `feature_importance_v5.png` | Top 20 feature importances from the best model |
| Feature Correlation | `top_features_corr_heatmap_v5.png` | Correlation heatmap of top 10 features |
| Actual vs Predicted (Best) | `actual_vs_predicted_v5.png` | Scatter plot with perfect prediction line |
| Actual vs Predicted (All) | `all_models_actual_vs_predicted_v5.png` | Multi-panel scatter plots for all models |
| Residual Plot | `residual_plot_v5.png` | Residuals vs predicted values |
| Error Distribution | `error_dist_v5.png` | Histogram of prediction errors with KDE |
| Regression Line | `regression_line_v5.png` | Regression trend line (rolling average) overlaid on actual vs predicted |

### Key Visual Observations

- **Actual vs Predicted**: The scatter plot shows predictions clustering around the diagonal (perfect prediction line) but with notable spread, especially for extreme scores (0 and 100). This suggests the model handles mid-range scores better.
- **Residual Plot**: Residuals are roughly centered around zero but show heteroscedasticity — variance increases for higher predicted scores.
- **Error Distribution**: The error distribution is approximately normal (centered at 0), confirming the model is unbiased overall.
- **Regression Line**: The rolling-average trend line closely follows the y=x diagonal for scores between 20–90, with divergence at extremes.

---

## 9. Conclusion

### Initial Intuitions

Before beginning this project, we had the following intuitions:

1. **"Students who engage more with the VLE will score higher"** — **Confirmed**. VLE click features (total_clicks, clicks_before_deadline, click_intensity) were among the top predictors, ranking in the top 10 by importance.

2. **"Prior performance predicts future performance"** — **Strongly confirmed**. `prior_mean_score` was by far the most important feature (importance = 56.35), followed by `prior_max_score` and `prior_std_score`. This aligns with educational research on the persistence of academic performance.

3. **"Demographic factors matter"** — **Partially confirmed**. While `highest_education_ord`, `imd_band_ord`, and `studied_credits` contributed to predictions, they were significantly less important than behavioral features (VLE engagement and prior scores). This suggests that what students *do* matters more than who they *are*.

4. **"Submission timing indicates engagement"** — **Confirmed**. `days_before_deadline` was a meaningful predictor. Late submissions correlated with lower scores, and the `late_x_clicks` interaction feature captured this relationship.

5. **"Non-linear models will greatly outperform linear regression"** — **Confirmed**. Tree-based models achieved ~44% R² vs 30% for linear regression, a substantial gap showing that the score prediction problem involves complex non-linear relationships.

### Key Takeaways

- The best model (**Stacking Ensemble**, R²=0.449) explains ~45% of the variance in assessment scores, representing a **+20.1% improvement** over the previous baseline.
- **63.4%** of predictions (ACC Blend + Threshold Snap) fall within ±10 points of the actual score.
- The most predictive features are **behavioral** (prior scores, VLE engagement) rather than demographic, suggesting that targeted interventions focusing on student behavior could be most effective.
- Ensemble methods (Stacking, Blending) consistently outperform individual models, and the improvement is stable across metrics.

### Limitations and Next Steps

- R²=0.449 means 55% of variance is unexplained. This is partly inherent to the problem — assessment scores depend on study effort, understanding, and test-day factors not captured in VLE logs.
- Temporal features (time-series modeling of daily VLE activity) could improve predictions.
- Student-level embeddings or graph neural networks capturing student-module interaction patterns could be explored.
- A dedicated validation set or nested cross-validation could provide more robust performance estimates.
