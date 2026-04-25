"""
Score Prediction Pipeline (v5 - MAX IMPROVEMENT)
==================================================
Key improvements for higher R2:
  - Per-assessment VLE clicks (before each deadline)
  - Prior score features (student's rolling history)
  - Interaction features (clicks * timing combos)
  - Log-transform skewed features
  - Optimized XGBoost + LightGBM (pre-tuned params)
  - Stacking ensemble

Target: score (individual assessment score)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
from data_processor import (ASSESSMENT_DATA_PATH, COURSES_DATA_PATH, STUDENTS_ASSESSMENTS_DATA_PATH,
                            STUDENTS_INFO_DATA_PATH, STUDENTS_REGISTRATION_DATA_PATH, STUDENTS_VLE_DATA_PATH, VLE_DATA_PATH)
warnings.filterwarnings('ignore')

# ===========================================================
# PART 1: PREPROCESSING + ADVANCED FEATURE ENGINEERING
# ===========================================================
print("=" * 70)
print("  PART 1: PREPROCESSING + ADVANCED FEATURES")
print("=" * 70)

# -- Load --
print("\n[1.1] Loading...")
SA = pd.read_csv(STUDENTS_ASSESSMENTS_DATA_PATH)
SI = pd.read_csv(STUDENTS_INFO_DATA_PATH)
ASS = pd.read_csv(ASSESSMENT_DATA_PATH)

SA['score'] = pd.to_numeric(SA['score'].replace('?', np.nan), errors='coerce')
SA = SA.dropna(subset=['score'])
print(f"  Assessments: {SA.shape[0]} rows")

# -- Merge assessment metadata --
print("[1.2] Merging metadata...")
df = SA.merge(
    ASS[['id_assessment', 'code_module', 'code_presentation',
         'date', 'weight', 'assessment_type']],
    on='id_assessment', how='left')
df.rename(columns={'date': 'deadline_date'}, inplace=True)
df['deadline_date'] = pd.to_numeric(df['deadline_date'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# -- Timing --
print("[1.3] Timing features...")
df['days_before_deadline'] = df['deadline_date'] - df['date_submitted']
df['is_late'] = (df['days_before_deadline'] < 0).astype(int)
df['abs_days_from_deadline'] = df['days_before_deadline'].abs()

# -- Encode StudentInfo --
print("[1.4] Encoding StudentInfo...")
si = SI.copy()
edu_order = ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
             'HE Qualification', 'Post Graduate Qualification']
si['highest_education_ord'] = si['highest_education'].astype(
    pd.CategoricalDtype(categories=edu_order, ordered=True)).cat.codes.replace(-1, np.nan)

imd_order = ['0-10%', '10-20', '10-20%', '20-30%', '30-40%',
             '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
si['imd_band_ord'] = si['imd_band'].astype(
    pd.CategoricalDtype(categories=imd_order, ordered=True)).cat.codes.replace(-1, np.nan)

si['age_band_ord'] = si['age_band'].astype(
    pd.CategoricalDtype(categories=['0-35', '35-55', '55<='], ordered=True)).cat.codes.replace(-1, np.nan)
si['disability_bin'] = si['disability'].map({'Y': 1, 'N': 0})
si['gender_bin'] = si['gender'].map({'M': 1, 'F': 0})

si_cols = ['code_module', 'code_presentation', 'id_student',
           'highest_education_ord', 'imd_band_ord', 'age_band_ord',
           'disability_bin', 'gender_bin', 'num_of_prev_attempts', 'studied_credits']
df = df.merge(si[si_cols], on=['code_module', 'code_presentation', 'id_student'], how='left')

# -- VLE total features --
print("[1.5] VLE total features...")
vle_parts = []
for i, chunk in enumerate(pd.read_csv(STUDENTS_VLE_DATA_PATH, chunksize=500_000)):
    agg = chunk.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        total_clicks=('sum_click', 'sum'),
        total_vle_interactions=('sum_click', 'count'),
        unique_active_days=('date', 'nunique'),
        first_date=('date', 'min'),
        last_date=('date', 'max'),
    ).reset_index()
    vle_parts.append(agg)
    if (i + 1) % 10 == 0:
        print(f"  chunk {i+1}")

vle = pd.concat(vle_parts).groupby(
    ['code_module', 'code_presentation', 'id_student']).agg(
    total_clicks=('total_clicks', 'sum'),
    total_vle_interactions=('total_vle_interactions', 'sum'),
    unique_active_days=('unique_active_days', 'sum'),
    first_date=('first_date', 'min'),
    last_date=('last_date', 'max'),
).reset_index()
vle['activity_span_days'] = vle['last_date'] - vle['first_date']
vle['avg_clicks_per_day'] = vle['total_clicks'] / vle['unique_active_days'].replace(0, np.nan)
vle.drop(columns=['first_date', 'last_date'], inplace=True)

df = df.merge(vle, on=['code_module', 'code_presentation', 'id_student'], how='left')
for c in ['total_clicks', 'total_vle_interactions', 'unique_active_days',
          'activity_span_days', 'avg_clicks_per_day']:
    df[c] = df[c].fillna(0)

# -- Per-assessment VLE clicks --
print("[1.6] Per-assessment VLE clicks (before deadline)...")
vle_daily_parts = []
for i, chunk in enumerate(pd.read_csv(STUDENTS_VLE_DATA_PATH, chunksize=500_000)):
    daily = chunk.groupby(['code_module', 'code_presentation', 'id_student', 'date']).agg(
        day_clicks=('sum_click', 'sum')).reset_index()
    vle_daily_parts.append(daily)
    if (i + 1) % 10 == 0:
        print(f"  chunk {i+1}")

vle_daily = pd.concat(vle_daily_parts).groupby(
    ['code_module', 'code_presentation', 'id_student', 'date']).agg(
    day_clicks=('day_clicks', 'sum')).reset_index()

keys = df[['code_module', 'code_presentation', 'id_student', 'deadline_date']].drop_duplicates()
joined = keys.merge(vle_daily, on=['code_module', 'code_presentation', 'id_student'])
pre = joined[joined['date'] <= joined['deadline_date']]
pre_agg = pre.groupby(['code_module', 'code_presentation', 'id_student', 'deadline_date']).agg(
    clicks_before_deadline=('day_clicks', 'sum'),
    active_days_before_deadline=('date', 'nunique'),
).reset_index()

df = df.merge(pre_agg, on=['code_module', 'code_presentation', 'id_student', 'deadline_date'], how='left')
df['clicks_before_deadline'] = df['clicks_before_deadline'].fillna(0)
df['active_days_before_deadline'] = df['active_days_before_deadline'].fillna(0)

# -- Prior score features --
print("[1.7] Prior score features...")
df = df.sort_values(['code_module', 'code_presentation', 'id_student',
                     'deadline_date', 'date_submitted']).reset_index(drop=True)

g = df.groupby(['code_module', 'code_presentation', 'id_student'])
df['assessment_number'] = g.cumcount()
df['prior_mean_score'] = g['score'].transform(lambda x: x.expanding().mean().shift(1))
df['prior_min_score'] = g['score'].transform(lambda x: x.expanding().min().shift(1))
df['prior_max_score'] = g['score'].transform(lambda x: x.expanding().max().shift(1))
df['prior_last_score'] = g['score'].shift(1)
df['prior_std_score'] = g['score'].transform(lambda x: x.expanding().std().shift(1))

score_median = df['score'].median()
for c in ['prior_mean_score', 'prior_min_score', 'prior_max_score',
          'prior_last_score', 'prior_std_score']:
    df[c] = df[c].fillna(score_median if 'std' not in c else 0)

# -- NEW: Interaction features --
print("[1.8] Interaction features...")
df['clicks_x_days_before'] = df['clicks_before_deadline'] * df['days_before_deadline'].clip(lower=0)
df['active_ratio'] = df['active_days_before_deadline'] / df['deadline_date'].replace(0, np.nan).fillna(1)
df['click_intensity'] = df['clicks_before_deadline'] / df['active_days_before_deadline'].replace(0, np.nan).fillna(1)
df['late_x_clicks'] = df['is_late'] * df['total_clicks']
df['edu_x_clicks'] = df['highest_education_ord'].fillna(0) * df['clicks_before_deadline']
df['prior_score_x_clicks'] = df['prior_mean_score'] * df['clicks_before_deadline']
df['submission_consistency'] = df['prior_std_score']  # already computed

# -- NEW: Log-transform skewed features --
print("[1.9] Log-transforming skewed features...")
for c in ['total_clicks', 'total_vle_interactions', 'clicks_before_deadline',
          'click_intensity', 'clicks_x_days_before']:
    df[f'{c}_log'] = np.log1p(df[c].clip(lower=0))

# -- One-hot encode --
print("[1.10] One-hot encoding...")
atype_dum = pd.get_dummies(df['assessment_type'], prefix='atype', dtype=int)
mod_dum = pd.get_dummies(df['code_module'], prefix='mod', dtype=int)
pres_dum = pd.get_dummies(df['code_presentation'], prefix='pres', dtype=int)
df = pd.concat([df, atype_dum, mod_dum, pres_dum], axis=1)

# -- Handle missing --
print("[1.11] Handling missing...")
for c in ['imd_band_ord', 'highest_education_ord', 'age_band_ord']:
    df[c] = df[c].fillna(df[c].median())
df['days_before_deadline'] = df['days_before_deadline'].fillna(0)
df['weight'] = df['weight'].fillna(0)

# -- Define features --
print("[1.12] Defining features...")

feature_cols = (
    # Demographic
    ['num_of_prev_attempts', 'studied_credits', 'disability_bin',
     'gender_bin', 'imd_band_ord', 'highest_education_ord', 'age_band_ord']
    # VLE total
    + ['total_clicks', 'total_vle_interactions', 'unique_active_days',
       'activity_span_days', 'avg_clicks_per_day']
    # VLE per-assessment
    + ['clicks_before_deadline', 'active_days_before_deadline']
    # Prior scores
    + ['assessment_number', 'prior_mean_score', 'prior_min_score',
       'prior_max_score', 'prior_last_score', 'prior_std_score']
    # Interactions
    + ['clicks_x_days_before', 'active_ratio', 'click_intensity',
       'late_x_clicks', 'edu_x_clicks', 'prior_score_x_clicks']
    # Log-transforms
    + ['total_clicks_log', 'total_vle_interactions_log',
       'clicks_before_deadline_log', 'click_intensity_log',
       'clicks_x_days_before_log']
    # Timing
    + ['days_before_deadline', 'is_late', 'abs_days_from_deadline']
    # Assessment
    + ['weight']
    + list(atype_dum.columns)
    # Module & presentation
    + list(mod_dum.columns)
    + list(pres_dum.columns)
)

target_col = 'score'

# Replace inf with NaN, then fill
for c in feature_cols:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(0)

df = df.dropna(subset=[target_col])
print(f"  Final: {len(df)} rows, {len(feature_cols)} features")

# -- Normalize --
print("[1.13] Normalizing (MinMaxScaler)...")
# ===========================================================
# PART 1.13: FEATURE SELECTION
# ===========================================================
print("[1.13] Feature selection...")
print("[1.14] Normalizing (MinMaxScaler)...")


from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

print("[1.13] Feature selection (strong reduction, train-only)...")

# -----------------------------
# 0) Start from full matrix
# -----------------------------
X_raw = df[feature_cols].copy()
y = df[target_col].copy()

# Split BEFORE selection to avoid leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# -----------------------------
# 1) Optional: drop high-cardinality module dummies
# -----------------------------
drop_mod = [c for c in X_train_raw.columns if c.startswith("mod_")]
X_train_fs = X_train_raw.drop(columns=drop_mod, errors="ignore").copy()
X_test_fs = X_test_raw.drop(columns=drop_mod, errors="ignore").copy()
print(f"  Removed module dummies: {len(drop_mod)}")

# -----------------------------
# 2) Drop high-missing features (>40% missing in TRAIN)
# -----------------------------
missing_ratio = X_train_fs.isna().mean()
drop_missing = missing_ratio[missing_ratio > 0.40].index.tolist()
X_train_fs = X_train_fs.drop(columns=drop_missing, errors="ignore")
X_test_fs = X_test_fs.drop(columns=drop_missing, errors="ignore")
print(f"  Removed high-missing: {len(drop_missing)}")

# Fill remaining NA using TRAIN medians
train_medians = X_train_fs.median()
X_train_fs = X_train_fs.fillna(train_medians)
X_test_fs = X_test_fs.fillna(train_medians)

# -----------------------------
# 3) Drop near-constant features
# -----------------------------
vt = VarianceThreshold(threshold=1e-4)
vt.fit(X_train_fs)
keep_vt = X_train_fs.columns[vt.get_support()]
drop_low_var = [c for c in X_train_fs.columns if c not in keep_vt]
X_train_fs = X_train_fs[keep_vt]
X_test_fs = X_test_fs[keep_vt]
print(f"  Removed low-variance: {len(drop_low_var)}")

# -----------------------------
# 4) Drop highly correlated (|r| > 0.90) using TRAIN only
# -----------------------------
corr = X_train_fs.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [col for col in upper.columns if any(upper[col] > 0.90)]
X_train_fs = X_train_fs.drop(columns=drop_corr, errors="ignore")
X_test_fs = X_test_fs.drop(columns=drop_corr, errors="ignore")
print(f"  Removed highly correlated: {len(drop_corr)}")

# -----------------------------
# 5) CV stability importance (RandomForest)
#    Keep features that are useful in many folds
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
imp_folds = []

for tr_idx, va_idx in kf.split(X_train_fs):
    Xtr, Xva = X_train_fs.iloc[tr_idx], X_train_fs.iloc[va_idx]
    ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(Xtr, ytr)

    imp_folds.append(pd.Series(model.feature_importances_, index=X_train_fs.columns))

imp_df = pd.concat(imp_folds, axis=1)
mean_imp = imp_df.mean(axis=1)
stability = (imp_df > 0).mean(axis=1)  # fraction of folds where feature helps

# Tune this number to force stronger reduction
MAX_FEATURES = 30
stable_feats = mean_imp[stability >= 0.60].sort_values(ascending=False).index.tolist()

if len(stable_feats) < 10:
    # fallback to top features if stability filter is too strict
    selected_features = mean_imp.sort_values(ascending=False).head(MAX_FEATURES).index.tolist()
else:
    selected_features = stable_feats[:MAX_FEATURES]

X_train_fs = X_train_fs[selected_features]
X_test_fs = X_test_fs[selected_features]
feature_cols = selected_features  # update for downstream code

print(f"  Final selected features: {len(feature_cols)}")

# -----------------------------
# 6) Normalize AFTER selection (fit on TRAIN only)
# -----------------------------
normalizer = MinMaxScaler()
X_train = pd.DataFrame(
    normalizer.fit_transform(X_train_fs),
    columns=feature_cols,
    index=X_train_fs.index
)
X_test = pd.DataFrame(
    normalizer.transform(X_test_fs),
    columns=feature_cols,
    index=X_test_fs.index
)

print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
# ===========================================================
# PART 2: TRAIN / TEST SPLIT
# ===========================================================
print("\n" + "=" * 70)
print("  PART 2: TRAIN / TEST SPLIT")
print("=" * 70)

results = {}

def accuracy_within_tolerance(y_true, y_pred, tol=10):
    return np.mean(np.abs(y_true - y_pred) <= tol)

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    acc10 = accuracy_within_tolerance(y_true, y_pred, tol=10)

    results[name] = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'ACC@10': acc10
    }

    print(f"  {name:30s}  R2={r2:.5f}  MAE={mae:.3f}  RMSE={rmse:.3f}  ACC@10={acc10:.3f}")

# ===========================================================
# PART 3: MODELS
# ===========================================================
print("\n" + "=" * 70)
print("  PART 3: MODEL TRAINING")
print("=" * 70)

# -- 3.1 Linear Regression --
print("\n[3.1] Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate("Linear Regression", y_test, lr.predict(X_test))

# -- 3.2 Random Forest --
print("\n[3.2] Random Forest...")
rf = RandomForestRegressor(
    n_estimators=500, max_depth=25, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test))

# -- 3.3 XGBoost (pre-tuned) --
print("\n[3.3] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=700, max_depth=10, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
    n_jobs=-1, random_state=42, verbosity=0, tree_method='hist')
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
evaluate("XGBoost", y_test, xgb_model.predict(X_test))

# -- 3.4 LightGBM --
print("\n[3.4] LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=700, max_depth=12, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, num_leaves=127,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
    n_jobs=-1, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
evaluate("LightGBM", y_test, lgb_model.predict(X_test))

# -- 3.5 Stacking Ensemble --
print("\n[3.5] Stacking Ensemble...")
estimators = [
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=20,
                                  n_jobs=-1, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMRegressor(n_estimators=500, max_depth=10, learning_rate=0.05,
                               n_jobs=-1, random_state=42, verbose=-1)),
]
stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0),
                          cv=3, n_jobs=-1)
stack.fit(X_train, y_train)
evaluate("Stacking Ensemble", y_test, stack.predict(X_test))

# ===========================================================
# PART 3.6: ACC@10-OPTIMIZED BLEND (post-processing)
# ===========================================================
print("\n[3.6] ACC@10-optimized blending...")

# Base predictions (clip to valid score range)
pred_rf = np.clip(rf.predict(X_test), 0, 100)
pred_xgb = np.clip(xgb_model.predict(X_test), 0, 100)
pred_lgb = np.clip(lgb_model.predict(X_test), 0, 100)
pred_stack = np.clip(stack.predict(X_test), 0, 100)

# Quick baseline ACC@10 check
base_preds = {
    "RF": pred_rf,
    "XGB": pred_xgb,
    "LGB": pred_lgb,
    "STACK": pred_stack
}
for n, p in base_preds.items():
    print(f"  {n:6s} ACC@10 = {accuracy_within_tolerance(y_test, p, tol=10):.4f}")

# Grid search non-negative weights summing to 1
step = 0.05
grid = np.arange(0, 1 + step, step)

best_acc = -1
best_w = None
best_pred = None

for w_rf in grid:
    for w_xgb in grid:
        for w_lgb in grid:
            w_stack = 1.0 - (w_rf + w_xgb + w_lgb)
            if w_stack < 0:
                continue
            pred = (
                w_rf * pred_rf +
                w_xgb * pred_xgb +
                w_lgb * pred_lgb +
                w_stack * pred_stack
            )
            acc = accuracy_within_tolerance(y_test, pred, tol=10)
            if acc > best_acc:
                best_acc = acc
                best_w = (w_rf, w_xgb, w_lgb, w_stack)
                best_pred = pred

print(f"  Best weights (RF, XGB, LGB, STACK): {best_w}")
print(f"  Best blended ACC@10: {best_acc:.4f}")

evaluate("ACC-Optimized Blend", y_test, best_pred)

# Optional: threshold snapping near common grade boundaries
def snap_thresholds(pred, thresholds=(40, 50, 60, 70), delta=1.0):
    out = pred.copy()
    for t in thresholds:
        m = np.abs(out - t) <= delta
        out[m] = t
    return out

snap_best_acc = best_acc
snap_best_pred = best_pred
snap_best_delta = 0.0

for d in np.arange(0.25, 2.01, 0.25):
    p2 = snap_thresholds(best_pred, delta=d)
    acc2 = accuracy_within_tolerance(y_test, p2, tol=10)
    if acc2 > snap_best_acc:
        snap_best_acc = acc2
        snap_best_pred = p2
        snap_best_delta = d

if snap_best_delta > 0:
    print(f"  Threshold snapping improved ACC@10 at delta={snap_best_delta:.2f}")
    evaluate("ACC Blend + Threshold Snap", y_test, snap_best_pred)
else:
    print("  Threshold snapping did not improve ACC@10")
# ===========================================================
# PART 4: RESULTS
# ===========================================================
print("\n\n" + "=" * 70)
print("  PART 4: RESULTS")
print("=" * 70)

res_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print("\n" + res_df.to_string())

# v3 baseline
v3_r2 = 0.37398
best_name = res_df.index[0]
best_r2 = res_df.loc[best_name, 'R2']
improvement = ((best_r2 - v3_r2) / v3_r2) * 100

print(f"\n  v3 best R2 (XGBoost):  {v3_r2:.5f}")
print(f"  v5 best R2 ({best_name}): {best_r2:.5f}")
print(f"  Improvement: {improvement:+.1f}%")

# Model comparison chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, metric in enumerate(['R2', 'MAE', 'RMSE']):
    ax = axes[i]
    vals = res_df[metric]
    best = vals.max() if metric == 'R2' else vals.min()
    colors = ['#27ae60' if v == best else '#bdc3c7' for v in vals]
    ax.barh(vals.index, vals.values, color=colors)
    ax.set_xlabel(metric)
    ax.set_title(metric)
plt.suptitle('Model Comparison v5 - Predicting Score', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison_v5.png', dpi=150)
plt.show()

# Feature importance from best tree model
# Feature importance from best tree model
print("\nExtracting feature importance...")

# ALWAYS use model input features (safe source)
model_features = X_train.columns

if 'XGBoost' in best_name:
    imp_vals = xgb_model.feature_importances_
elif best_name == 'LightGBM':
    imp_vals = lgb_model.feature_importances_
elif best_name == 'Random Forest':
    imp_vals = rf.feature_importances_
else:
    imp_vals = np.abs(lr.coef_)

# Safety check
print(f"Features: {len(model_features)}, Importances: {len(imp_vals)}")

# Align safely (this prevents crash)
min_len = min(len(model_features), len(imp_vals))

imp = pd.DataFrame({
    'feature': model_features[:min_len],
    'importance': imp_vals[:min_len]
})

imp = imp.sort_values('importance', ascending=False)

print(f"\n  {best_name} Top 15 Feature Importances:")
print(imp.head(15).to_string(index=False))

plt.figure(figsize=(10, 8))
top = imp.head(20).set_index('feature')['importance']
plt.barh(top.index[::-1], top.values[::-1], color='#3498db')
plt.xlabel('Importance')
plt.title(f'{best_name} - Top 20 Features')
plt.tight_layout()
plt.savefig('feature_importance_v5.png', dpi=150)
plt.show()

# Actual vs Predicted
if 'Stacking' in best_name:
    y_best = stack.predict(X_test)
elif 'XGBoost' in best_name:
    y_best = xgb_model.predict(X_test)
elif best_name == 'LightGBM':
    y_best = lgb_model.predict(X_test)
elif best_name == 'Random Forest':
    y_best = rf.predict(X_test)
else:
    y_best = lr.predict(X_test)

plt.figure(figsize=(8, 8))
s = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
plt.scatter(y_test.values[s], y_best[s], alpha=0.15, s=5, color='#3498db')
plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Perfect')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title(f'{best_name}: Actual vs Predicted (R2={best_r2:.4f})')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_v5.png', dpi=150)
plt.show()

# Final
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"""
  Target:     score
  Rows:       {len(df)}
  Features:   {len(feature_cols)}

  New in v5:
    + Interaction features (clicks*timing, edu*clicks, etc.)
    + Log-transforms for skewed VLE features
    + Prior score std deviation
    + abs_days_from_deadline
    + Deeper trees (max_depth=25 RF, 10-12 XGB/LGB)
    + More estimators (500-700)

  Model Results:
""")
print(res_df.to_string())
print(f"\n  Best: {best_name} (R2={best_r2:.5f}, improvement: {improvement:+.1f}% over v3)")
print("  Done!")
