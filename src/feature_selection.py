import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_processor import *

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
from scipy.stats import pearsonr, kendalltau

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
# 4.5) Pearson & Kendall tau filter (TRAIN only)
#       Keep features with significant linear OR monotonic
#       correlation to target
# -----------------------------
print("  Computing Pearson & Kendall-tau correlations with target...")

PEARSON_THRESH = 0.02    # minimum |r| to keep
KENDALL_THRESH = 0.02    # minimum |tau| to keep
PVALUE_THRESH  = 0.05    # max p-value for either test

pearson_results = {}
kendall_results = {}

for col in X_train_fs.columns:
    x_col = X_train_fs[col].values
    y_col = y_train.values

    # --- Pearson (linear) ---
    try:
        p_r, p_pval = pearsonr(x_col, y_col)
    except Exception:
        p_r, p_pval = 0.0, 1.0
    pearson_results[col] = {"r": p_r, "pval": p_pval}

    # --- Kendall tau (monotonic / rank-based) ---
    try:
        # Subsample for speed if dataset is large
        if len(x_col) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(x_col), size=10000, replace=False)
            k_tau, k_pval = kendalltau(x_col[idx], y_col[idx])
        else:
            k_tau, k_pval = kendalltau(x_col, y_col)
    except Exception:
        k_tau, k_pval = 0.0, 1.0
    kendall_results[col] = {"tau": k_tau, "pval": k_pval}

pearson_df = pd.DataFrame(pearson_results).T.rename(
    columns={"r": "pearson_r", "pval": "pearson_pval"}
)
kendall_df = pd.DataFrame(kendall_results).T.rename(
    columns={"tau": "kendall_tau", "pval": "kendall_pval"}
)
corr_stats = pearson_df.join(kendall_df)

# A feature passes if EITHER Pearson or Kendall is significant & above threshold
pass_pearson = (
    (corr_stats["pearson_r"].abs() >= PEARSON_THRESH) &
    (corr_stats["pearson_pval"] <= PVALUE_THRESH)
)
pass_kendall = (
    (corr_stats["kendall_tau"].abs() >= KENDALL_THRESH) &
    (corr_stats["kendall_pval"] <= PVALUE_THRESH)
)
keep_corr_filter = corr_stats[pass_pearson | pass_kendall].index.tolist()
drop_corr_filter = [c for c in X_train_fs.columns if c not in keep_corr_filter]

X_train_fs = X_train_fs[keep_corr_filter]
X_test_fs = X_test_fs[keep_corr_filter]

print(f"  Removed by Pearson/Kendall filter: {len(drop_corr_filter)}")
print(f"  Remaining after correlation filter: {len(keep_corr_filter)}")

# Print top features by each metric
top_pearson = corr_stats.loc[keep_corr_filter].reindex(
    corr_stats.loc[keep_corr_filter, "pearson_r"].abs().sort_values(ascending=False).index
)
top_kendall = corr_stats.loc[keep_corr_filter].reindex(
    corr_stats.loc[keep_corr_filter, "kendall_tau"].abs().sort_values(ascending=False).index
)
print("\n  Top 10 by |Pearson r|:")
print(top_pearson[["pearson_r", "pearson_pval"]].head(10).to_string(float_format="%.4f"))
print("\n  Top 10 by |Kendall tau|:")
print(top_kendall[["kendall_tau", "kendall_pval"]].head(10).to_string(float_format="%.4f"))
print()

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

# --- Print final Pearson & Kendall stats for selected features ---
print("\n  Final feature correlation summary:")
final_stats = corr_stats.loc[feature_cols, ["pearson_r", "kendall_tau"]].copy()
final_stats["rf_importance"] = mean_imp[feature_cols].values
final_stats = final_stats.sort_values("rf_importance", ascending=False)
print(final_stats.to_string(float_format="%.4f"))
print()

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