from src.data_processor import *

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