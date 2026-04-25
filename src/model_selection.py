from feature_selection import *

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
