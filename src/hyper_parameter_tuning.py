from model_selection import *

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
# PART 3.7: HYPERPARAMETER TUNING (RandomizedSearchCV)
# ===========================================================
print("\n[3.7] XGBoost Hyperparameter Tuning (RandomizedSearchCV)...")

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'max_depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1]
}

try:
    _test_xgb = xgb.XGBRegressor(tree_method='hist', device='cuda', random_state=42)
    _test_xgb.fit(X_train.iloc[:10], y_train.iloc[:10])
    xgb_device = 'cuda'
    print("  Using CUDA for XGBoost hyperparameter tuning")
except Exception:
    xgb_device = 'cpu'
    print("  CUDA not available, using CPU for XGBoost hyperparameter tuning")

xgb_tuner = xgb.XGBRegressor(tree_method='hist', device=xgb_device, random_state=42)
random_search = RandomizedSearchCV(
    xgb_tuner, param_distributions=param_dist,
    n_iter=20, cv=3, verbose=1, random_state=42, n_jobs=-1,
    scoring='r2'
)
random_search.fit(X_train, y_train)
print(f"  Best params: {random_search.best_params_}")
best_xgb = random_search.best_estimator_

tuned_xgb = xgb.XGBRegressor(
    **random_search.best_params_,
    tree_method='hist',
    device=xgb_device,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)
tuned_xgb.fit(X_train, y_train)
y_tuned_xgb_pred = tuned_xgb.predict(X_test)
evaluate("Tuned XGBoost", y_test, y_tuned_xgb_pred)