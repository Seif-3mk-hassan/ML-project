from hyper_parameter_tuning import *
import seaborn as sns
import os

os.makedirs('visuales', exist_ok=True)

# ===========================================================
# PART 4: RESULTS & COMPREHENSIVE MODEL COMPARISON
# ===========================================================
print("\n\n" + "=" * 70)
print("  PART 4: RESULTS & COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

res_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print("\n" + res_df.to_string())

v3_r2 = 0.37398
best_name = res_df.index[0]
best_r2 = res_df.loc[best_name, 'R2']
improvement = ((best_r2 - v3_r2) / v3_r2) * 100

print(f"\n  v3 best R2 (XGBoost):  {v3_r2:.5f}")
print(f"  v5 best R2 ({best_name}): {best_r2:.5f}")
print(f"  Improvement: {improvement:+.1f}%")

# -----------------------------------------------------------
# 1. Model Comparison Chart (Bar) - All metrics side by side
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
for i, metric in enumerate(['R2', 'MAE', 'RMSE', 'ACC@10']):
    ax = axes[i]
    vals = res_df[metric]
    best = vals.max() if metric in ['R2', 'ACC@10'] else vals.min()
    colors = ['#27ae60' if v == best else '#3498db' for v in vals]
    bars = ax.barh(vals.index, vals.values, color=colors, edgecolor='#2c3e50', linewidth=0.5)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=13, fontweight='bold')
plt.suptitle('Model Comparison - Predicting Assessment Score', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('visuales/model_comparison_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 2. Model Metrics Heatmap
# -----------------------------------------------------------
plt.figure(figsize=(10, max(5, len(res_df) * 0.7)))
sns.heatmap(res_df[['R2', 'MAE', 'RMSE', 'ACC@10']], annot=True, cmap='YlGnBu',
            fmt='.4f', linewidths=.5, cbar_kws={'shrink': 0.8})
plt.title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visuales/metrics_heatmap_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 3. Paired Model Comparison (R2 vs RMSE scatter)
# -----------------------------------------------------------
plt.figure(figsize=(9, 7))
for name, row in res_df.iterrows():
    plt.scatter(row['R2'], row['RMSE'], s=120, zorder=5)
    plt.annotate(name, (row['R2'], row['RMSE']),
                 textcoords="offset points", xytext=(8, 5), fontsize=9)
plt.xlabel('R² Score (higher is better)', fontsize=11)
plt.ylabel('RMSE (lower is better)', fontsize=11)
plt.title('R² vs RMSE Trade-off Across Models', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuales/r2_vs_rmse_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# Feature Importance Extraction
# -----------------------------------------------------------
print("\nExtracting feature importance...")

model_features = X_train.columns

if 'Tuned XGBoost' in best_name:
    imp_vals = tuned_xgb.feature_importances_
elif 'XGBoost' in best_name:
    imp_vals = xgb_model.feature_importances_
elif best_name == 'LightGBM':
    imp_vals = lgb_model.feature_importances_
elif best_name == 'Random Forest':
    imp_vals = rf.feature_importances_
else:
    imp_vals = np.abs(lr.coef_)

print(f"Features: {len(model_features)}, Importances: {len(imp_vals)}")

min_len = min(len(model_features), len(imp_vals))
imp = pd.DataFrame({
    'feature': model_features[:min_len],
    'importance': imp_vals[:min_len]
})
imp = imp.sort_values('importance', ascending=False)

print(f"\n  {best_name} Top 15 Feature Importances:")
print(imp.head(15).to_string(index=False))

# -----------------------------------------------------------
# 4. Feature Importance Chart (Bar)
# -----------------------------------------------------------
plt.figure(figsize=(10, 8))
top = imp.head(20).set_index('feature')['importance']
plt.barh(top.index[::-1], top.values[::-1], color='#3498db', edgecolor='#2c3e50', linewidth=0.5)
plt.xlabel('Importance', fontsize=11)
plt.title(f'{best_name} - Top 20 Feature Importances', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('visuales/feature_importance_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 5. Top Features Correlation Heatmap
# -----------------------------------------------------------
top_10_features = imp['feature'].head(10).tolist()
if len(top_10_features) > 1:
    plt.figure(figsize=(10, 8))
    corr_matrix = X_train[top_10_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap of Top 10 Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visuales/top_features_corr_heatmap_v5.png', dpi=150, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------------
# Prediction Generation (best model)
# -----------------------------------------------------------
if 'Tuned XGBoost' in best_name:
    y_best = tuned_xgb.predict(X_test)
elif 'Stacking' in best_name:
    y_best = stack.predict(X_test)
elif 'Blend' in best_name:
    y_best = best_pred  # from hyper_parameter_tuning
elif 'XGBoost' in best_name:
    y_best = xgb_model.predict(X_test)
elif best_name == 'LightGBM':
    y_best = lgb_model.predict(X_test)
elif best_name == 'Random Forest':
    y_best = rf.predict(X_test)
else:
    y_best = lr.predict(X_test)

residuals = y_test.values - y_best
s = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)

# -----------------------------------------------------------
# 6. Actual vs Predicted Scatter (best model)
# -----------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(y_test.values[s], y_best[s], alpha=0.15, s=5, color='#3498db')
plt.plot([0, 100], [0, 100], 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Actual Score', fontsize=11)
plt.ylabel('Predicted Score', fontsize=11)
plt.title(f'{best_name}: Actual vs Predicted (R²={best_r2:.4f})', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('visuales/actual_vs_predicted_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 7. Actual vs Predicted for ALL models (multi-panel)
# -----------------------------------------------------------
all_preds = {
    "Linear Regression": lr.predict(X_test),
    "Random Forest": rf.predict(X_test),
    "XGBoost": xgb_model.predict(X_test),
    "LightGBM": lgb_model.predict(X_test),
    "Stacking Ensemble": stack.predict(X_test),
    "Tuned XGBoost": tuned_xgb.predict(X_test),
}
n_models = len(all_preds)
ncols = 3
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
axes = axes.flatten()
for idx, (mname, preds) in enumerate(all_preds.items()):
    ax = axes[idx]
    ax.scatter(y_test.values[s], preds[s], alpha=0.15, s=5, color='#3498db')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=1)
    r2_val = results.get(mname, {}).get('R2', r2_score(y_test, preds))
    ax.set_title(f'{mname}\nR²={r2_val:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('Actual vs Predicted - All Models', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visuales/all_models_actual_vs_predicted_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 8. Residual Plot
# -----------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_best[s], residuals[s], alpha=0.15, s=5, color='#e74c3c')
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted Score', fontsize=11)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=11)
plt.title(f'{best_name}: Residuals vs Predictions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('visuales/residual_plot_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 9. Error Distribution Plot
# -----------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=50, color='#9b59b6')
plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title(f'{best_name}: Error Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('visuales/error_dist_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# 10. Regression Line Plot (best model)
# -----------------------------------------------------------
plt.figure(figsize=(9, 7))
sorted_idx = np.argsort(y_test.values[s])
actual_sorted = y_test.values[s][sorted_idx]
pred_sorted = y_best[s][sorted_idx]
window = max(1, len(actual_sorted) // 100)
pred_smooth = pd.Series(pred_sorted).rolling(window=window, center=True).mean()
plt.scatter(actual_sorted, pred_sorted, alpha=0.08, s=4, color='#bdc3c7', label='Individual')
plt.plot(actual_sorted, pred_smooth, color='#e74c3c', linewidth=2, label='Trend (rolling avg)')
plt.plot([0, 100], [0, 100], 'b--', linewidth=1, label='Perfect (y=x)')
plt.xlabel('Actual Score', fontsize=11)
plt.ylabel('Predicted Score', fontsize=11)
plt.title(f'{best_name}: Regression Line Plot', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('visuales/regression_line_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# ===========================================================
# Final Summary
# ===========================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"""
  Target:     score (assessment score prediction)
  Dataset:    OULAD (Open University Learning Analytics Dataset)
  Rows:       {len(df)}
  Features:   {len(feature_cols)} (after selection from {len(X_raw.columns)} original)

  Train size: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.0f}%)
  Test size:  {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.0f}%)

  Feature Engineering:
    + Interaction features (clicks*timing, edu*clicks, etc.)
    + Log-transforms for skewed VLE features
    + Prior score statistics (mean, min, max, last, std)
    + abs_days_from_deadline
    + One-hot encoding for module, presentation, assessment type

  Feature Selection:
    + Removed module dummies (high cardinality)
    + Removed high-missing features (>40%)
    + VarianceThreshold (removed near-constant)
    + Correlation pruning (|r| > 0.90)
    + 5-fold CV RandomForest importance + stability
    + Capped at {len(feature_cols)} features

  Models Trained:
    1. Linear Regression (baseline)
    2. Random Forest (n_estimators=500, max_depth=25)
    3. XGBoost (n_estimators=700, max_depth=10)
    4. LightGBM (n_estimators=700, max_depth=12)
    5. Stacking Ensemble (RF+XGB+LGB, Ridge meta)
    6. ACC@10-Optimized Blend (weighted ensemble)
    7. Tuned XGBoost (RandomizedSearchCV)

  Full Results:
""")
print(res_df.to_string())
print(f"\n  Best: {best_name} (R²={best_r2:.5f}, improvement: {improvement:+.1f}% over v3)")
print(f"  Selected features: {feature_cols}")
print("  All plots saved to visuales/")
print("  Done!")
